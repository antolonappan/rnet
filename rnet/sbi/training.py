from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from sbi.inference.posteriors import DirectPosterior
from sbi.neural_nets import posterior_nn
from sbi.utils import BoxUniform


def load_features(path, seed=2027):
    with np.load(path) as source:
        values = {key: source[key] for key in source.files}
    data = {}
    identities = []
    for split in ('train', 'validation', 'test'):
        h = values[f'{split}_h'].astype(np.float32)
        r = values[f'{split}_r'].astype(np.float64)
        ids = values[f'{split}_sim_idx'].astype(np.int64)
        if h.shape != (len(r), 64) or len(ids) != len(r):
            raise ValueError(f'Invalid {split} feature dimensions')
        if not np.isfinite(h).all() or not np.isfinite(r).all() or np.any((r < 0) | (r > .03)):
            raise ValueError(f'Nonfinite features or out-of-prior r in {split}')
        identities.append(set(ids.tolist()))
        if len(identities[-1]) != len(ids):
            raise ValueError(f'Duplicate IDs in {split}')
        use = np.ones(len(r), dtype=bool) if split == 'test' else (r > 0) & (r < .03)
        data[split] = {'h': h[use], 'r': r[use], 'sim_idx': ids[use]}
    if any(identities[i] & identities[j] for i in range(3) for j in range(i)):
        raise ValueError('Train, validation, and grid identities overlap')
    # Reserve a quarter of the original validation set for diagnostics of this head.
    order = np.random.default_rng(seed).permutation(len(data['validation']['r']))
    n_select = int(.75 * len(order))
    if n_select < 2 or len(order) - n_select < 2:
        raise ValueError('Too few validation maps')
    val = data.pop('validation')
    data['selection'] = {key: value[order[:n_select]] for key, value in val.items()}
    data['diagnostic'] = {key: value[order[n_select:]] for key, value in val.items()}
    return data


def build_density(data, config, device):

    theta = torch.as_tensor(data['train']['r'][:, None] / .01, dtype=torch.float32)
    x = torch.as_tensor(data['train']['h'])
    builder = posterior_nn(model='nsf', hidden_features=config['hidden_features'],
                           num_transforms=config['num_transforms'], num_bins=10,
                           z_score_theta='independent', z_score_x='independent')
    return builder(theta, x).to(device)


def atomic_save(value, path):
    temp = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temp)
    os.replace(temp, path)


def fit_density(data, config, output, device):
    net = build_density(data, config, device)
    optimizer = torch.optim.Adam(net.parameters(), lr=config['learning_rate'])
    arrays = {name: (torch.as_tensor(block['r'][:, None] / .01, dtype=torch.float32, device=device),
                     torch.as_tensor(block['h'], device=device))
              for name, block in data.items() if name in ('train', 'selection')}
    checkpoint = output / 'checkpoint.pt'
    first_epoch, best, stale, history = 0, float('inf'), 0, []
    if checkpoint.is_file():
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        net.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        first_epoch, best, stale, history = saved['epoch'], saved['best'], saved['stale'], saved['history']
        torch.set_rng_state(saved['rng'].cpu())
        if device == 'cuda':
            torch.cuda.set_rng_state_all([state.cpu() for state in saved['cuda_rng']])
    size = config['batch_size']
    for epoch in range(first_epoch, config['epochs']):
        if stale >= config['patience']:
            break
        started = time.monotonic()
        theta, x = arrays['train']
        net.train()
        order = torch.randperm(len(theta), device=device)
        total = 0.
        for indices in order.split(size):
            optimizer.zero_grad(set_to_none=True)
            loss = -net.log_prob(theta[indices].unsqueeze(0), condition=x[indices]).mean()
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training NLL')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.)
            optimizer.step()
            total += loss.item() * len(indices)
        net.eval()
        vt, vx = arrays['selection']
        with torch.no_grad():
            selection_loss = sum(-net.log_prob(t.unsqueeze(0), condition=h).sum().item()
                                 for t, h in zip(vt.split(size), vx.split(size))) / len(vt)
        if not np.isfinite(selection_loss):
            raise RuntimeError('Nonfinite selection NLL')
        improved = selection_loss < best
        if improved:
            best, stale = selection_loss, 0
            atomic_save(net.state_dict(), output / 'best_state.pt')
        else:
            stale += 1
        row = {'epoch': epoch + 1, 'train_nll': total / len(theta), 'selection_nll': selection_loss,
               'seconds': time.monotonic() - started, 'best': improved}
        history.append(row)
        print(json.dumps(row), flush=True)
        atomic_save({'epoch': epoch + 1, 'model': net.state_dict(), 'optimizer': optimizer.state_dict(),
                     'best': best, 'stale': stale, 'history': history, 'rng': torch.get_rng_state(),
                     'cuda_rng': torch.cuda.get_rng_state_all() if device == 'cuda' else []}, checkpoint)
        (output / 'history.json').write_text(json.dumps(history, indent=2) + '\n')
    net.load_state_dict(torch.load(output / 'best_state.pt', map_location=device, weights_only=True))
    net.eval()
    return net


def sample_predictions(posterior, block, output, label, device, samples, seed):
    destination = output / f'{label}_posterior_samples.npz'
    if destination.is_file():
        with np.load(destination) as cache:
            if not np.array_equal(cache['sim_idx'], block['sim_idx']) or not np.array_equal(cache['r'], block['r']):
                raise ValueError('Cached posterior identities differ')
            return cache['samples']
    torch.manual_seed(seed)
    chunks = []
    for start in range(0, len(block['r']), 16):
        x = torch.as_tensor(block['h'][start:start + 16], device=device)
        with torch.no_grad():
            draws = posterior.sample_batched((samples,), x=x, show_progress_bars=False)
        if draws.shape != (samples, len(x), 1):
            raise ValueError(f'Unexpected posterior sample shape {draws.shape}')
        chunks.append(draws[..., 0].cpu().numpy().T * .01)
        if start % 160 == 0:
            print(f'{label}: {start + len(x)}/{len(block["r"])}', flush=True)
    draws = np.concatenate(chunks)
    np.savez_compressed(destination, samples=draws, sim_idx=block['sim_idx'], r=block['r'])
    return draws


def summarize(block, draws, path):
    truth = block['r']
    mean = draws.mean(axis=1)
    std = draws.std(axis=1)
    q16, median, q84, upper95 = np.quantile(draws, [.16, .5, .84, .95], axis=1)
    rank = np.mean(draws < truth[:, None], axis=1)
    with path.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['sim_idx', 'r_true', 'r_mean', 'r_median', 'sigma_posterior',
                         'q16', 'q84', 'upper95', 'posterior_rank_fraction'])
        writer.writerows(zip(block['sim_idx'], truth, mean, median, std, q16, q84, upper95, rank))

    def metric(mask):
        residual = mean[mask] - truth[mask]
        return {'n': int(mask.sum()), 'bias': float(np.mean(residual)),
                'rmse': float(np.sqrt(np.mean(residual**2))),
                'mean_posterior_sigma': float(np.mean(std[mask])),
                'central_68_coverage': float(np.mean((truth[mask] >= q16[mask]) & (truth[mask] <= q84[mask]))),
                'symmetric_sigma_coverage': float(np.mean(np.abs(residual) <= .99446 * std[mask]))}
    return {'all': metric(np.ones(len(truth), dtype=bool)),
            'r_le_002': metric(truth <= .02),
            'per_r': {str(r): metric(truth == r) for r in np.unique(truth)} if len(np.unique(truth)) <= 20 else {}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--learning-rate', type=float, default=5e-4)
    parser.add_argument('--hidden-features', type=int, default=64)
    parser.add_argument('--num-transforms', type=int, default=3)
    parser.add_argument('--posterior-samples', type=int, default=2048)
    parser.add_argument('--seed', type=int, default=2027)
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.batch_size, args.posterior_samples) <= 0:
        raise ValueError('Epochs, patience, batch size, and posterior samples must be positive')
    torch.set_num_threads(4)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this job but is unavailable')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    data = load_features(args.features, args.seed)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(features_sha256=hashlib.sha256(args.features.read_bytes()).hexdigest(),
                  prior='Uniform(0, 0.03); no point mass at zero',
                  split_counts={key: len(value['r']) for key, value in data.items()},
                  versions={key: version(key) for key in ['sbi', 'torch', 'numpy', 'scipy', 'nflows', 'pyknos', 'zuko']})
    config_file = args.output / 'config.json'
    if config_file.is_file() and json.loads(config_file.read_text()) != config:
        raise ValueError('Existing run configuration differs; choose a new output directory')
    config_file.write_text(json.dumps(config, indent=2) + '\n')
    if (args.output / 'complete.json').is_file():
        print('Completed run exists; nothing to retrain.', flush=True)
        return
    print(json.dumps(config, indent=2), flush=True)
    started = time.monotonic()
    net = fit_density(data, config, args.output, args.device)
    prior = BoxUniform(low=torch.tensor([0.], device=args.device),
                       high=torch.tensor([3.], device=args.device), device=args.device)
    posterior = DirectPosterior(posterior_estimator=net, prior=prior, device=args.device)
    results = {'note': 'Frozen-encoder NPE; not an end-to-end map-level flow. No sigma rescaling.',
               'boundary_note': 'Central intervals of a continuous bounded posterior cannot cover exact r=0 or r=0.03. Report upper limits at the null; symmetric sigma coverage is a separate diagnostic.',
               'diagnostic_note': 'Diagnostic maps are withheld from flow fitting and selection, but the original CNN encoder used their parent validation split for early stopping.'}
    for label in ['diagnostic', 'test']:
        draws = sample_predictions(posterior, data[label], args.output, label, args.device,
                                   args.posterior_samples, args.seed + (1 if label == 'test' else 0))
        results[label] = summarize(data[label], draws, args.output / f'{label}_predictions.csv')
        if label == 'diagnostic':
            # Reuse posterior draws for sbi's rank and data-averaged posterior checks.
            from sbi.diagnostics import check_sbc
            torch.manual_seed(args.seed + 2)
            ranks = np.sum(draws < data[label]['r'][:, None], axis=1)
            checks = check_sbc(torch.tensor(ranks[:, None], dtype=torch.float32),
                               torch.rand((len(ranks), 1)) * .03,
                               torch.as_tensor(draws[:, :1]),
                               num_posterior_samples=args.posterior_samples)
            results['exploratory_sbc'] = {key: value.tolist() for key, value in checks.items()}
            results['diagnostic_central_coverage'] = {}
            for level in [.5, .68, .9, .95]:
                lo, hi = np.quantile(draws, [(1-level)/2, (1+level)/2], axis=1)
                results['diagnostic_central_coverage'][str(level)] = float(
                    np.mean((data[label]['r'] >= lo) & (data[label]['r'] <= hi)))
    results['elapsed_seconds'] = time.monotonic() - started
    (args.output / 'summary.json').write_text(json.dumps(results, indent=2) + '\n')
    (args.output / 'complete.json').write_text(json.dumps({'completed': True, 'elapsed_seconds': results['elapsed_seconds']}) + '\n')
    print(json.dumps(results, indent=2), flush=True)


if __name__ == '__main__':
    main()
