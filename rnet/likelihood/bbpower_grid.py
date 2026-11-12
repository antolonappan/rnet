from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


FREQUENCIES = (27, 39, 93, 145, 225, 280)
VERSION = 1
FIT_VERSION = 3


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def analysis_mask(
    mode: str, apodized: np.ndarray, scientific: np.ndarray, hits: np.ndarray,
    *, fraction: float = 0.5, weight_power: float = 1.0, weight_floor: float = 1.0e-3,
) -> tuple[np.ndarray, dict]:
    """Build the full, deep-region, or inverse-white-noise mask in RING order."""
    apodized = np.asarray(apodized, dtype=np.float64)
    scientific = np.asarray(scientific, dtype=np.float64)
    hits = np.asarray(hits, dtype=np.float64)
    if apodized.shape != scientific.shape or scientific.shape != hits.shape:
        raise ValueError("Apodized mask, scientific mask, and hits must have the same shape")
    valid = np.isfinite(scientific) & (scientific > 0) & np.isfinite(hits) & (hits > 0)
    if not np.any(valid):
        raise ValueError("Scientific mask and hits have no observed pixels in common")
    if mode == "full":
        mask = np.where(np.isfinite(apodized) & (apodized > 0), apodized, 0.0)
        threshold = None
        retained = 1.0
    elif mode == "high_hits":
        if not 0 < fraction <= 1:
            raise ValueError("fraction must lie in (0, 1]")
        indices = np.flatnonzero(valid)
        n_keep = max(1, int(np.ceil(fraction * len(indices))))
        order = np.lexsort((indices, -hits[indices]))
        kept = indices[order[:n_keep]]
        mask = np.zeros_like(hits)
        mask[kept] = 1.0
        threshold = float(np.min(hits[kept]))
        retained = float(n_keep / len(indices))
    elif mode == "hits_weighted":
        if not np.isfinite(weight_power) or weight_power < 0:
            raise ValueError("weight_power must be finite and non-negative")
        if not 0 < weight_floor <= 1:
            raise ValueError("weight_floor must lie in (0, 1]")
        mask = np.zeros_like(hits)
        mask[valid] = np.maximum(hits[valid] / np.max(hits[valid]), weight_floor) ** weight_power
        threshold = None
        retained = 1.0
    else:
        raise ValueError("mode must be 'full', 'high_hits', or 'hits_weighted'")
    second = np.mean(mask**2)
    fourth = np.mean(mask**4)
    summary = {
        "mode": mode,
        "valid_pixels": int(np.count_nonzero(valid)),
        "retained_pixels": int(np.count_nonzero(mask)),
        "retained_fraction": retained,
        "hit_threshold": threshold,
        "fsky_2": float(second),
        "fsky_effective": float(second**2 / fourth) if fourth > 0 else 0.0,
    }
    return mask, summary


def _sed_factors(frequencies: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Dust MBB and synchrotron K_RJ power law, both returned in K_CMB."""
    from scipy.constants import h, k

    nu = np.asarray(frequencies, dtype=np.float64)
    tcmb, tdust = 2.7255, 19.6

    def dblackbody_dt(freq: np.ndarray) -> np.ndarray:
        x = h * (freq * 1.0e9) / (k * tcmb)
        return freq**4 * np.exp(x) / np.expm1(x) ** 2

    def blackbody(freq: np.ndarray) -> np.ndarray:
        x = h * (freq * 1.0e9) / (k * tdust)
        return freq**3 / np.expm1(x)

    def rj_to_cmb(freq: np.ndarray) -> np.ndarray:
        x = h * (freq * 1.0e9) / (k * tcmb)
        return np.expm1(x) ** 2 / (x**2 * np.exp(x))

    dust = (nu / 353.0) ** 1.59 * blackbody(nu) / blackbody(np.asarray(353.0))
    dust *= dblackbody_dt(np.asarray(353.0)) / dblackbody_dt(nu)
    sync = (nu / 23.0) ** (-3.0) * rj_to_cmb(nu) / rj_to_cmb(np.asarray(23.0))
    return dust, sync


def foreground_design(ells: np.ndarray, frequencies=FREQUENCIES) -> np.ndarray:
    """Fixed-shape dust, synchrotron, and cross templates in pair-major C_ell."""
    ell = np.asarray(ells, dtype=np.float64)
    dust_sed, sync_sed = _sed_factors(np.asarray(frequencies, dtype=np.float64))
    to_cl = 2.0 * np.pi / (ell * (ell + 1.0))
    dust_cl = to_cl * (ell / 80.0) ** (-0.2)
    sync_cl = to_cl * (ell / 80.0) ** (-0.4)
    cross_cl = np.sqrt(dust_cl * sync_cl)
    rows = []
    for i in range(len(frequencies)):
        for j in range(i, len(frequencies)):
            rows.extend(zip(
                dust_sed[i] * dust_sed[j] * dust_cl,
                sync_sed[i] * sync_sed[j] * sync_cl,
                (dust_sed[i] * sync_sed[j] + sync_sed[i] * dust_sed[j]) * cross_cl,
            ))
    return np.asarray(rows, dtype=np.float64)


def fit_linear_posterior(
    data: np.ndarray, baseline: np.ndarray, design: np.ndarray,
    covariance: np.ndarray, r_bounds: tuple[float, float] = (0.0, 0.03),
) -> dict[str, np.ndarray]:
    """Marginalize unbounded linear nuisances and truncate the r posterior."""
    from scipy.linalg import cho_factor, cho_solve
    from scipy.stats import truncnorm

    values = np.atleast_2d(np.asarray(data, dtype=np.float64))
    baseline = np.asarray(baseline, dtype=np.float64)
    design = np.asarray(design, dtype=np.float64)
    if values.shape[1] != baseline.size or design.shape[0] != baseline.size:
        raise ValueError("Data, baseline, and design have incompatible dimensions")
    if not r_bounds[0] < r_bounds[1]:
        raise ValueError("r_bounds must be increasing")
    chol = cho_factor(np.asarray(covariance, dtype=np.float64), check_finite=True)
    inverse_design = cho_solve(chol, design)
    norms = np.sqrt(np.diag(design.T @ inverse_design))
    if not np.all(np.isfinite(norms)) or np.any(norms <= 0):
        raise ValueError("Likelihood design contains a degenerate component")
    normalized_design = design / norms
    inverse_design = inverse_design / norms
    fisher = normalized_design.T @ inverse_design
    fisher_chol = cho_factor(fisher, check_finite=True)
    parameter_covariance = cho_solve(fisher_chol, np.eye(fisher.shape[0]))
    projection = parameter_covariance @ inverse_design.T
    normalized_parameters = (values - baseline) @ projection.T
    parameters = normalized_parameters / norms
    sigma_unbounded = float(np.sqrt(parameter_covariance[0, 0]) / norms[0])
    r_unbounded = parameters[:, 0]
    lower = (r_bounds[0] - r_unbounded) / sigma_unbounded
    upper = (r_bounds[1] - r_unbounded) / sigma_unbounded
    quantiles = truncnorm.ppf(
        np.asarray([0.16, 0.5, 0.84])[:, None],
        lower[None, :], upper[None, :],
        loc=r_unbounded[None, :], scale=sigma_unbounded,
    )
    posterior_mean = truncnorm.mean(lower, upper, loc=r_unbounded, scale=sigma_unbounded)
    residual = values - baseline - normalized_parameters @ normalized_design.T
    chi2 = np.einsum("ij,ij->i", residual, cho_solve(chol, residual.T).T)
    return {
        "r_predicted": posterior_mean,
        "r_median": quantiles[1],
        "sigma_r": 0.5 * (quantiles[2] - quantiles[0]),
        "r_lower": quantiles[0],
        "r_upper": quantiles[2],
        "r_unbounded": r_unbounded,
        "sigma_unbounded": np.full(len(values), sigma_unbounded),
        "nuisance_parameters": parameters[:, 1:],
        "chi2_min_unbounded": chi2,
    }


class BBPowerGridLikelihood:
    """NaMaster bandpowers and simulation-calibrated Gaussian r likelihood."""

    def __init__(
        self, base: Path | str, *, ell_min: int = 30, ell_max: int = 300,
        bin_width: int = 30, n_covariance: int = 500, n_template: int = 500,
        n_test_per_r: int = 100, seed: int = 2027,
        mask_mode: str = "full", high_hits_fraction: float = 0.5,
        weight_power: float = 1.0, weight_floor: float = 1.0e-3,
    ) -> None:
        import healpy as hp
        import pymaster as nmt

        self.base = Path(base).expanduser().resolve()
        self.suite = self.base / "unified_sat_parallel_v1"
        self.cache_dir = self.base / "likelihood" / "bbpower_grid"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.grid_manifest = _read_json(self.suite / "grid" / "manifest.json")
        self.train_manifest = _read_json(self.suite / "uniform_train" / "manifest.json")
        if self.grid_manifest["settings_hash"] != self.train_manifest["settings_hash"]:
            raise ValueError("Training and grid maps have different simulation settings")
        if tuple(float(x) for x in self.grid_manifest["freq_values"]) != FREQUENCIES:
            raise ValueError("Expected the six SAT channels in their saved order")
        self.covariance_entries = [e for e in self.grid_manifest["entries"] if e["role"] == "covariance"]
        self.all_test_entries = [e for e in self.grid_manifest["entries"] if e["role"] == "test"]
        if len(self.covariance_entries) != 500 or any(float(e["r"]) != 0 for e in self.covariance_entries):
            raise ValueError("Expected 500 independent r=0 covariance maps")
        if not 10 <= n_covariance <= 500:
            raise ValueError("n_covariance must lie between 10 and 500")
        self.covariance_entries = self.covariance_entries[:n_covariance]
        if set(int(e["sim_idx"]) for e in self.covariance_entries) & set(int(e["sim_idx"]) for e in self.all_test_entries):
            raise ValueError("Covariance and test realizations overlap")
        if not 1 <= n_test_per_r <= 100:
            raise ValueError("n_test_per_r must be between 1 and 100")
        grouped: dict[float, list[dict]] = {}
        for entry in self.all_test_entries:
            grouped.setdefault(float(entry["r"]), []).append(entry)
        self.test_entries = [entry for r in sorted(grouped)
                             for entry in sorted(grouped[r], key=lambda e: int(e["sim_idx"]))[:n_test_per_r]]
        self.test_positions = {int(e["sim_idx"]): i for i, e in enumerate(self.all_test_entries)}
        if len(self.test_positions) != len(self.all_test_entries):
            raise ValueError("Grid test simulation IDs must be unique")
        train_entries = self.train_manifest["entries"]
        if not 10 <= n_template <= len(train_entries):
            raise ValueError("n_template must lie between 10 and the uniform training pool size")
        selected = np.random.default_rng(seed).choice(len(train_entries), n_template, replace=False)
        self.template_entries = [train_entries[int(i)] for i in selected]
        if set(int(e["sim_idx"]) for e in self.template_entries) & set(int(e["sim_idx"]) for e in self.test_entries):
            raise ValueError("Tensor template and test realizations overlap")

        with np.load(self.suite / "geometry.npz") as geometry:
            self.pixel_idx = np.asarray(geometry["pixel_idx"], dtype=np.int64)
            self.mask_state = np.asarray(geometry["mask_state"], dtype=np.uint8)
            self.nside = int(geometry["nside"])
        self.npix = hp.nside2npix(self.nside)
        apodized = np.asarray(hp.read_map(str(self.base / "data" / "mask_apo.fits")), dtype=np.float64)
        scientific = np.asarray(hp.read_map(str(self.base / "data" / "mask.fits")), dtype=np.float64)
        hits = np.asarray(hp.read_map(str(self.base / "data" / "hitsmap.fits.gz")), dtype=np.float64)
        if hp.get_nside(hits) != self.nside:
            hits = hp.ud_grade(hits, nside_out=self.nside)
        if len(apodized) != self.npix or len(scientific) != self.npix:
            raise ValueError("Analysis masks have the wrong resolution")
        self.mask, self.mask_info = analysis_mask(
            mask_mode, apodized, scientific, hits, fraction=high_hits_fraction,
            weight_power=weight_power, weight_floor=weight_floor,
        )
        if not 0 < ell_min < ell_max <= 3 * self.nside - 1 or bin_width <= 0:
            raise ValueError("Invalid multipole cuts or bin width")
        if (ell_max - ell_min) % bin_width:
            raise ValueError("Multipole range must be divisible by bin_width")
        lows = np.arange(ell_min, ell_max, bin_width)
        highs = lows + bin_width
        self.binning = nmt.NmtBin.from_edges(lows, highs, is_Dell=False)
        self.ells = self.binning.get_effective_ells()
        self.lmax_field = self.binning.lmax
        self.pairs = [(i, j) for i in range(6) for j in range(i, 6)]
        self.n_data = len(self.pairs) * len(self.ells)
        signature = {
            "version": VERSION, "settings_hash": self.grid_manifest["settings_hash"],
            "mask_sha256": hashlib.sha256(self.mask.tobytes()).hexdigest(),
            "bin_edges": [lows.tolist(), highs.tolist()], "n_template": n_template,
            "n_covariance": n_covariance,
            "template_ids": [int(e["sim_idx"]) for e in self.template_entries],
        }
        self.signature = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:16]
        self.test_signature = hashlib.sha256(json.dumps(
            [int(e["sim_idx"]) for e in self.test_entries]
        ).encode()).hexdigest()[:12]
        self.full_test_signature = hashlib.sha256(json.dumps(
            [int(e["sim_idx"]) for e in self.all_test_entries]
        ).encode()).hexdigest()[:12]
        self.workspace = nmt.NmtWorkspace()
        workspace_path = self.cache_dir / f"workspace_{self.signature}.fits"
        if workspace_path.is_file():
            self.workspace.read_from(str(workspace_path))
        else:
            zero_field = nmt.NmtField(self.mask, [np.zeros(self.npix)],
                                      lmax=self.lmax_field, lmax_mask=self.lmax_field)
            self.workspace.compute_coupling_matrix(zero_field, zero_field, self.binning)
            self.workspace.write_to(str(workspace_path))
        self.prepared = False

    def _bandpowers(self, entry: dict, variant: str) -> np.ndarray:
        import h5py

        with h5py.File(str(entry["h5_file"]), "r") as shard:
            row = int(entry["h5_row"])
            if not shard["done"][row] or int(shard["sim_idx"][row]) != int(entry["sim_idx"]):
                raise ValueError(f"Incomplete or mismatched map: {entry['sim_idx']}")
            maps = np.asarray(shard[variant][row], dtype=np.float64)
        return self.bandpowers_selected(maps)

    def bandpowers_selected(self, maps: np.ndarray) -> np.ndarray:
        """NaMaster C_ell bandpowers for six saved NEST-order selected-pixel B maps."""
        import healpy as hp
        import pymaster as nmt

        if maps.shape != (6, len(self.pixel_idx)):
            raise ValueError("Stored map shape does not match six-band geometry")
        fields = []
        for selected in maps:
            full_nest = np.zeros(self.npix, dtype=np.float64)
            full_nest[self.pixel_idx] = selected
            ring = hp.reorder(full_nest, n2r=True)
            fields.append(nmt.NmtField(self.mask, [ring], lmax=self.lmax_field,
                                       lmax_mask=self.lmax_field))
        spectra = [self.workspace.decouple_cell(nmt.compute_coupled_cell(fields[i], fields[j]))[0]
                   for i, j in self.pairs]
        return np.asarray(spectra, dtype=np.float64).reshape(-1)

    def _cached_bandpowers(self, name: str, entries: list[dict], variant: str) -> np.ndarray:
        from tqdm.auto import tqdm

        path = self.cache_dir / f"{name}_{variant}_{self.signature}.npy"
        shape = (len(self.all_test_entries) if name == "test" else len(entries), self.n_data)
        if path.is_file():
            values = np.load(path, mmap_mode="r+")
            if values.shape != shape:
                raise ValueError(f"Cached spectra have wrong shape: {path}")
        else:
            values = np.lib.format.open_memmap(path, mode="w+", dtype=np.float64, shape=shape)
            values[:] = np.nan
            if name == "test":
                legacy = self.cache_dir / f"test_{variant}_{self.signature}_{self.full_test_signature}.npy"
                if legacy.is_file():
                    previous = np.load(legacy, mmap_mode="r")
                    if previous.shape == shape:
                        selected_order = sorted(self.all_test_entries, key=lambda e: (float(e["r"]), int(e["sim_idx"])))
                        for row, entry in zip(previous, selected_order):
                            values[self.test_positions[int(entry["sim_idx"])]] = row
            values.flush()
        positions = ([self.test_positions[int(e["sim_idx"])] for e in entries]
                     if name == "test" else list(range(len(entries))))
        incomplete = [i for i, pos in enumerate(positions) if not np.isfinite(values[pos]).all()]
        for count, index in enumerate(tqdm(incomplete, desc=f"{name} {variant}", unit="map"), 1):
            values[positions[index]] = self._bandpowers(entries[index], variant)
            if count % 10 == 0:
                values.flush()
        values.flush()
        return np.asarray(values[positions])

    def prepare(self) -> "BBPowerGridLikelihood":
        from sklearn.covariance import LedoitWolf

        path = self.cache_dir / f"prepared_{self.signature}.npz"
        if path.is_file():
            with np.load(path) as saved:
                if "fg_baseline" in saved:
                    self.baseline = saved["baseline"]
                    self.fg_baseline = saved["fg_baseline"]
                    self.tensor_template = saved["tensor_template"]
                    self.covariances = {name: saved[f"cov_{name}"] for name in ("nofg", "fg_d11s6")}
                    self.shrinkage = {name: float(saved[f"shrinkage_{name}"]) for name in self.covariances}
                    self.prepared = True
                    return self

        nofg_cov = self._cached_bandpowers("covariance", self.covariance_entries, "nofg")
        fg_cov = self._cached_bandpowers("covariance", self.covariance_entries, "fg_d11s6")
        template_maps = self._cached_bandpowers("tensor_template", self.template_entries, "nofg")
        self.baseline = nofg_cov.mean(axis=0)
        self.fg_baseline = fg_cov.mean(axis=0)
        r = np.asarray([float(entry["r"]) for entry in self.template_entries])
        design = np.column_stack([np.ones(len(r)), r])
        self.tensor_template = np.linalg.lstsq(design, template_maps, rcond=None)[0][1]
        self.covariances = {}
        self.shrinkage = {}
        for name, samples in (("nofg", nofg_cov), ("fg_d11s6", fg_cov)):
            scale = samples.std(axis=0, ddof=1)
            if not np.all(scale > 0):
                raise ValueError(f"{name} covariance has zero-variance bandpowers")
            standardized = (samples - samples.mean(axis=0)) / scale
            estimate = LedoitWolf().fit(standardized)
            covariance = estimate.covariance_ * scale[:, None] * scale[None, :]
            if np.linalg.eigvalsh(estimate.covariance_)[0] <= 0:
                raise ValueError(f"{name} covariance is not positive definite")
            self.covariances[name] = covariance
            self.shrinkage[name] = float(estimate.shrinkage_)
        np.savez_compressed(
            path, baseline=self.baseline, fg_baseline=self.fg_baseline,
            tensor_template=self.tensor_template,
            cov_nofg=self.covariances["nofg"], cov_fg_d11s6=self.covariances["fg_d11s6"],
            shrinkage_nofg=self.shrinkage["nofg"],
            shrinkage_fg_d11s6=self.shrinkage["fg_d11s6"],
        )
        self.prepared = True
        return self

    def diagnostics(self) -> dict:
        if not self.prepared:
            raise RuntimeError("Call prepare() first")
        return {
            "n_covariance": len(self.covariance_entries),
            "n_template": len(self.template_entries),
            "n_test": len(self.test_entries),
            "n_bandpowers": self.n_data,
            "mask": self.mask_info,
            "mask_fsky_2": self.mask_info["fsky_2"],
            "shrinkage": self.shrinkage,
            "condition_correlation": {
                name: float(np.linalg.cond(cov / np.sqrt(np.outer(np.diag(cov), np.diag(cov)))))
                for name, cov in self.covariances.items()
            },
        }

    def fit_grid(self, variant: str, r_bounds=(0.0, 0.03)):
        import pandas as pd

        if not self.prepared:
            raise RuntimeError("Call prepare() first")
        if variant not in self.covariances:
            raise ValueError("variant must be 'nofg' or 'fg_d11s6'")
        suffix = f"v{FIT_VERSION}_{variant}_{self.signature}_{self.test_signature}_{r_bounds[0]:g}_{r_bounds[1]:g}"
        path = self.cache_dir / f"fits_{suffix}.csv"
        if path.is_file():
            result = pd.read_csv(path)
            expected = np.asarray([int(e["sim_idx"]) for e in self.test_entries])
            if np.array_equal(result["sim_idx"].to_numpy(dtype=int), expected):
                return result
            raise ValueError(f"Cached likelihood fit IDs do not match: {path}")
        samples = self._cached_bandpowers("test", self.test_entries, variant)
        columns = [self.tensor_template]
        if variant == "fg_d11s6":
            columns.extend(foreground_design(self.ells).T)
        design = np.column_stack(columns)
        baseline = self.baseline if variant == "nofg" else self.fg_baseline
        fit = fit_linear_posterior(samples, baseline, design,
                                   self.covariances[variant], r_bounds)
        result = pd.DataFrame({
            "sim_idx": [int(e["sim_idx"]) for e in self.test_entries],
            "r_true": [float(e["r"]) for e in self.test_entries],
            "r_predicted": fit["r_predicted"],
            "r_median": fit["r_median"],
            "sigma_r": fit["sigma_r"],
            "r_lower": fit["r_lower"], "r_upper": fit["r_upper"],
            "r_unbounded": fit["r_unbounded"],
            "sigma_unbounded": fit["sigma_unbounded"],
            "chi2_min_unbounded": fit["chi2_min_unbounded"],
        })
        if variant == "fg_d11s6":
            for index, name in enumerate(("delta_A_d", "delta_A_s", "delta_A_ds")):
                result[name] = fit["nuisance_parameters"][:, index]
        result.to_csv(path, index=False)
        return result
