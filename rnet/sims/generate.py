from __future__ import annotations

import argparse
from pathlib import Path

from .suite import build_training_manifest
from .unified import UnifiedSATSimulator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basedir", type=Path, required=True,
                        help="Directory containing the existing simulation inputs")
    parser.add_argument("--output-root", type=Path,
                        help="Suite root; defaults to BASEDIR/unified_sat_v3")
    parser.add_argument("--pool", required=True,
                        choices=("all", "grid", "none", *UnifiedSATSimulator.POOLS),
                        help="One resumable pool, the fixed grid, or the full suite")
    parser.add_argument("--n-sims", type=int,
                        help="Generate only this many rows of the selected pool")
    parser.add_argument("--build-recipes", action="store_true",
                        help="Create both training manifests for each requested size")
    parser.add_argument("--sizes", type=int, nargs="+",
                        default=(5_000, 10_000, 15_000, 20_000))
    args = parser.parse_args()
    if args.n_sims is not None and args.pool not in UnifiedSATSimulator.POOLS:
        parser.error("--n-sims requires a named training or validation pool")
    if args.pool == "none" and not args.build_recipes:
        parser.error("--pool none requires --build-recipes")

    root = args.output_root or args.basedir / "unified_sat_v3"
    if args.pool != "none":
        sim = UnifiedSATSimulator(args.basedir, output_root=root)
        if args.pool == "all":
            paths = sim.run_full_suite()
            for name, path in paths.items():
                print(f"{name}: {path}")
        elif args.pool == "grid":
            print(sim.run_grid(n_per_r=100, n_zero_covariance=500))
        else:
            print(sim.run_pool(args.pool, n_sims=args.n_sims))
    if args.build_recipes:
        for size in args.sizes:
            for enriched in (False, True):
                print(build_training_manifest(root, size, enriched=enriched))


if __name__ == "__main__":
    main()
