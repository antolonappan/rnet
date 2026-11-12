"""Paired six-band SAT B-mode suites with a common Q/U processing route."""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from pathlib import Path
from typing import Sequence

import h5py
import healpy as hp
import numpy as np
import pymaster as nmt
import pysm3.units as u
from pysm3 import utils as pysm_utils

from .cmb import CMBLensing
from .foreground_suite import ForegroundAugmentedSimulator
from .noise import Noise


class UnifiedSATSimulator:
    """Save matched NOFG and stochastic d11/s6 maps at nside 128.

    CMB is lensed at nside 1024. Noise and foregrounds are made at nside 128.
    Both products use the same NaMaster pure-B operator, transfer, and mask.
    Completed HDF5 rows are reusable after an interrupted run.
    """

    FREQUENCIES = (27, 39, 93, 145, 225, 280)
    GRID_R = (0.0, 0.0005, 0.001, 0.002, 0.003, 0.004, 0.005,
              0.0075, 0.01, 0.015, 0.02, 0.03)
    POOLS = {
        "uniform_train": (20_000, 10_000_000),
        "uniform_validation": (2_000, 11_000_000),
        "low_train": (5_000, 12_000_000),
        "low_validation": (500, 13_000_000),
        "zero_train": (300, 14_000_000),
        "zero_validation": (100, 15_000_000),
    }
    LOW_R_MAX = 0.006

    def __init__(
        self,
        basedir: Path | str,
        output_root: Path | str | None = None,
        r_prior: tuple[float, float] = (0.0, 0.03),
        nside_cmb: int = 1024,
        nside_out: int = 128,
        lmax_out: int = 300,
        lmin: int = 30,
        shard_size: int = 64,
        seed: int = 2029,
        lensing_threads: int = 0,
    ) -> None:
        self.basedir = Path(basedir).expanduser().resolve()
        self.root = (Path(output_root).expanduser().resolve() if output_root
                     else self.basedir / "unified_sat_v3")
        self.root.mkdir(parents=True, exist_ok=True)
        self.r_prior = tuple(map(float, r_prior))
        self.nside_cmb = int(nside_cmb)
        self.nside_out = int(nside_out)
        self.lmax_out = int(lmax_out)
        self.lmin = int(lmin)
        self.shard_size = int(shard_size)
        self.seed = int(seed)
        self.lensing_threads = int(lensing_threads)
        if self.lensing_threads < 0:
            raise ValueError("lensing_threads must be >= 0")
        if not (0 <= self.r_prior[0] < self.r_prior[1]):
            raise ValueError("r_prior must be increasing and nonnegative")
        if not (0 <= self.lmin <= self.lmax_out <= 3 * self.nside_out - 1):
            raise ValueError("output lmax is inconsistent with output nside")
        if self.nside_cmb < self.nside_out or self.shard_size < 1:
            raise ValueError("invalid resolution or shard size")

        data = self.basedir / "data"
        required = ("camb.ini", "hitsmap.fits.gz", "mask.fits",
                    "mask_apo.fits", "mask_padded.fits")
        if data.exists():
            missing = [name for name in required if not (data / name).is_file()]
            if missing:
                raise FileNotFoundError(f"Incomplete simulation inputs in {data}: {missing}")
        else:
            data = Path(__file__).resolve().parents[1] / "data"
        inputs_manifest = self.basedir / "inputs" / "inputs_manifest.json"
        if inputs_manifest.is_file():
            payload = json.loads(inputs_manifest.read_text(encoding="utf-8"))
            pysm_root = self.basedir / "inputs" / "pysm_data"
            missing = [name for name in payload["pysm_files_sha256"]
                       if not (pysm_root / name).is_file()]
            if missing:
                raise FileNotFoundError(f"Missing staged PySM inputs: {missing}")
            os.environ["PYSM_LOCAL_DATA"] = str(pysm_root)
        self.mask_apo = self._read_mask(data / "mask_apo.fits")
        self.mask_bin = (self._read_mask(data / "mask.fits") > 0).astype(np.float64)
        padded = self._read_mask(data / "mask_padded.fits", nest=True)
        self.pixel_idx = np.flatnonzero(padded > 0).astype(np.int64)
        self.mask_state = hp.reorder(self.mask_bin, r2n=True)[self.pixel_idx].astype(np.uint8)
        self.transfer = np.ones(self.lmax_out + 1)
        self.transfer[:self.lmin] = 0
        self.cmb = CMBLensing(nside=self.nside_cmb, basedir=self.basedir, seed=1234)
        self.noise = Noise(atm_noise=True, decon=True, inhom=True,
                           nside=self.nside_out, lmax=self.lmax_out,
                           hits_path=data / "hitsmap.fits.gz")
        self.fg = ForegroundAugmentedSimulator(
            basedir=self.basedir, output_root=self.root / "foreground_cache",
            fg_preset=("d11", "s6"), nside=self.nside_out,
            foreground_nside=self.nside_out, lmax=self.lmax_out,
            lmin=self.lmin, seed_base=50000, map2alm_lsq_maxiter=0,
        )
        settings = {
            "version": 2, "nside_cmb": self.nside_cmb,
            "nside_out": self.nside_out, "lmax_out": self.lmax_out,
            "lmin": self.lmin, "r_prior": self.r_prior,
            "seed": self.seed, "frequencies": self.FREQUENCIES,
            "shard_size": self.shard_size,
            "foreground": ("d11", "s6"), "foreground_seed_base": 50000,
            "b_composition": "separate_pure_b_then_add",
            "cmb_seed_base": 1234, "noise_seed_base": 7777,
            "noise": {"atm": True, "decon": True, "inhom": True},
            "mask_sha256": hashlib.sha256(
                self.mask_apo.tobytes() + self.mask_bin.tobytes()
                + self.pixel_idx.tobytes()).hexdigest(),
        }
        if inputs_manifest.is_file():
            settings["inputs_manifest_sha256"] = hashlib.sha256(
                inputs_manifest.read_bytes()).hexdigest()
        self.settings = settings
        self.settings_hash = hashlib.sha256(
            json.dumps(settings, sort_keys=True).encode()).hexdigest()
        self._save_geometry()

    def _read_mask(self, path: Path, nest: bool = False) -> np.ndarray:
        arr = np.asarray(hp.read_map(str(path), field=0, nest=nest), dtype=np.float64)
        if hp.get_nside(arr) != self.nside_out:
            arr = hp.ud_grade(arr, nside_out=self.nside_out, order_in="NEST" if nest else "RING",
                              order_out="NEST" if nest else "RING")
        return arr

    def _save_geometry(self) -> None:
        path = self.root / "geometry.npz"
        if path.exists():
            with np.load(path) as saved:
                if not np.array_equal(saved["pixel_idx"], self.pixel_idx):
                    raise ValueError("Stored geometry differs from current selected pixels")
            return
        temporary = path.with_name(
            f"{path.stem}.{os.getpid()}.{uuid.uuid4().hex}.partial.npz"
        )
        try:
            np.savez(temporary, pixel_idx=self.pixel_idx, mask_state=self.mask_state,
                     nside=np.int64(self.nside_out))
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def _cmb_qu(self, idx: int, r: float) -> tuple[np.ndarray, np.ndarray]:
        high = self.cmb.get_lensed_qu(idx, r=r, nthreads=self.lensing_threads)
        alms = hp.map2alm_spin(high, 2, lmax=self.lmax_out)
        return tuple(hp.alm2map_spin(alms, self.nside_out, 2, lmax=self.lmax_out))

    def _foreground_qu(self, idx: int, freq: int) -> tuple[np.ndarray, np.ndarray]:
        emission = self.fg._make_sky(idx).get_emission(float(freq) * u.GHz)
        if emission.unit != u.uK_CMB:
            emission = emission.to(u.uK_CMB,
                                   equivalencies=u.cmb_equivalencies(float(freq) * u.GHz))
        celestial = pysm_utils.apply_smoothing_and_coord_transform(
            emission, rot=hp.Rotator(coord=("G", "C")), lmax=self.lmax_out,
            map2alm_lsq_maxiter=0, output_nside=self.nside_out,
            return_healpix=True, return_car=False,
        )
        maps = np.asarray(celestial.value if hasattr(celestial, "value") else celestial)
        return maps[1], maps[2]

    def _b_selected(self, q: np.ndarray, u_map: np.ndarray) -> np.ndarray:
        field = nmt.NmtField(self.mask_apo, [q, u_map],
                             lmax=self.lmax_out, lmax_mask=self.lmax_out,
                             masked_on_input=False, purify_b=True)
        alms = field._purify(field.mask, field.get_mask_alms(),
                             [q, u_map], n_iter=field.n_iter,
                             task=[False, True], return_maps=False)
        b_alm = hp.almxfl(np.asarray(alms[1]), self.transfer)
        b_ring = hp.alm2map(b_alm, self.nside_out, lmax=self.lmax_out,
                            pixwin=False) * self.mask_bin
        return hp.reorder(b_ring, r2n=True)[self.pixel_idx].astype(np.float32)

    def _maps(self, idx: int, r: float,
              timing: dict[str, float] | None = None) -> tuple[np.ndarray, np.ndarray]:
        started = time.perf_counter() if timing is not None else 0.0
        q_cmb, u_cmb = self._cmb_qu(idx, r)
        if timing is not None:
            timing["cmb_lensing_and_downgrade"] = time.perf_counter() - started
        nofg = np.empty((len(self.FREQUENCIES), len(self.pixel_idx)), dtype=np.float32)
        fg = np.empty_like(nofg)
        for band_idx, freq in enumerate(self.FREQUENCIES):
            stage = time.perf_counter() if timing is not None else 0.0
            q_noise, u_noise = self.noise.get_freq_qu(idx, freq)
            if timing is not None:
                timing[f"noise_{freq}"] = time.perf_counter() - stage
            q_base = q_cmb + q_noise
            u_base = u_cmb + u_noise
            stage = time.perf_counter() if timing is not None else 0.0
            q_fg, u_fg = self._foreground_qu(idx, freq)
            if timing is not None:
                timing[f"foreground_{freq}"] = time.perf_counter() - stage
            stage = time.perf_counter() if timing is not None else 0.0
            nofg[band_idx] = self._b_selected(q_base, u_base)
            if timing is not None:
                timing[f"pure_b_nofg_{freq}"] = time.perf_counter() - stage
            stage = time.perf_counter() if timing is not None else 0.0
            fg[band_idx] = nofg[band_idx] + self._b_selected(q_fg, u_fg)
            if timing is not None:
                timing[f"pure_b_fg_{freq}"] = time.perf_counter() - stage
        if timing is not None:
            timing["map_generation_total"] = time.perf_counter() - started
        return nofg, fg

    def _file(self, suite: str, slot: int) -> tuple[Path, int]:
        shard, row = divmod(int(slot), self.shard_size)
        path = self.root / suite / "shards" / f"shard_{shard:06d}.h5"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path, row

    def _store(self, suite: str, slot: int, idx: int, r: float) -> tuple[Path, int]:
        path, row = self._file(suite, slot)
        if path.is_file():
            with h5py.File(path, "r") as handle:
                if handle.attrs["settings_hash"] != self.settings_hash:
                    raise ValueError(f"Incompatible simulation settings in {path}")
                if handle["done"][row]:
                    if int(handle["sim_idx"][row]) != idx or not np.isclose(
                        handle["r"][row], r, rtol=0, atol=1e-14
                    ):
                        raise ValueError(f"Occupied row has different realization: {path}:{row}")
                    return path, row
        with h5py.File(path, "a") as handle:
            if "settings_hash" in handle.attrs:
                if handle.attrs["settings_hash"] != self.settings_hash:
                    raise ValueError(f"Incompatible simulation settings in {path}")
            else:
                handle.attrs["settings_hash"] = self.settings_hash
                shape = (self.shard_size, len(self.FREQUENCIES), len(self.pixel_idx))
                for variant in ("nofg", "fg_d11s6"):
                    handle.create_dataset(variant, shape=shape, dtype="f4",
                                          chunks=(1, 1, len(self.pixel_idx)))
                handle.create_dataset("done", shape=(self.shard_size,), dtype="bool")
                handle.create_dataset("sim_idx", shape=(self.shard_size,), dtype="i8")
                handle.create_dataset("r", shape=(self.shard_size,), dtype="f8")
            if handle["done"][row]:
                if int(handle["sim_idx"][row]) != idx or not np.isclose(
                    handle["r"][row], r, rtol=0, atol=1e-14
                ):
                    raise ValueError(f"Occupied row has different realization: {path}:{row}")
                return path, row
            nofg, fg = self._maps(idx, r)
            handle["nofg"][row] = nofg
            handle["fg_d11s6"][row] = fg
            handle["sim_idx"][row] = idx
            handle["r"][row] = r
            handle.flush()
            handle["done"][row] = True
            handle.flush()
        return path, row

    def _continuous_r(self, idx: int) -> float:
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, idx]))
        return float(rng.uniform(*self.r_prior))

    def _pool_r(self, pool: str, idx: int) -> float:
        distribution = pool.rsplit("_", 1)[0]
        if distribution == "zero":
            return 0.0
        upper = self.LOW_R_MAX if distribution == "low" else self.r_prior[1]
        lower = self.r_prior[0]
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, idx]))
        return float(rng.uniform(lower, upper))

    def _check_existing_manifest(self, suite: str, design: dict | None) -> None:
        path = self.root / suite / "manifest.json"
        if path.exists():
            previous = json.loads(path.read_text(encoding="utf-8"))
            if previous.get("settings_hash") != self.settings_hash:
                raise ValueError(f"Existing manifest has different simulation settings: {path}")
            if design is not None and previous.get("design") != design:
                raise ValueError(f"{suite.capitalize()} design changed; use a new output root")

    def _write_manifest(self, suite: str, entries: list[dict],
                        design: dict | None = None) -> Path:
        self._check_existing_manifest(suite, design)
        manifest = {
            "version": 2, "format": "hdf5_paired_bmode", "product": "bmode",
            "ordering": "NEST", "dataset_kind": suite,
            "freq_values": [f"{f:.3f}" for f in self.FREQUENCIES],
            "nside": self.nside_out, "mask_cache": str(self.root / "geometry.npz"),
            "variants": ["nofg", "fg_d11s6"], "settings": self.settings,
            "settings_hash": self.settings_hash, "design": design, "entries": entries,
        }
        path = self.root / suite / "manifest.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".partial.json")
        temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        temporary.replace(path)
        return path

    def run_continuous(self, n_sims: int, progress: bool = True) -> Path:
        """Generate continuous-r train/validation pool; no grid entries appear here."""
        if n_sims < 1:
            raise ValueError("n_sims must be positive")
        self._check_existing_manifest("continuous", None)
        entries = []
        iterator = range(int(n_sims))
        if progress:
            from tqdm.auto import tqdm
            iterator = tqdm(iterator, desc="Continuous r")
        for idx in iterator:
            r = self._continuous_r(idx)
            path, row = self._store("continuous", idx, idx, r)
            entries.append({"sim_idx": idx, "r": r, "h5_file": str(path),
                            "h5_row": row, "role": "train_pool"})
        return self._write_manifest("continuous", entries)

    def run_pool(self, pool: str, n_sims: int | None = None,
                 progress: bool = True) -> Path:
        """Generate or extend one disjoint training/validation pool."""
        if pool not in self.POOLS:
            raise ValueError(f"Unknown pool {pool!r}; choose from {tuple(self.POOLS)}")
        capacity, base_id = self.POOLS[pool]
        count = capacity if n_sims is None else int(n_sims)
        if not 1 <= count <= capacity:
            raise ValueError(f"n_sims for {pool} must be within [1, {capacity}]")
        distribution, role = pool.rsplit("_", 1)
        if distribution == "low" and not self.r_prior[0] < self.LOW_R_MAX <= self.r_prior[1]:
            raise ValueError("Low-r interval must lie within the configured r prior")
        design = {"distribution": distribution, "role": role,
                  "capacity": capacity, "base_id": base_id,
                  "low_r_max": self.LOW_R_MAX if distribution == "low" else None}
        self._check_existing_manifest(pool, design)
        manifest_path = self.root / pool / "manifest.json"
        if manifest_path.is_file():
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
            if len(previous["entries"]) > count:
                raise ValueError("n_sims cannot shrink an existing pool manifest")

        entries = []
        iterator = range(count)
        if progress:
            from tqdm.auto import tqdm
            iterator = tqdm(iterator, desc=pool)
        for slot in iterator:
            idx = base_id + slot
            r = self._pool_r(pool, idx)
            path, row = self._store(pool, slot, idx, r)
            entries.append({"sim_idx": idx, "r": r, "h5_file": str(path),
                            "h5_row": row, "role": role, "source_pool": pool})
        return self._write_manifest(pool, entries, design=design)

    def run_full_suite(self, progress: bool = True) -> dict[str, Path]:
        """Produce all pools plus the disjoint fixed-r test/covariance grid."""
        paths = {pool: self.run_pool(pool, progress=progress) for pool in self.POOLS}
        paths["grid"] = self.run_grid(n_per_r=100, n_zero_covariance=500,
                                      progress=progress)
        return paths

    def run_grid(
        self, n_per_r: int = 100, n_zero_covariance: int = 500,
        r_values: Sequence[float] = GRID_R, progress: bool = True,
    ) -> Path:
        """Generate grid-only test maps plus an independent r=0 covariance pool."""
        if n_per_r < 1 or n_zero_covariance < 1:
            raise ValueError("grid counts must be positive")
        values = tuple(map(float, r_values))
        if len(values) != len(set(values)) or 0.0 not in values:
            raise ValueError("grid must contain unique r values including zero")
        design = {"r_values": list(values), "n_per_r": n_per_r,
                  "n_zero_covariance": n_zero_covariance}
        self._check_existing_manifest("grid", design)
        entries = []
        jobs = [(r, i, j) for i, r in enumerate(values)
                for j in range(n_per_r + (n_zero_covariance if r == 0 else 0))]
        iterator = jobs
        if progress:
            from tqdm.auto import tqdm
            iterator = tqdm(jobs, desc="Grid test and covariance")
        for r, r_index, j in iterator:
            if j >= 100000:
                raise ValueError("per-r count exceeds reserved index block")
            slot = r_index * 100000 + j
            idx = 1_000_000 + slot
            path, row = self._store("grid", slot, idx, r)
            role = "covariance" if r == 0 and j < n_zero_covariance else "test"
            entries.append({"sim_idx": idx, "r": r, "h5_file": str(path),
                            "h5_row": row, "role": role})
        return self._write_manifest("grid", entries, design=design)
