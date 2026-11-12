from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import healpy as hp
from .hl import hl_transform




def _rkey(value: object) -> str:
    return f"{float(value):.6f}"


def _json_file(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _manifest_entries_by_r(manifest: Mapping[str, Any]) -> Dict[str, list[Mapping[str, Any]]]:
    result: Dict[str, list[Mapping[str, Any]]] = {}
    for entry in manifest["entries"]:
        result.setdefault(_rkey(entry["r"]), []).append(entry)
    for entries in result.values():
        entries.sort(key=lambda item: int(item["sim_idx"]))
    return result


def _pack_symmetric(matrices: np.ndarray) -> np.ndarray:
    values = np.asarray(matrices, dtype=np.float64)
    nfreq = values.shape[-1]
    stacked = np.stack([values[..., i, j] for i in range(nfreq) for j in range(i, nfreq)], axis=-1)
    if values.ndim >= 4:
        return stacked.reshape(values.shape[0], -1)
    return stacked.reshape(-1)


def _stable_digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _stat_signature(path: Path) -> tuple[int, int]:
    try:
        info = path.stat()
        return info.st_mtime_ns, info.st_size
    except FileNotFoundError:
        return 0, 0


def select_balanced_entries(manifest_path: Path | str, n: int, seed: int) -> np.ndarray:
    """Choose a reproducible, approximately r-balanced set of manifest rows."""
    manifest = _json_file(Path(manifest_path))
    groups: Dict[str, list[int]] = {}
    for index, entry in enumerate(manifest["entries"]):
        groups.setdefault(_rkey(entry["r"]), []).append(index)
    rng = np.random.default_rng(int(seed))
    chosen: list[int] = []
    order_by_r: Dict[str, np.ndarray] = {}
    for key, indices in groups.items():
        order_by_r[key] = rng.permutation(indices)
    while len(chosen) < min(int(n), len(manifest["entries"])):
        progressed = False
        for key in groups:
            order = order_by_r[key]
            if order.size:
                selected_index = int(order[0])
                order_by_r[key] = order[1:]
                chosen.append(selected_index)
                progressed = True
                if len(chosen) >= n:
                    break
        if not progressed:
            break
    return np.asarray(chosen, dtype=np.int64)


def select_balanced_positions(r_values: Sequence[float], n: int, seed: int) -> np.ndarray:
    """Select row positions round-robin over the available true-r values."""
    groups: Dict[str, list[int]] = {}
    for index, value in enumerate(r_values):
        groups.setdefault(_rkey(value), []).append(index)
    rng = np.random.default_rng(int(seed))
    orders = {key: rng.permutation(indices) for key, indices in groups.items()}
    chosen: list[int] = []
    while len(chosen) < min(int(n), len(r_values)):
        progressed = False
        for key in groups:
            if orders[key].size:
                chosen.append(int(orders[key][0]))
                orders[key] = orders[key][1:]
                progressed = True
                if len(chosen) >= n:
                    break
        if not progressed:
            break
    return np.asarray(chosen, dtype=np.int64)


class MultifrequencyLikelihoods:
    """Simulation-calibrated HL, Gaussian likelihoods.

    All likelihoods use the full upper-triangle multifrequency BB vector and
    covariance from the same ``n_covariance`` r=0 maps. The SO mode is a
    BBPower-inspired Gaussian component fit with one foreground amplitude,
    appropriate for the limited two-frequency data vector.
    """

    def __init__(self, config: Mapping[str, Any]):
        if hp is None:
            raise ImportError("healpy is required for multifrequency likelihoods")
        self.config = dict(config)
        paths = self.config["paths"]
        self.covariance_manifest_path = Path(paths["covariance_manifest"]).expanduser().resolve()
        self.signal_manifest_path = Path(paths.get("signal_manifest", paths["covariance_manifest"])).expanduser().resolve()
        self.data_manifest_path = Path(paths["data_manifest"]).expanduser().resolve()
        self.augmented_manifest_path = (
            Path(paths["augmented_manifest"]).expanduser().resolve()
            if paths.get("augmented_manifest") else None
        )
        self.cache_dir = Path(self.config["cache_dir"]).expanduser().resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.cov_manifest = _json_file(self.covariance_manifest_path)
        self.signal_manifest = _json_file(self.signal_manifest_path)
        self.data_manifest = _json_file(self.data_manifest_path)
        self.aug_manifest = _json_file(self.augmented_manifest_path) if self.augmented_manifest_path else None
        for name, manifest in (
            ("covariance", self.cov_manifest), ("signal", self.signal_manifest),
            ("data", self.data_manifest), ("augmented", self.aug_manifest),
        ):
            if manifest is None:
                continue
            if manifest.get("product") != "bmode" or str(manifest.get("ordering", "")).upper() != "NEST":
                raise ValueError(f"{name} manifest must contain NEST-ordered B-mode maps")
            if int(manifest.get("nside", 0)) != int(self.cov_manifest.get("nside", 0)):
                raise ValueError(f"{name} manifest has a different HEALPix nside")
        self.frequencies = tuple(str(x) for x in self.cov_manifest["freq_values"])
        if len(self.frequencies) < 2:
            raise ValueError("Multifrequency likelihood requires at least two frequency channels")
        for name, manifest in (
            ("signal", self.signal_manifest), ("data", self.data_manifest),
            ("augmented", self.aug_manifest),
        ):
            if manifest is not None and tuple(str(x) for x in manifest["freq_values"]) != self.frequencies:
                raise ValueError(f"{name} manifest has different frequency channels")
        self.nside = int(self.cov_manifest["nside"])
        self.n_covariance = int(self.config.get("n_covariance", 500))
        self.n_template = int(self.config.get("n_template_per_r", 50))
        self.lmin = int(self.config.get("lmin", 30))
        self.lmax = int(self.config.get("lmax", 300))
        self.bin_width = int(self.config.get("bin_width", 20))
        self.map2alm_iter = int(self.config.get("map2alm_iter", 0))
        if not 0 <= self.lmin <= self.lmax <= 3 * self.nside - 1:
            raise ValueError("Invalid likelihood multipole range for map nside")
        self.bin_edges = np.arange(self.lmin, self.lmax + 1, self.bin_width, dtype=int)
        if self.bin_edges.size == 0 or self.bin_edges[0] != self.lmin:
            self.bin_edges = np.insert(self.bin_edges, 0, self.lmin)
        if self.bin_edges[-1] != self.lmax + 1:
            self.bin_edges = np.append(self.bin_edges, self.lmax + 1)
        self.n_bins = len(self.bin_edges) - 1
        self.n_spectra = len(self.frequencies) * (len(self.frequencies) + 1) // 2

        self.pixel_indices = self._load_pixel_indices(self.cov_manifest)
        hits_path = Path(self.config.get("hits_map", Path(__file__).resolve().parents[1] / "data" / "hitsmap.fits.gz"))
        if not hits_path.is_file():
            raise FileNotFoundError(f"Hits map not found: {hits_path}")
        hits = np.asarray(hp.read_map(str(hits_path)), dtype=np.float64)
        if hp.get_nside(hits) != self.nside:
            hits = hp.ud_grade(hits, nside_out=self.nside)
        hits = np.where(np.isfinite(hits) & (hits > 0), hits, 0.0)
        mask_options = dict(self.config.get("hits_mask", {}))
        mode = str(mask_options.get("mode", "highest_fraction"))
        binary_path = Path(mask_options.get("binary_mask", Path(__file__).resolve().parents[1] / "data" / "mask.fits")).expanduser().resolve()
        if not binary_path.is_file():
            raise FileNotFoundError(f"Binary scientific mask not found: {binary_path}")
        binary_mask = np.asarray(hp.read_map(str(binary_path)), dtype=np.float64)
        if hp.get_nside(binary_mask) != self.nside:
            binary_mask = hp.ud_grade(binary_mask, nside_out=self.nside)
        valid = np.isfinite(binary_mask) & (binary_mask > 0) & (hits > 0)
        if not np.any(valid):
            raise ValueError("The binary scientific mask and hits map have no observed pixels in common")
        self.mask_threshold: Optional[float] = None
        if mode == "highest_fraction":
            fraction = float(mask_options.get("fraction", 0.5))
            if not 0.0 < fraction <= 1.0:
                raise ValueError("hits_mask.fraction must lie in (0, 1]")
            valid_indices = np.flatnonzero(valid)
            n_keep = max(1, int(np.ceil(fraction * valid_indices.size)))
            order = np.lexsort((valid_indices, -hits[valid_indices]))
            kept = valid_indices[order[:n_keep]]
            selected = np.zeros(hits.size, dtype=bool)
            selected[kept] = True
            self.mask_threshold = float(np.min(hits[kept]))
        elif mode == "scientific":
            selected = valid
            fraction = 1.0
        elif mode == "inverse_variance_weight":
            fraction = 1.0
            power = float(mask_options.get("weight_power", 1.0))
            floor = float(mask_options.get("floor", 1.0e-3))
            if not np.isfinite(power) or power < 0:
                raise ValueError("hits_mask.weight_power must be finite and non-negative")
            if not np.isfinite(floor) or not 0 < floor <= 1:
                raise ValueError("hits_mask.floor must lie in (0, 1]")
            selected = valid
        else:
            raise ValueError("hits_mask.mode must be 'highest_fraction', 'scientific', or 'inverse_variance_weight'")
        if mode == "inverse_variance_weight":
            self.mask_ring = np.zeros_like(hits)
            self.mask_ring[valid] = np.maximum(hits[valid] / np.max(hits[valid]), floor) ** power
            self.mask_fsky = float(np.mean(self.mask_ring ** 2) ** 2 / np.mean(self.mask_ring ** 4))
        else:
            self.mask_ring = selected.astype(np.float64)
            self.mask_fsky = float(np.mean(self.mask_ring))
        self.valid_sky_fraction = float(np.mean(valid))
        self.mask_options = {"mode": mode, "fraction": fraction, "threshold": self.mask_threshold}
        if mode == "inverse_variance_weight":
            self.mask_options.update({"weight_power": power, "floor": floor})
        self.binary_mask_path = binary_path
        self.hits_path = hits_path.resolve()
        expected_pixels = self._load_pixel_indices(self.signal_manifest)
        if not np.array_equal(expected_pixels, self.pixel_indices):
            raise ValueError("Signal and covariance manifests use different selected-pixel geometry")
        for name, manifest in (("data", self.data_manifest), ("augmented", self.aug_manifest)):
            if manifest is not None and not np.array_equal(self._load_pixel_indices(manifest), self.pixel_indices):
                raise ValueError(f"{name} manifest uses different selected-pixel geometry")
        self.prepared = False

    def show_mask(self, title: Optional[str] = None, cmap: str = "viridis") -> None:
        """Display the exact mask or weight that will be applied to all maps."""
        import matplotlib.pyplot as plt

        hp.mollview(
            self.mask_ring, coord="C", title=title or "",
            min=0.0, max=1.0, cmap=cmap, cbar=True,
        )
        plt.show()

    def mask_summary(self) -> Dict[str, float | str | None]:
        return {
            "mode": str(self.mask_options["mode"]),
            "retained_fraction_of_valid_hits": float(self.mask_options["fraction"]),
            "hit_threshold": self.mask_threshold,
            "fsky": self.mask_fsky,
            "fraction_of_scientific_observed_sky": self.mask_fsky / max(self.valid_sky_fraction, 1e-30),
            "retained_pixels": int(np.count_nonzero(self.mask_ring)),
            "mean_weight_on_valid_sky": float(np.mean(self.mask_ring[self.mask_ring > 0])),
        }

    @staticmethod
    def _load_pixel_indices(manifest: Mapping[str, Any]) -> np.ndarray:
        cache = manifest.get("mask_cache")
        nside = int(manifest["nside"])
        if not cache:
            return np.arange(12 * nside * nside, dtype=np.int64)
        with np.load(Path(str(cache)).expanduser()) as payload:
            return np.asarray(payload["pixel_idx"], dtype=np.int64)

    def _cache_signature(self) -> Dict[str, Any]:
        return {
            "calibration_selection": "exclude_benchmark_v1",
            "excluded_realizations": _stable_digest({
                "ids": sorted(self._evaluation_ids()),
            }),
            "covariance_manifest": str(self.covariance_manifest_path),
            "covariance_dataset_key": self.cov_manifest.get("dataset_key"),
            "signal_manifest": str(self.signal_manifest_path),
            "signal_dataset_key": self.signal_manifest.get("dataset_key"),
            "augmented_manifest": None if self.augmented_manifest_path is None else str(self.augmented_manifest_path),
            "augmented_dataset_key": None if self.aug_manifest is None else self.aug_manifest.get("dataset_key"),
            "n_covariance": self.n_covariance,
            "n_template": self.n_template,
            "lmin": self.lmin,
            "lmax": self.lmax,
            "bin_width": self.bin_width,
            "hits_mask": self.mask_options,
            "mask_checksum": hashlib.sha256(
                (self.mask_ring if self.mask_options["mode"] == "inverse_variance_weight"
                 else self.mask_ring.astype(np.uint8)).tobytes()
            ).hexdigest()[:16],
            "binary_mask_path": str(self.binary_mask_path),
            "hits_path": str(self.hits_path),
        }

    def _entries(self, manifest: Mapping[str, Any]) -> Dict[str, list[Mapping[str, Any]]]:
        return _manifest_entries_by_r(manifest)

    def _evaluation_ids(self) -> set[tuple[str, int]]:
        # Augmented and base maps share the same underlying realization IDs.
        return {(_rkey(entry["r"]), int(entry["sim_idx"]))
                for entry in self.data_manifest["entries"]}

    def _calibration_entries(self, manifest: Mapping[str, Any]) -> Dict[str, list[Mapping[str, Any]]]:
        excluded = self._evaluation_ids()
        return {
            key: [entry for entry in entries if (key, int(entry["sim_idx"])) not in excluded]
            for key, entries in self._entries(manifest).items()
        }

    def _bandpowers(self, entry: Mapping[str, Any], manifest: Mapping[str, Any]) -> np.ndarray:
        nside = int(manifest["nside"])
        pixels = self._load_pixel_indices(manifest)
        npix = 12 * nside * nside
        alms = []
        for frequency in self.frequencies:
            path = Path(str(entry["files"][frequency])).expanduser()
            with np.load(path) as payload:
                selected = np.asarray(payload["b_selected"], dtype=np.float64)
            if selected.shape != pixels.shape:
                raise ValueError(f"Selected-pixel geometry mismatch in {path}")
            full_nest = np.zeros(npix, dtype=np.float64)
            full_nest[pixels] = selected
            ring = hp.reorder(full_nest, n2r=True) * self.mask_ring
            alms.append(hp.map2alm(ring, lmax=self.lmax, iter=self.map2alm_iter, pol=False))
        ell = np.arange(self.lmax + 1, dtype=np.float64)
        factor = ell * (ell + 1.0) / (2.0 * np.pi)
        result = np.zeros((self.n_bins, len(alms), len(alms)), dtype=np.float64)
        for i in range(len(alms)):
            for j in range(i, len(alms)):
                dl = hp.alm2cl(alms[i], alms[j], lmax=self.lmax) * factor
                for b, (lo, hi) in enumerate(zip(self.bin_edges[:-1], self.bin_edges[1:])):
                    multipoles = np.arange(lo, hi)
                    result[b, i, j] = result[b, j, i] = np.average(dl[lo:hi], weights=2 * multipoles + 1)
        return result

    def _cached_mean(
        self, cache_key: str, manifest: Mapping[str, Any],
        entries: Sequence[Mapping[str, Any]], force: bool = False,
    ) -> np.ndarray:
        path = self.cache_dir / f"{cache_key}_{_stable_digest(self._cache_signature())}.npz"
        if path.is_file() and not force:
            with np.load(path) as payload:
                return np.asarray(payload["mean"], dtype=np.float64)
        if not entries:
            raise ValueError(f"No simulations available for {cache_key}")
        powers = [self._bandpowers(entry, manifest) for entry in entries]
        mean = np.mean(powers, axis=0)
        np.savez_compressed(path, mean=mean, n=np.int64(len(entries)))
        return mean

    def _cached_covariance_samples(self, entries: Sequence[Mapping[str, Any]], force: bool = False) -> np.ndarray:
        path = self.cache_dir / f"covariance_samples_{_stable_digest(self._cache_signature())}.npy"
        if len(entries) < self.n_covariance:
            raise ValueError(f"Need {self.n_covariance} covariance simulations, found {len(entries)}")
        shape = (self.n_covariance, self.n_bins, len(self.frequencies), len(self.frequencies))
        if force and path.exists():
            path.unlink()
        if path.is_file():
            powers = np.load(path, mmap_mode="r+")
            if powers.shape != shape:
                raise ValueError(f"Cached covariance shape {powers.shape} does not match requested {shape}")
        else:
            powers = np.lib.format.open_memmap(path, mode="w+", dtype=np.float64, shape=shape)
            powers[:] = np.nan
            powers.flush()
        incomplete = np.flatnonzero(~np.all(np.isfinite(powers), axis=(1, 2, 3)))
        from tqdm.auto import tqdm
        progress = tqdm(total=self.n_covariance, initial=self.n_covariance - incomplete.size,
                        desc="Likelihood covariance maps", unit="set")
        try:
            for index in incomplete:
                powers[index] = self._bandpowers(entries[int(index)], self.cov_manifest)
                powers.flush()
                progress.update(1)
        finally:
            progress.close()
        return np.asarray(powers, dtype=np.float64)

    def prepare(self, force: bool = False) -> "MultifrequencyLikelihoods":
        cache_path = self.cache_dir / f"prepared_{_stable_digest(self._cache_signature())}.npz"
        if cache_path.is_file() and not force:
            with np.load(cache_path) as payload:
                self.covariance_samples = np.asarray(payload["covariance_samples"], dtype=np.float64)
                self.baseline = np.asarray(payload["baseline"], dtype=np.float64)
                self.tensor_template = np.asarray(payload["tensor_template"], dtype=np.float64)
                self.foreground_template = np.asarray(payload["foreground_template"], dtype=np.float64)
                self.covariance = np.asarray(payload["covariance"], dtype=np.float64)
                self.inverse_covariance = np.asarray(payload["inverse_covariance"], dtype=np.float64)
                self.reference = np.asarray(payload["reference"], dtype=np.float64)
                self.hl_covariance = np.asarray(payload["hl_covariance"], dtype=np.float64)
                self.hl_inverse_covariance = np.asarray(payload["hl_inverse_covariance"], dtype=np.float64)
                self.hl_mean = np.asarray(payload["hl_mean"], dtype=np.float64)
                self.effective_ells = np.asarray(payload["effective_ells"], dtype=np.float64)
                self.covariance_sim_idx = np.asarray(payload["covariance_sim_idx"], dtype=np.int64)
            self._prepared_signature = _stable_digest(self._cache_signature())
            self.prepared = True
            return self

        covariance_by_r = self._calibration_entries(self.cov_manifest)
        signal_by_r = self._calibration_entries(self.signal_manifest)
        zero_key = _rkey(0.0)
        if zero_key not in covariance_by_r or zero_key not in signal_by_r:
            raise ValueError("Covariance and signal manifests must both include r=0 simulations")
        covariance_rows = covariance_by_r[zero_key]
        if len(covariance_rows) < self.n_covariance:
            raise ValueError(
                f"Need {self.n_covariance} independent r=0 covariance maps after excluding "
                f"the entire benchmark; found {len(covariance_rows)}"
            )
        for key, entries in signal_by_r.items():
            if len(entries) < self.n_template:
                raise ValueError(
                    f"Need {self.n_template} independent template maps at r={key} "
                    f"after excluding the benchmark; found {len(entries)}"
                )
        self.covariance_samples = self._cached_covariance_samples(covariance_rows, force=force)
        self.covariance_sim_idx = np.asarray(
            [int(e["sim_idx"]) for e in covariance_rows[:self.n_covariance]], dtype=np.int64
        )
        self.baseline = self._cached_mean(
            "baseline", self.signal_manifest, signal_by_r[zero_key][:self.n_template], force=force
        )

        # Estimate the tensor response from the signal-only suite means. This
        # naturally includes its mask, transfer, geometry, and hit weighting.
        means, r_values = [], []
        for key, entries in sorted(signal_by_r.items(), key=lambda pair: float(pair[0])):
            r_values.append(float(key))
            means.append(self._cached_mean(
                f"signal_r{key.replace('.', 'p')}", self.signal_manifest,
                entries[:self.n_template], force=force,
            ))
        design = np.column_stack([np.ones(len(r_values)), np.asarray(r_values)])
        flattened = np.asarray(means).reshape(len(means), -1)
        coeff = np.linalg.lstsq(design, flattened, rcond=None)[0]
        self.tensor_template = coeff[1].reshape(self.baseline.shape)

        self.foreground_template = np.zeros_like(self.baseline)
        if self.aug_manifest is not None:
            aug_by_r = self._calibration_entries(self.aug_manifest)
            if zero_key not in aug_by_r:
                raise ValueError("Augmented manifest must include r=0 simulations")
            aug_entries = {int(item["sim_idx"]): item for item in aug_by_r[zero_key]}
            base_entries = {int(item["sim_idx"]): item for item in signal_by_r[zero_key]}
            common = sorted(set(aug_entries).intersection(base_entries))[:self.n_template]
            if len(common) < self.n_template:
                raise ValueError("Not enough matched augmented/base r=0 maps for the foreground template")
            fg_cache = self.cache_dir / f"foreground_template_{_stable_digest(self._cache_signature())}.npz"
            if fg_cache.is_file() and not force:
                with np.load(fg_cache) as payload:
                    self.foreground_template = np.asarray(payload["template"], dtype=np.float64)
            else:
                differences = []
                from tqdm.auto import tqdm
                for sim_idx in tqdm(common, desc="Foreground template", unit="set"):
                    differences.append(
                        self._bandpowers(aug_entries[sim_idx], self.aug_manifest)
                        - self._bandpowers(base_entries[sim_idx], self.signal_manifest)
                    )
                self.foreground_template = np.mean(differences, axis=0)
                np.savez_compressed(fg_cache, template=self.foreground_template, sim_idx=np.asarray(common))

        self.covariance = np.cov(_pack_symmetric(self.covariance_samples).T, ddof=1)
        ridge = float(self.config.get("covariance_ridge", 1.0e-6))
        scale = max(float(np.median(np.diag(self.covariance))), 1e-30)
        self.covariance += ridge * scale * np.eye(self.covariance.shape[0])
        self.inverse_covariance = np.linalg.pinv(self.covariance, hermitian=True)
        if bool(self.config.get("apply_hartlap", True)):
            p = self.covariance.shape[0]
            hartlap = (self.n_covariance - p - 2.0) / (self.n_covariance - 1.0)
            if hartlap <= 0:
                raise ValueError(f"500 samples are insufficient for {p}-element covariance Hartlap correction")
            self.inverse_covariance *= hartlap
        self.reference = self.baseline + self.foreground_template
        if np.any(np.linalg.eigvalsh(self.reference) <= 0):
            raise ValueError("Fiducial multifrequency spectrum is not positive definite")
        transformed = np.asarray([
            hl_transform(sample, self.reference, self.reference)
            for sample in self.covariance_samples
        ])
        self.hl_mean = transformed.mean(axis=0)
        self.hl_covariance = np.cov(transformed, rowvar=False, ddof=1)
        hscale = max(float(np.median(np.diag(self.hl_covariance))), 1e-30)
        self.hl_covariance += ridge * hscale * np.eye(self.hl_covariance.shape[0])
        self.hl_inverse_covariance = np.linalg.pinv(self.hl_covariance, hermitian=True)
        if bool(self.config.get("apply_hartlap", True)):
            self.hl_inverse_covariance *= hartlap
        ell = np.arange(self.lmax + 1, dtype=float)
        self.effective_ells = np.asarray([
            np.average(np.arange(lo, hi), weights=2 * np.arange(lo, hi) + 1)
            for lo, hi in zip(self.bin_edges[:-1], self.bin_edges[1:])
        ])
        np.savez_compressed(
            cache_path, covariance_samples=self.covariance_samples, baseline=self.baseline,
            covariance_sim_idx=np.asarray([int(e["sim_idx"]) for e in covariance_rows[:self.n_covariance]]),
            tensor_template=self.tensor_template, foreground_template=self.foreground_template,
            covariance=self.covariance, inverse_covariance=self.inverse_covariance,
            reference=self.reference, hl_covariance=self.hl_covariance,
            hl_inverse_covariance=self.hl_inverse_covariance, hl_mean=self.hl_mean,
            effective_ells=self.effective_ells,
        )
        self._prepared_signature = _stable_digest(self._cache_signature())
        self.prepared = True
        return self

    def verify_calibration(self) -> Dict[str, int]:
        """Audit the IDs used in the prepared covariance, including cache loads."""
        if not self.prepared or getattr(self, "_prepared_signature", None) != _stable_digest(self._cache_signature()):
            raise RuntimeError("Stale likelihood object: reload and prepare it before inference")
        actual = getattr(self, "covariance_sim_idx", None)
        expected = np.asarray([
            int(e["sim_idx"]) for e in
            self._calibration_entries(self.cov_manifest)["0.000000"][:self.n_covariance]
        ], dtype=np.int64)
        if actual is None or len(expected) != self.n_covariance or not np.array_equal(actual, expected):
            raise RuntimeError("Prepared covariance IDs do not match the independent selection; rebuild with prepare(force=True)")
        overlap = {("0.000000", int(i)) for i in actual} & self._evaluation_ids()
        if overlap:
            raise ValueError(f"Prepared covariance overlaps benchmark realizations: {sorted(overlap)[:5]}")
        return {"covariance_maps": len(actual), "benchmark_overlap": len(overlap)}

    def bandpowers_for_index(self, index: int) -> np.ndarray:
        entry = self.data_manifest["entries"][int(index)]
        return self._bandpowers(entry, self.data_manifest)

    def _model(self, r: float, foreground_amplitude: float = 1.0) -> np.ndarray:
        model = self.baseline + float(r) * self.tensor_template
        if self.aug_manifest is not None:
            model = model + float(foreground_amplitude) * self.foreground_template
        return model

    def _score(self, data: np.ndarray, r: float, foreground_amplitude: float, kind: str) -> float:
        model = self._model(r, foreground_amplitude)
        if any(np.min(np.linalg.eigvalsh(matrix)) <= 0.0 for matrix in model):
            return -np.inf
        if kind == "hl":
            try:
                transformed = hl_transform(data, model, self.reference) - self.hl_mean
            except (ValueError, np.linalg.LinAlgError):
                return -np.inf
            vector = np.asarray(transformed, dtype=np.float64)
            return -0.5 * float(vector @ self.hl_inverse_covariance @ vector)
        residual = _pack_symmetric(data - model).reshape(-1)
        return -0.5 * float(residual @ self.inverse_covariance @ residual)

    def fit_index(
        self,
        index: int,
        kind: Optional[str] = None,
        return_samples: bool = False,
    ) -> Dict[str, Any]:
        """Sample the configured likelihood posterior with emcee."""
        if not self.prepared:
            raise RuntimeError("Call prepare() first")
        if kind is None:
            kind = str(self.config.get("mode", "gaussian")).lower()
        if kind not in {"hl", "so", "gaussian"}:
            raise ValueError("kind must be 'hl', 'so', or 'gaussian'")
        try:
            import emcee
        except ImportError as exc:  # pragma: no cover - depends on optional package
            raise ImportError("emcee is required for likelihood sampling; install emcee") from exc

        entry = self.data_manifest["entries"][int(index)]
        data = self.bandpowers_for_index(index)
        r_bounds = tuple(map(float, self.config.get("r_bounds", (0.0, 0.03))))
        has_fg_amplitude = self.aug_manifest is not None and bool(
            self.config.get("sample_foreground_amplitude", True)
        )
        bounds = [r_bounds]
        if has_fg_amplitude:
            bounds.append(tuple(map(float, self.config.get("foreground_bounds", (0.0, 3.0)))))
        ndim = len(bounds)
        nwalkers = int(self.config.get("mcmc_walkers", max(24, 4 * ndim)))
        steps = int(self.config.get("mcmc_steps", 1200))
        burnin = int(self.config.get("mcmc_burnin", 300))
        thin = int(self.config.get("mcmc_thin", 4))
        if nwalkers < 2 * ndim or steps <= burnin or burnin < 0 or thin < 1:
            raise ValueError("Invalid MCMC settings: require walkers >= 2*ndim and steps > burnin >= 0")
        for lower, upper in bounds:
            if not lower < upper:
                raise ValueError("MCMC prior bounds must be strictly increasing")

        def log_probability(theta: np.ndarray) -> float:
            if any(value < lo or value > hi for value, (lo, hi) in zip(theta, bounds)):
                return -np.inf
            r_value = float(theta[0])
            amplitude = float(theta[1]) if has_fg_amplitude else float(
                self.config.get("fixed_foreground_amplitude", 1.0)
            )
            return self._score(data, r_value, amplitude, kind)

        seed = int(self.config.get("mcmc_seed", 1729)) + int(index)
        rng = np.random.default_rng(seed)
        center = np.asarray([(lo + hi) / 2.0 for lo, hi in bounds])
        width = np.asarray([hi - lo for lo, hi in bounds])
        initial = np.empty((nwalkers, ndim), dtype=np.float64)
        for walker in range(nwalkers):
            for _ in range(10000):
                candidate = center + rng.normal(0.0, 0.08, size=ndim) * width
                candidate = np.clip(candidate, [item[0] for item in bounds], [item[1] for item in bounds])
                if np.isfinite(log_probability(candidate)):
                    initial[walker] = candidate
                    break
            else:
                raise RuntimeError("Could not initialize finite emcee walkers inside the prior")

        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability)
        sampler.random_state = np.random.RandomState(seed).get_state()
        sampler.run_mcmc(initial, steps, progress=bool(self.config.get("mcmc_progress", False)))
        samples = sampler.get_chain(discard=burnin, thin=thin, flat=True)
        log_samples = sampler.get_log_prob(discard=burnin, thin=thin, flat=True)
        if samples.size == 0 or not np.any(np.isfinite(log_samples)):
            raise RuntimeError(f"emcee returned no finite posterior samples for entry {index}")
        quantiles = np.quantile(samples[:, 0], [0.16, 0.5, 0.84])
        best = samples[int(np.nanargmax(log_samples))]
        result: Dict[str, Any] = {
            "entry_index": int(index), "sim_idx": int(entry["sim_idx"]),
            "r_true": float(entry["r"]), "r_median": float(quantiles[1]),
            "r_lower": float(quantiles[0]), "r_upper": float(quantiles[2]),
            "sigma_r": float(0.5 * (quantiles[2] - quantiles[0])),
            "r_mean": float(np.mean(samples[:, 0])),
            "r_map": float(best[0]),
            "acceptance_fraction": float(np.mean(sampler.acceptance_fraction)),
            "n_posterior_samples": int(samples.shape[0]),
        }
        # Check chain length in autocorrelation times, without treating walkers
        # or thinned posterior samples as independent convergence tests.
        tau = emcee.autocorr.integrated_time(
            sampler.get_chain(discard=burnin), tol=0, quiet=True
        )
        tau_max = float(np.max(tau))
        result["mcmc_tau_max"] = tau_max
        result["mcmc_length_in_tau"] = (
            float((steps - burnin) / tau_max) if np.isfinite(tau_max) and tau_max > 0 else 0.0
        )
        result["mcmc_converged"] = bool(result["mcmc_length_in_tau"] >= 50.0)
        if has_fg_amplitude:
            fg_quantiles = np.quantile(samples[:, 1], [0.16, 0.5, 0.84])
            result.update({
                "foreground_amplitude_median": float(fg_quantiles[1]),
                "foreground_amplitude_lower": float(fg_quantiles[0]),
                "foreground_amplitude_upper": float(fg_quantiles[2]),
            })
        if return_samples:
            result["samples"] = samples
            result["log_probability"] = log_samples
            result["chain"] = sampler.get_chain()
        return result


def load_case_config(path: Path | str) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise ImportError("PyYAML is required to load comparison YAML; install pyyaml") from exc
    config_path = Path(path).expanduser().resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid comparison YAML: {config_path}")
    payload["_config_path"] = str(config_path)
    return payload


def _predict_selected(config: Mapping[str, Any], model_kind: str, manifest_path: Path, indices: np.ndarray) -> Dict[str, np.ndarray]:
    model_options = config["models"][model_kind]
    run_dir = Path(model_options["run_dir"]).expanduser().resolve()
    if model_kind == "cnn":
        from ..cnn.config import DataConfig, ModelConfig
        from ..cnn.data import SimulationDataset, Standardizer
        from ..cnn.inference import configure_cpu
        from ..cnn.model import build_model, prediction_to_r

        configure_cpu(int(model_options.get("cpu_threads", 24)))
        run_config = _json_file(run_dir / "config.json")
        train_data = DataConfig(**run_config["data"])
        train_dataset = SimulationDataset(train_data)
        standardizer = Standardizer.load(run_dir / "normalization.npz")
        model = build_model(
            ModelConfig(**run_config["model"]),
            train_dataset.pixel_indices,
        )
        model.load_weights(run_dir / "final.weights.h5")
        external = SimulationDataset(DataConfig(
            manifest=manifest_path, validation_fraction=0.0, test_fraction=0.0,
            normalization_samples=1, target_scale=train_data.target_scale,
        ))
        if external.nside != train_dataset.nside or external.frequencies != train_dataset.frequencies or not np.array_equal(external.pixel_indices, train_dataset.pixel_indices):
            raise ValueError("Prediction manifest geometry differs from the CNN training geometry")
        raw = []
        batch_size = int(model_options.get("batch_size", 8))
        from tqdm.auto import tqdm
        for start in tqdm(range(0, len(indices), batch_size), desc="CNN cached predictions", unit="batch"):
            batch_ids = indices[start:start + batch_size]
            batch = np.stack([external.load_sample(int(i), standardizer)[0] for i in batch_ids])
            raw.append(np.asarray(model.predict(batch, verbose=0)))
        raw_prediction = np.concatenate(raw, axis=0)
        calibration = float(_json_file(run_dir / "calibration.json").get("scale", 1.0))
        mean, sigma = prediction_to_r(raw_prediction, train_data.target_scale, calibration)
        sigma_predicted = sigma
    elif model_kind == "bnn":
        from ..bnn.inference import load_run
        from ..bnn.training import _mc_predict
        from ..bnn.model import mc_prediction_to_r
        from ..cnn.config import DataConfig
        from ..cnn.data import SimulationDataset

        model, train_dataset, standardizer = load_run(
            run_dir, cpu_threads=int(model_options.get("cpu_threads", 24))
        )
        external = SimulationDataset(DataConfig(
            manifest=manifest_path, validation_fraction=0.0, test_fraction=0.0,
            normalization_samples=1, target_scale=train_dataset.config.target_scale,
        ))
        if external.nside != train_dataset.nside or external.frequencies != train_dataset.frequencies or not np.array_equal(external.pixel_indices, train_dataset.pixel_indices):
            raise ValueError("Prediction manifest geometry differs from the BNN training geometry")
        raw = _mc_predict(
            model, external, indices, standardizer,
            int(model_options.get("batch_size", 4)), int(model_options.get("mc_samples", 32)),
        )
        calibration = float(_json_file(run_dir / "calibration.json").get("scale", 1.0))
        converted = mc_prediction_to_r(raw, train_dataset.config.target_scale, calibration)
        mean = converted["r_predicted"]
        sigma_predicted = converted["sigma_r"]
    else:
        raise ValueError("model_kind must be cnn or bnn")

    entries = _json_file(manifest_path)["entries"]
    selected = [entries[int(index)] for index in indices]
    truth = np.asarray([float(entry["r"]) for entry in selected])
    sim_idx = np.asarray([int(entry["sim_idx"]) for entry in selected])
    return {
        "entry_index": np.asarray(indices, dtype=np.int64),
        "sim_idx": sim_idx,
        "r_true": truth,
        "r_predicted": np.asarray(mean, dtype=np.float64),
        "sigma_predicted": np.asarray(sigma_predicted, dtype=np.float64),
    }


def cached_predictive_results(config: Mapping[str, Any], model_kind: str, force: bool = False) -> Dict[str, np.ndarray]:
    """Predict a fixed 500-entry ensemble once, then reuse its cached metrics."""
    manifest_path = Path(config["paths"]["data_manifest"]).expanduser().resolve()
    model_options = config["models"][model_kind]
    run_dir = Path(model_options["run_dir"]).expanduser().resolve()
    weights = run_dir / "final.weights.h5"
    n = int(config.get("evaluation", {}).get("n_sigma_simulations", 500))
    cache_dir = Path(config["cache_dir"]).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = {
        "model": model_kind, "run": str(run_dir),
        "weights_mtime": weights.stat().st_mtime_ns, "weights_size": weights.stat().st_size,
        "run_config_signature": _stat_signature(run_dir / "config.json"),
        "calibration_signature": _stat_signature(run_dir / "calibration.json"),
        "manifest": str(manifest_path), "dataset_key": _json_file(manifest_path).get("dataset_key"),
        "n": n, "seed": int(config.get("evaluation", {}).get("selection_seed", 2027)),
        "mc_samples": int(model_options.get("mc_samples", 1)),
    }
    path = cache_dir / f"prediction_{model_kind}_{_stable_digest(cache_key)}.npz"
    if path.is_file() and not force:
        with np.load(path) as payload:
            return {name: np.asarray(payload[name]) for name in payload.files}
    indices = select_balanced_entries(
        manifest_path, n, int(config.get("evaluation", {}).get("selection_seed", 2027))
    )
    predictions = _predict_selected(config, model_kind, manifest_path, indices)
    error = predictions["r_predicted"] - predictions["r_true"]
    predictions["error"] = error
    predictions["sigma_predicted_mean"] = np.asarray(float(np.mean(predictions["sigma_predicted"])))
    predictions["bias"] = np.asarray(float(np.mean(error)))
    predictions["sigma_empirical"] = np.asarray(float(np.std(error, ddof=1)))
    predictions["rmse"] = np.asarray(float(np.sqrt(np.mean(error**2))))
    per_r = {}
    for r in np.unique(predictions["r_true"]):
        select = np.isclose(predictions["r_true"], r)
        per_r[_rkey(r)] = {
            "n": int(np.count_nonzero(select)),
            "sigma_predicted_mean": float(np.mean(predictions["sigma_predicted"][select])),
            "sigma_empirical": float(np.std(error[select], ddof=1)) if np.count_nonzero(select) > 1 else float("nan"),
            "bias": float(np.mean(error[select])),
        }
    predictions["per_r_json"] = np.asarray(json.dumps(per_r, sort_keys=True))
    np.savez_compressed(path, **predictions)
    return predictions


def cached_likelihood_bias(
    likelihood: MultifrequencyLikelihoods,
    n: int = 50,
    seed: int = 31415,
    force: bool = False,
    entry_indices: Optional[Sequence[int]] = None,
    kind: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """Fit the configured likelihood to a reproducible balanced set of maps."""
    kind = str(kind or likelihood.config.get("mode", "gaussian")).lower()
    if kind not in {"hl", "so", "gaussian"}:
        raise ValueError("kind must be 'hl', 'so', or 'gaussian'")
    indices = (
        np.asarray(entry_indices, dtype=np.int64)
        if entry_indices is not None
        else select_balanced_entries(likelihood.data_manifest_path, n, seed)
    )
    if indices.ndim != 1 or indices.size != int(n):
        raise ValueError(f"Expected exactly {n} likelihood bias indices")
    signature = _stable_digest({
        **likelihood._cache_signature(), "n_bias": int(n), "seed": int(seed),
        "entry_indices": indices.tolist(),
        "mode": kind,
        "mcmc": {key: likelihood.config.get(key) for key in (
            "mcmc_walkers", "mcmc_steps", "mcmc_burnin", "mcmc_thin", "mcmc_seed",
            "sample_foreground_amplitude", "foreground_bounds", "fixed_foreground_amplitude",
            "r_bounds",
        )},
    })
    path = likelihood.cache_dir / f"likelihood_bias_{signature}.npz"
    if path.is_file() and not force:
        with np.load(path) as payload:
            return {name: np.asarray(payload[name]) for name in payload.files}
    output: Dict[str, np.ndarray] = {"entry_index": indices}
    rows = [likelihood.fit_index(int(index), kind) for index in indices]
    for name in ("r_true", "r_median", "r_lower", "r_upper", "sigma_r", "r_map", "r_mean", "sim_idx", "acceptance_fraction"):
        output[f"{kind}_{name}"] = np.asarray([row[name] for row in rows])
    if rows and "foreground_amplitude_median" in rows[0]:
        for name in ("foreground_amplitude_median", "foreground_amplitude_lower", "foreground_amplitude_upper"):
            output[f"{kind}_{name}"] = np.asarray([row[name] for row in rows])
    np.savez_compressed(path, **output)
    return output
