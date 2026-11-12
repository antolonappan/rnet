from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    import healpy as hp  # type: ignore
except ImportError:  # pragma: no cover - environment dependent
    hp = None

try:
    from tqdm.auto import tqdm  # type: ignore
except ImportError:  # pragma: no cover - environment dependent
    tqdm = None


@dataclass(frozen=True)
class HLConfig:
    """Configuration for a simulation-calibrated, binned HL likelihood."""

    manifest: Path
    lmin: int = 30
    lmax: int = 300
    bin_width: int = 20
    n_covariance: int = 300
    n_template: int = 300
    covariance_r: float = 0.0
    map2alm_iter: int = 0
    covariance_ridge: float = 1.0e-6
    matrix_floor: float = 1.0e-8
    apply_hartlap: bool = True
    cache_dir: Optional[Path] = None
    dataset_key: Optional[str] = None
    cmb_spectra_file: Optional[Path] = None
    alens_fiducial: float = 1.0
    base_manifest: Optional[Path] = None
    use_theory_cmb: bool = False
    use_theory_foreground: bool = False
    foreground_preset: Tuple[str, str] = ("d10", "s5")
    foreground_sim_idx: int = 0
    foreground_seed_base: Optional[int] = None
    foreground_basedir: Optional[Path] = None
    foreground_output_root: Optional[Path] = None

    def __post_init__(self) -> None:
        manifest = Path(self.manifest).expanduser().resolve()
        object.__setattr__(self, "manifest", manifest)
        if not manifest.is_file():
            raise FileNotFoundError(f"Dataset manifest not found: {manifest}")
        if self.lmin < 0 or self.lmax < self.lmin:
            raise ValueError("Require 0 <= lmin <= lmax")
        if self.bin_width <= 0:
            raise ValueError("bin_width must be > 0")
        if self.n_covariance < 3 or self.n_template < 1:
            raise ValueError("n_covariance must be >= 3 and n_template must be >= 1")
        if self.covariance_r < 0:
            raise ValueError("covariance_r must be >= 0")
        if self.map2alm_iter < 0:
            raise ValueError("map2alm_iter must be >= 0")
        if self.covariance_ridge < 0 or self.matrix_floor <= 0:
            raise ValueError("covariance_ridge must be >= 0 and matrix_floor must be > 0")
        if self.cache_dir is not None:
            object.__setattr__(self, "cache_dir", Path(self.cache_dir).expanduser().resolve())
        if self.cmb_spectra_file is not None:
            spectra_file = Path(self.cmb_spectra_file).expanduser().resolve()
            if not spectra_file.is_file():
                raise FileNotFoundError(f"CMB spectra cache not found: {spectra_file}")
            object.__setattr__(self, "cmb_spectra_file", spectra_file)
        if self.base_manifest is not None:
            base_manifest = Path(self.base_manifest).expanduser().resolve()
            if not base_manifest.is_file():
                raise FileNotFoundError(f"Base CMB+noise manifest not found: {base_manifest}")
            object.__setattr__(self, "base_manifest", base_manifest)
        if len(self.foreground_preset) != 2:
            raise ValueError("foreground_preset must contain dust and synchrotron presets")
        if self.foreground_sim_idx < 0:
            raise ValueError("foreground_sim_idx must be non-negative")
        if self.foreground_seed_base is not None and self.foreground_seed_base < 0:
            raise ValueError("foreground_seed_base must be non-negative")
        for name in ("foreground_basedir", "foreground_output_root"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, Path(value).expanduser().resolve())
        if self.alens_fiducial <= 0:
            raise ValueError("alens_fiducial must be > 0")


@dataclass(frozen=True)
class MCMCConfig:
    """Configuration for emcee sampling of r and optionally A_lens."""

    n_walkers: int = 32
    n_steps: int = 3000
    burn_in: int = 500
    thin: int = 5
    r_bounds: Tuple[float, float] = (-0.01, 0.1)
    alens_bounds: Tuple[float, float] = (0.0, 3.0)
    vary_alens: bool = True
    fixed_alens: float = 1.0
    foreground_amplitude_bounds: Tuple[float, float] = (0.0, 3.0)
    vary_foreground: bool = False
    fixed_foreground_amplitude: float = 1.0
    seed: int = 2028
    progress: bool = True

    def __post_init__(self) -> None:
        ndim = int(self.vary_alens) + int(self.vary_foreground) + 1
        if self.n_walkers < 2 * ndim or self.n_walkers % 2:
            raise ValueError("n_walkers must be even and at least twice the sampled dimension")
        if self.n_steps <= 0 or self.burn_in < 0 or self.thin <= 0:
            raise ValueError("n_steps and thin must be > 0; burn_in must be >= 0")
        if self.burn_in >= self.n_steps:
            raise ValueError("burn_in must be smaller than n_steps")
        for name, bounds in (("r_bounds", self.r_bounds), ("alens_bounds", self.alens_bounds)):
            if len(bounds) != 2 or bounds[0] >= bounds[1]:
                raise ValueError(f"{name} must contain increasing lower and upper bounds")
        if self.alens_bounds[0] < 0:
            raise ValueError("The A_lens prior must be non-negative")
        if self.fixed_alens < 0:
            raise ValueError("fixed_alens must be non-negative")
        if len(self.foreground_amplitude_bounds) != 2 or self.foreground_amplitude_bounds[0] >= self.foreground_amplitude_bounds[1]:
            raise ValueError("foreground_amplitude_bounds must contain increasing limits")
        if self.foreground_amplitude_bounds[0] < 0 or self.fixed_foreground_amplitude < 0:
            raise ValueError("Foreground amplitude prior and fixed value must be non-negative")


@dataclass(frozen=True)
class HLMCMCResult:
    """Flattened emcee chain with helpers for GetDist analysis."""

    samples: np.ndarray
    log_probability: np.ndarray
    parameter_names: Tuple[str, ...]
    parameter_labels: Tuple[str, ...]
    bounds: Tuple[Tuple[float, float], ...]
    r_true: float
    alens_true: float
    sim_idx: int
    entry_index: int
    data_index_within_r: int
    backend_file: Optional[Path]
    acceptance_fraction: float
    autocorrelation_time: Optional[np.ndarray]

    def upper_limit(self, level: float = 0.95) -> float:
        """Return a one-sided upper limit after restricting the chain to r >= 0."""
        if not 0 < level < 1:
            raise ValueError("level must lie strictly between 0 and 1")
        positive = np.asarray(self.samples[:, 0], dtype=np.float64)
        positive = positive[positive >= 0]
        if positive.size == 0:
            return 0.0
        return float(np.quantile(positive, level))

    def getdist_samples(self, label: str = "HL"):
        """Convert the flattened chain to a GetDist MCSamples object."""
        try:
            from getdist import MCSamples
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError("GetDist is required; install it with `pip install getdist`") from exc
        ranges = {name: list(bound) for name, bound in zip(self.parameter_names, self.bounds)}
        return MCSamples(
            samples=self.samples,
            loglikes=-self.log_probability,
            names=list(self.parameter_names),
            labels=list(self.parameter_labels),
            ranges=ranges,
            label=label,
        )

    def summary(self) -> Dict[str, object]:
        out: Dict[str, object] = {
            "r_true": self.r_true,
            "alens_true": self.alens_true,
            "sim_idx": self.sim_idx,
            "entry_index": self.entry_index,
            "samples": len(self.samples),
            "acceptance_fraction": self.acceptance_fraction,
            "backend_file": None if self.backend_file is None else str(self.backend_file),
        }
        for index, name in enumerate(self.parameter_names):
            q16, q50, q84 = np.quantile(self.samples[:, index], [0.16, 0.5, 0.84])
            out[name] = {
                "median": float(q50),
                "lower_68": float(q16),
                "upper_68": float(q84),
                "minus": float(q50 - q16),
                "plus": float(q84 - q50),
            }
        if self.autocorrelation_time is not None:
            out["autocorrelation_time"] = self.autocorrelation_time.tolist()
        out["upper_limit_95"] = self.upper_limit(0.95)
        out["prior_interpretation"] = (
            "upper_limit" if abs(float(np.mean(self.samples[:, 0]))) < 2.0 * float(np.std(self.samples[:, 0]))
            else "interval"
        )
        return out


@dataclass(frozen=True)
class HLFitResult:
    """One-dimensional posterior and compact summaries from an HL fit."""

    r_true: float
    sim_idx: int
    entry_index: int
    data_index_within_r: int
    r_grid: np.ndarray
    chi2: np.ndarray
    loglike: np.ndarray
    posterior: np.ndarray
    r_map: float
    r_mean: float
    r_median: float
    interval_68: Tuple[float, float]
    upper_95: float
    data_bandpowers: np.ndarray

    def summary(self) -> Dict[str, object]:
        return {
            "r_true": self.r_true,
            "sim_idx": self.sim_idx,
            "entry_index": self.entry_index,
            "data_index_within_r": self.data_index_within_r,
            "r_map": self.r_map,
            "r_mean": self.r_mean,
            "r_median": self.r_median,
            "interval_68": self.interval_68,
            "upper_95": self.upper_95,
        }


def _canonical_r(value: float | str) -> str:
    r_value = float(value)
    if r_value < 0:
        raise ValueError("r must be >= 0")
    return f"{r_value:.6f}"


class _EntryPair(tuple):
    """Internal ``(entry_index, entry)`` pair with convenient field access."""

    def __new__(cls, entry_index: int, entry: Mapping[str, object]):
        return super().__new__(cls, (entry_index, entry))

    def __getitem__(self, key):
        if isinstance(key, str):
            return self[1][key]
        return super().__getitem__(key)


def _symmetric_matrix_power(matrix: np.ndarray, power: float, floor: float) -> np.ndarray:
    sym = 0.5 * (np.asarray(matrix, dtype=np.float64) + np.asarray(matrix).T)
    eigenvalues, eigenvectors = np.linalg.eigh(sym)
    scale = max(float(np.max(np.abs(eigenvalues))), float(np.trace(np.abs(sym))) / sym.shape[0], 1.0e-30)
    eigenvalues = np.maximum(eigenvalues, floor * scale)
    return (eigenvectors * np.power(eigenvalues, power)) @ eigenvectors.T


def _hl_g(eigenvalues: np.ndarray) -> np.ndarray:
    values = np.maximum(np.asarray(eigenvalues, dtype=np.float64), 1.0e-15)
    argument = np.maximum(2.0 * (values - np.log(values) - 1.0), 0.0)
    return np.sign(values - 1.0) * np.sqrt(argument)


def _vectorize_symmetric(matrices: np.ndarray) -> np.ndarray:
    """Vectorize symmetric matrices, scaling off-diagonals by sqrt(2)."""
    mats = np.asarray(matrices, dtype=np.float64)
    nfreq = mats.shape[-1]
    values = []
    for i in range(nfreq):
        for j in range(i, nfreq):
            factor = 1.0 if i == j else np.sqrt(2.0)
            values.append(factor * mats[..., i, j])
    return np.stack(values, axis=-1).reshape(-1)


def hl_transform(
    data: np.ndarray,
    model: np.ndarray,
    reference: np.ndarray,
    matrix_floor: float = 1.0e-8,
) -> np.ndarray:
    """Return the flattened Hamimeche-Lewis transform for binned spectra matrices."""
    data_arr = np.asarray(data, dtype=np.float64)
    model_arr = np.asarray(model, dtype=np.float64)
    reference_arr = np.asarray(reference, dtype=np.float64)
    if data_arr.shape != model_arr.shape or data_arr.shape != reference_arr.shape:
        raise ValueError("data, model, and reference must have identical shapes")
    if data_arr.ndim != 3 or data_arr.shape[-1] != data_arr.shape[-2]:
        raise ValueError("Expected arrays with shape (nbin, nfreq, nfreq)")

    transformed = np.empty_like(data_arr)
    for band in range(data_arr.shape[0]):
        model_inv_sqrt = _symmetric_matrix_power(model_arr[band], -0.5, matrix_floor)
        ratio = model_inv_sqrt @ data_arr[band] @ model_inv_sqrt
        ratio = 0.5 * (ratio + ratio.T)
        eigenvalues, eigenvectors = np.linalg.eigh(ratio)
        g_matrix = (eigenvectors * _hl_g(eigenvalues)) @ eigenvectors.T
        reference_sqrt = _symmetric_matrix_power(reference_arr[band], 0.5, matrix_floor)
        transformed[band] = reference_sqrt @ g_matrix @ reference_sqrt
    return _vectorize_symmetric(transformed)


class HLLikelihood:
    """Binned multifrequency HL likelihood with hit-map pixel treatment."""

    def __init__(
        self,
        config: HLConfig,
        hit_power: float = 0.5,
        hits_map_file: Optional[Path] = None,
    ) -> None:
        if hp is None:
            raise ImportError("healpy is required for HLLikelihood")
        self.config = config
        with open(config.manifest, "r", encoding="utf-8") as handle:
            self.manifest = json.load(handle)
        self.base_manifest = None
        if config.base_manifest is not None:
            with open(config.base_manifest, "r", encoding="utf-8") as handle:
                self.base_manifest = json.load(handle)
        if self.manifest.get("product") != "bmode":
            raise ValueError("HL likelihood requires a manifest with product='bmode'")
        if str(self.manifest.get("ordering", "")).upper() != "NEST":
            raise ValueError("HL likelihood requires NEST-ordered saved maps")
        if config.dataset_key is not None and self.manifest.get("dataset_key") != config.dataset_key:
            raise ValueError("Manifest dataset_key does not match HLConfig.dataset_key")

        self.entries: List[Mapping[str, object]] = list(self.manifest.get("entries", []))
        self.frequencies = tuple(str(value) for value in self.manifest.get("freq_values", []))
        self.nside = int(self.manifest.get("nside", 0))
        if not self.entries or not self.frequencies or self.nside <= 0:
            raise ValueError("Manifest is missing entries, frequencies, or nside")
        self.base_entries_by_key = {}
        if self.base_manifest is not None:
            if self.base_manifest.get("freq_values") != self.manifest.get("freq_values") or self.base_manifest.get("nside") != self.nside:
                raise ValueError("base_manifest must have the same frequencies and nside as manifest")
            for base_entry in self.base_manifest.get("entries", []):
                key = (_canonical_r(base_entry["r"]), int(base_entry["sim_idx"]))
                self.base_entries_by_key[key] = base_entry
            missing = [
                (_canonical_r(entry["r"]), int(entry["sim_idx"]))
                for entry in self.entries
                if (_canonical_r(entry["r"]), int(entry["sim_idx"])) not in self.base_entries_by_key
            ]
            if missing:
                raise ValueError(f"base_manifest is missing {len(missing)} matching augmented entries")
        if config.lmax > 3 * self.nside - 1:
            raise ValueError(f"lmax={config.lmax} exceeds the nside={self.nside} HEALPix limit")

        self.entries_by_r: Dict[str, List[Tuple[int, Mapping[str, object]]]] = {}
        for entry_index, entry in enumerate(self.entries):
            self.entries_by_r.setdefault(_canonical_r(str(entry["r"])), []).append(_EntryPair(entry_index, entry))
        self.r_values = np.asarray(sorted((float(key) for key in self.entries_by_r), key=float))
        if self.r_values.size < 2 or not np.any(self.r_values > 0):
            raise ValueError("HL r fitting requires at least two r values, including one r > 0")
        self.covariance_r_key = _canonical_r(config.covariance_r)
        if self.covariance_r_key not in self.entries_by_r:
            raise KeyError(
                f"covariance_r={self.covariance_r_key} is unavailable; "
                f"choose from {tuple(self.entries_by_r)}"
            )

        required = max(config.n_covariance, config.n_template)
        short = {key: len(values) for key, values in self.entries_by_r.items() if len(values) < required}
        if short:
            raise ValueError(f"Not enough simulations for requested covariance/template sizes: {short}")

        self.pixel_indices = self._load_pixel_indices()
        if hit_power < 0:
            raise ValueError("hit_power must be >= 0")
        self.hit_power = float(hit_power)
        if hits_map_file is None and self.manifest.get("mask_cache") is None:
            self.hits_map_file = None
            self.hit_map = np.ones(12 * self.nside**2, dtype=np.float64)
        else:
            self.hits_map_file = (
                Path(hits_map_file).expanduser().resolve()
                if hits_map_file is not None
                else Path(__file__).resolve().parent.parent / "data" / "hitsmap.fits.gz"
            )
            if not self.hits_map_file.is_file():
                raise FileNotFoundError(f"Hit-count map not found: {self.hits_map_file}")
            hits = hp.read_map(str(self.hits_map_file))
            if hp.get_nside(hits) != self.nside:
                hits = hp.ud_grade(hits, nside_out=self.nside)
            observed = (hits > 0).astype(np.float64)
            hits_norm = np.clip(hits / hits.max(), 1.0e-3, None)
            self.hit_map = observed * np.power(hits_norm, self.hit_power)
        self.bin_edges = self._make_bin_edges()
        self.effective_ells = self._effective_ells()
        cache_root = config.cache_dir or (config.manifest.parent / "cache" / "hl")
        self.cache_dir = Path(cache_root)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cmb_spectra_file = self._resolve_cmb_spectra_file()

        self.template_bandpowers: Optional[np.ndarray] = None
        self.template_means: Optional[np.ndarray] = None
        self.model_intercept: Optional[np.ndarray] = None
        self.model_slope: Optional[np.ndarray] = None
        self.tensor_template: Optional[np.ndarray] = None
        self.lensing_template: Optional[np.ndarray] = None
        self.non_cmb_baseline: Optional[np.ndarray] = None
        self.theory_tensor_bandpowers: Optional[np.ndarray] = None
        self.theory_lensing_bandpowers: Optional[np.ndarray] = None
        self.foreground_template: Optional[np.ndarray] = None
        self.foreground_template_files: Optional[Dict[str, Path]] = None
        self.transfer_response: Optional[np.ndarray] = None
        self.reference: Optional[np.ndarray] = None
        self.transform_mean: Optional[np.ndarray] = None
        self.covariance: Optional[np.ndarray] = None
        self.inverse_covariance: Optional[np.ndarray] = None
        self.hartlap_factor: Optional[float] = None

    @property
    def n_bins(self) -> int:
        return len(self.bin_edges) - 1

    @property
    def n_spectra(self) -> int:
        nfreq = len(self.frequencies)
        return nfreq * (nfreq + 1) // 2

    @property
    def data_vector_size(self) -> int:
        return self.n_bins * self.n_spectra

    def _load_pixel_indices(self) -> np.ndarray:
        cache_file = self.manifest.get("mask_cache")
        if cache_file is None:
            return np.arange(12 * self.nside**2, dtype=np.int64)
        path = Path(str(cache_file)).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Mask cache referenced by manifest does not exist: {path}")
        with np.load(path) as payload:
            return np.asarray(payload["pixel_idx"], dtype=np.int64)

    def _resolve_cmb_spectra_file(self) -> Path:
        if self.config.cmb_spectra_file is not None:
            return self.config.cmb_spectra_file
        candidates = [
            self.config.manifest.parent / "cmb_spectra.pkl",
            self.config.manifest.parents[2] / "cmb_spectra.pkl",
        ]
        first_files = self.entries[0].get("files")
        if isinstance(first_files, Mapping) and first_files:
            sample_path = Path(str(next(iter(first_files.values())))).expanduser().resolve()
            for parent in sample_path.parents:
                if parent.name == "simulations":
                    candidates.append(parent.parent / "cmb_spectra.pkl")
                    break
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        checked = "\n".join(f"- {candidate}" for candidate in candidates)
        raise FileNotFoundError(
            "Could not locate the cmb_spectra.pkl used by the simulations. "
            "Pass HLConfig(cmb_spectra_file=...). Checked:\n" + checked
        )

    def _make_bin_edges(self) -> np.ndarray:
        edges = np.arange(self.config.lmin, self.config.lmax + 1, self.config.bin_width, dtype=int)
        if edges.size == 0 or edges[0] != self.config.lmin:
            edges = np.insert(edges, 0, self.config.lmin)
        if edges[-1] != self.config.lmax + 1:
            edges = np.append(edges, self.config.lmax + 1)
        return edges

    def _effective_ells(self) -> np.ndarray:
        ell = np.arange(self.config.lmax + 1, dtype=np.float64)
        out = []
        for lo, hi in zip(self.bin_edges[:-1], self.bin_edges[1:]):
            select = (ell >= lo) & (ell < hi)
            weights = 2.0 * ell[select] + 1.0
            out.append(np.average(ell[select], weights=weights))
        return np.asarray(out)

    def _bin_theory_cl(self, cl: np.ndarray) -> np.ndarray:
        values = np.asarray(cl, dtype=np.float64)
        if values.size <= self.config.lmax:
            raise ValueError(
                f"Theory spectrum has lmax={values.size - 1}, below requested lmax={self.config.lmax}"
            )
        ell = np.arange(self.config.lmax + 1, dtype=np.float64)
        dl = values[: self.config.lmax + 1] * ell * (ell + 1.0) / (2.0 * np.pi)
        binned = []
        for lo, hi in zip(self.bin_edges[:-1], self.bin_edges[1:]):
            multipoles = np.arange(lo, hi)
            weights = 2.0 * multipoles + 1.0
            binned.append(float(np.average(dl[lo:hi], weights=weights)))
        return np.asarray(binned)

    def _build_cmb_templates(self) -> None:
        if self.model_intercept is None or self.model_slope is None:
            raise RuntimeError("Empirical r templates must be fitted before CMB decomposition")
        empirical_intercept = np.asarray(self.model_intercept, dtype=np.float64)
        if self.config.use_theory_cmb:
            with self.cmb_spectra_file.open("rb") as handle:
                spectra = pickle.load(handle)
            tensor_cl = np.asarray(spectra["cls_r1"]["tensor"][:, 2], dtype=np.float64)
            lensing_cl = np.asarray(spectra["cls"]["lensed_scalar"][:, 2], dtype=np.float64)
            tensor_band = self._bin_theory_cl(tensor_cl)
            lensing_band = self._bin_theory_cl(lensing_cl)
            self.theory_tensor_bandpowers = tensor_band
            self.theory_lensing_bandpowers = lensing_band
            # The simulations contain hit weighting and a partial-sky
            # pseudo-spectrum response. Calibrate the full-sky theory to the
            # same response using the empirical r-slope cross-spectrum.
            empirical_slope = np.asarray(self.model_slope, dtype=np.float64)
            response = empirical_slope[:, 0, 1] / np.maximum(tensor_band, self.config.matrix_floor)
            positive = np.isfinite(response) & (response > 0)
            fallback = float(np.median(response[positive])) if np.any(positive) else 1.0
            response = np.where(positive, response, fallback)
            self.transfer_response = response
            self.tensor_template = response[:, None, None] * tensor_band[:, None, None] * np.ones_like(empirical_intercept)
            self.lensing_template = response[:, None, None] * lensing_band[:, None, None] * np.ones_like(empirical_intercept)
            self.non_cmb_baseline = empirical_intercept - self.lensing_template
        else:
            self.tensor_template = np.asarray(self.model_slope, dtype=np.float64)
            self.lensing_template = np.zeros_like(self.tensor_template)
            self.non_cmb_baseline = empirical_intercept
            self.theory_tensor_bandpowers = None
            self.theory_lensing_bandpowers = None
    def _build_foreground_template(self) -> None:
        """Build a fixed PySM multifrequency foreground pseudo-spectrum."""
        self.foreground_template = None
        self.foreground_template_files = None
        if not self.config.use_theory_foreground:
            return
        if self.base_manifest is None:
            raise ValueError("use_theory_foreground=True requires HLConfig(base_manifest=...)")
        try:
            from ..sims.foreground_suite import ForegroundAugmentedSimulator
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("PySM foregrounds require the foreground simulation dependencies") from exc

        basedir = self.config.foreground_basedir or self.config.manifest.parent
        simulator = ForegroundAugmentedSimulator(
            basedir=basedir,
            fg_preset=self.config.foreground_preset,
            nside=self.nside,
            lmax=self.config.lmax,
            lmin=self.config.lmin,
            seed_base=(
                int(self.config.foreground_seed_base)
                if self.config.foreground_seed_base is not None
                else int(self.manifest.get("foreground", {}).get("seed_base", 50000))
            ),
            output_root=self.config.foreground_output_root,
        )
        maps = []
        files: Dict[str, Path] = {}
        for frequency in self.frequencies:
            cache_file = simulator._foreground_cache_file(float(frequency), self.config.foreground_sim_idx)
            maps.append(simulator.foreground_bmode(float(frequency), self.config.foreground_sim_idx))
            files[frequency] = cache_file
        self.foreground_template_files = files
        # The cached PySM maps already contain the binary scientific mask;
        # apply the same hit weighting used for every likelihood map.
        self.foreground_template = self._bandpowers_from_maps(np.asarray(maps) * self.hit_map[None, :])

    def _base_zero_mean(self) -> Optional[np.ndarray]:
        """Return the matched CMB+noise r=0 mean for an augmented suite."""
        if self.base_manifest is None:
            return None
        cache_file = self._base_mean_cache_file()
        if cache_file.is_file():
            with np.load(cache_file) as payload:
                return np.asarray(payload["mean"], dtype=np.float64)
        entries = [
            entry for entry in self.base_manifest.get("entries", [])
            if _canonical_r(entry["r"]) == _canonical_r(0.0)
        ]
        if len(entries) < self.config.n_template:
            raise ValueError("Base manifest has too few r=0 entries for the requested template")
        entries.sort(key=lambda entry: int(entry["sim_idx"]))
        mean = np.mean(
            [self.bandpowers_for_entry(entry) for entry in entries[: self.config.n_template]],
            axis=0,
        )
        np.savez_compressed(cache_file, mean=mean)
        return np.asarray(mean, dtype=np.float64)

    def _cache_file(self) -> Path:
        payload = {
            "dataset_key": self.manifest.get("dataset_key"),
            "manifest": str(self.config.manifest),
            "r_values": self.r_values.tolist(),
            "frequencies": self.frequencies,
            "lmin": self.config.lmin,
            "lmax": self.config.lmax,
            "bin_width": self.config.bin_width,
            "map2alm_iter": self.config.map2alm_iter,
            "n_template": self.config.n_template,
            "n_covariance": self.config.n_covariance,
            "hit_power": self.hit_power,
            "hits_map_file": None if self.hits_map_file is None else str(self.hits_map_file),
            "base_manifest": None if self.config.base_manifest is None else str(self.config.base_manifest),
            "use_theory_cmb": self.config.use_theory_cmb,
            "use_theory_foreground": self.config.use_theory_foreground,
            "foreground_preset": self.config.foreground_preset,
            "foreground_sim_idx": self.config.foreground_sim_idx,
            "foreground_seed_base": self.config.foreground_seed_base,
            "foreground_basedir": None if self.config.foreground_basedir is None else str(self.config.foreground_basedir),
            "foreground_output_root": None if self.config.foreground_output_root is None else str(self.config.foreground_output_root),
        }
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
        return self.cache_dir / f"bandpowers_{digest}.npz"

    def _base_mean_cache_file(self) -> Path:
        payload = {
            "base_manifest": None if self.config.base_manifest is None else str(self.config.base_manifest),
            "frequencies": self.frequencies,
            "nside": self.nside,
            "lmin": self.config.lmin,
            "lmax": self.config.lmax,
            "bin_width": self.config.bin_width,
            "map2alm_iter": self.config.map2alm_iter,
            "n_template": self.config.n_template,
            "hit_power": self.hit_power,
            "hits_map_file": None if self.hits_map_file is None else str(self.hits_map_file),
        }
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
        return self.cache_dir / f"base_mean_{digest}.npz"

    def _load_selected_maps(self, entry: Mapping[str, object], pixel_indices: Optional[np.ndarray] = None) -> np.ndarray:
        files = entry.get("files")
        if not isinstance(files, Mapping):
            raise ValueError("Simulation entry has no valid frequency file mapping")
        maps = []
        selected_pixels = self.pixel_indices if pixel_indices is None else np.asarray(pixel_indices, dtype=np.int64)
        npix = 12 * self.nside**2
        for frequency in self.frequencies:
            path = Path(str(files[frequency])).expanduser()
            if not path.is_file():
                raise FileNotFoundError(f"Simulation map does not exist: {path}")
            with np.load(path) as payload:
                selected = np.asarray(payload["b_selected"], dtype=np.float64)
            if selected.shape != selected_pixels.shape:
                raise ValueError(f"Selected map shape mismatch in {path}")
            full_nest = np.zeros(npix, dtype=np.float64)
            full_nest[selected_pixels] = selected
            maps.append(hp.reorder(full_nest, n2r=True) * self.hit_map)
        return np.asarray(maps)

    def _bandpowers_from_maps(self, maps: np.ndarray) -> np.ndarray:
        """Compute binned spectra from full-sky RING maps already weighted by hits."""
        alms = [
            hp.map2alm(map_values, lmax=self.config.lmax, iter=self.config.map2alm_iter,
                       pol=False, use_pixel_weights=False)
            for map_values in np.asarray(maps)
        ]
        ell = np.arange(self.config.lmax + 1, dtype=np.float64)
        dl_factor = ell * (ell + 1.0) / (2.0 * np.pi)
        matrices = np.empty((self.n_bins, len(alms), len(alms)), dtype=np.float64)
        for i, alm_i in enumerate(alms):
            for j in range(i, len(alms)):
                dl = hp.alm2cl(alm_i, alms[j], lmax=self.config.lmax) * dl_factor
                for band, (lo, hi) in enumerate(zip(self.bin_edges[:-1], self.bin_edges[1:])):
                    multipoles = np.arange(lo, hi)
                    weights = 2.0 * multipoles + 1.0
                    value = float(np.average(dl[lo:hi], weights=weights))
                    matrices[band, i, j] = value
                    matrices[band, j, i] = value
        return matrices

    def bandpowers_for_entry(self, entry: Mapping[str, object]) -> np.ndarray:
        """Compute binned auto/cross pseudo-spectra for one multifrequency entry."""
        return self._bandpowers_from_maps(self._load_selected_maps(entry))

    def _build_bandpower_library(self, verbose: bool) -> np.ndarray:
        n_use = max(self.config.n_covariance, self.config.n_template)
        shape = (len(self.r_values), n_use, self.n_bins, len(self.frequencies), len(self.frequencies))
        library = np.empty(shape, dtype=np.float64)
        jobs = [(r_index, sim_index) for r_index in range(len(self.r_values)) for sim_index in range(n_use)]
        iterator = tqdm(jobs, desc="HL bandpowers", unit="map set") if verbose and tqdm is not None else jobs
        for r_index, sim_index in iterator:
            r_key = _canonical_r(self.r_values[r_index])
            _, entry = self.entries_by_r[r_key][sim_index]
            library[r_index, sim_index] = self.bandpowers_for_entry(entry)
        np.savez_compressed(
            self._cache_file(),
            bandpowers=library,
            r_values=self.r_values,
            effective_ells=self.effective_ells,
            bin_edges=self.bin_edges,
        )
        return library

    def prepare(self, force: bool = False, verbose: bool = True) -> "HLLikelihood":
        """Build/load templates and estimate the fiducial transformed covariance."""
        cache_file = self._cache_file()
        if cache_file.is_file() and not force:
            with np.load(cache_file) as payload:
                library = np.asarray(payload["bandpowers"], dtype=np.float64)
        else:
            library = self._build_bandpower_library(verbose=verbose)
        self.template_bandpowers = library
        self.template_means = library[:, : self.config.n_template].mean(axis=1)

        matrix_shape = self.template_means.shape[1:]
        zero_indices = np.flatnonzero(np.isclose(self.r_values, 0.0))
        flattened_means = self.template_means.reshape(len(self.r_values), -1)
        if zero_indices.size:
            intercept = flattened_means[int(zero_indices[0])]
            denominator = float(self.r_values @ self.r_values)
            slope = (self.r_values @ (flattened_means - intercept)) / denominator
            self.model_intercept = intercept.reshape(matrix_shape)
            self.model_slope = slope.reshape(matrix_shape)
        else:
            design = np.column_stack([np.ones_like(self.r_values), self.r_values])
            coefficients = np.linalg.lstsq(design, flattened_means, rcond=None)[0]
            self.model_intercept = coefficients[0].reshape(matrix_shape)
            self.model_slope = coefficients[1].reshape(matrix_shape)
        base_mean = self._base_zero_mean()
        if base_mean is not None:
            # The matched base suite provides the CMB+noise baseline directly;
            # do not estimate it by subtracting a foreground realization.
            self.model_intercept = np.asarray(base_mean, dtype=np.float64)
        self._build_cmb_templates()
        self._build_foreground_template()
        self.reference = self.model_at(float(self.covariance_r_key))

        covariance_index = int(np.where(np.isclose(self.r_values, float(self.covariance_r_key)))[0][0])
        transformed = np.asarray(
            [
                hl_transform(sample, self.reference, self.reference, self.config.matrix_floor)
                for sample in library[covariance_index, : self.config.n_covariance]
            ]
        )
        self.transform_mean = transformed.mean(axis=0)
        covariance = np.cov(transformed, rowvar=False, ddof=1)
        diagonal_scale = max(float(np.median(np.diag(covariance))), 1.0e-30)
        covariance = covariance + self.config.covariance_ridge * diagonal_scale * np.eye(covariance.shape[0])
        inverse = np.linalg.pinv(covariance, hermitian=True)

        n_sim = self.config.n_covariance
        n_data = covariance.shape[0]
        hartlap = (n_sim - n_data - 2.0) / (n_sim - 1.0)
        if self.config.apply_hartlap:
            if hartlap <= 0:
                raise ValueError(
                    f"Hartlap correction is undefined for {n_sim} simulations and {n_data} data bins"
                )
            inverse *= hartlap
        else:
            hartlap = 1.0
        self.covariance = covariance
        self.inverse_covariance = inverse
        self.hartlap_factor = float(hartlap)
        return self

    def _require_prepared(self) -> None:
        if self.inverse_covariance is None or self.reference is None or self.transform_mean is None:
            raise RuntimeError("Call prepare() before evaluating or fitting the likelihood")

    def model_at(self, r: float, alens: float = 1.0, foreground_amplitude: float = 1.0) -> np.ndarray:
        """Evaluate the model at ``(r, A_lens, A_fg)``."""
        if self.non_cmb_baseline is None or self.tensor_template is None or self.lensing_template is None:
            raise RuntimeError("Call prepare() before requesting the model")
        if alens < 0:
            raise ValueError("A_lens must be non-negative")
        if foreground_amplitude < 0:
            raise ValueError("Foreground amplitude must be non-negative")
        model = (
            self.non_cmb_baseline
            + float(r) * self.tensor_template
            + float(alens) * self.lensing_template
        )
        if self.foreground_template is not None:
            model = model + float(foreground_amplitude) * self.foreground_template
        regularized = np.empty_like(model)
        for band, matrix in enumerate(model):
            eigenvalues = np.linalg.eigvalsh(0.5 * (matrix + matrix.T))
            if float(np.min(eigenvalues)) <= 0:
                raise ValueError(
                    f"The spectral model is not positive definite at r={r}, A_lens={alens}"
                )
            regularized[band] = _symmetric_matrix_power(matrix, 1.0, self.config.matrix_floor)
        return regularized

    def chi2_at(self, data_bandpowers: np.ndarray, r: float, alens: float = 1.0, foreground_amplitude: float = 1.0) -> float:
        self._require_prepared()
        vector = hl_transform(
            data_bandpowers,
            self.model_at(r, alens=alens, foreground_amplitude=foreground_amplitude),
            self.reference,
            self.config.matrix_floor,
        )
        vector = vector - self.transform_mean
        return float(vector @ self.inverse_covariance @ vector)

    def _resolve_data_entry(
        self,
        r: float | str,
        data_index: Optional[int],
    ) -> Tuple[int, int, Mapping[str, object]]:
        r_key = _canonical_r(r)
        if r_key not in self.entries_by_r:
            raise KeyError(f"r={r_key} is unavailable; choose from {tuple(self.entries_by_r)}")
        index = max(self.config.n_covariance, self.config.n_template) if data_index is None else int(data_index)
        entries = self.entries_by_r[r_key]
        if index < 0 or index >= len(entries):
            raise IndexError(f"data_index={index} is outside [0, {len(entries) - 1}] for r={r_key}")
        entry_index, entry = entries[index]
        return index, entry_index, entry

    def fit(
        self,
        r: float | str,
        data_index: Optional[int] = None,
        r_grid: Optional[Sequence[float]] = None,
        alens: float = 1.0,
    ) -> HLFitResult:
        """Fit an independent realization selected by its r key and within-r index."""
        self._require_prepared()
        within_r, entry_index, entry = self._resolve_data_entry(r, data_index)
        data_bandpowers = self.bandpowers_for_entry(entry)
        if r_grid is None:
            upper = max(0.05, 1.25 * float(self.r_values.max()))
            scan = np.linspace(-0.01, upper, 601)
        else:
            scan = np.asarray(r_grid, dtype=np.float64)
        if scan.ndim != 1 or scan.size < 3 or np.any(np.diff(scan) <= 0):
            raise ValueError("r_grid must be a strictly increasing one-dimensional array")

        chi2_values = []
        for value in scan:
            try:
                chi2_values.append(self.chi2_at(data_bandpowers, value, alens=alens))
            except (ValueError, np.linalg.LinAlgError):
                chi2_values.append(np.inf)
        chi2 = np.asarray(chi2_values)
        if not np.any(np.isfinite(chi2)):
            raise RuntimeError("No r-grid point produced a positive-definite spectral model")
        loglike = -0.5 * (chi2 - chi2.min())
        density = np.exp(loglike)
        normalization = float(np.trapezoid(density, scan))
        if not np.isfinite(normalization) or normalization <= 0:
            raise RuntimeError("Could not normalize the r posterior")
        posterior = density / normalization
        cumulative = np.zeros_like(scan)
        cumulative[1:] = np.cumsum(0.5 * (posterior[1:] + posterior[:-1]) * np.diff(scan))
        cumulative /= cumulative[-1]

        quantile = lambda probability: float(np.interp(probability, cumulative, scan))
        r_mean = float(np.trapezoid(scan * posterior, scan))
        return HLFitResult(
            r_true=float(entry["r"]),
            sim_idx=int(entry["sim_idx"]),
            entry_index=entry_index,
            data_index_within_r=within_r,
            r_grid=scan,
            chi2=chi2,
            loglike=loglike,
            posterior=posterior,
            r_map=float(scan[int(np.argmin(chi2))]),
            r_mean=r_mean,
            r_median=quantile(0.5),
            interval_68=(quantile(0.16), quantile(0.84)),
            upper_95=quantile(0.95),
            data_bandpowers=data_bandpowers,
        )

    def sample(
        self,
        r: float | str,
        data_index: Optional[int] = None,
        mcmc_config: Optional[MCMCConfig] = None,
        backend_file: Optional[Path | str] = None,
        resume: bool = True,
    ) -> HLMCMCResult:
        """Sample r and optionally A_lens with emcee for one saved realization."""
        self._require_prepared()
        try:
            import emcee
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError("emcee is required; install it with `pip install emcee`") from exc

        settings = MCMCConfig() if mcmc_config is None else mcmc_config
        within_r, entry_index, entry = self._resolve_data_entry(r, data_index)
        data_bandpowers = self.bandpowers_for_entry(entry)
        if settings.vary_foreground and self.foreground_template is None:
            raise ValueError("vary_foreground=True requires HLConfig(base_manifest=...) for an augmented manifest")
        names = ["r"]
        labels = ["r"]
        bounds = [settings.r_bounds]
        if settings.vary_alens:
            names.append("Alens"); labels.append("A_{\\rm lens}"); bounds.append(settings.alens_bounds)
        if settings.vary_foreground:
            names.append("Afg"); labels.append("A_{\\rm fg}"); bounds.append(settings.foreground_amplitude_bounds)
        names = tuple(names); labels = tuple(labels); bounds = tuple(bounds)
        ndim = len(names)

        def log_probability(parameters: np.ndarray) -> float:
            values = np.asarray(parameters, dtype=np.float64)
            if values.shape != (ndim,):
                return -np.inf
            if any(not (lower < value < upper) for value, (lower, upper) in zip(values, bounds)):
                return -np.inf
            r_value = float(values[0])
            offset = 1
            if settings.vary_alens:
                alens_value = float(values[offset]); offset += 1
            else:
                alens_value = settings.fixed_alens
            if settings.vary_foreground:
                foreground_value = float(values[offset])
            else:
                foreground_value = settings.fixed_foreground_amplitude
            try:
                chi2 = self.chi2_at(data_bandpowers, r_value, alens=alens_value, foreground_amplitude=foreground_value)
            except (ValueError, np.linalg.LinAlgError):
                return -np.inf
            return -0.5 * chi2 if np.isfinite(chi2) else -np.inf

        if backend_file is None:
            mode = "r" + ("_alens" if settings.vary_alens else f"_alens{settings.fixed_alens:.3f}") + ("_fg" if settings.vary_foreground else f"_fg{settings.fixed_foreground_amplitude:.3f}")
            r_tag = _canonical_r(entry["r"]).replace(".", "p")
            chain_signature = hashlib.sha256(
                json.dumps(
                    {
                        "dataset_key": self.manifest.get("dataset_key"),
                        "entry_index": entry_index,
                        "covariance_r": self.covariance_r_key,
                        "lmin": self.config.lmin,
                        "lmax": self.config.lmax,
                        "bin_width": self.config.bin_width,
                        "n_covariance": self.config.n_covariance,
                        "n_template": self.config.n_template,
                        "n_walkers": settings.n_walkers,
                        "vary_alens": settings.vary_alens,
                        "fixed_alens": settings.fixed_alens,
                        "vary_foreground": settings.vary_foreground,
                        "fixed_foreground_amplitude": settings.fixed_foreground_amplitude,
                        "foreground_amplitude_bounds": settings.foreground_amplitude_bounds,
                        "base_manifest": None if self.config.base_manifest is None else str(self.config.base_manifest),
                        "use_theory_cmb": self.config.use_theory_cmb,
                        "use_theory_foreground": self.config.use_theory_foreground,
                        "foreground_preset": self.config.foreground_preset,
                        "foreground_sim_idx": self.config.foreground_sim_idx,
                        "foreground_seed_base": self.config.foreground_seed_base,
                        "foreground_basedir": None if self.config.foreground_basedir is None else str(self.config.foreground_basedir),
                        "foreground_output_root": None if self.config.foreground_output_root is None else str(self.config.foreground_output_root),
                        "r_bounds": settings.r_bounds,
                        "alens_bounds": settings.alens_bounds,
                        "hit_power": self.hit_power,
                        "hits_map_file": None if self.hits_map_file is None else str(self.hits_map_file),
                    },
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest()[:10]
            backend_path = self.cache_dir / "chains" / (
                f"hl_{mode}_r{r_tag}_data{within_r:04d}_{chain_signature}.h5"
            )
        else:
            backend_path = Path(backend_file).expanduser().resolve()
        backend_path.parent.mkdir(parents=True, exist_ok=True)

        backend_existed = backend_path.is_file()
        backend = emcee.backends.HDFBackend(str(backend_path))
        rng = np.random.default_rng(settings.seed)
        initial_state = None
        previous_steps = 0
        if backend_existed and resume:
            try:
                shape = backend.shape
                previous_steps = int(backend.iteration)
            except (AttributeError, OSError, KeyError):
                shape = None
            if shape is not None and shape != (settings.n_walkers, ndim):
                raise ValueError(
                    f"Existing emcee backend has shape {shape}, expected "
                    f"({settings.n_walkers}, {ndim}): {backend_path}"
                )
        else:
            backend.reset(settings.n_walkers, ndim)

        if previous_steps == 0:
            initial_state = np.column_stack(
                [rng.uniform(lower, upper, size=settings.n_walkers) for lower, upper in bounds]
            )
        sampler = emcee.EnsembleSampler(
            settings.n_walkers,
            ndim,
            log_probability,
            backend=backend,
        )
        remaining = max(0, settings.n_steps - previous_steps)
        if remaining:
            sampler.run_mcmc(initial_state, remaining, progress=settings.progress)

        completed_steps = int(backend.iteration)
        if completed_steps <= settings.burn_in:
            raise RuntimeError(
                f"Chain has {completed_steps} steps, not enough for burn_in={settings.burn_in}"
            )
        samples = np.asarray(
            backend.get_chain(discard=settings.burn_in, thin=settings.thin, flat=True),
            dtype=np.float64,
        )
        log_prob = np.asarray(
            backend.get_log_prob(discard=settings.burn_in, thin=settings.thin, flat=True),
            dtype=np.float64,
        )
        finite = np.isfinite(log_prob) & np.all(np.isfinite(samples), axis=1)
        samples = samples[finite]
        log_prob = log_prob[finite]
        if samples.size == 0:
            raise RuntimeError("The emcee chain contains no finite post-burn-in samples")
        try:
            autocorrelation = np.asarray(backend.get_autocorr_time(tol=0), dtype=np.float64)
        except Exception:
            autocorrelation = None

        return HLMCMCResult(
            samples=samples,
            log_probability=log_prob,
            parameter_names=names,
            parameter_labels=labels,
            bounds=bounds,
            r_true=float(entry["r"]),
            alens_true=self.config.alens_fiducial,
            sim_idx=int(entry["sim_idx"]),
            entry_index=entry_index,
            data_index_within_r=within_r,
            backend_file=backend_path,
            acceptance_fraction=float(np.mean(sampler.acceptance_fraction)),
            autocorrelation_time=autocorrelation,
        )

    def metadata(self) -> Dict[str, object]:
        """Return notebook-friendly likelihood and dataset metadata."""
        config = asdict(self.config)
        config["manifest"] = str(config["manifest"])
        if config["cache_dir"] is not None:
            config["cache_dir"] = str(config["cache_dir"])
        if config["cmb_spectra_file"] is not None:
            config["cmb_spectra_file"] = str(config["cmb_spectra_file"])
        if config["base_manifest"] is not None:
            config["base_manifest"] = str(config["base_manifest"])
        return {
            "config": config,
            "dataset_key": self.manifest.get("dataset_key"),
            "frequencies_ghz": tuple(float(value) for value in self.frequencies),
            "r_values": self.r_values.copy(),
            "n_bins": self.n_bins,
            "data_vector_size": self.data_vector_size,
            "effective_ells": self.effective_ells.copy(),
            "cache_file": str(self._cache_file()),
            "cmb_spectra_file": str(self.cmb_spectra_file),
            "alens_fiducial": self.config.alens_fiducial,
            "hartlap_factor": self.hartlap_factor,
        }

    def covariance_r_sweep(
        self,
        data_r: float | str,
        covariance_values: Sequence[float],
        data_index: Optional[int] = None,
        r_grid: Optional[Sequence[float]] = None,
    ) -> List[Dict[str, float]]:
        """Refit fixed data while rebuilding the covariance at each supplied r."""
        results: List[Dict[str, float]] = []
        original_r = self.config.covariance_r
        for covariance_r in covariance_values:
            if not np.isclose(float(covariance_r), original_r):
                config = HLConfig(**{**asdict(self.config), "covariance_r": float(covariance_r)})
                replacement = HLLikelihood(config).prepare(force=True, verbose=False)
            else:
                replacement = self
            fit = replacement.fit(data_r, data_index=data_index, r_grid=r_grid)
            sigma = 0.5 * (fit.interval_68[1] - fit.interval_68[0])
            results.append({
                "covariance_r": float(covariance_r),
                "r_map": float(fit.r_map),
                "r_mean": float(fit.r_mean),
                "sigma_r": float(sigma),
            })
        return results
