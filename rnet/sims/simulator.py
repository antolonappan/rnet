from __future__ import annotations

import json
import hashlib
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import healpy as hp
from deepsphere import utils as ds_utils
from tqdm.auto import tqdm
from .sky import Sky


class RGridSimulator:
	"""Run and persist map simulations over an extensible tensor-to-scalar r grid."""

	DEFAULT_R_VALUES: Sequence[float] = (
		0.0,
		0.0005,
		0.001,
		0.002,
		0.003,
		0.004,
		0.005,
		0.0075,
		0.010,
		0.015,
		0.020,
		0.030,
	)

	def __init__(
		self,
		basedir: Optional[Path | str] = None,
		sky: Optional[Sky] = None,
		r_values: Optional[Sequence[float]] = None,
		sky_kwargs: Optional[Dict[str, object]] = None,
		padded_mask_file: Optional[Path | str] = None,
		shuffle_seed: int = 12345,
		mask_threshold: float = 0.5,
		padding_rings: int = 1,
		deepsphere_pad_nside_out: int = 8,
		alens: float = 1.0,
		r_prior: Optional[Tuple[float, float]] = None,
	) -> None:
		self.basedir = Path(basedir) if basedir is not None else Path.cwd()
		self.basedir.mkdir(parents=True, exist_ok=True)

		if sky is not None:
			self.sky = sky
		else:
			kwargs = {} if sky_kwargs is None else dict(sky_kwargs)
			kwargs.setdefault("basedir", self.basedir)
			self.sky = Sky(**kwargs)

		self._root = self.basedir / "simulations" / "r_grid"
		self._root.mkdir(parents=True, exist_ok=True)
		self._registry_file = self._root / "registry.json"
		self._mask_cache_dir = self._root / "cache" / "masks"
		self._mask_cache_dir.mkdir(parents=True, exist_ok=True)
		self._dataset_cache_dir = self._root / "cache" / "datasets"
		self._dataset_cache_dir.mkdir(parents=True, exist_ok=True)
		self._active_dataset_file = self._root / "active_dataset.json"
		self._padded_mask_file = self._resolve_padded_mask_file(padded_mask_file)

		self._r_values = self._normalize_r_values(
			self.DEFAULT_R_VALUES if r_values is None else r_values
		)
		self._shuffle_seed = int(shuffle_seed)
		self._mask_threshold = float(mask_threshold)
		if self._mask_threshold < 0.0:
			raise ValueError("mask_threshold must be >= 0")
		self._padding_rings = int(padding_rings)
		if self._padding_rings < 0:
			raise ValueError("padding_rings must be >= 0")
		self._deepsphere_pad_nside_out = int(deepsphere_pad_nside_out)
		self.alens = float(alens)
		if self.alens < 0:
			raise ValueError("alens must be >= 0")
		self.r_prior = None if r_prior is None else (float(r_prior[0]), float(r_prior[1]))
		if self.r_prior is not None and (self.r_prior[0] < 0 or self.r_prior[0] >= self.r_prior[1]):
			raise ValueError("r_prior must be an increasing non-negative pair")
		if self._deepsphere_pad_nside_out <= 0:
			raise ValueError("deepsphere_pad_nside_out must be > 0")

		self._registry = self._load_registry()
		self._ensure_r_keys(self._r_values)


	def _resolve_padded_mask_file(self, padded_mask_file: Optional[Path | str]) -> Optional[Path]:
		if padded_mask_file is not None:
			path = Path(padded_mask_file).expanduser().resolve()
			if not path.is_file():
				raise FileNotFoundError(f"padded_mask_file not found: {path}")
			return path

		candidates = [
			Path(__file__).resolve().parents[1] / "data" / "mask_padded.fits",
			self.basedir / "data" / "mask_padded.fits",
			self.basedir / "mask_padded.fits",
		]
		for c in candidates:
			if c.is_file():
				return c.resolve()
		return None

	@staticmethod
	def _canonical_r(r: float) -> str:
		r_use = float(r)
		if r_use < 0.0:
			raise ValueError("r must be >= 0")
		return f"{r_use:.6f}"

	@staticmethod
	def _r_tag(r: float) -> str:
		return RGridSimulator._canonical_r(r).replace(".", "p")

	def _assignment_key(self, r: float) -> str:
		base = self._canonical_r(r) if self.r_prior is None else (
			f"prior:{self.r_prior[0]:.6f}:{self.r_prior[1]:.6f}"
		)
		return base if self.alens == 1.0 else f"alens:{self.alens:.6f}|{base}"

	def _prior_r(self, sim_idx: int) -> float:
		if self.r_prior is None:
			raise RuntimeError("continuous r prior is not configured")
		seed = (12345 + int(sim_idx) * 1664525) % (np.iinfo(np.int32).max - 1)
		return float(np.random.default_rng(seed + 1).uniform(*self.r_prior))

	@staticmethod
	def _freq_tag(freq: float) -> str:
		return f"{float(freq):.3f}".replace(".", "p")

	@staticmethod
	def _canonical_freq(freq: float) -> str:
		return f"{float(freq):.3f}"

	@staticmethod
	def _normalize_freq_values(freq: float | Sequence[float]) -> List[float]:
		if isinstance(freq, (int, float, np.integer, np.floating)):
			return [float(freq)]
		if isinstance(freq, (str, bytes)):
			raise ValueError("freq must be a number or a sequence of numbers")

		out: List[float] = []
		seen = set()
		for value in freq:
			fval = float(value)
			if fval in seen:
				continue
			seen.add(fval)
			out.append(fval)

		if len(out) == 0:
			raise ValueError("freq sequence must not be empty")
		return out

	@staticmethod
	def _normalize_r_values(r_values: Iterable[float]) -> List[float]:
		tags = {RGridSimulator._canonical_r(r) for r in r_values}
		return [float(tag) for tag in sorted(tags, key=float)]

	@staticmethod
	def _normalize_int_list(values: Iterable[int]) -> List[int]:
		seen = set()
		out: List[int] = []
		for v in values:
			iv = int(v)
			if iv in seen:
				continue
			seen.add(iv)
			out.append(iv)
		return out

	def _new_registry(self) -> Dict[str, object]:
		now = datetime.now(timezone.utc).isoformat()
		return {
			"version": 1,
			"next_idx": 0,
			"created_utc": now,
			"updated_utc": now,
			"r_assignments": {},
		}

	def _load_registry(self) -> Dict[str, object]:
		if not self._registry_file.is_file():
			registry = self._new_registry()
			self._save_registry(registry)
			return registry

		with open(self._registry_file, "r", encoding="utf-8") as handle:
			registry = json.load(handle)

		if not isinstance(registry, dict):
			raise ValueError(f"Invalid registry format in {self._registry_file}")
		if "r_assignments" not in registry or "next_idx" not in registry:
			raise ValueError(f"Registry missing required keys in {self._registry_file}")

		return registry

	def _save_registry(self, registry: Optional[Dict[str, object]] = None) -> None:
		data = self._registry if registry is None else registry
		data["updated_utc"] = datetime.now(timezone.utc).isoformat()
		with open(self._registry_file, "w", encoding="utf-8") as handle:
			json.dump(data, handle, indent=2, sort_keys=True)

	def _ensure_r_keys(self, r_values: Sequence[float]) -> None:
		r_assignments = self._registry.setdefault("r_assignments", {})
		if not isinstance(r_assignments, dict):
			raise ValueError("Registry key `r_assignments` must be a dictionary")

		changed = False
		for r in r_values:
			key = self._canonical_r(r)
			if key not in r_assignments:
				r_assignments[key] = []
				changed = True

		if changed:
			self._save_registry()

	def _assign_indices(self, r: float, n_sims: int) -> List[int]:
		n_use = int(n_sims)
		if n_use < 0:
			raise ValueError("n_sims must be >= 0")

		key = self._assignment_key(r)
		r_assignments = self._registry["r_assignments"]
		if not isinstance(r_assignments, dict):
			raise ValueError("Registry key `r_assignments` must be a dictionary")

		assigned = r_assignments.setdefault(key, [])
		if not isinstance(assigned, list):
			raise ValueError(f"Registry key r_assignments[{key}] must be a list")

		if len(assigned) < n_use:
			next_idx = int(self._registry["next_idx"])
			for _ in range(n_use - len(assigned)):
				assigned.append(next_idx)
				next_idx += 1
			self._registry["next_idx"] = next_idx
			self._save_registry()

		return [int(v) for v in assigned[:n_use]]

	def _product_dir(
		self,
		freq: float,
		product: str,
		masked: bool,
		purify: bool,
		lmin: int,
		selection_sig: str = "",
	) -> Path:
		base = self._root / "maps" / f"freq_{self._freq_tag(freq)}"
		if self.alens != 1.0:
			base = base / f"alens_{self.alens:.6f}"
		if product == "bmode":
			tag = f"bmode_mask{int(masked)}_purify{int(purify)}_lmin{int(lmin)}"
			if selection_sig:
				tag = f"{tag}_{selection_sig}"
		else:
			tag = "qu"
		out = base / tag
		out.mkdir(parents=True, exist_ok=True)
		return out

	def _map_file(
		self,
		freq: float,
		product: str,
		r: float,
		sim_idx: int,
		masked: bool,
		purify: bool,
		lmin: int,
		selection_sig: str = "",
	) -> Path:
		product_dir = self._product_dir(
			freq,
			product,
			masked,
			purify,
			lmin,
			selection_sig=selection_sig,
		)
		r_dir = product_dir / f"r_{self._r_tag(r)}"
		r_dir.mkdir(parents=True, exist_ok=True)
		return r_dir / f"idx_{int(sim_idx):08d}.npz"

	def _selection_signature(self, product: str, masked: bool, mask_threshold: float, padding_rings: int) -> str:
		if product != "bmode" or not masked:
			return ""

		if self._padded_mask_file is not None and self._padded_mask_file.is_file():
			st = self._padded_mask_file.stat()
			payload = f"external|{self._padded_mask_file}|{st.st_size}|{st.st_mtime_ns}"
			return "sel" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

		payload = f"fallback|thr={float(mask_threshold):.6f}|pad={int(padding_rings)}"
		return "sel" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

	def _mask_cache_file(self, mask_threshold: float, padding_rings: int) -> Path:
		nside = int(self.sky.nside_out)
		thr_tag = f"{float(mask_threshold):.6f}".replace(".", "p")
		if self._padded_mask_file is not None:
			if self._padded_mask_file.is_file():
				st = self._padded_mask_file.stat()
				payload = f"{self._padded_mask_file}|{st.st_size}|{st.st_mtime_ns}"
			else:
				payload = str(self._padded_mask_file)
			sig = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
			return self._mask_cache_dir / f"padded_mask_external_nside{nside}_{sig}_nest.npz"
		return self._mask_cache_dir / f"padded_mask_nside{nside}_thr{thr_tag}_pad{int(padding_rings)}_nest.npz"

	def _compute_padded_mask_info(self, mask_threshold: float, padding_rings: int) -> Dict[str, np.ndarray]:
	
		if padding_rings < 0:
			raise ValueError("padding_rings must be >= 0")

		# Primary strategy: use externally prepared padded mask and select only mask > 0.
		if self._padded_mask_file is not None:
			mask_padded = np.asarray(
				hp.read_map(str(self._padded_mask_file), field=0, nest=True, verbose=False),
				dtype=np.float64,
			)
			nside = hp.get_nside(mask_padded)
			if int(nside) != int(self.sky.nside_out):
				raise ValueError(
					f"mask_padded nside={nside} does not match simulator nside_out={self.sky.nside_out}"
				)

			selected = np.asarray(mask_padded > 0.0, dtype=bool)
			pixel_idx = np.flatnonzero(selected).astype(np.int64)
			selected_state = np.ones(pixel_idx.shape[0], dtype=np.int8)
			return {
				"pixel_idx": pixel_idx,
				"core_idx": pixel_idx.copy(),
				"padded_idx": np.asarray([], dtype=np.int64),
				"selected_state": selected_state,
			}

		_, mask = self.sky._load_masks()
		mask_ring = np.asarray(mask, dtype=np.float64)
		nside = hp.get_nside(mask_ring)
		mask_nest = hp.reorder(mask_ring, r2n=True)

		core = np.asarray(mask_nest > float(mask_threshold), dtype=bool)
		selected = core.copy()

		# Prefer the exact DeepSphere-style padding helper when available.
		if ds_utils is not None and int(padding_rings) > 0:
			core_idx = np.flatnonzero(core).astype(np.int64)
			if core_idx.size > 0:
				pad_idx = np.asarray(
					ds_utils.extend_indices(
						core_idx,
						nside_in=int(nside),
						nside_out=int(self._deepsphere_pad_nside_out),
					),
					dtype=np.int64,
				)
				pad_idx = pad_idx[(pad_idx >= 0) & (pad_idx < selected.size)]
				selected[pad_idx] = True
		else:
			# Fallback: nearest-neighbor expansion in NEST ordering.
			for _ in range(int(padding_rings)):
				pix = np.flatnonzero(selected)
				if pix.size == 0:
					break
				neighbors = hp.get_all_neighbours(nside, pix, nest=True)
				neighbors = np.asarray(neighbors, dtype=np.int64).reshape(-1)
				neighbors = neighbors[neighbors >= 0]
				selected[neighbors] = True

		pixel_idx = np.flatnonzero(selected).astype(np.int64)
		core_idx = np.flatnonzero(core).astype(np.int64)
		padded_idx = np.flatnonzero(selected & (~core)).astype(np.int64)
		selected_state = np.zeros(pixel_idx.shape[0], dtype=np.int8)
		selected_state[core[pixel_idx]] = 1
		return {
			"pixel_idx": pixel_idx,
			"core_idx": core_idx,
			"padded_idx": padded_idx,
			"selected_state": selected_state,
		}

	def _get_or_create_mask_info(self, mask_threshold: float, padding_rings: int) -> Dict[str, np.ndarray]:
		cache_file = self._mask_cache_file(mask_threshold=mask_threshold, padding_rings=padding_rings)
		if cache_file.is_file():
			cached = np.load(cache_file)
			return {
				"pixel_idx": np.asarray(cached["pixel_idx"], dtype=np.int64),
				"core_idx": np.asarray(cached["core_idx"], dtype=np.int64),
				"padded_idx": np.asarray(cached["padded_idx"], dtype=np.int64),
				"selected_state": np.asarray(cached["selected_state"], dtype=np.int8),
			}

		mask_info = self._compute_padded_mask_info(mask_threshold=mask_threshold, padding_rings=padding_rings)
		np.savez_compressed(
			cache_file,
			pixel_idx=np.asarray(mask_info["pixel_idx"], dtype=np.int64),
			core_idx=np.asarray(mask_info["core_idx"], dtype=np.int64),
			padded_idx=np.asarray(mask_info["padded_idx"], dtype=np.int64),
			selected_state=np.asarray(mask_info["selected_state"], dtype=np.int8),
			nside=np.int64(self.sky.nside_out),
			ordering=np.asarray("NEST"),
			mask_threshold=np.float64(mask_threshold),
			padding_rings=np.int64(padding_rings),
		)
		return mask_info

	def _select_pixels(self, map_nest: np.ndarray, pixel_idx: np.ndarray) -> np.ndarray:
		arr = np.asarray(map_nest, dtype=np.float64)
		idx = np.asarray(pixel_idx, dtype=np.int64)
		return arr[idx]

	def _dataset_key(
		self,
		freq_values: Sequence[float],
		r_values: Sequence[float],
		n_sims: int,
		product: str,
		masked: bool,
		purify: bool,
		lmin: int,
		mask_threshold: float,
		padding_rings: int,
		shuffle_seed: int,
		selection_sig: str,
		assignment_sig: str,
	) -> str:
		freq_key = ",".join(self._canonical_freq(v) for v in freq_values)
		r_key = ",".join(self._canonical_r(v) for v in r_values)
		return (
			f"freq={freq_key}|r={r_key}|n={int(n_sims)}|product={product}|"
			f"masked={int(masked)}|purify={int(purify)}|lmin={int(lmin)}|"
			f"thr={float(mask_threshold):.6f}|pad={int(padding_rings)}|seed={int(shuffle_seed)}|"
			f"prior={self.r_prior}|alens={self.alens:.6f}|sel={selection_sig}|assign={assignment_sig}"
		)

	def _build_dataset_index(
		self,
		freq_values: Sequence[float],
		r_values: Sequence[float],
		n_sims: int,
		product: str,
		masked: bool,
		purify: bool,
		lmin: int,
		mask_threshold: float,
		padding_rings: int,
		shuffle_seed: int,
		selection_sig: str,
	) -> Dict[str, Any]:
		assigned_by_r: Dict[str, List[int]] = {}
		for r in r_values:
			r_key = self._assignment_key(r)
			assigned_by_r[r_key] = self._assign_indices(r, int(n_sims))

		assignment_sig = hashlib.sha256(
			json.dumps(assigned_by_r, sort_keys=True).encode("utf-8")
		).hexdigest()[:12]

		dataset_key = self._dataset_key(
			freq_values=freq_values,
			r_values=r_values,
			n_sims=n_sims,
			product=product,
			masked=masked,
			purify=purify,
			lmin=lmin,
			mask_threshold=mask_threshold,
			padding_rings=padding_rings,
			shuffle_seed=shuffle_seed,
			selection_sig=selection_sig,
			assignment_sig=assignment_sig,
		)
		dataset_name = hashlib.sha256(dataset_key.encode("utf-8")).hexdigest()
		dataset_file = self._dataset_cache_dir / f"dataset_{dataset_name}.json"

		if dataset_file.is_file():
			with open(dataset_file, "r", encoding="utf-8") as handle:
				return json.load(handle)

		entries: List[Dict[str, Any]] = []
		for r in r_values:
			r_key = self._assignment_key(r)
			indices = assigned_by_r[r_key]
			for sim_idx in indices:
				r_entry = self._prior_r(sim_idx) if self.r_prior is not None else float(r)
				entry_files: Dict[str, str] = {}
				for freq in freq_values:
					path = self._map_file(
						freq=freq,
						product=product,
						r=r_entry,
						sim_idx=sim_idx,
						masked=masked,
						purify=purify,
						lmin=lmin,
						selection_sig=selection_sig,
					)
					entry_files[self._canonical_freq(freq)] = str(path)

				entries.append(
					{
						"r": self._canonical_r(r_entry),
						"sim_idx": int(sim_idx),
						"files": entry_files,
					}
				)

		rng = np.random.default_rng(int(shuffle_seed))
		perm = rng.permutation(len(entries)).tolist() if len(entries) > 0 else []
		dataset = {
			"version": 2 if self.r_prior is not None or self.alens != 1.0 else 1,
			"created_utc": datetime.now(timezone.utc).isoformat(),
			"dataset_key": dataset_key,
			"product": product,
			"ordering": "NEST",
			"masked": bool(masked),
			"purify": bool(purify),
			"lmin": int(lmin),
			"n_sims": int(n_sims),
			"r_values": [self._canonical_r(r) for r in r_values],
			"r_prior": list(self.r_prior) if self.r_prior is not None else None,
			"alens": float(self.alens),
			"freq_values": [self._canonical_freq(f) for f in freq_values],
			"shuffle_seed": int(shuffle_seed),
			"permutation": perm,
			"entries": entries,
			"mask_threshold": float(mask_threshold),
			"padding_rings": int(padding_rings),
			"nside": int(self.sky.nside_out),
			"padded_mask_file": str(self._padded_mask_file) if self._padded_mask_file is not None else None,
			"selection_sig": selection_sig,
			"mask_cache": str(self._mask_cache_file(mask_threshold, padding_rings)) if masked else None,
		}

		with open(dataset_file, "w", encoding="utf-8") as handle:
			json.dump(dataset, handle, indent=2, sort_keys=True)

		with open(self._active_dataset_file, "w", encoding="utf-8") as handle:
			json.dump(dataset, handle, indent=2, sort_keys=True)

		return dataset

	def _set_active_dataset(self, dataset: Dict[str, Any]) -> None:
		with open(self._active_dataset_file, "w", encoding="utf-8") as handle:
			json.dump(dataset, handle, indent=2, sort_keys=True)

	def _load_active_dataset(self) -> Dict[str, Any]:
		if not self._active_dataset_file.is_file():
			raise FileNotFoundError(
				"No active dataset found. Run `run(...)` first to generate/index data."
			)
		with open(self._active_dataset_file, "r", encoding="utf-8") as handle:
			return json.load(handle)

	@staticmethod
	def selected_to_healpy_map(
		selected_values: np.ndarray,
		pixel_idx: np.ndarray,
		nside: int,
		fill_value: float = np.nan,
		ordering: str = "NEST",
		to_ring: bool = False,
	) -> np.ndarray:
		if hp is None:
			raise ImportError("healpy is required. Install with `pip install healpy`.")
		npix = hp.nside2npix(int(nside))
		full_nest = np.full(npix, float(fill_value), dtype=np.float64)
		idx = np.asarray(pixel_idx, dtype=np.int64)
		vals = np.asarray(selected_values, dtype=np.float64)
		if vals.shape[0] != idx.shape[0]:
			raise ValueError("selected_values and pixel_idx must have identical lengths")
		full_nest[idx] = vals

		ord_tag = str(ordering).upper()
		if ord_tag == "NEST":
			return hp.reorder(full_nest, n2r=True) if to_ring else full_nest
		if ord_tag == "RING":
			full_ring = full_nest
			return full_ring if to_ring else hp.reorder(full_ring, r2n=True)
		raise ValueError("ordering must be 'NEST' or 'RING'")

	def bmode_selected_to_healpy(
		self,
		selected_values: np.ndarray,
		pixel_idx: Optional[np.ndarray] = None,
		fill_value: float = np.nan,
		to_ring: bool = False,
	) -> np.ndarray:
		"""Reconstruct a full healpy map from selected NEST B-mode pixels."""
		if pixel_idx is None:
			dataset = self._load_active_dataset()
			cache_file = dataset.get("mask_cache")
			if cache_file is None:
				raise ValueError("Active dataset has no mask cache; pass `pixel_idx` explicitly")
			cached = np.load(str(cache_file))
			pixel_idx = np.asarray(cached["pixel_idx"], dtype=np.int64)

		return self.selected_to_healpy_map(
			selected_values=np.asarray(selected_values),
			pixel_idx=np.asarray(pixel_idx),
			nside=int(self.sky.nside_out),
			fill_value=float(fill_value),
			ordering="NEST",
			to_ring=bool(to_ring),
		)

	def run(
		self,
		freq: float | Sequence[float],
		n_sims: int,
		r_values: Optional[Sequence[float]] = None,
		product: str = "bmode",
		masked: bool = True,
		purify: bool = True,
		lmin: int = 30,
		overwrite: bool = False,
		return_alm: bool = False,
		shuffle_seed: Optional[int] = None,
		mask_threshold: Optional[float] = None,
		padding_rings: Optional[int] = None,
		alens: Optional[float] = None,
	) -> Dict[str, List[Path]] | Dict[str, Dict[str, List[Path]]]:
		"""
		Generate and save maps for each r value, preserving unique simulation indices.

		`n_sims` can be increased later; old maps are reused and only missing indices are added.
		`r_values` can be extended later; newly added r values receive new global indices.
		Saved B-mode products are stored as NEST-ordered selected arrays.
		"""
		if product not in {"bmode", "qu"}:
			raise ValueError("product must be one of {'bmode', 'qu'}")

		n_use = int(n_sims)
		if n_use < 0:
			raise ValueError("n_sims must be >= 0")

		r_use = self._r_values if r_values is None else self._normalize_r_values(r_values)
		if self.r_prior is not None:
			r_use = [self.r_prior[0]]
		alens_use = self.alens if alens is None else float(alens)
		if alens_use < 0:
			raise ValueError("alens must be >= 0")
		self._ensure_r_keys(r_use)
		freq_values = self._normalize_freq_values(freq)
	

		shuffle_seed_use = self._shuffle_seed if shuffle_seed is None else int(shuffle_seed)
		mask_threshold_use = self._mask_threshold if mask_threshold is None else float(mask_threshold)
		padding_rings_use = self._padding_rings if padding_rings is None else int(padding_rings)
		if padding_rings_use < 0:
			raise ValueError("padding_rings must be >= 0")

		selection_sig = self._selection_signature(
			product=product,
			masked=masked,
			mask_threshold=mask_threshold_use,
			padding_rings=padding_rings_use,
		)

		mask_info: Optional[Dict[str, np.ndarray]] = None
		if product == "bmode" and masked:
			mask_info = self._get_or_create_mask_info(mask_threshold_use, padding_rings_use)

		all_saved: Dict[str, Dict[str, List[Path]]] = {}
		total_jobs = len(freq_values) * len(r_use) * n_use
		progress_cm = (
			tqdm(total=total_jobs, desc="sim.run", unit="map")
			if tqdm is not None
			else nullcontext(None)
		)
		with progress_cm as pbar:
			for fval in freq_values:
				freq_key = self._canonical_freq(fval)
				saved: Dict[str, List[Path]] = {}
				for r in r_use:
					indices = self._assign_indices(r, n_use)
					saved_key = self._canonical_r(r) if self.r_prior is None else "continuous"
					saved[saved_key] = []

					for sim_idx in indices:
						r_drawn = self._prior_r(sim_idx) if self.r_prior is not None else float(r)
						outfile = self._map_file(
							freq=fval,
							product=product,
							r=r_drawn,
							sim_idx=sim_idx,
							masked=masked,
							purify=purify,
							lmin=lmin,
							selection_sig=selection_sig,
						)
						if outfile.is_file() and not overwrite:
							saved[saved_key].append(outfile)
							if pbar is not None:
								pbar.update(1)
							continue

						if product == "bmode":
							result = self.sky.get_freq_bmode(
								idx=sim_idx,
								freq=float(fval),
								r=float(r_drawn),
								alens=alens_use,
								masked=bool(masked),
								purify=bool(purify),
								lmin=int(lmin),
								return_alm=bool(return_alm),
							)

							if return_alm:
								b_map, b_alm = result
								b_map_nest = hp.reorder(np.asarray(b_map, dtype=np.float64), r2n=True)
								if masked and mask_info is not None:
									pixel_idx = np.asarray(mask_info["pixel_idx"], dtype=np.int64)
									selected_state = np.asarray(mask_info["selected_state"], dtype=np.int8)
									b_selected = self._select_pixels(b_map_nest, pixel_idx)
								else:
									selected_state = np.ones(b_map_nest.shape[0], dtype=np.int8)
									b_selected = np.asarray(b_map_nest)
								np.savez_compressed(
									outfile,
									b_selected=np.asarray(b_selected, dtype=np.float64),
									mask_state_selected=np.asarray(selected_state, dtype=np.int8),
									ordering=np.asarray("NEST"),
									selected=np.asarray(bool(masked)),
									b_alm=np.asarray(b_alm),
								)
							else:
								b_map_nest = hp.reorder(np.asarray(result, dtype=np.float64), r2n=True)
								if masked and mask_info is not None:
									pixel_idx = np.asarray(mask_info["pixel_idx"], dtype=np.int64)
									selected_state = np.asarray(mask_info["selected_state"], dtype=np.int8)
									b_selected = self._select_pixels(b_map_nest, pixel_idx)
								else:
									selected_state = np.ones(b_map_nest.shape[0], dtype=np.int8)
									b_selected = np.asarray(b_map_nest)
								np.savez_compressed(
									outfile,
									b_selected=np.asarray(b_selected, dtype=np.float64),
									mask_state_selected=np.asarray(selected_state, dtype=np.int8),
									ordering=np.asarray("NEST"),
									selected=np.asarray(bool(masked)),
								)
						else:
							q_map, u_map = self.sky.get_freq_qu(idx=sim_idx, freq=float(fval), r=float(r_drawn), alens=alens_use)
							q_nest = hp.reorder(np.asarray(q_map, dtype=np.float64), r2n=True)
							u_nest = hp.reorder(np.asarray(u_map, dtype=np.float64), r2n=True)
							np.savez_compressed(
								outfile,
								q_map=np.asarray(q_nest, dtype=np.float64),
								u_map=np.asarray(u_nest, dtype=np.float64),
								ordering=np.asarray("NEST"),
							)

						saved[saved_key].append(outfile)
						if pbar is not None:
							pbar.update(1)

				all_saved[freq_key] = saved

		dataset = self._build_dataset_index(
			freq_values=freq_values,
			r_values=r_use,
			n_sims=n_use,
			product=product,
			masked=masked,
			purify=purify,
			lmin=lmin,
			mask_threshold=mask_threshold_use,
			padding_rings=padding_rings_use,
			shuffle_seed=shuffle_seed_use,
			selection_sig=selection_sig,
		)
		self._set_active_dataset(dataset)

		if len(freq_values) == 1:
			return all_saved[self._canonical_freq(freq_values[0])]
		return all_saved

	def get_data(self, idx: int) -> Dict[str, Any]:
		"""
		Return one shuffled sample from the active dataset.

		The returned payload contains selected NEST B-mode arrays for requested frequencies and the sample r.
		"""
		dataset = self._load_active_dataset()
		entries = dataset.get("entries", [])
		perm = dataset.get("permutation", [])
		if not isinstance(entries, list) or len(entries) == 0:
			raise ValueError("Active dataset has no entries")
		if not isinstance(perm, list) or len(perm) != len(entries):
			raise ValueError("Active dataset permutation is invalid")

		idx_use = int(idx)
		if idx_use < 0:
			raise ValueError("idx must be >= 0")
		entry = entries[int(perm[idx_use % len(perm)])]

		freq_values = dataset.get("freq_values", [])
		if not isinstance(freq_values, list) or len(freq_values) == 0:
			raise ValueError("Active dataset has no frequency metadata")

		files = entry.get("files", {})
		if not isinstance(files, dict):
			raise ValueError("Active dataset entry has invalid file mapping")

		bmode_list: List[np.ndarray] = []
		mask_state_selected: Optional[np.ndarray] = None
		for freq in freq_values:
			path = files.get(str(freq))
			if path is None:
				raise KeyError(f"Missing file for frequency {freq}")
			loaded = np.load(str(path))
			if "b_selected" not in loaded:
				raise KeyError(f"Missing `b_selected` in {path}")
			if "mask_state_selected" not in loaded:
				raise KeyError(f"Missing `mask_state_selected` in {path}")
			bmode_list.append(np.asarray(loaded["b_selected"], dtype=np.float64))
			state = np.asarray(loaded["mask_state_selected"], dtype=np.int8)
			if mask_state_selected is None:
				mask_state_selected = state
			elif not np.array_equal(mask_state_selected, state):
				raise ValueError("Mask-state mismatch across frequencies for a single sample")

		if mask_state_selected is None:
			raise ValueError("Could not load mask state for sample")

		core_idx = np.asarray([], dtype=np.int64)
		padded_idx = np.asarray([], dtype=np.int64)
		pixel_idx = np.asarray([], dtype=np.int64)
		cache_file = dataset.get("mask_cache")
		if cache_file is not None:
			cached = np.load(str(cache_file))
			core_idx = np.asarray(cached["core_idx"], dtype=np.int64)
			padded_idx = np.asarray(cached["padded_idx"], dtype=np.int64)
			pixel_idx = np.asarray(cached["pixel_idx"], dtype=np.int64)

		return {
			"bmode": np.asarray(bmode_list, dtype=np.float64),
			"mask_state_selected": np.asarray(mask_state_selected, dtype=np.int8),
			"pixel_idx": pixel_idx,
			"core_idx": core_idx,
			"padded_idx": padded_idx,
			"r": float(entry["r"]),
			"freq": np.asarray([float(v) for v in freq_values], dtype=np.float64),
			"sim_idx": int(entry["sim_idx"]),
			"ordering": "NEST",
			"selected": bool(dataset.get("masked", True)),
		}

	def _padded_pixels_nest_from_dataset(self, dataset: Dict[str, Any]) -> np.ndarray:

		nside = int(dataset.get("nside", self.sky.nside_out))
		npix = hp.nside2npix(nside)

		cache_file = dataset.get("mask_cache")
		if cache_file is not None and Path(str(cache_file)).is_file():
			cached = np.load(str(cache_file))
			idx = np.asarray(cached["pixel_idx"], dtype=np.int64)
			mask = np.zeros(npix, dtype=bool)
			mask[idx] = True
			return mask

		if self._padded_mask_file is not None and self._padded_mask_file.is_file():
			mask_padded = np.asarray(
				hp.read_map(str(self._padded_mask_file), field=0, nest=True, verbose=False),
				dtype=np.float64,
			)
			if hp.get_nside(mask_padded) != nside:
				raise ValueError(
					f"padded mask nside={hp.get_nside(mask_padded)} does not match dataset nside={nside}"
				)
			return np.asarray(mask_padded > 0.0, dtype=bool)

		raise ValueError("Could not determine padded pixels; missing mask cache and padded mask file")

	def _selected_to_healpy_ring_map(
		self,
		selected_values: np.ndarray,
		padded_pixels_nest: np.ndarray,
	) -> np.ndarray:

		vals = np.asarray(selected_values, dtype=np.float64)
		padded = np.asarray(padded_pixels_nest, dtype=bool)
		npix = padded.shape[0]
		if vals.shape[0] != int(padded.sum()):
			raise ValueError(
				"selected map length does not match number of padded pixels "
				f"({vals.shape[0]} vs {int(padded.sum())})"
			)

		map_to_return = np.full(npix, 0.0, dtype=np.float64)
		map_to_return[padded] = vals
		return hp.reorder(map_to_return, n2r=True)

	def get_data_by_r(
		self,
		r: str | float,
		idx: int,
		return_type: str = "map",
		format: str = "d",
		lmax: Optional[int] = None,
	) -> Dict[str, Any]:
		"""
		Return the i-th simulation for a specific r from the active dataset.

		Parameters
		----------
		r : str | float
			R value (for example "0.001000" from summary keys, or 0.001).
		idx : int
			0-based simulation index within this r bin.
		return_type : str
			"map" or "spectra".
		format : str
			"d" (DeepSphere selected format) or "h" (healpy full-sky RING map).
		lmax : Optional[int]
			Optional lmax for spectra mode.
		"""
		dataset = self._load_active_dataset()

		return_type_use = str(return_type).strip().lower()
		if return_type_use not in {"map", "spectra"}:
			raise ValueError("return_type must be 'map' or 'spectra'")

		format_use = str(format).strip().lower()
		if format_use not in {"d", "h"}:
			raise ValueError("format must be 'd' or 'h'")

		try:
			r_key = self._canonical_r(float(r))
		except Exception as exc:
			raise ValueError(f"Could not parse r={r}") from exc

		entries = dataset.get("entries", [])
		if not isinstance(entries, list) or len(entries) == 0:
			raise ValueError("Active dataset has no entries")

		entries_for_r = [entry for entry in entries if str(entry.get("r")) == r_key]
		if len(entries_for_r) == 0:
			available = sorted({str(entry.get("r")) for entry in entries})
			raise KeyError(f"r={r_key} not found in active dataset. Available r keys: {available}")

		idx_use = int(idx)
		if idx_use < 0 or idx_use >= len(entries_for_r):
			raise IndexError(
				f"idx={idx_use} out of range for r={r_key}. "
				f"Valid range: [0, {len(entries_for_r) - 1}]"
			)

		entry = entries_for_r[idx_use]
		freq_values = dataset.get("freq_values", [])
		if not isinstance(freq_values, list) or len(freq_values) == 0:
			raise ValueError("Active dataset has no frequency metadata")

		files = entry.get("files", {})
		if not isinstance(files, dict):
			raise ValueError("Active dataset entry has invalid file mapping")

		bmode_selected: List[np.ndarray] = []
		for freq in freq_values:
			path = files.get(str(freq))
			if path is None:
				raise KeyError(f"Missing file for frequency {freq}")
			path_obj = Path(str(path))
			if path_obj.is_file():
				loaded = np.load(str(path_obj))
				if "b_selected" not in loaded:
					raise KeyError(f"Missing `b_selected` in {path}")
				bmode_selected.append(np.asarray(loaded["b_selected"], dtype=np.float64))
				continue

			# Rebuild the missing sample directly from the simulation pipeline.
			fresh_ring = np.asarray(
				self.sky.get_freq_bmode(
					idx=int(entry["sim_idx"]),
					freq=float(freq),
					r=float(entry["r"]),
					masked=True,
					purify=bool(dataset.get("purify", True)),
					lmin=int(dataset.get("lmin", 30)),
				),
				dtype=np.float64,
			)
			fresh_nest = hp.reorder(fresh_ring, r2n=True)
			padded_pixels_nest = self._padded_pixels_nest_from_dataset(dataset)
			fresh_selected = fresh_nest[padded_pixels_nest]
			path_obj.parent.mkdir(parents=True, exist_ok=True)
			np.savez_compressed(
				path_obj,
				b_selected=np.asarray(fresh_selected, dtype=np.float64),
				mask_state_selected=np.asarray(
					np.ones(fresh_selected.shape[0], dtype=np.int8),
					dtype=np.int8,
				),
				ordering=np.asarray("NEST"),
				selected=np.asarray(True),
			)
			bmode_selected.append(np.asarray(fresh_selected, dtype=np.float64))

		bmode_selected_arr = np.asarray(bmode_selected, dtype=np.float64)

		if return_type_use == "map" and format_use == "d":
			return {
				"map": bmode_selected_arr,
				"r": float(entry["r"]),
				"freq": np.asarray([float(v) for v in freq_values], dtype=np.float64),
				"sim_idx": int(entry["sim_idx"]),
				"return_type": "map",
				"format": "d",
			}

		padded_pixels_nest = self._padded_pixels_nest_from_dataset(dataset)
		maps_ring = np.asarray(
			[
				self._selected_to_healpy_ring_map(values, padded_pixels_nest)
				for values in bmode_selected_arr
			],
			dtype=np.float64,
		)

		if return_type_use == "map":
			return {
				"map": maps_ring,
				"r": float(entry["r"]),
				"freq": np.asarray([float(v) for v in freq_values], dtype=np.float64),
				"sim_idx": int(entry["sim_idx"]),
				"return_type": "map",
				"format": "h",
			}

		nside = int(dataset.get("nside", self.sky.nside_out))
		lmax_use = 3 * nside - 1 if lmax is None else int(lmax)
		if lmax_use < 0:
			raise ValueError("lmax must be >= 0")
		cls = np.asarray([hp.anafast(m, lmax=lmax_use) for m in maps_ring], dtype=np.float64)
		return {
			"spectra": cls,
			"ell": np.arange(cls.shape[1], dtype=np.int64),
			"r": float(entry["r"]),
			"freq": np.asarray([float(v) for v in freq_values], dtype=np.float64),
			"sim_idx": int(entry["sim_idx"]),
			"return_type": "spectra",
			"format": format_use,
		}

	def check_saved_vs_direct(self, r: str | float, idx: int) -> Dict[str, Any]:
		"""Compare saved selected vectors against freshly computed sky.get_freq_bmode for the same sample."""
		d = self.get_data_by_r(r=r, idx=idx, return_type="map", format="d")
		r_val = float(d["r"])
		sim_idx = int(d["sim_idx"])
		freqs = np.asarray(d["freq"], dtype=np.float64)
		saved = np.asarray(d["map"], dtype=np.float64)

		dataset = self._load_active_dataset()
		padded_pixels_nest = self._padded_pixels_nest_from_dataset(dataset)
		pixel_idx = np.flatnonzero(padded_pixels_nest).astype(np.int64)

		diffs: List[Dict[str, Any]] = []
		for i, freq in enumerate(freqs):
			fresh_ring = np.asarray(
				self.sky.get_freq_bmode(
					idx=sim_idx,
					freq=float(freq),
					r=r_val,
					masked=True,
					purify=True,
					lmin=int(dataset.get("lmin", 30)),
				),
				dtype=np.float64,
			)
			fresh_nest = hp.reorder(fresh_ring, r2n=True)
			fresh_selected = fresh_nest[pixel_idx]
			delta = saved[i] - fresh_selected
			diffs.append(
				{
					"freq": float(freq),
					"max_abs": float(np.max(np.abs(delta))),
					"mean_abs": float(np.mean(np.abs(delta))),
				}
			)

		return {
			"r": r_val,
			"requested_idx": int(idx),
			"sim_idx": sim_idx,
			"diffs": diffs,
		}

	def summary(self) -> Dict[str, int]:
		"""Return number of assigned simulation indices for each r currently tracked."""
		r_assignments = self._registry["r_assignments"]
		if not isinstance(r_assignments, dict):
			raise ValueError("Registry key `r_assignments` must be a dictionary")

		out: Dict[str, int] = {}
		for key, values in r_assignments.items():
			if isinstance(values, list):
				out[str(key)] = len(values)
		return out
