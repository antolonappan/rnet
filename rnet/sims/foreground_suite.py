from __future__ import annotations

import copy
import hashlib
import json
import multiprocessing as mp
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
from tqdm.auto import tqdm
import healpy as hp
import pysm3  # type: ignore
import pysm3.units as u  # type: ignore
from pysm3.sky import PRESET_MODELS  # type: ignore
from pysm3 import utils as pysm_utils  # type: ignore




class ForegroundAugmentedSimulator:
    """Foreground-augmented simulator for CMB B-mode maps.

    This class generates and caches PySM3 foreground Q/U maps for a fixed set of presets,
    and provides utilities for adding these foregrounds to existing CMB simulations.
    """

    def __init__(
        self,
        basedir: Path | str,
        fg_preset: Sequence[str] = ("d11", "s6"),
        nside: int = 128,
        lmax: int = 300,
        lmin: int = 30,
        seed_base: int = 50000,
        output_root: Optional[Path | str] = None,
        foreground_nside: Optional[int] = None,
        binary_mask_file: Optional[Path | str] = None,
        compress: bool = True,
        map2alm_lsq_maxiter: Optional[int] = 0,
    ) -> None:
        self.basedir = Path(basedir).expanduser().resolve()
        self.output_root = Path(output_root).expanduser().resolve() if output_root else self.basedir / "simulations" / "r_grid_fg"
        self.fg_preset = tuple(str(x) for x in fg_preset)
        if len(self.fg_preset) != 2:
            raise ValueError("fg_preset must contain dust and synchrotron presets")
        self.nside = int(nside)
        self.foreground_nside = int(foreground_nside or self.nside)
        self.lmax = int(lmax)
        self.lmin = int(lmin)
        self.seed_base = int(seed_base)
        self.binary_mask_file = Path(binary_mask_file).expanduser().resolve() if binary_mask_file else None
        self.compress = bool(compress)
        self.map2alm_lsq_maxiter = map2alm_lsq_maxiter
        if self.nside <= 0 or self.foreground_nside <= 0:
            raise ValueError("nside values must be positive")
        if self.lmin < 0 or self.lmax < self.lmin:
            raise ValueError("require 0 <= lmin <= lmax")
        if PRESET_MODELS is not None:
            for preset in self.fg_preset:
                if preset not in PRESET_MODELS:
                    raise KeyError(f"Unknown PySM preset {preset!r}")
        self._sky_cache: Dict[int, Any] = {}


    @staticmethod
    def _read_manifest(path: Path | str) -> Dict[str, Any]:
        manifest_path = Path(path).expanduser().resolve()
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Input manifest not found: {manifest_path}")
        with manifest_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict) or not payload.get("entries"):
            raise ValueError(f"Manifest has no entries: {manifest_path}")
        return payload

    def _seed_config(self, sim_idx: int) -> tuple[list[int], list[int]]:
        base = self.seed_base + int(sim_idx) * 10
        return [base, base + 1, base + 2], [base + 3, base + 4]

    def _sky_cache_index(self, sim_idx: int) -> int:
        # Fixed PySM presets have no realization seeds; reuse their map across entries.
        stochastic = PRESET_MODELS is not None and any(
            str(PRESET_MODELS[preset].get("class", "")).endswith("Realization")
            for preset in self.fg_preset
        )
        return int(sim_idx) if stochastic else 0

    def _cache_tag(self) -> str:
        preset = "_".join(self.fg_preset)
        lsq_tag = "default" if self.map2alm_lsq_maxiter is None else f"lsq{self.map2alm_lsq_maxiter}"
        return f"{preset}_nside{self.foreground_nside}_lmax{self.lmax}_lmin{self.lmin}_{lsq_tag}_binarymask1"

    def _resolve_binary_mask_source(self) -> Path:
        candidates = []
        if self.binary_mask_file is not None:
            candidates.append(self.binary_mask_file)
        candidates.extend(
            [
                self.basedir / "data" / "mask.fits",
                self.basedir / "mask.fits",
                Path(__file__).resolve().parents[1] / "data" / "mask.fits",
            ]
        )
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        raise FileNotFoundError("Could not locate the binary celestial mask (mask.fits)")

    def _binary_mask_cache_file(self) -> Path:
        path = self.output_root / "masks" / f"binary_mask_nside{self.nside}.fits"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def binary_mask_ring(self) -> np.ndarray:

        cache_file = self._binary_mask_cache_file()
        if cache_file.is_file():
            mask = hp.read_map(str(cache_file), field=0, nest=False)
            if hp.get_nside(mask) == self.nside:
                return np.asarray(mask > 0.0, dtype=np.float64)
        source = self._resolve_binary_mask_source()
        mask = np.asarray(hp.read_map(str(source), field=0, nest=False), dtype=np.float64)
        if hp.get_nside(mask) != self.nside:
            mask = hp.ud_grade(mask, nside_out=self.nside)
        mask = np.asarray(mask > 0.0, dtype=np.float64)
        hp.write_map(str(cache_file), mask, dtype=np.float32, overwrite=True, fits_IDL=False)
        return mask

    def _foreground_cache_file(self, freq: float, sim_idx: int) -> Path:
        cache_idx = self._sky_cache_index(sim_idx)
        path = self.output_root / "foreground_bmode" / self._cache_tag() / f"freq_{float(freq):.3f}" / f"idx_{cache_idx:08d}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _make_sky(self, sim_idx: int):

        sim_idx = self._sky_cache_index(sim_idx)
        # A PySM Sky retains large component maps. Keep it only for the two
        # frequencies of the active realization instead of accumulating one
        # object per simulation.
        if sim_idx not in self._sky_cache:
            self._sky_cache.clear()
            dust_seeds, sync_seeds = self._seed_config(sim_idx)
            dust = copy.deepcopy(PRESET_MODELS[self.fg_preset[0]])
            synch = copy.deepcopy(PRESET_MODELS[self.fg_preset[1]])
            if str(dust.get("class", "")).endswith("Realization"):
                dust["seeds"] = dust_seeds
            if str(synch.get("class", "")).endswith("Realization"):
                synch["seeds"] = sync_seeds
            self._sky_cache[sim_idx] = pysm3.Sky(
                nside=self.foreground_nside,
                component_config={"dust": dust, "synch": synch},
                output_unit=u.uK_CMB,
            )
        return self._sky_cache[sim_idx]

    def foreground_bmode(self, freq: float, sim_idx: int, overwrite: bool = False) -> np.ndarray:
        """Return a celestial RING foreground B map with the common transfer."""

        cache_file = self._foreground_cache_file(freq, sim_idx)
        if cache_file.is_file() and not overwrite:
            with np.load(cache_file) as payload:
                return np.asarray(payload["b_map"], dtype=np.float64)

        emission = self._make_sky(sim_idx).get_emission(float(freq) * u.GHz)
        if emission.unit != u.uK_CMB:
            emission = emission.to(u.uK_CMB, equivalencies=u.cmb_equivalencies(float(freq) * u.GHz))
        # PySM is Galactic. This utility performs the polarization-aware
        # Galactic -> celestial rotation before the spin-2 B transform.
        celestial = pysm_utils.apply_smoothing_and_coord_transform(
            emission,
            rot=hp.Rotator(coord=("G", "C")),
            lmax=self.lmax,
            map2alm_lsq_maxiter=self.map2alm_lsq_maxiter,
            output_nside=self.nside,
            return_healpix=True,
            return_car=False,
        )
        celestial_maps = np.asarray(celestial.value if hasattr(celestial, "value") else celestial, dtype=np.float64)
        if celestial_maps.shape[0] < 3:
            raise ValueError("PySM emission must contain I, Q, and U")
        _, b_alm = hp.map2alm_spin([celestial_maps[1], celestial_maps[2]], 2, lmax=self.lmax)
        transfer = np.ones(self.lmax + 1, dtype=np.float64)
        transfer[:self.lmin] = 0.0
        b_alm = hp.almxfl(b_alm, transfer, inplace=False)
        b_map = hp.alm2map(b_alm, nside=self.nside, lmax=self.lmax, pixwin=False)
        # The selected graph geometry includes padded pixels. Foregrounds must
        # be zero outside the scientific binary mask before they are added.
        b_map *= self.binary_mask_ring()
        save = np.savez_compressed if self.compress else np.savez
        save(
            cache_file,
            b_map=np.asarray(b_map, dtype=np.float64),
            b_alm=np.asarray(b_alm),
            transfer=transfer,
            freq=np.float64(freq),
            sim_idx=np.int64(sim_idx),
            dust_seeds=np.asarray(self._seed_config(sim_idx)[0], dtype=np.int64),
            sync_seeds=np.asarray(self._seed_config(sim_idx)[1], dtype=np.int64),
        )
        return np.asarray(b_map, dtype=np.float64)

    @staticmethod
    def _selected_to_ring(selected: np.ndarray, pixel_indices: np.ndarray, nside: int) -> np.ndarray:
        full_nest = np.zeros(hp.nside2npix(int(nside)), dtype=np.float64)
        full_nest[np.asarray(pixel_indices, dtype=np.int64)] = np.asarray(selected, dtype=np.float64)
        return hp.reorder(full_nest, n2r=True)

    def _entry_output_path(self, suite_root: Path, freq: float, entry: Mapping[str, Any]) -> Path:
        r_tag = f"{float(entry['r']):.6f}".replace(".", "p")
        path = suite_root / "maps" / f"freq_{float(freq):.3f}" / f"bmode_fg_{'_'.join(self.fg_preset)}_lmin{self.lmin}_binarymask1" / f"r_{r_tag}" / f"idx_{int(entry['sim_idx']):08d}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _augment_entry(
        self,
        entry: Mapping[str, Any],
        frequencies: Sequence[str],
        pixel_indices: np.ndarray,
        suite_root: Path,
        overwrite: bool,
    ) -> Dict[str, str]:
        output_files: Dict[str, str] = {}
        for freq_key in frequencies:
            freq = float(freq_key)
            output = self._entry_output_path(suite_root, freq, entry)
            output_files[freq_key] = str(output)
            if output.is_file() and not overwrite:
                continue
            source_path = Path(str(entry["files"][freq_key])).expanduser()
            if not source_path.is_file():
                raise FileNotFoundError(f"Source simulation map not found: {source_path}")
            with np.load(source_path) as payload:
                base_selected = np.asarray(payload["b_selected"], dtype=np.float64)
                state = np.asarray(payload.get("mask_state_selected", np.ones(base_selected.size)), dtype=np.int8)
            base_ring = self._selected_to_ring(base_selected, pixel_indices, self.nside)
            fg_ring = self.foreground_bmode(freq, int(entry["sim_idx"]), overwrite=overwrite)
            combined_ring = base_ring + fg_ring
            combined_nest = hp.reorder(combined_ring, r2n=True)
            combined_selected = combined_nest[np.asarray(pixel_indices, dtype=np.int64)]
            save = np.savez_compressed if self.compress else np.savez
            save(
                output,
                b_selected=np.asarray(combined_selected, dtype=np.float64),
                mask_state_selected=state,
                ordering=np.asarray("NEST"),
                selected=np.asarray(True),
            )
        return output_files

    def run(
        self,
        input_manifest: Path | str,
        output_manifest: Path | str,
        suite_name: str,
        indices: Optional[Sequence[int]] = None,
        overwrite: bool = False,
        show_progress: bool = True,
    ) -> Path:
        manifest = self._read_manifest(input_manifest)
        if manifest.get("product") != "bmode" or str(manifest.get("ordering", "")).upper() != "NEST":
            raise ValueError("input manifest must be an NEST bmode suite")
        nside = int(manifest.get("nside", self.nside))
        if nside != self.nside:
            raise ValueError(f"manifest nside={nside} does not match simulator nside={self.nside}")
        frequencies = tuple(str(x) for x in manifest.get("freq_values", []))
        if not frequencies:
            raise ValueError("input manifest has no frequencies")
        cache_path = manifest.get("mask_cache")
        if cache_path is None:
            pixel_indices = np.arange(hp.nside2npix(self.nside), dtype=np.int64)
        else:
            with np.load(Path(str(cache_path)).expanduser()) as payload:
                pixel_indices = np.asarray(payload["pixel_idx"], dtype=np.int64)
        entries = list(manifest["entries"])
        use = list(range(len(entries))) if indices is None else [int(i) for i in indices]
        if any(i < 0 or i >= len(entries) for i in use):
            raise IndexError("requested entry index is outside the manifest")
        suite_root = self.output_root / str(suite_name)
        new_entries = []
        selected_indices = sorted(set(use))
        iterator = selected_indices
        if show_progress and tqdm is not None:
            iterator = tqdm(selected_indices, total=len(selected_indices), desc=f"Foreground {suite_name}")
        for index in iterator:
            entry = entries[index]
            files = self._augment_entry(entry, frequencies, pixel_indices, suite_root, overwrite)
            new_entries.append({"r": entry["r"], "sim_idx": int(entry["sim_idx"]), "files": files})
        output = Path(output_manifest).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(manifest)
        payload.update(
            {
                "dataset_kind": f"{suite_name}_foreground_augmented",
                "dataset_key": f"{manifest.get('dataset_key', input_manifest)}|fg={'/'.join(self.fg_preset)}|coord=G2C|lmin={self.lmin}",
                "entries": new_entries,
                "n_sims": len(new_entries),
                "source_manifest": str(Path(input_manifest).expanduser().resolve()),
                "foreground": {
                    "presets": list(self.fg_preset),
                    "coordinate_in": "G",
                    "coordinate_out": "C",
                    "seed_base": self.seed_base,
                    "lmax": self.lmax,
                    "lmin": self.lmin,
                    "transfer": "T_ell=0 for ell<lmin, 1 otherwise",
                    "addition": "filtered foreground B map added to existing saved B map before selected-pixel mask",
                    "binary_mask": str(self._binary_mask_cache_file()),
                },
            }
        )
        with output.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        return output

    def run_parallel(
        self,
        input_manifest: Path | str,
        output_manifest: Path | str,
        suite_name: str,
        workers: int = 4,
        chunks: Optional[int] = None,
        overwrite: bool = False,
    ) -> Path:
        
        manifest = self._read_manifest(input_manifest)
        entries = list(manifest["entries"])
        workers = max(1, min(int(workers), len(entries)))
        if workers == 1:
            return self.run(input_manifest, output_manifest, suite_name, overwrite=overwrite)
        nchunks = max(workers, int(chunks or workers * 4))
        index_chunks = [chunk.tolist() for chunk in np.array_split(np.arange(len(entries)), nchunks) if len(chunk)]
        config = {
            "basedir": str(self.basedir), "fg_preset": self.fg_preset,
            "nside": self.nside, "lmax": self.lmax, "lmin": self.lmin,
            "seed_base": self.seed_base, "output_root": str(self.output_root),
            "foreground_nside": self.foreground_nside,
            "binary_mask_file": str(self.binary_mask_file) if self.binary_mask_file else None,
            "compress": self.compress, "map2alm_lsq_maxiter": self.map2alm_lsq_maxiter,
        }
        input_path = str(Path(input_manifest).expanduser().resolve())
        output_path = Path(output_manifest).expanduser().resolve()
        with tempfile.TemporaryDirectory(prefix="rnet_fg_") as tmpdir:
            jobs = []
            for number, index_chunk in enumerate(index_chunks):
                jobs.append((config, input_path, str(Path(tmpdir) / f"chunk_{number}.json"), suite_name, index_chunk, overwrite))
            context = mp.get_context("spawn")
            with context.Pool(processes=workers) as pool:
                iterator = pool.imap(_run_foreground_chunk, jobs)
                if tqdm is not None:
                    iterator = tqdm(iterator, total=len(jobs), desc=f"Foreground {suite_name} chunks")
                for _ in iterator:
                    pass
            combined = dict(manifest)
            combined["entries"] = []
            for number in range(len(jobs)):
                with (Path(tmpdir) / f"chunk_{number}.json").open("r", encoding="utf-8") as handle:
                    combined["entries"].extend(json.load(handle)["entries"])
            combined["entries"].sort(key=lambda entry: (float(entry["r"]), int(entry["sim_idx"])))
            combined["foreground"] = {
                "presets": list(self.fg_preset), "seed_base": self.seed_base,
                "lmax": self.lmax, "lmin": self.lmin,
                "map2alm_lsq_maxiter": self.map2alm_lsq_maxiter,
                "parallel_workers": workers,
            }
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as handle:
                json.dump(combined, handle, indent=2, sort_keys=True)
        return output_path


def _run_foreground_chunk(job: tuple) -> str:
    config, input_manifest, output_manifest, suite_name, indices, overwrite = job
    simulator = ForegroundAugmentedSimulator(**config)
    simulator.run(input_manifest, output_manifest, suite_name, indices=indices, overwrite=overwrite, show_progress=False)
    return output_manifest
