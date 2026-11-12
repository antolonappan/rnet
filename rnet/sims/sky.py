from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable
from typing import Dict, List, Optional

import numpy as np
import healpy as hp
import pymaster as nmt_lib


from .cmb import CMBLensing
from .fg import Pysm3Foregrounds
from .noise import Noise


class Sky:
	"""Compose sky Q/U maps from CMB, foregrounds, and instrumental noise."""

	def __init__(
		self,
		basedir: Optional[Path | str] = None,
		nside: int = 1024,
		nside_out: int = 128,
		dlmax: int = 512,
		addfg: bool = True,
		addnoise: bool = True,
		atm_noise: bool = False,
		decon: bool = True,
		inhom: bool = False,
		fg_preset = None,
		noise_sensitivity: str = "goal",
		verbose: bool = False,
	) -> None:
		self.basedir = Path(basedir) if basedir is not None else Path.cwd()
		self.basedir.mkdir(parents=True, exist_ok=True)

		self.nside = int(nside)
		if self.nside <= 0:
			raise ValueError("nside must be > 0")
		self.lmax = 3 * self.nside - 1

		self.nside_out = int(nside_out)
		if self.nside_out <= 0:
			raise ValueError("nside_out must be > 0")
		# Conservative harmonic bandlimit for stable transforms at output nside.
		self.lmax_out = 300
		if self.lmax_out > self.lmax:
			raise ValueError("nside_out must be <= nside so that lmax_out <= lmax")

		self.dlmax = int(dlmax)
		if self.dlmax < 0:
			raise ValueError("dlmax must be >= 0")

		self.addfg = bool(addfg)
		self.addnoise = bool(addnoise)
		self.atm_noise = bool(atm_noise)
		self.decon = bool(decon)
		self.inhom = bool(inhom)

		if fg_preset is None:
			fg_preset = ["d10", "s5"]

		self.cmb = CMBLensing(
			nside=self.nside,
			dlmax=self.dlmax,
			basedir=self.basedir,
			verbose=verbose,
		)

		self.fg = Pysm3Foregrounds(
			basedir=self.basedir,
			nside=self.nside,
			preset_string=fg_preset,
		)

		self.noise = Noise(
			atm_noise=self.atm_noise,
			sensitivity=noise_sensitivity,
			decon=self.decon,
			inhom=self.inhom,
		)

		if self.noise.nside != self.nside:
			raise ValueError(
				f"Noise model nside={self.noise.nside} does not match sky nside={self.nside}."
			)

	@staticmethod
	def slice_alms(teb: np.ndarray, lmax_new: int) -> np.ndarray:
		"""Return input T/E/B alms sliced to a smaller lmax."""

		nfields = len(teb)
		lmax = hp.Alm.getlmax(len(teb[0]))
		if nfields > 3:
			lmax = nfields
			nfields = 1
			dtype = teb.dtype
		else:
			dtype = teb[0].dtype

		if lmax_new > lmax:
			raise ValueError("lmax_new must be smaller or equal to lmax")
		if lmax_new == lmax:
			return teb

		teb_new = np.zeros((nfields, hp.Alm.getsize(lmax_new)), dtype=dtype)
		indices_full = hp.Alm.getidx(lmax, *hp.Alm.getlm(lmax_new))
		indices_new = hp.Alm.getidx(lmax_new, *hp.Alm.getlm(lmax_new))
		teb_new[:, indices_new] = teb[:, indices_full]
		return teb_new

	def component_qu(self, idx: int, freq: float, r: Optional[float] = None, alens: float = 1.0) -> Dict[str, np.ndarray]:
		"""Return component-wise Q/U maps for one realization and frequency."""
		r_use = 0.0 if r is None else float(r)
		if r_use < 0.0:
			raise ValueError("r must be >= 0")

		q_cmb, u_cmb = self.cmb.get_lensed_qu(idx, r=r_use, alens=alens)
		components: Dict[str, np.ndarray] = {
			"cmb": np.asarray([q_cmb, u_cmb]),
		}

		if self.addfg:
			q_fg, u_fg = self.fg.get_freq_qu(float(freq))
			components["fg"] = np.asarray([q_fg, u_fg])

		if self.addnoise:
			components["noise"] = self.noise.get_freq_qu(idx, int(freq))

		return components

	def get_freq_qu(
		self,
		idx: int,
		freq: float,
		r: float = 0.0,
		alens: float = 1.0,
	) -> List[np.ndarray]:
		"""Return total Q/U for one realization and frequency at nside_out."""

		r_use = float(r)
		if r_use < 0.0:
			raise ValueError("r must be >= 0")

		q_cmb, u_cmb = self.cmb.get_lensed_qu(idx, r=r_use, alens=alens)
		q_tot = np.asarray(q_cmb, dtype=np.float64).copy()
		u_tot = np.asarray(u_cmb, dtype=np.float64).copy()

		if self.addfg:
			q_fg, u_fg = self.fg.get_freq_qu(float(freq))
			q_tot += np.asarray(q_fg, dtype=np.float64)
			u_tot += np.asarray(u_fg, dtype=np.float64)

		if self.addnoise:
			q_n, u_n = self.noise.get_freq_qu(idx, int(freq))
			q_tot += np.asarray(q_n, dtype=np.float64)
			u_tot += np.asarray(u_n, dtype=np.float64)

		e_alm, b_alm = hp.map2alm_spin([q_tot, u_tot], spin=2, lmax=self.lmax)
		eb_slice = self.slice_alms(np.asarray([e_alm, b_alm]), self.lmax_out)
		q_out, u_out = hp.alm2map_spin(eb_slice, self.nside_out, 2, lmax=self.lmax_out)
		return [np.asarray(q_out), np.asarray(u_out)]

	def _load_masks(self) -> tuple[np.ndarray, np.ndarray]:
		"""Load mask_apo.fits and mask.fits at nside_out from discovered data folders."""


		def candidates(name: str) -> Iterable[Path]:
			# First prefer explicit run directory data paths.
			yield self.basedir / "data" / name
			yield self.basedir / name

			# Then package-level expected data location.
			yield Path(__file__).resolve().parents[1] / "data" / name

			# Finally walk cwd and module parent trees for generic data folders.
			for root in [Path.cwd(), *Path.cwd().resolve().parents]:
				yield root / "data" / name
			for root in Path(__file__).resolve().parents:
				yield root / "data" / name

		def resolve_mask(name: str) -> Path:
			seen = set()
			checked: List[Path] = []
			for path in candidates(name):
				if path in seen:
					continue
				seen.add(path)
				checked.append(path)
				if path.is_file():
					return path.resolve()

			looked_in = "\n".join(f"- {p}" for p in checked)
			raise FileNotFoundError(f"Could not locate {name}. Checked:\n{looked_in}")

		mask_apo_file = resolve_mask("mask_apo.fits")
		mask_file = resolve_mask("mask.fits")

		mask_apo = np.asarray(hp.read_map(str(mask_apo_file), field=0, verbose=False), dtype=np.float64)
		mask = np.asarray(hp.read_map(str(mask_file), field=0, verbose=False), dtype=np.float64)

		if hp.get_nside(mask_apo) != self.nside_out:
			mask_apo = hp.ud_grade(mask_apo, nside_out=self.nside_out)
		if hp.get_nside(mask) != self.nside_out:
			mask = hp.ud_grade(mask, nside_out=self.nside_out)

		return mask_apo, mask

	def get_freq_bmode(
		self,
		idx: int,
		freq: float,
		r: float = 0.0,
		alens: float = 1.0,
		masked: bool = True,
		purify: bool = True,
		lmin: int = 30,
		save: bool = False,
		return_alm: bool = False,
	) -> np.ndarray:
		"""Return B-mode map with optional masking, purification, lmin filter, and cache reuse."""


		lmin_use = int(lmin)
		if lmin_use < 0:
			raise ValueError("lmin must be >= 0")
		if lmin_use > self.lmax_out:
			raise ValueError("lmin must be <= lmax_out")

		r_use = float(r)
		if r_use < 0.0:
			raise ValueError("r must be >= 0")

		cache_file: Optional[Path] = None
		if save:
			cache_dir = self.basedir / "cache" / "bmode"
			cache_dir.mkdir(parents=True, exist_ok=True)
			cache_key = (
				f"idx={int(idx)}|freq={float(freq):.10g}|r={r_use:.10g}|alens={float(alens):.10g}|"
				f"masked={int(masked)}|purify={int(purify)}|lmin={lmin_use}|"
				f"nside_out={self.nside_out}|lmax_out={self.lmax_out}|"
				f"addfg={int(self.addfg)}|addnoise={int(self.addnoise)}|atm_noise={int(self.atm_noise)}|"
				f"decon={int(self.decon)}|inhom={int(self.inhom)}"
			)
			cache_name = hashlib.sha256(cache_key.encode("utf-8")).hexdigest() + ".npz"
			cache_file = cache_dir / cache_name
			if cache_file.is_file():
				cached = np.load(cache_file)
				b_map_cached = np.asarray(cached["b_map"])
				if return_alm:
					b_alm_cached = np.asarray(cached["b_alm"])
					return b_map_cached, b_alm_cached
				return b_map_cached

		q_map, u_map = self.get_freq_qu(idx, freq, r=r, alens=alens)
		if masked:
			mask_apo, _ = self._load_masks()
			fp = nmt_lib.NmtField(
				mask_apo,
				[q_map, u_map],
				lmax=self.lmax_out,
				lmax_mask=self.lmax_out,
				masked_on_input=False,
				purify_b=purify,
			)
			alms_p = fp._purify(
				fp.mask,
				fp.get_mask_alms(),
				[q_map, u_map],
				n_iter=fp.n_iter,
				task=[False, purify],
				return_maps=False,
			)
			b_alm = alms_p[1]
		else:
			b_alm = hp.map2alm_spin([q_map, u_map], 2, lmax=self.lmax_out)[1]

		# Suppress large-scale B modes below the requested harmonic cutoff.
		fl = np.ones(self.lmax_out + 1, dtype=np.float64)
		fl[:lmin_use] = 0.0
		b_alm = hp.almxfl(np.asarray(b_alm), fl, inplace=False)
		b_map = hp.alm2map(b_alm, nside=self.nside_out, lmax=self.lmax_out, pixwin=False)

		if save and cache_file is not None:
			np.savez_compressed(cache_file, b_map=np.asarray(b_map), b_alm=np.asarray(b_alm))

		if return_alm:
			return b_map, b_alm
		return np.asarray(b_map)
