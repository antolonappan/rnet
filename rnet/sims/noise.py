from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import healpy as hp 
from so_models_v3 import SO_Noise_Calculator_Public_v3_1_2 as so_models


def NoiseSpectra(
	sensitivity_mode: str,
	fsky: float,
	lmax: int,
	atm_noise: bool,
) -> Dict[str, np.ndarray]:
	"""Return SAT noise spectra from SO v3.1.2 models."""
	if so_models is None:
		raise ImportError(
			"so_models_v3 is required. Make sure `so_noise_models` is installed "
			"and import `SO_Noise_Calculator_Public_v3_1_2` is available."
		)

	teles = so_models.SOSatV3point1(
		sensitivity_mode=sensitivity_mode,
		N_tubes=[7.0, 35.5, 9.0]
		survey_years=1.0,
		one_over_f_mode=1,
		survey_efficiency=0.18,
	)

	corr_pairs = [(0, 1), (2, 3), (4, 5)]
	ell, _, n_ell_pol_full = teles.get_noise_curves(
		fsky, lmax, 1, full_covar=True, deconv_beam=False
	)
	bands = teles.get_bands().astype(int)
	nbands = len(bands)

	n_ell_pol = n_ell_pol_full[range(nbands), range(nbands)]  # type: ignore[index]
	n_ell_pol_x = [n_ell_pol_full[i, j] for i, j in corr_pairs]  # type: ignore[index]

	nell_dict: Dict[str, np.ndarray] = {}
	nell_dict["ell"] = np.asarray(ell)

	if atm_noise:
		for i in range(3):
			for j in range(3):
				if j < 2:
					nell_dict[f"{bands[i * 2 + j]}"] = np.asarray(n_ell_pol[i * 2 + j])
				else:
					k = i * 2 + j
					nell_dict[f"{bands[k - 2]}x{bands[k - 1]}"] = np.asarray(n_ell_pol_x[i])
	else:
		white_noise = np.radians(teles.get_white_noise(fsky) ** 0.5 * np.sqrt(2) / 60.0) ** 2
		for i in range(nbands):
			nell_dict[f"{bands[i]}"] = np.asarray(white_noise[i] * np.ones_like(ell))

	return nell_dict


class Noise:
	"""SAT-only noise simulator with hardcoded survey setup and optional atm/1f noise."""

	def __init__(
		self,
		atm_noise: bool | str = False,
		sensitivity: str = "goal",
		atm: bool | str | None = None,
		decon: bool | str = True,
		inhom: bool | str = True,
		nside: int = 1024,
		lmax: int | None = None,
		hits_path: Path | str | None = None,
	) -> None:


		def _parse_bool(value: bool | str) -> bool:
			if isinstance(value, bool):
				return value
			text = str(value).strip().lower()
			if text in {"true", "t", "1", "yes", "y", "tre"}:
				return True
			if text in {"false", "f", "0", "no", "n"}:
				return False
			raise ValueError(f"Cannot parse boolean value from: {value}")

		self.nside = int(nside)
		if self.nside <= 0:
			raise ValueError("nside must be > 0")
		self.fsky = 0.4
		self.lmax = 3 * self.nside - 1 if lmax is None else int(lmax)
		if not 2 <= self.lmax <= 3 * self.nside - 1:
			raise ValueError("lmax must be between 2 and 3*nside-1")
		self.telescope = "SAT"
		self.sensitivity_mode = sensitivity
		self.atm_noise = _parse_bool(atm if atm is not None else atm_noise)
		self.decon = _parse_bool(decon)
		self.inhom = _parse_bool(inhom)
		self.hits_path = Path(hits_path).expanduser() if hits_path is not None else None

		self.bands = np.array([27, 39, 93, 145, 225, 280], dtype=int)
		self.nell = NoiseSpectra(self.sensitivity_mode, self.fsky, self.lmax, self.atm_noise)
		self._seed_base = 7777
		self._seed_step = 1664525
		self._seed_modulus = np.iinfo(np.int32).max - 1
		self._atm_cache_idx: int | None = None
		self._atm_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] | None = None

		# SAT beam FWHM (arcmin) per band, used to deconvolve E/B alms before map making.
		self.fwhm_arcmin = {"27": 91.0, "39": 63.0, "93": 30.0, "145": 17.0, "225": 11.0, "280": 9.0}
		self._bl_inv = {band: self._inverse_beam(fwhm) for band, fwhm in self.fwhm_arcmin.items()} if self.decon else None
		self._inhom_weight = self._load_inhom_weight() if self.inhom else None


	_DECON_LCUT = 350

	def _inverse_beam(self, fwhm_arcmin: float) -> np.ndarray:
		bl = hp.gauss_beam(np.radians(fwhm_arcmin / 60.0), lmax=self.lmax)
		lcut = min(self._DECON_LCUT, self.lmax)
		inv = np.ones_like(bl)
		good = bl[: lcut + 1] > 1e-300
		inv[: lcut + 1][good] = 1.0 / bl[: lcut + 1][good]
		inv[lcut + 1 :] = inv[lcut]
		return inv

	def _load_inhom_weight(self) -> np.ndarray:
		hits_path = self.hits_path or Path(__file__).resolve().parent.parent / "data" / "hitsmap.fits.gz"
		hits = hp.read_map(str(hits_path))
		if hp.get_nside(hits) != self.nside:
			hits = hp.ud_grade(hits, nside_out=self.nside)
		observed = (hits > 0).astype(float)
		nhits = np.clip(hits / hits.max(), 1e-3, None)
		return observed / np.sqrt(nhits)

	def _seed_idx(self, idx: int) -> int:
		if idx < 0:
			raise ValueError("idx must be >= 0")
		return int(idx)

	def _seed_from_idx(self, idx: int) -> int:
		seed_idx = self._seed_idx(idx)
		seed = (self._seed_base + seed_idx * self._seed_step) % self._seed_modulus
		return int(seed + 1)

	def _draw_rand_alm(self, seed: int) -> np.ndarray:
		state = np.random.get_state()
		np.random.seed(int(seed))
		try:
			cl = np.repeat(1.0e3, self.lmax + 1)
			alm = hp.synalm(cl, lmax=self.lmax, new=True)
			return hp.almxfl(alm, 1.0 / np.sqrt(cl))
		finally:
			np.random.set_state(state)

	def _cl_from_nell(self, key: str) -> np.ndarray:
		out = np.zeros(self.lmax + 1, dtype=float)
		ell = np.asarray(self.nell["ell"], dtype=int)
		vals = np.asarray(self.nell[key], dtype=float)
		n = min(len(ell), len(vals))
		i = ell[:n]
		v = vals[:n]
		valid = (i >= 0) & (i <= self.lmax)
		out[i[valid]] = v[valid]
		return out

	def _atm_cholesky_terms(self) -> Dict[str, np.ndarray]:
		auto_27 = self._cl_from_nell("27")
		auto_39 = self._cl_from_nell("39")
		auto_93 = self._cl_from_nell("93")
		auto_145 = self._cl_from_nell("145")
		auto_225 = self._cl_from_nell("225")
		auto_280 = self._cl_from_nell("280")

		cross_27x39 = self._cl_from_nell("27x39")
		cross_93x145 = self._cl_from_nell("93x145")
		cross_225x280 = self._cl_from_nell("225x280")

		def terms(cl_a: np.ndarray, cl_b: np.ndarray, cl_ab: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
			l11 = np.sqrt(np.maximum(cl_a, 0.0))
			l21 = np.zeros_like(l11)
			good = l11 > 0
			l21[good] = cl_ab[good] / l11[good]
			arg = np.maximum(cl_b - l21**2, 0.0)
			l22 = np.sqrt(arg)
			return l11, l21, l22

		l11, l21, l22 = terms(auto_27, auto_39, cross_27x39)
		l33, l43, l44 = terms(auto_93, auto_145, cross_93x145)
		l55, l65, l66 = terms(auto_225, auto_280, cross_225x280)

		return {
			"27": l11,
			"39_a": l21,
			"39_b": l22,
			"93": l33,
			"145_a": l43,
			"145_b": l44,
			"225": l55,
			"280_a": l65,
			"280_b": l66,
		}

	def _white_noise_map(self, band: str, seed: int) -> np.ndarray:
		cl = self._cl_from_nell(band)
		state = np.random.get_state()
		np.random.seed(int(seed))
		try:
			return hp.synalm(cl, lmax=self.lmax, new=True)
		finally:
			np.random.set_state(state)

	def _atm_eb_alms(self, idx: int) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
		base = self._seed_from_idx(idx)
		terms = self._atm_cholesky_terms()

		# E-mode correlated pairs
		e_a1 = self._draw_rand_alm(base)
		e_a2 = self._draw_rand_alm(base + 1)
		e_c1 = self._draw_rand_alm(base + 2)
		e_c2 = self._draw_rand_alm(base + 3)
		e_e1 = self._draw_rand_alm(base + 4)
		e_e2 = self._draw_rand_alm(base + 5)

		# B-mode correlated pairs, independent from E
		b_a1 = self._draw_rand_alm(base + 100)
		b_a2 = self._draw_rand_alm(base + 101)
		b_c1 = self._draw_rand_alm(base + 102)
		b_c2 = self._draw_rand_alm(base + 103)
		b_e1 = self._draw_rand_alm(base + 104)
		b_e2 = self._draw_rand_alm(base + 105)

		e27 = hp.almxfl(e_a1, terms["27"])
		e39 = hp.almxfl(e_a1, terms["39_a"]) + hp.almxfl(e_a2, terms["39_b"])
		e93 = hp.almxfl(e_c1, terms["93"])
		e145 = hp.almxfl(e_c1, terms["145_a"]) + hp.almxfl(e_c2, terms["145_b"])
		e225 = hp.almxfl(e_e1, terms["225"])
		e280 = hp.almxfl(e_e1, terms["280_a"]) + hp.almxfl(e_e2, terms["280_b"])

		b27 = hp.almxfl(b_a1, terms["27"])
		b39 = hp.almxfl(b_a1, terms["39_a"]) + hp.almxfl(b_a2, terms["39_b"])
		b93 = hp.almxfl(b_c1, terms["93"])
		b145 = hp.almxfl(b_c1, terms["145_a"]) + hp.almxfl(b_c2, terms["145_b"])
		b225 = hp.almxfl(b_e1, terms["225"])
		b280 = hp.almxfl(b_e1, terms["280_a"]) + hp.almxfl(b_e2, terms["280_b"])

		return {
			"27": (e27, b27),
			"39": (e39, b39),
			"93": (e93, b93),
			"145": (e145, b145),
			"225": (e225, b225),
			"280": (e280, b280),
		}

	def get_freq_qu(self, idx: int, freq: int | str) -> np.ndarray:
		"""Return Q/U noise maps for a single SAT frequency band."""
		band = str(freq).split("-")[0]
		if band not in {"27", "39", "93", "145", "225", "280"}:
			raise ValueError("Invalid SAT frequency band. Choose from 27, 39, 93, 145, 225, 280")

		seed_idx = self._seed_idx(idx)
		base = self._seed_from_idx(seed_idx)

		if self.atm_noise:
			if self.nside <= 128:
				if self._atm_cache_idx != seed_idx:
					self._atm_cache = self._atm_eb_alms(idx)
					self._atm_cache_idx = seed_idx
				e_alm, b_alm = self._atm_cache[band]
			else:
				e_alm, b_alm = self._atm_eb_alms(idx)[band]
		else:
			e_alm = self._white_noise_map(band, base)
			b_alm = self._white_noise_map(band, base + 1000)

		if self.decon:
			bl_inv = self._bl_inv[band]
			e_alm = hp.almxfl(e_alm, bl_inv)
			b_alm = hp.almxfl(b_alm, bl_inv)
		q_map, u_map = hp.alm2map_spin([e_alm, b_alm], self.nside, 2, lmax=self.lmax)
		if self.inhom:
			q_map = q_map * self._inhom_weight
			u_map = u_map * self._inhom_weight
		return np.asarray([q_map, u_map])

	def get_all_freq_qu(self, idx: int) -> np.ndarray:
		"""Return Q/U noise maps for all SAT frequency bands in band order."""
		maps: List[np.ndarray] = []
		for band in self.bands:
			maps.append(self.get_freq_qu(idx, int(band)))
		return np.asarray(maps)
