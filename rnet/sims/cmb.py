from __future__ import annotations

import pickle as pl
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import camb
import healpy as hp
import lenspyx
from mpi4py import MPI



class _MPIProxy:
	"""Small adapter to keep a simple mpi.rank / mpi.barrier API."""

	def __init__(self) -> None:
		if MPI is None:
			self.rank = 0
			self._comm = None
		else:
			self._comm = MPI.COMM_WORLD
			self.rank = self._comm.Get_rank()

	def barrier(self) -> None:
		if self._comm is not None:
			self._comm.Barrier()


mpi = _MPIProxy()


class CMBPowerSpectra:
	"""Compute and cache CMB power spectra from a CAMB ini file."""

	def __init__(
		self,
		basedir: Optional[Path | str] = None,
		ini_file: Optional[Path | str] = None,
		spectra_file: str = "cmb_spectra.pkl",
	) -> None:
		self.basedir = Path(basedir) if basedir is not None else Path.cwd()
		self.basedir.mkdir(parents=True, exist_ok=True)
		self.ini_file = self._resolve_ini_file(ini_file)
		self.__spectra_file__ = self.basedir / spectra_file

	def _resolve_ini_file(self, ini_file: Optional[Path | str]) -> Path:
		if ini_file is not None:
			candidate = Path(ini_file).expanduser().resolve()
			if not candidate.exists():
				raise FileNotFoundError(f"CAMB ini file not found: {candidate}")
			return candidate

		candidates = list(self._ini_candidates())
		for candidate in candidates:
			if candidate.exists():
				return candidate.resolve()

		looked_in = "\n".join(f"- {path}" for path in candidates)
		raise FileNotFoundError(
			"Could not locate camb.ini automatically. Checked:\n"
			f"{looked_in}"
		)

	def _ini_candidates(self) -> Iterable[Path]:
		yield self.basedir / "data" / "camb.ini"
		# Fall back to the packaged default for older suites.
		yield Path(__file__).resolve().parents[1] / "data" / "camb.ini"

		for root in [Path.cwd(), *Path.cwd().resolve().parents]:
			yield root / "data" / "camb.ini"

		for root in Path(__file__).resolve().parents:
			yield root / "data" / "camb.ini"

	def compute_powers(self) -> Dict[str, Any]:
		"""
		Compute CMB power spectra using CAMB.
		"""
		def _valid_payload(payload: Any) -> bool:
			return isinstance(payload, dict) and {"cls", "dls", "cls_r1"}.issubset(payload.keys())

		if self.__spectra_file__.is_file():
			try:
				with open(self.__spectra_file__, "rb") as handle:
					cached = pl.load(handle)
				if _valid_payload(cached):
					return cached
			except Exception:
				# Fall back to recomputing if cache is missing/corrupted/incompatible.
				pass

		if camb is None:
			raise ImportError(
				"CAMB is required to compute CMB power spectra. "
				"Install it with `pip install camb`."
			)

		# In MPI runs, only rank 0 computes and writes; all ranks then read the same cache.
		if mpi.rank != 0:
			mpi.barrier()
			with open(self.__spectra_file__, "rb") as handle:
				cached = pl.load(handle)
			if not _valid_payload(cached):
				raise RuntimeError("Cached CMB spectra file is invalid after MPI synchronization")
			return cached

		cwd = Path.cwd()
		os.chdir(self.basedir)
		try:
			params = camb.read_ini(str(self.ini_file))
			results = camb.get_results(params)

			# Build an explicit r=1 tensor reference, independent of ini settings.
			params_r1 = camb.read_ini(str(self.ini_file))
			params_r1.InitPower.r = 1.0
			results_r1 = camb.get_results(params_r1)
		finally:
			os.chdir(cwd)

		powers: Dict[str, Any] = {}
		powers["cls"] = results.get_cmb_power_spectra(
			params, CMB_unit="muK", raw_cl=True
		)
		powers["dls"] = results.get_cmb_power_spectra(
			params, CMB_unit="muK", raw_cl=False
		)
		powers["cls_r1"] = results_r1.get_cmb_power_spectra(
			params_r1, CMB_unit="muK", raw_cl=True
		)

		with open(self.__spectra_file__, "wb") as handle:
			pl.dump(powers, handle)

		mpi.barrier()
		return powers


class CMBLensing:
	"""Generate CMB lensing potential alms and lensed Q/U maps."""

	def __init__(
		self,
		nside: int = 1024,
		dlmax: int = 512,
		basedir: Optional[Path | str] = None,
		power_spectra: Optional[CMBPowerSpectra] = None,
		seed: int = 1234,
		nseed_pool: int = 10000,
		verbose: bool = False,
	) -> None:
		self.nside = int(nside)
		if self.nside <= 0:
			raise ValueError("nside must be > 0")
		self.dlmax = int(dlmax)
		if self.dlmax < 0:
			raise ValueError("dlmax must be >= 0")

		# Standard harmonic limit for maps at this resolution.
		self.lmax = 3 * self.nside - 1
		# Internal synthesis/lensing bandwidth with configurable buffer.
		self.lmax_lens = self.lmax + self.dlmax
		self.verbose = bool(verbose)

		self.basedir = Path(basedir) if basedir is not None else Path.cwd()
		self.basedir.mkdir(parents=True, exist_ok=True)

		self._powers_obj = (
			power_spectra if power_spectra is not None else CMBPowerSpectra(basedir=self.basedir)
		)
		self._powers_cache: Optional[Dict[str, Any]] = None

		# Keep deterministic stream seeds based on idx without fixed-size seed pools.
		self._seed_base = int(seed)
		self._seed_step = 1103515245
		self._seed_modulus = np.iinfo(np.int32).max - 1

	def _seed_from_idx(self, idx: int, stream: int) -> int:
		if idx < 0:
			raise ValueError("idx must be >= 0")
		if stream < 0:
			raise ValueError("stream must be >= 0")

		base = self._seed_base % self._seed_modulus
		shift = (stream + 1) * 362437
		seed = (base + shift + int(idx) * self._seed_step) % self._seed_modulus
		return int(seed + 1)


	def _powers(self) -> Dict[str, Any]:
		if self._powers_cache is None:
			self._powers_cache = self._powers_obj.compute_powers()
		return self._powers_cache

	def __get_cmb_seed_idx__(self, idx: int) -> int:
		if idx < 0:
			raise ValueError("idx must be >= 0")
		return int(idx)

	def _tensor_r1_bb(self, dl: bool = False) -> np.ndarray:
		"""Return r=1 tensor BB spectrum up to lmax_lens."""
		if dl:
			raise ValueError("r=1 tensor reference is currently available only for raw_cl")

		payload = self._powers()["cls_r1"]
		key = "tensor"
		if key not in payload:
			raise KeyError(
				"CAMB r=1 output does not contain `tensor` spectra required for r > 0"
			)

		return np.asarray(payload[key][: self.lmax_lens + 1, 2], dtype=np.float64)

	def cl_pp(self) -> np.ndarray:
		"""Return C_L^{phiphi} up to lmax from CAMB lens_potential output."""
		cls = self._powers()["cls"]
		if "lens_potential" not in cls:
			raise KeyError("CAMB output does not contain `lens_potential` spectra")
		return np.asarray(cls["lens_potential"][: self.lmax_lens + 1, 0], dtype=np.float64)

	def get_unlensed_spectra(self, dl: bool = False) -> Dict[str, np.ndarray]:
		"""Return unlensed TT/EE/BB/TE spectra arrays up to lmax."""
		payload = self._powers()["dls" if dl else "cls"]
		key = "unlensed_scalar"
		if key not in payload:
			raise KeyError(f"CAMB output does not contain `{key}` spectra")

		arr = np.asarray(payload[key][: self.lmax_lens + 1, :4], dtype=np.float64)
		return {
			"tt": arr[:, 0],
			"ee": arr[:, 1],
			"bb": arr[:, 2],
			"te": arr[:, 3],
		}

	def phi_alm(self, idx: int, alens: float = 1.0) -> np.ndarray:
		"""Generate the lensing potential alm for simulation index idx."""

		seed_idx = self.__get_cmb_seed_idx__(idx)

		if alens < 0:
			raise ValueError("alens must be >= 0")
		cl_pp = float(alens) * self.cl_pp()
		rng_state = np.random.get_state()
		np.random.seed(self._seed_from_idx(seed_idx, stream=0))
		try:
			alm = hp.synalm(cl_pp, lmax=self.lmax_lens, new=True)
		finally:
			np.random.set_state(rng_state)

		return alm

	def grad_phi_alm(self, idx: int, alens: float = 1.0) -> np.ndarray:
		"""Return deflection alm from phi alm via sqrt(l(l+1)) filter."""

		phi_alm = self.phi_alm(idx, alens=alens)
		ells = np.arange(self.lmax_lens + 1, dtype=np.float64)
		filt = np.sqrt(ells * (ells + 1.0))
		return hp.almxfl(phi_alm, filt, None, False)

	def get_lensed_qu(self, idx: int, r: float = 0.0, alens: float = 1.0,
				  nthreads: int = 0) -> List[np.ndarray]:
		"""Generate deterministic lensed real-space Q/U maps for a simulation index."""

		if r < 0.0 or alens < 0.0:
			raise ValueError("r and alens must be >= 0")

		seed_idx = self.__get_cmb_seed_idx__(idx)

		spectra = self.get_unlensed_spectra(dl=False)
		if r > 0.0:
			r1_bb = self._tensor_r1_bb(dl=False)
			spectra = {
				"tt": spectra["tt"],
				"ee": spectra["ee"],
				"bb": spectra["bb"] + r * r1_bb,
				"te": spectra["te"],
			}

		rng_state = np.random.get_state()
		np.random.seed(self._seed_from_idx(seed_idx, stream=1))
		try:
			alms = hp.synalm(
				[
					spectra["tt"],
					spectra["ee"],
					spectra["bb"],
					spectra["te"],
				],
				lmax=self.lmax_lens,
				new=True,
			)
		finally:
			np.random.set_state(rng_state)

		defl = self.grad_phi_alm(idx, alens=alens)
		geom_info = ("healpix", {"nside": self.nside})
		qlen, ulen = lenspyx.alm2lenmap_spin(
			[alms[1], alms[2]],
			defl,
			2,
			geometry=geom_info,
			verbose=int(self.verbose),
			nthreads=int(nthreads),
		)

		return [np.asarray(qlen), np.asarray(ulen)]

