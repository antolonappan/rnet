from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import healpy as hp
import pysm3 
import pysm3.units as u


class Pysm3Foregrounds:
	"""Generate and cache PySM3 foreground Q/U maps for a fixed frequency."""

	def __init__(
		self,
		basedir: Optional[Path | str] = None,
		nside: int = 1024,
		preset_string: str = ["d10","s5"],
		output_unit = u.uK_RJ if u is not None else None,
	) -> None:
		self.nside = int(nside)
		if self.nside <= 0:
			raise ValueError("nside must be > 0")

		self.preset_string = preset_string
		self.output_unit = output_unit
		self.basedir = Path(basedir) if basedir is not None else Path.cwd()
		self.basedir.mkdir(parents=True, exist_ok=True)
		self.fgdir = self.basedir / "fg"
		self.fgdir.mkdir(parents=True, exist_ok=True)

		self._sky = None



	def _get_sky(self):
		if self._sky is None:
			self._sky = pysm3.Sky(
				nside=self.nside,
				preset_strings=self.preset_string,
				output_unit=self.output_unit,
			)
		return self._sky

	def _freq_tag(self, freq_ghz: float) -> str:
		freq = float(freq_ghz)
		return f"{freq:.3f}".replace(".", "p")

	def _freq_file(self, freq_ghz: float) -> Path:
		return self.fgdir / f"fg_{self.preset_string}_nside{self.nside}_freq{self._freq_tag(freq_ghz)}.fits"

	def get_freq_qu(self, freq_ghz: float) -> List[np.ndarray]:
		"""Return cached Q/U maps for a single frequency in GHz."""
		savefile = self._freq_file(freq_ghz)
		if savefile.is_file():
			q_map, u_map = hp.read_map(str(savefile), field=[0, 1])
			return [np.asarray(q_map), np.asarray(u_map)]

		sky = self._get_sky()
		freq = float(freq_ghz) * u.GHz
		emission = sky.get_emission(freq)
		if self.output_unit is not None and emission.unit != self.output_unit:
			emission = emission.to(self.output_unit, equivalencies=u.cmb_equivalencies(freq))
		emission = np.asarray(emission)

		if emission.shape[0] < 3:
			raise ValueError("PySM emission did not return I, Q, U components")

		q_map = np.asarray(emission[1])
		u_map = np.asarray(emission[2])
		hp.write_map(str(savefile), [q_map, u_map], dtype=np.float64, overwrite=True)
		return [q_map, u_map]
