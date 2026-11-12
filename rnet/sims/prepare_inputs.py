rom __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path


STATIC_FILES = (
    "camb.ini", "hitsmap.fits.gz", "mask.fits", "mask_apo.fits",
    "mask_padded.fits",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_checked(source: Path, destination: Path) -> str:
    source_hash = _sha256(source)
    if destination.is_file():
        if _sha256(destination) != source_hash:
            raise ValueError(f"Existing staged input differs from source: {destination}")
        return source_hash
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f"{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.partial"
    )
    try:
        shutil.copy2(source, temporary)
        if _sha256(temporary) != source_hash:
            raise OSError(f"Copy verification failed: {destination}")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return source_hash


def stage_inputs(root: Path | str) -> Path:
    """Copy masks, CAMB settings, hits, and every d11/s6 PySM template."""
    from pysm3.sky import PRESET_MODELS
    from pysm3.utils import RemoteData

    root = Path(root).expanduser().resolve()
    source_data = Path(__file__).resolve().parents[1] / "data"
    static = {}
    for name in STATIC_FILES:
        source = source_data / name
        if not source.is_file():
            raise FileNotFoundError(source)
        static[name] = _copy_checked(source, root / "data" / name)

    pysm_root = root / "inputs" / "pysm_data"
    os.environ["PYSM_LOCAL_DATA"] = str(pysm_root)
    remote = RemoteData()
    templates = sorted({value for preset in ("d11", "s6")
                        for value in PRESET_MODELS[preset].values()
                        if isinstance(value, str) and "/" in value})
    pysm_files = {}
    for name in templates:
        source = Path(remote.get(name))
        pysm_files[name] = _copy_checked(source, pysm_root / name)

    manifest = {
        "version": 1,
        "foreground_presets": ["d11", "s6"],
        "static_files_sha256": static,
        "pysm_files_sha256": pysm_files,
    }
    path = root / "inputs" / "inputs_manifest.json"
    payload = json.dumps(manifest, sort_keys=True, indent=2) + "\n"
    if path.is_file():
        if path.read_text(encoding="utf-8") != payload:
            raise ValueError(f"Existing input manifest differs: {path}")
    else:
        temporary = path.with_name(f"{path.name}.{os.getpid()}.partial")
        try:
            temporary.write_text(payload, encoding="utf-8")
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(stage_inputs(args.root))


if __name__ == "__main__":
    main()
