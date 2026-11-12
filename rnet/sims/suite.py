from __future__ import annotations

import hashlib
import json
from pathlib import Path


def training_quotas(n_train: int, enriched: bool) -> dict[str, int]:
    """Keep one percent exact-zero maps and optionally 25 percent low-r maps."""
    n_train = int(n_train)
    if not 100 <= n_train <= 20_000:
        raise ValueError("n_train must be between 100 and 20,000")
    zero = round(n_train / 100)
    low = round(n_train / 4) if enriched else 0
    return {"uniform_train": n_train - low - zero,
            "low_train": low, "zero_train": zero}


def build_training_manifest(
    root: Path | str, n_train: int, enriched: bool = False,
    output_path: Path | str | None = None,
) -> Path:
    """Build a fixed-ratio, nested train/validation manifest without copying maps.

    Pool manifests are in generation order. Their prefixes are used for every
    training size, so smaller selections remain subsets after pool extension.
    """
    root = Path(root).expanduser().resolve()
    quotas = training_quotas(n_train, enriched)
    pool_names = ("uniform_train", "low_train", "zero_train",
                  "uniform_validation", "low_validation", "zero_validation")
    pools = {}
    for name in pool_names:
        path = root / name / "manifest.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing simulation pool: {path}")
        pools[name] = json.loads(path.read_text(encoding="utf-8"))

    reference = pools["uniform_train"]
    for name, manifest in pools.items():
        if manifest.get("settings_hash") != reference.get("settings_hash"):
            raise ValueError(f"Incompatible simulation settings in {name}")
        if manifest.get("freq_values") != reference.get("freq_values"):
            raise ValueError(f"Incompatible frequencies in {name}")

    entries = []
    for name in pool_names:
        source = pools[name]["entries"]
        wanted = quotas.get(name, len(source))
        if len(source) < wanted:
            raise ValueError(f"{name} has {len(source)} maps but {wanted} are required")
        role = "validation" if name.endswith("_validation") else "train"
        for entry in source[:wanted]:
            if entry.get("source_pool") != name or entry.get("role") != role:
                raise ValueError(f"Invalid pool entry in {name}")
            entries.append(dict(entry))

    ids = [int(entry["sim_idx"]) for entry in entries]
    if len(ids) != len(set(ids)):
        raise ValueError("Training and validation pools overlap")
    recipe = "enriched" if enriched else "uniform"
    selected = dict(reference)
    selected.update({
        "dataset_kind": "continuous",
        "split_policy": "explicit_roles",
        "recipe": recipe,
        "quotas": quotas,
        "n_train": int(n_train),
        "n_validation": len(entries) - int(n_train),
        "source_manifests": [str(root / name / "manifest.json") for name in pool_names],
        "selection_hash": hashlib.sha256(
            json.dumps(ids, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "entries": entries,
    })
    if output_path is None:
        output_path = root / "recipes" / f"{recipe}_n{int(n_train):05d}.json"
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.is_file():
        previous = json.loads(output_path.read_text(encoding="utf-8"))
        if previous.get("selection_hash") != selected["selection_hash"]:
            raise ValueError(f"Existing training manifest has different maps: {output_path}")
    temporary = output_path.with_suffix(".partial.json")
    temporary.write_text(json.dumps(selected, indent=2), encoding="utf-8")
    temporary.replace(output_path)
    return output_path
