from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .config import DataConfig


def _read_manifest(path: Path | str) -> Dict[str, object]:
    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Dataset manifest not found: {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid dataset manifest: {manifest_path}")
    return payload


def build_mixed_manifests(
    continuous_manifest: Path | str,
    grid_manifest: Path | str,
    mixed_manifest: Path | str,
    benchmark_manifest: Path | str,
    benchmark_fraction: float = 0.2,
    seed: int = 2027,
) -> Tuple[Path, Path]:
    """Combine continuous simulations with part of a grid suite.

    The grid is split independently at every exact ``r`` value. The benchmark
    portion is never included in the mixed training manifest, which keeps a
    clean, balanced final evaluation set for CNN-vs-HL comparisons.
    """
    if not 0.0 < float(benchmark_fraction) < 1.0:
        raise ValueError("benchmark_fraction must lie strictly between 0 and 1")

    continuous = _read_manifest(continuous_manifest)
    grid = _read_manifest(grid_manifest)
    required = ("product", "ordering", "freq_values", "nside", "entries")
    for name, manifest in (("continuous", continuous), ("grid", grid)):
        missing = [key for key in required if key not in manifest]
        if missing:
            raise ValueError(f"{name} manifest is missing keys: {missing}")
        if manifest["product"] != "bmode" or str(manifest["ordering"]).upper() != "NEST":
            raise ValueError(f"{name} manifest must contain NEST bmode data")

    for key in ("freq_values", "nside", "mask_cache"):
        if continuous.get(key) != grid.get(key):
            raise ValueError(f"Continuous and grid manifests disagree on {key!r}")

    continuous_entries = list(continuous["entries"])
    grid_entries = list(grid["entries"])
    if not continuous_entries or not grid_entries:
        raise ValueError("Both manifests must contain at least one entry")

    grouped: Dict[str, List[Mapping[str, object]]] = {}
    for entry in grid_entries:
        if not isinstance(entry, Mapping) or "r" not in entry:
            raise ValueError("Grid manifest contains an invalid entry")
        grouped.setdefault(f"{float(entry['r']):.6f}", []).append(entry)

    rng = np.random.default_rng(int(seed))
    mixed_grid: List[Mapping[str, object]] = []
    benchmark_entries: List[Mapping[str, object]] = []
    for r_key in sorted(grouped, key=float):
        entries = list(grouped[r_key])
        order = rng.permutation(len(entries))
        n_benchmark = max(1, int(round(len(entries) * float(benchmark_fraction))))
        if n_benchmark >= len(entries):
            n_benchmark = len(entries) - 1
        benchmark_indices = set(int(index) for index in order[:n_benchmark])
        benchmark_entries.extend(entry for index, entry in enumerate(entries) if index in benchmark_indices)
        mixed_grid.extend(entry for index, entry in enumerate(entries) if index not in benchmark_indices)

    mixed_path = Path(mixed_manifest).expanduser().resolve()
    benchmark_path = Path(benchmark_manifest).expanduser().resolve()
    mixed_path.parent.mkdir(parents=True, exist_ok=True)
    benchmark_path.parent.mkdir(parents=True, exist_ok=True)

    def make_manifest(entries: Sequence[Mapping[str, object]], kind: str) -> Dict[str, object]:
        payload = dict(continuous)
        payload["entries"] = list(entries)
        payload["dataset_kind"] = kind
        payload["source_manifests"] = [str(Path(continuous_manifest).expanduser().resolve()), str(Path(grid_manifest).expanduser().resolve())]
        payload["grid_benchmark_fraction"] = float(benchmark_fraction)
        payload["split_seed"] = int(seed)
        return payload

    with open(mixed_path, "w", encoding="utf-8") as handle:
        json.dump(make_manifest(continuous_entries + mixed_grid, "continuous_plus_grid_train"), handle, indent=2, sort_keys=True)
    with open(benchmark_path, "w", encoding="utf-8") as handle:
        json.dump(make_manifest(benchmark_entries, "grid_benchmark"), handle, indent=2, sort_keys=True)
    return mixed_path, benchmark_path


@dataclass(frozen=True)
class Standardizer:
    """Per-frequency affine normalization fitted on the training split only."""

    mean: np.ndarray
    std: np.ndarray

    def transform(self, maps: np.ndarray) -> np.ndarray:
        return (np.asarray(maps, dtype=np.float32) - self.mean) / self.std

    def save(self, path: Path | str) -> None:
        np.savez(Path(path), mean=self.mean, std=self.std)

    @classmethod
    def load(cls, path: Path | str) -> "Standardizer":
        with np.load(Path(path)) as payload:
            return cls(
                mean=np.asarray(payload["mean"], dtype=np.float32),
                std=np.asarray(payload["std"], dtype=np.float32),
            )


class SimulationDataset:

    def __init__(self, config: DataConfig) -> None:
        self.config = config
        with open(config.manifest, "r", encoding="utf-8") as handle:
            self.manifest = json.load(handle)

        if self.manifest.get("product") != "bmode":
            raise ValueError("CNN input requires a manifest with product='bmode'")
        if str(self.manifest.get("ordering", "")).upper() != "NEST":
            raise ValueError("DeepSphere input must use NEST ordering")

        self.entries: List[Mapping[str, object]] = list(self.manifest.get("entries", []))
        available = tuple(str(v) for v in self.manifest.get("freq_values", []))
        self.is_paired_hdf5 = self.manifest.get("format") == "hdf5_paired_bmode"
        if config.frequencies is None:
            self.frequencies = available
        else:
            wanted = tuple(f"{band:.3f}" for band in config.frequencies)
            if any(band not in available for band in wanted):
                raise ValueError(f"Requested frequencies {wanted} not present in {available}")
            self.frequencies = wanted
        self.channel_indices = tuple(available.index(band) for band in self.frequencies)
        if self.is_paired_hdf5 and config.variant not in self.manifest.get("variants", []):
            raise ValueError(f"Variant {config.variant!r} is missing from the manifest")
        if not self.is_paired_hdf5 and config.variant != "nofg":
            raise ValueError("Legacy NPZ manifests do not support selectable variants")
        if self.is_paired_hdf5:
            identity = {
                "settings_hash": self.manifest.get("settings_hash"),
                "selection_hash": self.manifest.get("selection_hash"),
                "manifest": str(config.manifest),
                "entries": len(self.entries),
                "frequencies": self.frequencies,
                "variant": config.variant,
            }
            self.dataset_key = hashlib.sha256(
                json.dumps(identity, sort_keys=True).encode("utf-8")
            ).hexdigest()
        else:
            self.dataset_key = self.manifest.get("dataset_key")
        self.nside = int(self.manifest.get("nside", 0))
        if not self.entries or not self.frequencies or self.nside <= 0:
            raise ValueError("Manifest is missing entries, frequencies, or nside")

        self.pixel_indices = self._load_pixel_indices()
        self.n_pixels = int(self.pixel_indices.size)
        self._validate_first_entry()
        self.splits = self._stratified_splits()

    def _load_pixel_indices(self) -> np.ndarray:
        cache = self.manifest.get("mask_cache")
        if cache is None:
            return np.arange(12 * self.nside**2, dtype=np.int64)
        cache_path = Path(str(cache)).expanduser()
        if not cache_path.is_file():
            raise FileNotFoundError(f"Mask cache referenced by manifest does not exist: {cache_path}")
        with np.load(cache_path) as payload:
            return np.asarray(payload["pixel_idx"], dtype=np.int64)

    def _entry_paths(self, entry_index: int) -> Tuple[Path, ...]:
        if self.is_paired_hdf5:
            path = Path(str(self.entries[int(entry_index)]["h5_file"])).expanduser()
            if not path.is_file():
                raise FileNotFoundError(f"Missing simulation shard: {path}")
            return (path,)
        files = self.entries[int(entry_index)].get("files")
        if not isinstance(files, Mapping):
            raise ValueError(f"Entry {entry_index} has no valid frequency file mapping")
        paths = tuple(Path(str(files[freq])).expanduser() for freq in self.frequencies)
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError("Missing simulation map(s): " + ", ".join(missing))
        return paths

    def _validate_first_entry(self) -> None:
        maps, _ = self.load_sample(0, standardizer=None)
        if maps.shape != (self.n_pixels, len(self.frequencies)):
            raise ValueError(
                f"Map shape {maps.shape} does not match expected "
                f"({self.n_pixels}, {len(self.frequencies)})"
            )

    def _stratified_splits(self) -> Dict[str, np.ndarray]:
        if self.manifest.get("split_policy") == "explicit_roles":
            roles = [entry.get("role") for entry in self.entries]
            if any(role not in {"train", "validation"} for role in roles):
                raise ValueError("Explicit-role manifests require train/validation entries only")
            return {
                "train": np.flatnonzero(np.asarray(roles) == "train"),
                "validation": np.flatnonzero(np.asarray(roles) == "validation"),
                "test": np.empty(0, dtype=np.int64),
            }
        if self.is_paired_hdf5 and self.manifest.get("dataset_kind") == "grid":
            return {
                "train": np.empty(0, dtype=np.int64),
                "validation": np.empty(0, dtype=np.int64),
                "test": np.asarray(
                    [i for i, entry in enumerate(self.entries) if entry.get("role") == "test"],
                    dtype=np.int64,
                ),
            }
        values = np.asarray([float(entry["r"]) for entry in self.entries])
        n_bins = min(20, len(values) // 50)
        if n_bins >= 2:
            order = np.argsort(values, kind="stable")
            labels = np.empty(len(values), dtype=np.int64)
            labels[order] = np.minimum(np.arange(len(values)) * n_bins // len(values), n_bins - 1)
            groups = {int(label): np.flatnonzero(labels == label).tolist() for label in range(n_bins)}
        else:
            groups = {}
            for index, entry in enumerate(self.entries):
                groups.setdefault(str(entry["r"]), []).append(index)

        rng = np.random.default_rng(self.config.split_seed)
        split: Dict[str, List[int]] = {"train": [], "validation": [], "test": []}
        for indices in groups.values():
            shuffled = np.asarray(indices, dtype=np.int64)
            rng.shuffle(shuffled)
            n = len(shuffled)
            n_test = int(np.floor(n * self.config.test_fraction))
            n_validation = int(np.floor(n * self.config.validation_fraction))
            if self.config.test_fraction > 0 and n >= 3:
                n_test = max(1, n_test)
            if self.config.validation_fraction > 0 and n >= 3:
                n_validation = max(1, n_validation)
            if n_test + n_validation >= n:
                n_validation = max(0, n - n_test - 1)
            split["test"].extend(shuffled[:n_test])
            split["validation"].extend(shuffled[n_test : n_test + n_validation])
            split["train"].extend(shuffled[n_test + n_validation :])

        for name, values in split.items():
            arr = np.asarray(values, dtype=np.int64)
            np.random.default_rng(self.config.split_seed + len(name)).shuffle(arr)
            split[name] = arr.tolist()
        return {name: np.asarray(values, dtype=np.int64) for name, values in split.items()}

    def load_sample(
        self,
        entry_index: int,
        standardizer: Optional[Standardizer],
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.is_paired_hdf5:
            import h5py

            entry = self.entries[int(entry_index)]
            with h5py.File(self._entry_paths(entry_index)[0], "r") as shard:
                row = int(entry["h5_row"])
                if not shard["done"][row] or int(shard["sim_idx"][row]) != int(entry["sim_idx"]):
                    raise ValueError("Incomplete or mismatched paired simulation row")
                maps = np.stack(
                    [np.asarray(shard[self.config.variant][row, band, :], dtype=np.float32)
                     for band in self.channel_indices], axis=-1,
                )
        else:
            channels: List[np.ndarray] = []
            for path in self._entry_paths(entry_index):
                with np.load(path) as payload:
                    if "b_selected" not in payload:
                        raise KeyError(f"Missing b_selected in {path}")
                    channels.append(np.asarray(payload["b_selected"], dtype=np.float32))
            maps = np.stack(channels, axis=-1)
        if standardizer is not None:
            maps = standardizer.transform(maps)
        target = np.asarray(
            [float(self.entries[int(entry_index)]["r"]) / self.config.target_scale],
            dtype=np.float32,
        )
        return maps, target

    def fit_standardizer(self, indices: Optional[Sequence[int]] = None) -> Standardizer:
        use_indices = self.splits["train"] if indices is None else np.asarray(indices, dtype=np.int64)
        if len(use_indices) == 0:
            raise ValueError("Cannot fit normalization with an empty training split")
        rng = np.random.default_rng(self.config.split_seed + 91)
        if len(use_indices) > self.config.normalization_samples:
            use_indices = rng.choice(use_indices, self.config.normalization_samples, replace=False)

        total = np.zeros(len(self.frequencies), dtype=np.float64)
        total_sq = np.zeros_like(total)
        count = 0
        for index in use_indices:
            maps, _ = self.load_sample(int(index), standardizer=None)
            total += maps.sum(axis=0, dtype=np.float64)
            total_sq += np.square(maps, dtype=np.float64).sum(axis=0)
            count += maps.shape[0]
        mean = total / count
        variance = np.maximum(total_sq / count - mean**2, 1.0e-20)
        return Standardizer(mean.astype(np.float32), np.sqrt(variance).astype(np.float32))

    def iter_samples(
        self,
        split: str,
        standardizer: Standardizer,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        if split not in self.splits:
            raise ValueError(f"Unknown split {split!r}; choose from {tuple(self.splits)}")
        for index in self.splits[split]:
            yield self.load_sample(int(index), standardizer)

    def make_tf_dataset(
        self,
        split: str,
        standardizer: Standardizer,
        batch_size: int,
        training: bool = False,
    ):
        import tensorflow as tf

        output_signature = (
            tf.TensorSpec((self.n_pixels, len(self.frequencies)), tf.float32),
            tf.TensorSpec((1,), tf.float32),
        )
        dataset = tf.data.Dataset.from_generator(
            lambda: self.iter_samples(split, standardizer),
            output_signature=output_signature,
        )
        dataset = dataset.apply(
            tf.data.experimental.assert_cardinality(len(self.splits[split]))
        )
        if self.config.cache_in_memory:
            dataset = dataset.cache()
        if training:
            buffer = min(len(self.splits[split]), self.config.shuffle_buffer)
            dataset = dataset.shuffle(
                buffer,
                seed=self.config.split_seed,
                reshuffle_each_iteration=True,
            )
        return dataset.batch(int(batch_size), drop_remainder=False).prefetch(1)

    def make_signed_tf_dataset(
        self, split: str, standardizer: Standardizer, batch_size: int,
        zero_samples_per_batch: int, training: bool = False,
    ):
        """Sample every positive-r map once per epoch with zero-r maps in each batch."""
        import tensorflow as tf

        if split not in {"train", "validation"}:
            raise ValueError("Signed batches are only defined for train and validation")
        if not 2 <= zero_samples_per_batch < batch_size:
            raise ValueError("Invalid number of zero-r samples per batch")
        indices = self.splits[split]
        zero_indices = [int(i) for i in indices if float(self.entries[int(i)]["r"]) == 0.0]
        positive_indices = [int(i) for i in indices if float(self.entries[int(i)]["r"]) > 0.0]
        if len(zero_indices) < zero_samples_per_batch or not positive_indices:
            raise ValueError(f"{split} needs enough zero-r and positive-r maps for signed batches")
        signature = (
            tf.TensorSpec((self.n_pixels, len(self.frequencies)), tf.float32),
            tf.TensorSpec((1,), tf.float32),
        )

        def pool(pool_indices: list[int], repeat: bool):
            ds = tf.data.Dataset.from_generator(
                lambda: (self.load_sample(i, standardizer) for i in pool_indices),
                output_signature=signature,
            ).apply(tf.data.experimental.assert_cardinality(len(pool_indices)))
            if self.config.cache_in_memory:
                ds = ds.cache()
            if training:
                ds = ds.shuffle(min(len(pool_indices), self.config.shuffle_buffer),
                                seed=self.config.split_seed, reshuffle_each_iteration=True)
            return ds.repeat() if repeat else ds

        positives = pool(positive_indices, repeat=False).batch(batch_size - zero_samples_per_batch)
        zeros = pool(zero_indices, repeat=True).batch(zero_samples_per_batch)
        mixed = tf.data.Dataset.zip((positives, zeros)).map(
            lambda positive, zero: (
                tf.concat((positive[0], zero[0]), axis=0),
                tf.concat((positive[1], zero[1]), axis=0),
            ),
            num_parallel_calls=1,
        )
        return mixed.prefetch(1)

    def summary(self) -> Dict[str, object]:
        bytes_per_sample = self.n_pixels * len(self.frequencies) * np.dtype(np.float32).itemsize
        return {
            "dataset_key": self.dataset_key,
            "entries": len(self.entries),
            "frequencies_ghz": tuple(float(v) for v in self.frequencies),
            "variant": self.config.variant,
            "nside": self.nside,
            "pixels": self.n_pixels,
            "input_shape": (self.n_pixels, len(self.frequencies)),
            "float32_mib_per_sample": bytes_per_sample / 2**20,
            "split_sizes": {name: len(values) for name, values in self.splits.items()},
        }
