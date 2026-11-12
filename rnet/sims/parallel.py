from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import time
import uuid
from pathlib import Path

import h5py
import numpy as np

from .unified import UnifiedSATSimulator


GRID_N_PER_R = 100
GRID_N_ZERO_COVARIANCE = 500
POOL_ORDER = (
    "uniform_train", "uniform_validation", "low_train", "low_validation",
    "zero_train", "zero_validation", "grid",
)


def job_count(sim: UnifiedSATSimulator, pool: str) -> int:
    if pool == "grid":
        return len(sim.GRID_R) * GRID_N_PER_R + GRID_N_ZERO_COVARIANCE
    if pool not in sim.POOLS:
        raise ValueError(f"Unknown pool {pool!r}")
    return sim.POOLS[pool][0]


def task_count(sim: UnifiedSATSimulator, pool: str, block_size: int) -> int:
    if block_size < 1:
        raise ValueError("block_size must be >= 1")
    return math.ceil(job_count(sim, pool) / block_size)


def worker_tasks(sim: UnifiedSATSimulator, rank: int, world_size: int,
                 block_size: int):
    """Assign every part to exactly one rank in a single multi-node job."""
    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("rank must be in [0, world_size)")
    global_id = 0
    for pool in POOL_ORDER:
        for task_id in range(task_count(sim, pool, block_size)):
            if global_id % world_size == rank:
                yield pool, task_id
            global_id += 1


def run_worker(sim: UnifiedSATSimulator, rank: int, world_size: int,
               block_size: int = 64) -> None:
    for pool, task_id in worker_tasks(sim, rank, world_size, block_size):
        path = run_task(sim, pool, task_id, block_size, progress=True)
        print(f"rank={rank}/{world_size} completed {path}", flush=True)


def _job(sim: UnifiedSATSimulator, pool: str, position: int) -> tuple[int, float, str]:
    if not 0 <= position < job_count(sim, pool):
        raise IndexError(position)
    if pool != "grid":
        capacity, base_id = sim.POOLS[pool]
        assert position < capacity
        idx = base_id + position
        return idx, sim._pool_r(pool, idx), pool.rsplit("_", 1)[1]
    zero_total = GRID_N_ZERO_COVARIANCE + GRID_N_PER_R
    if position < zero_total:
        grid_slot = position
        role = "covariance" if position < GRID_N_ZERO_COVARIANCE else "test"
        return 1_000_000 + grid_slot, 0.0, role
    remainder = position - zero_total
    r_index = 1 + remainder // GRID_N_PER_R
    within_r = remainder % GRID_N_PER_R
    grid_slot = r_index * 100_000 + within_r
    return 1_000_000 + grid_slot, float(sim.GRID_R[r_index]), "test"


def _part_paths(sim: UnifiedSATSimulator, pool: str, task_id: int) -> tuple[Path, Path]:
    directory = sim.root / pool / "parts"
    directory.mkdir(parents=True, exist_ok=True)
    final = directory / f"part_{task_id:05d}.h5"
    return final, directory / f"part_{task_id:05d}.partial.h5"


def prepare_shared_inputs(sim: UnifiedSATSimulator) -> None:
    """Serialize creation of the common CAMB cache before independent workers."""
    lock_path = sim.basedir / "cmb_spectra.pkl.lock"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        sim.cmb.get_unlensed_spectra(dl=False)


def _validate_part(handle: h5py.File, sim: UnifiedSATSimulator, pool: str,
                   task_id: int, block_size: int, require_done: bool) -> None:
    start = task_id * block_size
    stop = min(start + block_size, job_count(sim, pool))
    expected = {"settings_hash": sim.settings_hash, "pool": pool,
                "task_id": task_id, "block_size": block_size,
                "start": start, "stop": stop}
    for key, value in expected.items():
        if handle.attrs.get(key) != value:
            raise ValueError(f"Part {handle.filename} has incompatible {key}")
    count = stop - start
    shape = (count, len(sim.FREQUENCIES), len(sim.pixel_idx))
    for name in ("nofg", "fg_d11s6"):
        if name not in handle or handle[name].shape != shape:
            raise ValueError(f"Part {handle.filename} is missing {name} or has wrong shape")
    for name in ("done", "sim_idx", "r"):
        if name not in handle or handle[name].shape != (count,):
            raise ValueError(f"Part {handle.filename} is missing {name}")
    for row in range(count):
        done = bool(handle["done"][row])
        if require_done and not done:
            raise ValueError(f"Incomplete part {handle.filename}, row {row}")
        if done:
            idx, r, _ = _job(sim, pool, start + row)
            if int(handle["sim_idx"][row]) != idx or not np.isclose(
                float(handle["r"][row]), r, rtol=0, atol=1e-14
            ):
                raise ValueError(f"Part {handle.filename} has wrong realization at row {row}")


def _initialize_part(path: Path, sim: UnifiedSATSimulator, pool: str,
                     task_id: int, block_size: int) -> None:
    start = task_id * block_size
    stop = min(start + block_size, job_count(sim, pool))
    count = stop - start
    temporary = path.with_name(
        f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.init"
    )
    try:
        with h5py.File(temporary, "w") as handle:
            handle.attrs.update({"settings_hash": sim.settings_hash, "pool": pool,
                                 "task_id": task_id, "block_size": block_size,
                                 "start": start, "stop": stop})
            shape = (count, len(sim.FREQUENCIES), len(sim.pixel_idx))
            for name in ("nofg", "fg_d11s6"):
                handle.create_dataset(name, shape=shape, dtype="f4",
                                      chunks=(1, 1, len(sim.pixel_idx)))
            handle.create_dataset("done", shape=(count,), dtype="bool")
            handle.create_dataset("sim_idx", shape=(count,), dtype="i8")
            handle.create_dataset("r", shape=(count,), dtype="f8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run_task(sim: UnifiedSATSimulator, pool: str, task_id: int,
             block_size: int = 64,
             timing_records: list[dict[str, float]] | None = None,
             progress: bool = False) -> Path:
    """Compute one exclusively owned part, resuming complete rows if possible."""
    n_tasks = task_count(sim, pool, block_size)
    if not 0 <= task_id < n_tasks:
        raise ValueError(f"task_id must be in [0, {n_tasks - 1}]")
    serial_shards = sim.root / pool / "shards"
    if serial_shards.is_dir() and next(serial_shards.glob("*.h5"), None) is not None:
        raise ValueError(f"Serial shards already exist in {pool}; use a separate output root")
    final, partial = _part_paths(sim, pool, task_id)
    lock_path = final.with_suffix(".lock")
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if final.is_file():
            with h5py.File(final, "r") as handle:
                _validate_part(handle, sim, pool, task_id, block_size, True)
            return final
        if not partial.is_file():
            _initialize_part(partial, sim, pool, task_id, block_size)
        with h5py.File(partial, "a") as handle:
            _validate_part(handle, sim, pool, task_id, block_size, False)
            start = task_id * block_size
            for row in range(handle["done"].shape[0]):
                if handle["done"][row]:
                    continue
                idx, r, _ = _job(sim, pool, start + row)
                row_started = time.perf_counter() if timing_records is not None or progress else 0.0
                timing: dict[str, float] = {}
                if timing_records is None:
                    nofg, fg = sim._maps(idx, r)
                else:
                    nofg, fg = sim._maps(idx, r, timing=timing)
                write_started = time.perf_counter() if timing_records is not None else 0.0
                handle["nofg"][row] = nofg
                handle["fg_d11s6"][row] = fg
                handle["sim_idx"][row] = idx
                handle["r"][row] = r
                handle.flush()
                handle["done"][row] = True
                handle.flush()
                if timing_records is not None:
                    timing["hdf5_write"] = time.perf_counter() - write_started
                    timing["row_total"] = time.perf_counter() - row_started
                    timing_records.append({"sim_idx": idx, "r": r, **timing})
                if progress:
                    print(f"{pool} part={task_id} row={row + 1}/{handle['done'].shape[0]} "
                          f"sim_idx={idx} r={r:.8g} elapsed={time.perf_counter() - row_started:.1f}s",
                          flush=True)
            _validate_part(handle, sim, pool, task_id, block_size, True)
        os.replace(partial, final)
    return final


def finalize_pool(sim: UnifiedSATSimulator, pool: str,
                  block_size: int = 64) -> Path:
    """Validate all parts and publish a manifest without copying map arrays."""
    entries = []
    for task_id in range(task_count(sim, pool, block_size)):
        final, _ = _part_paths(sim, pool, task_id)
        if not final.is_file():
            raise FileNotFoundError(f"Missing completed part: {final}")
        with h5py.File(final, "r") as handle:
            _validate_part(handle, sim, pool, task_id, block_size, True)
            for row in range(handle["done"].shape[0]):
                idx, r, role = _job(sim, pool, task_id * block_size + row)
                entry = {"sim_idx": idx, "r": r, "h5_file": str(final),
                         "h5_row": row, "role": role}
                if pool != "grid":
                    entry["source_pool"] = pool
                entries.append(entry)
    if len(entries) != job_count(sim, pool):
        raise ValueError(f"Wrong number of completed maps in {pool}")
    if pool == "grid":
        design = {"r_values": list(sim.GRID_R), "n_per_r": GRID_N_PER_R,
                  "n_zero_covariance": GRID_N_ZERO_COVARIANCE,
                  "parallel_block_size": block_size}
    else:
        capacity, base_id = sim.POOLS[pool]
        distribution, role = pool.rsplit("_", 1)
        design = {"distribution": distribution, "role": role,
                  "capacity": capacity, "base_id": base_id,
                  "low_r_max": sim.LOW_R_MAX if distribution == "low" else None,
                  "parallel_block_size": block_size}
    manifest_path = sim.root / pool / "manifest.json"
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("entries") != entries:
            raise ValueError(f"Existing manifest differs from completed parts: {manifest_path}")
    return sim._write_manifest(pool, entries, design=design)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "task", "worker", "finalize", "count"))
    parser.add_argument("--basedir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--pool", choices=("grid", *UnifiedSATSimulator.POOLS))
    parser.add_argument("--task-id", type=int)
    parser.add_argument("--rank", type=int)
    parser.add_argument("--world-size", type=int)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--lensing-threads", type=int, default=1)
    args = parser.parse_args()
    if args.action not in ("prepare", "worker") and args.pool is None:
        parser.error("--pool is required for count/task/finalize")
    if args.action == "task" and args.task_id is None:
        parser.error("--task-id is required for task")
    if args.action == "worker":
        args.rank = int(os.environ["SLURM_PROCID"]) if args.rank is None else args.rank
        args.world_size = int(os.environ["SLURM_NTASKS"]) if args.world_size is None else args.world_size
    root = args.output_root or args.basedir / "unified_sat_parallel_v1"
    sim = UnifiedSATSimulator(args.basedir, output_root=root,
                              lensing_threads=args.lensing_threads)
    if args.action == "prepare":
        prepare_shared_inputs(sim)
        print(f"Prepared common inputs at {sim.basedir}")
    elif args.action == "count":
        print(task_count(sim, args.pool, args.block_size))
    elif args.action == "task":
        prepare_shared_inputs(sim)
        print(run_task(sim, args.pool, args.task_id, args.block_size, progress=True))
    elif args.action == "worker":
        run_worker(sim, args.rank, args.world_size, args.block_size)
    else:
        print(finalize_pool(sim, args.pool, args.block_size))


if __name__ == "__main__":
    main()
