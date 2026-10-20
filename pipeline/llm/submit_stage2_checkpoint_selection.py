"""Submit one Step-05 array plus its afterok CPU aggregation job.

No submission occurs with --dry-run. Each array task evaluates every eligible
checkpoint on its disjoint validation shard. A fresh run ID isolates old results.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess
import uuid

from common import load_yaml, resolve_project_path


def submission_commands(config, *, num_shards=8, batch_size=4, expected_users=109258,
                        partition="batch", nodelist=None, max_parallel=4,
                        wall_time="30:00:00", run_id=None):
    if num_shards < 1 or batch_size < 1 or expected_users < num_shards:
        raise ValueError("Require positive batch/shard counts and at least one user per shard")
    concurrency = min(num_shards, 4 if max_parallel is None else max_parallel)
    if concurrency < 1:
        raise ValueError("max_parallel must be positive")
    run_id = run_id or uuid.uuid4().hex
    launcher = resolve_project_path("slurm/llm/select_video_games_recent5_three_stage2_sharded.slurm")
    base = [str(launcher), str(config), str(num_shards), run_id, str(batch_size), str(expected_users)]
    resources = ["--partition", partition]
    if nodelist:
        resources += ["--nodelist", nodelist]
    array = ["sbatch", "--parsable", *resources, "--time", wall_time,
             "--array", f"0-{num_shards - 1}%{concurrency}", *base, "worker"]
    aggregate = ["sbatch", "--parsable", "--partition", partition,
                 "--time", "02:00:00", "--dependency", "afterok:{array_job_id}",
                 *base, "aggregate"]
    return run_id, array, aggregate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--generation-batch-size", type=int, default=4)
    parser.add_argument("--expected-users", type=int, default=109258)
    parser.add_argument("--partition", default="batch")
    parser.add_argument("--nodelist", help="Optional user-verified node list; no node is pinned by default")
    parser.add_argument("--max-parallel", type=int, default=4,
                        help="Maximum simultaneous array workers; Slurm may run fewer")
    parser.add_argument("--time", default="30:00:00")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config_path = resolve_project_path(args.config)
    config = load_yaml(config_path)
    selection = config.get("checkpoint_selection", {})
    if (config.get("tuning", {}).get("stage") != 2
            or selection.get("max_validation_users", "missing") is not None
            or selection.get("num_beams") != 10
            or config.get("generation", {}).get("num_beams") != 10):
        parser.error("This submitter requires Stage 2, full validation and 10 beams in both checkpoint_selection and generation")
    run_id, array, aggregate = submission_commands(
        config_path, num_shards=args.num_shards, batch_size=args.generation_batch_size,
        expected_users=args.expected_users, partition=args.partition, nodelist=args.nodelist,
        max_parallel=args.max_parallel, wall_time=args.time)
    print(f"Shard run ID: {run_id}", flush=True)
    print(shlex.join(array), flush=True)
    if args.dry_run:
        print(shlex.join(aggregate), flush=True)
        return
    result = subprocess.run(array, check=True, capture_output=True, text=True)
    array_id = result.stdout.strip().split(";")[0]
    if not array_id.isdigit():
        raise RuntimeError(f"Unrecognized sbatch job ID: {result.stdout!r}")
    aggregate = [argument.replace("{array_job_id}", array_id) for argument in aggregate]
    print(f"Array job ID: {array_id}", flush=True)
    print(shlex.join(aggregate), flush=True)
    try:
        result = subprocess.run(aggregate, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            "Array submitted, but aggregation submission failed. Retry this command: "
            + shlex.join(aggregate) + "\n" + (error.stderr or "")
        ) from error
    print(f"Aggregation job ID: {result.stdout.strip()}", flush=True)


if __name__ == "__main__":
    main()
