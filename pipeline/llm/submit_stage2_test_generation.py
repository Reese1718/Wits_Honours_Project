"""Submit the Step-06 generation array and its afterok CPU-only merge.

Does not run Step 07/08. --dry-run prints commands without submitting jobs.
Every submission gets a fresh UUID. mscluster74 is always excluded.
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import uuid

from common import load_yaml, resolve_project_path


def submission_commands(config, *, num_shards=8, max_parallel=4, batch_size=4,
                        expected_users=109258, partition="bigbatch", exclude="",
                        overwrite=False, run_id=None):
    if min(num_shards, max_parallel, batch_size) < 1 or expected_users < num_shards:
        raise ValueError("Require positive shard/concurrency/batch counts and nonempty shards")
    run_id = run_id or uuid.uuid4().hex
    exclusions = "mscluster74" + ("," + exclude if exclude else "")
    launcher = resolve_project_path("slurm/llm/generate_video_games_recent5_three_stage2_sharded.slurm")
    base = [str(launcher), str(config), str(num_shards), run_id, str(batch_size),
            str(expected_users)]
    resources = ["--partition", partition, "--exclude", exclusions]
    worker = ["sbatch", "--parsable", *resources, "--exclusive",
              "--array", f"0-{num_shards-1}%{min(num_shards, max_parallel)}",
              *base, "worker", "yes" if overwrite else "no"]
    merge = ["sbatch", "--parsable", *resources, "--cpus-per-task", "2", "--mem", "16G",
             "--time", "02:00:00", "--dependency", "afterok:{array_job_id}",
             *base, "merge", "yes" if overwrite else "no"]
    return run_id, worker, merge


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--max-parallel", type=int, default=4)
    parser.add_argument("--generation-batch-size", type=int, default=None,
                        help="Optional override; otherwise preserve generation.batch_size")
    parser.add_argument("--expected-users", type=int, default=109258)
    parser.add_argument("--partition", default="bigbatch")
    parser.add_argument("--exclude", default="", help="Additional excluded nodes; mscluster74 always excluded")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly permit replacement of final outputs at merge")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config_path = resolve_project_path(args.config)
    config = load_yaml(config_path)
    if config.get("tuning", {}).get("stage") != 2 or config.get("generation", {}).get("num_beams") != 10:
        parser.error("Requires a promoted Stage-2 config with generation.num_beams=10")
    batch_size = (args.generation_batch_size if args.generation_batch_size is not None
                  else int(config["generation"].get("batch_size", 4)))
    run_id, worker, merge = submission_commands(
        config_path, num_shards=args.num_shards, max_parallel=args.max_parallel,
        batch_size=batch_size, expected_users=args.expected_users, partition=args.partition,
        exclude=args.exclude, overwrite=args.overwrite)
    print(f"Shard run ID: {run_id}", flush=True)
    print(shlex.join(worker), flush=True)
    if args.dry_run:
        print(shlex.join(merge), flush=True)
        return
    result = subprocess.run(worker, check=True, capture_output=True, text=True)
    array_id = result.stdout.strip().split(";")[0]
    if not array_id.isdigit():
        raise RuntimeError(f"Unrecognized sbatch job ID: {result.stdout!r}")
    merge = [argument.replace("{array_job_id}", array_id) for argument in merge]
    print(f"Array job ID: {array_id}", flush=True)
    print(shlex.join(merge), flush=True)
    try:
        result = subprocess.run(merge, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as error:
        raise RuntimeError("Workers submitted but merge submission failed. Retry: "
                           + shlex.join(merge) + "\n" + (error.stderr or "")) from error
    print(f"Merge job ID: {result.stdout.strip()}", flush=True)


if __name__ == "__main__":
    main()
