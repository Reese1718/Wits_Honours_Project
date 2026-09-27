#!/usr/bin/env python3

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean, median


def load_user_histories(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyse original Video Games user interaction counts."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to original Video Games user histories JSON.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    data = load_user_histories(input_path)

    # Support either:
    # {"user1": [...], "user2": [...], ...}
    # or {"users": {"user1": [...], ...}}
    if "users" in data and isinstance(data["users"], dict):
        users = data["users"]
    else:
        users = data

    counts = []

    for user_id, history in users.items():
        if not isinstance(history, list):
            raise ValueError(
                f"Expected a list of interactions for user {user_id}, "
                f"got {type(history).__name__}"
            )
        counts.append(len(history))

    counts.sort()

    n_users = len(counts)

    if n_users == 0:
        raise ValueError("No users found.")

    frequency = Counter(counts)

    print("=" * 60)
    print("VIDEO GAMES ORIGINAL USER INTERACTION DISTRIBUTION")
    print("=" * 60)

    print(f"Users:    {n_users:,}")
    print(f"Minimum:  {min(counts):,}")
    print(f"Maximum:  {max(counts):,}")
    print(f"Mean:     {mean(counts):.2f}")
    print(f"Median:   {median(counts):.2f}")

    # Simple percentile calculation.
    def percentile(values, p):
        index = (len(values) - 1) * p
        lower = int(index)
        upper = min(lower + 1, len(values) - 1)
        fraction = index - lower
        return values[lower] + (values[upper] - values[lower]) * fraction

    print(f"Q1 (25%): {percentile(counts, 0.25):.2f}")
    print(f"Q3 (75%): {percentile(counts, 0.75):.2f}")

    print("\nInteraction count distribution:")
    print("-" * 60)
    print(f"{'Interactions':>15} {'Users':>15} {'Percentage':>15}")
    print("-" * 60)

    for interaction_count in sorted(frequency):
        user_count = frequency[interaction_count]
        percentage = 100 * user_count / n_users

        print(
            f"{interaction_count:>15,} "
            f"{user_count:>15,} "
            f"{percentage:>14.2f}%"
        )

    print("\nCumulative thresholds:")
    print("-" * 60)
    print(f"{'Threshold':>15} {'Users':>15} {'Percentage':>15}")
    print("-" * 60)

    for threshold in [1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 100]:
        user_count = sum(c <= threshold for c in counts)
        percentage = 100 * user_count / n_users

        print(
            f"≤ {threshold:>12,} "
            f"{user_count:>15,} "
            f"{percentage:>14.2f}%"
        )


if __name__ == "__main__":
    main()
