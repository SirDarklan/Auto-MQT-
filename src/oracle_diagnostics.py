from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from statistics import mean
from typing import Any

from data_utils import load_jsonl


def row_score(item: dict[str, Any], score_key: str) -> float:
    if score_key in item:
        return float(item[score_key])
    if "score" in item:
        return float(item["score"])
    if "dataset_score" in item:
        return float(item["dataset_score"])
    if "relaxed_match" in item:
        return float(item["relaxed_match"])
    if "exact_match" in item:
        return float(item["exact_match"])
    return 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick diagnostics for oracle label JSONL.")
    parser.add_argument("--data", required=True, help="Oracle JSONL path")
    args = parser.parse_args()

    rows = load_jsonl(args.data)
    if not rows:
        print("rows: 0")
        return

    budget_counts = Counter(int(row["oracle_budget"]) for row in rows if "oracle_budget" in row)
    by_dataset = defaultdict(Counter)
    all_zero_count = 0
    all_zero_budget_counts = Counter()
    avg_max_score = []

    for row in rows:
        dataset = str(row.get("dataset", "unknown"))
        if "oracle_budget" in row:
            by_dataset[dataset][int(row["oracle_budget"])] += 1

        score_key = str(row.get("oracle_score_key", "dataset_score"))
        budget_results = row.get("budget_results")
        if not isinstance(budget_results, list) or not budget_results:
            continue
        scores = [row_score(item, score_key) for item in budget_results if isinstance(item, dict)]
        if not scores:
            continue
        max_score = max(scores)
        avg_max_score.append(max_score)
        if max_score <= 0.0:
            all_zero_count += 1
            if "oracle_budget" in row:
                all_zero_budget_counts[int(row["oracle_budget"])] += 1

    print(f"rows: {len(rows)}")
    print(f"oracle_budget_dist: {dict(sorted(budget_counts.items()))}")
    print(f"all_zero_rows: {all_zero_count}")
    if all_zero_count:
        print(f"all_zero_oracle_budget_dist: {dict(sorted(all_zero_budget_counts.items()))}")
    if avg_max_score:
        print(f"avg_max_score: {mean(avg_max_score):.4f}")

    print("oracle_budget_dist_by_dataset:")
    for dataset, counter in sorted(by_dataset.items()):
        print(f"- {dataset}: {dict(sorted(counter.items()))}")


if __name__ == "__main__":
    main()
