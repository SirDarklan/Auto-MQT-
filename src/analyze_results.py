from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import math
from pathlib import Path
from statistics import mean
from typing import Any

from data_utils import filter_rows_by_dataset, load_jsonl, parse_dataset_names


def parse_inputs(specs: list[str]) -> list[tuple[str, Path]]:
    runs: list[tuple[str, Path]] = []
    for spec in specs:
        if "=" in spec:
            name, path_str = spec.split("=", 1)
            name = name.strip()
            path = Path(path_str.strip())
        else:
            path = Path(spec)
            name = path.stem
        if not path.exists():
            raise FileNotFoundError(f"Result file not found: {path}")
        runs.append((name, path))
    return runs


def safe_mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(mean(values))


def round_or_nan(value: float, digits: int = 4) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.{digits}f}"


def oracle_map(path: str | Path) -> dict[str, int]:
    rows = load_jsonl(path)
    mapping: dict[str, int] = {}
    for row in rows:
        if "example_id" not in row or "oracle_budget" not in row:
            continue
        mapping[str(row["example_id"])] = int(row["oracle_budget"])
    return mapping


def summarize_result(rows: list[dict[str, Any]], oracle_budget_by_id: dict[str, int] | None) -> dict[str, float]:
    exact = [float(row["exact_match"]) for row in rows if "exact_match" in row]
    relaxed = [float(row["relaxed_match"]) for row in rows if "relaxed_match" in row]
    dataset_scores = [float(row["dataset_score"]) for row in rows if "dataset_score" in row]
    tokens = [float(row["visual_tokens"]) for row in rows if "visual_tokens" in row]
    latencies = [float(row["latency_s"]) for row in rows if "latency_s" in row]

    summary: dict[str, float] = {
        "examples": float(len(rows)),
        "exact_match": safe_mean(exact),
        "relaxed_match": safe_mean(relaxed),
        "dataset_score": safe_mean(dataset_scores),
        "avg_visual_tokens": safe_mean(tokens),
        "avg_latency_s": safe_mean(latencies),
    }

    if oracle_budget_by_id is None:
        return summary

    regrets: list[float] = []
    under_budget_flags: list[float] = []
    matched = 0
    for row in rows:
        if "example_id" not in row or "visual_tokens" not in row:
            continue
        key = str(row["example_id"])
        oracle_budget = oracle_budget_by_id.get(key)
        if oracle_budget is None:
            continue
        pred_budget = int(row["visual_tokens"])
        regrets.append(float(max(0, pred_budget - oracle_budget)))
        under_budget_flags.append(float(pred_budget < oracle_budget))
        matched += 1

    summary["oracle_overlap"] = float(matched)
    summary["budget_regret"] = safe_mean(regrets)
    summary["under_budget_rate"] = safe_mean(under_budget_flags)
    return summary


def print_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def fmt_line(cells: list[str]) -> str:
        return " | ".join(cell.ljust(widths[index]) for index, cell in enumerate(cells))

    print(fmt_line(headers))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(fmt_line(row))


def summarize_manifest(path: str | Path) -> dict[str, Any]:
    rows = load_jsonl(path)
    dataset_counts = Counter(str(row.get("dataset", "unknown")) for row in rows)
    split_counts = Counter(str(row.get("split", "unknown")) for row in rows)
    task_counts = Counter(str(row.get("task", "unknown")) for row in rows)
    prompt_lengths = [len(str(row.get("prompt", "")).split()) for row in rows]

    answer_counts = []
    for row in rows:
        answers = row.get("answers")
        if isinstance(answers, list):
            answer_counts.append(len(answers))
        elif "answer" in row:
            answer_counts.append(1)

    return {
        "path": str(path),
        "rows": len(rows),
        "dataset_counts": dict(dataset_counts),
        "split_counts": dict(split_counts),
        "task_counts": dict(task_counts),
        "avg_prompt_words": safe_mean(prompt_lengths),
        "avg_answers_per_example": safe_mean(answer_counts),
    }


def summarize_oracle(path: str | Path) -> dict[str, Any]:
    rows = load_jsonl(path)
    oracle_budgets = [int(row["oracle_budget"]) for row in rows if "oracle_budget" in row]
    score_keys = Counter(str(row.get("oracle_score_key", "unknown")) for row in rows)

    candidate_budgets: list[int] = []
    for row in rows:
        budget_results = row.get("budget_results")
        if isinstance(budget_results, list) and budget_results:
            extracted = [
                int(item["visual_tokens"])
                for item in budget_results
                if isinstance(item, dict) and "visual_tokens" in item
            ]
            if extracted:
                candidate_budgets = sorted(set(extracted))
                break

    return {
        "path": str(path),
        "rows": len(rows),
        "avg_oracle_budget": safe_mean(oracle_budgets),
        "oracle_budget_distribution": dict(Counter(oracle_budgets)),
        "candidate_budgets": candidate_budgets,
        "score_keys": dict(score_keys),
    }


def parse_named_paths(specs: list[str] | None) -> list[tuple[str, Path]]:
    if not specs:
        return []
    parsed: list[tuple[str, Path]] = []
    for spec in specs:
        if "=" in spec:
            name, path_str = spec.split("=", 1)
            label = name.strip()
            path = Path(path_str.strip())
        else:
            path = Path(spec)
            label = path.stem
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        parsed.append((label, path))
    return parsed


def load_checkpoint_summaries(checkpoint_specs: list[str] | None) -> list[dict[str, Any]]:
    specs = parse_named_paths(checkpoint_specs)
    if not specs:
        return []

    import torch

    summaries: list[dict[str, Any]] = []
    for label, path in specs:
        checkpoint = torch.load(path, map_location="cpu")
        router_config = checkpoint.get("router_config", {})
        raw_config = checkpoint.get("raw_config", {})
        summaries.append(
            {
                "name": label,
                "path": str(path),
                "mode": checkpoint.get("mode", "unknown"),
                "best_val_accuracy": checkpoint.get("best_val_accuracy"),
                "best_val_metric": checkpoint.get("best_val_metric"),
                "selection_metric": checkpoint.get("selection_metric"),
                "budgets": router_config.get("budgets", []),
                "prompt_dim": router_config.get("prompt_dim"),
                "image_dim": router_config.get("image_dim"),
                "hidden_dim": router_config.get("hidden_dim"),
                "dropout": router_config.get("dropout"),
                "epochs": raw_config.get("epochs"),
                "learning_rate": raw_config.get("learning_rate"),
                "batch_size": raw_config.get("batch_size"),
                "lambda_cost": raw_config.get("lambda_cost"),
                "cost_power": raw_config.get("cost_power"),
                "tuned_cost_bias": checkpoint.get("tuned_cost_bias"),
                "tuned_fallback_threshold": checkpoint.get("tuned_fallback_threshold"),
            }
        )
    return summaries


def format_counter(counter_like: dict[Any, Any]) -> str:
    if not counter_like:
        return "n/a"
    items = sorted(counter_like.items(), key=lambda pair: str(pair[0]))
    return ", ".join(f"{key}:{value}" for key, value in items)


def proposal_coverage_rows() -> list[tuple[str, str, str]]:
    return [
        ("Oracle budget labeling", "implemented", "src/oracle_labeling.py"),
        ("Tolerance-based labels (delta)", "implemented", "--tolerance in oracle_labeling.py"),
        ("Prompt-only router", "implemented", "--mode prompt in train_router.py"),
        ("Image-only router", "implemented", "--mode image in train_router.py"),
        ("Multimodal router", "implemented", "--mode multimodal in train_router.py"),
        ("Tiny cross-attention router variant", "implemented", "--mode cross_attention in train_router.py"),
        ("Cost-aware objective", "implemented", "cost_aware_loss in train_router.py"),
        ("Confidence fallback policy", "implemented", "predict_budget threshold in router_model.py"),
        ("Calibration loss term", "implemented", "gamma_calibration in cost_aware_loss"),
        ("Soft budget target distribution", "implemented", "soft_target_distribution from budget_results"),
        ("Official dataset-specific metrics", "partial", "dataset_score implemented (VQA-soft + exact), not external leaderboard scripts"),
    ]


def write_markdown_report(
    out_path: str | Path,
    title: str,
    run_records: list[dict[str, Any]],
    headers: list[str],
    table_rows: list[list[str]],
    train_manifest_summary: dict[str, Any] | None,
    eval_manifest_summary: dict[str, Any] | None,
    oracle_summary: dict[str, Any] | None,
    backend_label: str | None,
    checkpoint_summaries: list[dict[str, Any]],
) -> None:
    lines: list[str] = []
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"- Generated: {now}")
    if backend_label:
        lines.append(f"- Backend: {backend_label}")
    lines.append("")

    lines.append("## Results")
    lines.append("")
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in table_rows:
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines.append("## Dataset Summary")
    lines.append("")
    if train_manifest_summary:
        lines.append(f"- Train manifest: `{train_manifest_summary['path']}`")
        lines.append(f"- Train rows: {train_manifest_summary['rows']}")
        lines.append(f"- Train datasets: {format_counter(train_manifest_summary['dataset_counts'])}")
        lines.append(f"- Train tasks: {format_counter(train_manifest_summary['task_counts'])}")
        lines.append(
            f"- Train avg prompt words: {round_or_nan(float(train_manifest_summary['avg_prompt_words']), 2)}"
        )
    if eval_manifest_summary:
        lines.append(f"- Eval manifest: `{eval_manifest_summary['path']}`")
        lines.append(f"- Eval rows: {eval_manifest_summary['rows']}")
        lines.append(f"- Eval datasets: {format_counter(eval_manifest_summary['dataset_counts'])}")
        lines.append(f"- Eval tasks: {format_counter(eval_manifest_summary['task_counts'])}")
        lines.append(
            f"- Eval avg prompt words: {round_or_nan(float(eval_manifest_summary['avg_prompt_words']), 2)}"
        )
    lines.append("")

    lines.append("## Oracle Summary")
    lines.append("")
    if oracle_summary:
        lines.append(f"- Oracle file: `{oracle_summary['path']}`")
        lines.append(f"- Oracle rows: {oracle_summary['rows']}")
        lines.append(
            f"- Candidate budgets: {oracle_summary['candidate_budgets'] if oracle_summary['candidate_budgets'] else 'n/a'}"
        )
        lines.append(
            f"- Oracle budget distribution: {format_counter(oracle_summary['oracle_budget_distribution'])}"
        )
        lines.append(
            f"- Avg oracle budget: {round_or_nan(float(oracle_summary['avg_oracle_budget']), 2)}"
        )
        lines.append(f"- Oracle score keys: {format_counter(oracle_summary['score_keys'])}")
    else:
        lines.append("- Oracle summary not provided.")
    lines.append("")

    lines.append("## Method/Settings Summary")
    lines.append("")
    lines.append("- Metrics in this report: exact match, relaxed match, dataset_score, avg visual tokens, avg latency, regret, under-budget rate.")
    lines.append("- Regret is computed as `max(0, chosen_budget - oracle_budget)` averaged over overlapping example IDs.")
    lines.append("- Under-budget rate is the fraction of examples where `chosen_budget < oracle_budget`.")
    if checkpoint_summaries:
        lines.append("")
        lines.append("### Router Checkpoints")
        lines.append("")
        for checkpoint in checkpoint_summaries:
            lines.append(
                "- "
                f"{checkpoint['name']}: mode={checkpoint['mode']}, "
                f"best_val_metric={round_or_nan(float(checkpoint['best_val_metric']), 4) if checkpoint['best_val_metric'] is not None else 'n/a'}"
                f" ({checkpoint.get('selection_metric', 'accuracy')}), "
                f"budgets={checkpoint['budgets']}, hidden_dim={checkpoint['hidden_dim']}, "
                f"lr={checkpoint['learning_rate']}, batch_size={checkpoint['batch_size']}, "
                f"lambda_cost={checkpoint['lambda_cost']}, cost_power={checkpoint['cost_power']}, "
                f"bias={checkpoint.get('tuned_cost_bias', 'n/a')}, "
                f"threshold={checkpoint.get('tuned_fallback_threshold', 'n/a')}"
            )
    lines.append("")

    lines.append("## Proposal Coverage")
    lines.append("")
    lines.append("| Item | Status | Evidence |")
    lines.append("| --- | --- | --- |")
    for item, status, evidence in proposal_coverage_rows():
        lines.append(f"| {item} | {status} | {evidence} |")
    lines.append("")

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize Auto-MQT result JSONL files into comparable metrics. "
            "Use `name=path` to label each run."
        )
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="One or more result files, optionally labeled as name=path",
    )
    parser.add_argument(
        "--oracle",
        default=None,
        help="Optional oracle-label JSONL (must include example_id, oracle_budget) for regret diagnostics",
    )
    parser.add_argument("--out-csv", default=None, help="Optional CSV output path for the summary table")
    parser.add_argument("--train-manifest", default=None, help="Optional train manifest JSONL for dataset summary")
    parser.add_argument("--eval-manifest", default=None, help="Optional eval manifest JSONL for dataset summary")
    parser.add_argument(
        "--checkpoints",
        nargs="*",
        default=None,
        help="Optional checkpoints, format: name=path",
    )
    parser.add_argument("--backend-label", default=None, help="Optional backend label shown in markdown report")
    parser.add_argument(
        "--out-markdown",
        default=None,
        help="Optional markdown report output path with dataset/method/settings summary",
    )
    parser.add_argument("--title", default="Auto-MQT Final Summary", help="Title for markdown report")
    parser.add_argument(
        "--include-datasets",
        nargs="+",
        default=None,
        help="Optional dataset allow-list (space/comma separated)",
    )
    parser.add_argument(
        "--exclude-datasets",
        nargs="+",
        default=None,
        help="Optional dataset deny-list (space/comma separated), e.g. textvqa",
    )
    args = parser.parse_args()

    runs = parse_inputs(args.inputs)
    oracle_budget_by_id = oracle_map(args.oracle) if args.oracle else None
    include_datasets = parse_dataset_names(args.include_datasets)
    exclude_datasets = parse_dataset_names(args.exclude_datasets)
    if include_datasets is not None:
        print(f"include_datasets: {sorted(include_datasets)}")
    if exclude_datasets is not None:
        print(f"exclude_datasets: {sorted(exclude_datasets)}")

    records: list[dict[str, Any]] = []
    table_rows: list[list[str]] = []

    for run_name, path in runs:
        rows = filter_rows_by_dataset(
            load_jsonl(path),
            include_datasets=include_datasets,
            exclude_datasets=exclude_datasets,
        )
        print(f"run={run_name} rows_after_dataset_filter={len(rows)}")
        metrics = summarize_result(rows, oracle_budget_by_id)
        record = {"run": run_name, "path": str(path), **metrics}
        records.append(record)

        table_rows.append(
            [
                run_name,
                str(int(metrics["examples"])),
                round_or_nan(metrics["exact_match"], 4),
                round_or_nan(metrics["relaxed_match"], 4),
                round_or_nan(metrics["dataset_score"], 4),
                round_or_nan(metrics["avg_visual_tokens"], 2),
                round_or_nan(metrics["avg_latency_s"], 3),
                round_or_nan(metrics.get("budget_regret", float("nan")), 2),
                round_or_nan(metrics.get("under_budget_rate", float("nan")), 4),
            ]
        )

    headers = [
        "run",
        "examples",
        "exact",
        "relaxed",
        "dataset_score",
        "avg_tokens",
        "avg_latency_s",
        "regret",
        "under_rate",
    ]
    print_table(headers, table_rows)

    if args.out_csv:
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)
        print(f"wrote_csv: {out_path}")

    train_manifest_summary = summarize_manifest(args.train_manifest) if args.train_manifest else None
    eval_manifest_summary = summarize_manifest(args.eval_manifest) if args.eval_manifest else None
    oracle_summary_dict = summarize_oracle(args.oracle) if args.oracle else None
    checkpoint_summaries = load_checkpoint_summaries(args.checkpoints)

    if args.out_markdown:
        write_markdown_report(
            out_path=args.out_markdown,
            title=args.title,
            run_records=records,
            headers=headers,
            table_rows=table_rows,
            train_manifest_summary=train_manifest_summary,
            eval_manifest_summary=eval_manifest_summary,
            oracle_summary=oracle_summary_dict,
            backend_label=args.backend_label,
            checkpoint_summaries=checkpoint_summaries,
        )
        print(f"wrote_markdown: {args.out_markdown}")


if __name__ == "__main__":
    main()
