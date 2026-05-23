from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from data_utils import filter_rows_by_dataset, load_jsonl, parse_dataset_names


def parse_inputs(specs: list[str]) -> list[tuple[str, Path]]:
    runs: list[tuple[str, Path]] = []
    for spec in specs:
        if "=" in spec:
            name, path_str = spec.split("=", 1)
            run_name = name.strip()
            path = Path(path_str.strip())
        else:
            path = Path(spec.strip())
            run_name = path.stem
        if not path.exists():
            raise FileNotFoundError(f"Result file not found: {path}")
        runs.append((run_name, path))
    return runs


def _safe_mean(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna().tolist()
    if not values:
        return float("nan")
    return float(mean(values))


def _round_or_na(value: float, digits: int = 4) -> str:
    if value is None or math.isnan(value):
        return "n/a"
    return f"{value:.{digits}f}"


def _pct_or_na(value: float, digits: int = 2) -> str:
    if value is None or math.isnan(value):
        return "n/a"
    return f"{value * 100.0:.{digits}f}"


def summarize_run(run: str, rows: list[dict[str, Any]]) -> tuple[dict[str, Any], pd.DataFrame]:
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"{run}: no rows in result file")

    if "dataset" not in df.columns:
        df["dataset"] = "unknown"
    df["dataset"] = df["dataset"].fillna("unknown").astype(str)

    numeric_cols = ["exact_match", "relaxed_match", "dataset_score", "visual_tokens", "latency_s"]
    for col in numeric_cols:
        if col not in df.columns:
            df[col] = float("nan")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    overall = {
        "run": run,
        "examples": int(len(df)),
        "exact_match": _safe_mean(df["exact_match"]),
        "relaxed_match": _safe_mean(df["relaxed_match"]),
        "dataset_score": _safe_mean(df["dataset_score"]),
        "avg_visual_tokens": _safe_mean(df["visual_tokens"]),
        "avg_latency_s": _safe_mean(df["latency_s"]),
    }

    by_dataset = (
        df.groupby("dataset", dropna=False)
        .agg(
            examples=("dataset", "size"),
            exact_match=("exact_match", "mean"),
            relaxed_match=("relaxed_match", "mean"),
            dataset_score=("dataset_score", "mean"),
            avg_visual_tokens=("visual_tokens", "mean"),
            avg_latency_s=("latency_s", "mean"),
        )
        .reset_index()
    )
    by_dataset.insert(0, "run", run)
    return overall, by_dataset


def build_markdown_report(
    out_path: Path,
    overall_df: pd.DataFrame,
    by_dataset_df: pd.DataFrame,
    datasets_order: list[str],
) -> None:
    lines: list[str] = []
    lines.append("# Auto-MQT Per-Dataset Benchmark Breakdown")
    lines.append("")
    lines.append("## Paper-Style Dataset Score Table (%)")
    lines.append("")

    pivot = (
        by_dataset_df.pivot(index="run", columns="dataset", values="dataset_score")
        .reindex(columns=datasets_order)
        .copy()
    )
    pivot["overall"] = overall_df.set_index("run")["dataset_score"]
    pivot["avg_tokens"] = overall_df.set_index("run")["avg_visual_tokens"]
    pivot = pivot.reindex(overall_df["run"].tolist())

    headers = ["run"] + datasets_order + ["overall", "avg_tokens"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for run in pivot.index.tolist():
        row_cells = [run]
        for ds in datasets_order:
            row_cells.append(_pct_or_na(float(pivot.loc[run, ds]), 2))
        row_cells.append(_pct_or_na(float(pivot.loc[run, "overall"]), 2))
        row_cells.append(_round_or_na(float(pivot.loc[run, "avg_tokens"]), 2))
        lines.append("| " + " | ".join(row_cells) + " |")
    lines.append("")

    lines.append("## Overall Metrics")
    lines.append("")
    overall_headers = ["run", "examples", "exact", "relaxed", "dataset_score", "avg_tokens", "avg_latency_s"]
    lines.append("| " + " | ".join(overall_headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(overall_headers)) + " |")
    for _, row in overall_df.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["run"]),
                    str(int(row["examples"])),
                    _pct_or_na(float(row["exact_match"]), 2),
                    _pct_or_na(float(row["relaxed_match"]), 2),
                    _pct_or_na(float(row["dataset_score"]), 2),
                    _round_or_na(float(row["avg_visual_tokens"]), 2),
                    _round_or_na(float(row["avg_latency_s"]), 3),
                ]
            )
            + " |"
        )
    lines.append("")

    lines.append("## Per-Dataset Detail")
    lines.append("")
    detail_headers = ["run", "dataset", "examples", "exact", "relaxed", "dataset_score", "avg_tokens", "avg_latency_s"]
    lines.append("| " + " | ".join(detail_headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(detail_headers)) + " |")
    for _, row in by_dataset_df.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["run"]),
                    str(row["dataset"]),
                    str(int(row["examples"])),
                    _pct_or_na(float(row["exact_match"]), 2),
                    _pct_or_na(float(row["relaxed_match"]), 2),
                    _pct_or_na(float(row["dataset_score"]), 2),
                    _round_or_na(float(row["avg_visual_tokens"]), 2),
                    _round_or_na(float(row["avg_latency_s"]), 3),
                ]
            )
            + " |"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def plot_overall_score(overall_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ordered = overall_df.sort_values("dataset_score", ascending=False).reset_index(drop=True)
    values = ordered["dataset_score"].to_numpy(dtype=float) * 100.0
    ax.bar(ordered["run"], values, color="#2A6F97")
    ax.set_title("Overall Dataset Score by Run")
    ax.set_ylabel("Dataset Score (%)")
    ax.set_ylim(0, max(100.0, float(values.max()) + 5.0))
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_grouped_dataset_scores(by_dataset_df: pd.DataFrame, out_path: Path, datasets_order: list[str]) -> None:
    pivot = (
        by_dataset_df.pivot(index="run", columns="dataset", values="dataset_score")
        .reindex(columns=datasets_order)
        .copy()
    )
    runs = pivot.index.tolist()
    x = list(range(len(runs)))
    width = 0.18 if len(datasets_order) >= 4 else 0.24
    offsets = [((i - (len(datasets_order) - 1) / 2.0) * width) for i in range(len(datasets_order))]
    colors = ["#2A6F97", "#468FAF", "#61A5C2", "#89C2D9", "#A9D6E5"]

    fig, ax = plt.subplots(figsize=(11, 5.2))
    for idx, dataset in enumerate(datasets_order):
        y = (pivot[dataset].to_numpy(dtype=float) * 100.0) if dataset in pivot.columns else []
        ax.bar([pos + offsets[idx] for pos in x], y, width=width, label=dataset, color=colors[idx % len(colors)])

    ax.set_title("Per-Dataset Score by Run")
    ax.set_ylabel("Dataset Score (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(runs, rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(loc="best", ncol=2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_fixed_budget_curves(by_dataset_df: pd.DataFrame, out_path: Path, datasets_order: list[str]) -> None:
    fixed = by_dataset_df[by_dataset_df["run"].str.match(r"^fixed_\d+$", na=False)].copy()
    if fixed.empty:
        return

    fixed["budget"] = fixed["run"].str.extract(r"^fixed_(\d+)$").astype(float)
    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    colors = ["#184E77", "#1E6091", "#1A759F", "#34A0A4", "#52B69A"]

    for idx, dataset in enumerate(datasets_order):
        subset = fixed[fixed["dataset"] == dataset].sort_values("budget")
        if subset.empty:
            continue
        ax.plot(
            subset["budget"],
            subset["dataset_score"] * 100.0,
            marker="o",
            linewidth=2.0,
            color=colors[idx % len(colors)],
            label=dataset,
        )

    ax.set_title("Dataset Score vs Visual Tokens (Fixed-Budget Runs)")
    ax.set_xlabel("Visual Tokens")
    ax.set_ylabel("Dataset Score (%)")
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_tradeoff(overall_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 5.4))
    x = overall_df["avg_visual_tokens"].to_numpy(dtype=float)
    y = overall_df["dataset_score"].to_numpy(dtype=float) * 100.0
    ax.scatter(x, y, color="#0B525B", s=52)
    for _, row in overall_df.iterrows():
        ax.annotate(str(row["run"]), (float(row["avg_visual_tokens"]), float(row["dataset_score"]) * 100.0), fontsize=8)
    ax.set_title("Token/Accuracy Tradeoff")
    ax.set_xlabel("Average Visual Tokens")
    ax.set_ylabel("Overall Dataset Score (%)")
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def run_order_key(name: str) -> tuple[int, int, str]:
    fixed_match = re.match(r"^fixed_(\d+)$", name)
    if fixed_match:
        return (0, int(fixed_match.group(1)), name)
    router_rank = {
        "router_prompt": 10,
        "router_image": 11,
        "router_multimodal": 12,
        "router_cross_attention": 13,
    }
    return (1, router_rank.get(name, 99), name)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build paper-style per-dataset benchmark tables and plots from existing result JSONLs "
            "(no re-inference required)."
        )
    )
    parser.add_argument("--inputs", nargs="+", required=True, help="Run files as name=path or plain path")
    parser.add_argument("--out-dir", default="results", help="Output directory for tables, markdown, and figures")
    parser.add_argument("--out-prefix", default="dataset_breakdown", help="Filename prefix for generated outputs")
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
    include_datasets = parse_dataset_names(args.include_datasets)
    exclude_datasets = parse_dataset_names(args.exclude_datasets)
    if include_datasets is not None:
        print(f"include_datasets: {sorted(include_datasets)}")
    if exclude_datasets is not None:
        print(f"exclude_datasets: {sorted(exclude_datasets)}")

    overall_rows: list[dict[str, Any]] = []
    by_dataset_parts: list[pd.DataFrame] = []

    for run_name, path in runs:
        rows = filter_rows_by_dataset(
            load_jsonl(path),
            include_datasets=include_datasets,
            exclude_datasets=exclude_datasets,
        )
        print(f"run={run_name} rows_after_dataset_filter={len(rows)}")
        overall, by_dataset = summarize_run(run_name, rows)
        overall_rows.append(overall)
        by_dataset_parts.append(by_dataset)

    overall_df = pd.DataFrame(overall_rows)
    overall_df = overall_df.sort_values(by="run", key=lambda s: s.map(run_order_key)).reset_index(drop=True)
    by_dataset_df = pd.concat(by_dataset_parts, ignore_index=True)
    by_dataset_df["run"] = by_dataset_df["run"].astype(str)
    by_dataset_df["dataset"] = by_dataset_df["dataset"].astype(str)
    by_dataset_df["run_order"] = by_dataset_df["run"].map(run_order_key)
    by_dataset_df = by_dataset_df.sort_values(["run_order", "dataset"]).drop(columns=["run_order"]).reset_index(drop=True)

    preferred_order = ["vqav2", "gqa", "textvqa", "scienceqa_img"]
    datasets_present = [ds for ds in preferred_order if ds in by_dataset_df["dataset"].unique().tolist()]
    extra = sorted(ds for ds in by_dataset_df["dataset"].unique().tolist() if ds not in datasets_present)
    datasets_order = datasets_present + extra

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    overall_csv = out_dir / f"{args.out_prefix}_overall.csv"
    by_dataset_csv = out_dir / f"{args.out_prefix}_by_dataset.csv"
    paper_csv = out_dir / f"{args.out_prefix}_paper_style.csv"
    report_md = out_dir / f"{args.out_prefix}.md"

    overall_df.to_csv(overall_csv, index=False)
    by_dataset_df.to_csv(by_dataset_csv, index=False)

    paper_df = by_dataset_df.pivot(index="run", columns="dataset", values="dataset_score").reindex(columns=datasets_order)
    paper_df["overall"] = overall_df.set_index("run")["dataset_score"]
    paper_df["avg_tokens"] = overall_df.set_index("run")["avg_visual_tokens"]
    paper_df = paper_df.reindex(overall_df["run"].tolist())
    paper_df.to_csv(paper_csv)

    build_markdown_report(
        out_path=report_md,
        overall_df=overall_df,
        by_dataset_df=by_dataset_df,
        datasets_order=datasets_order,
    )

    fig_overall = out_dir / f"{args.out_prefix}_overall_score.png"
    fig_grouped = out_dir / f"{args.out_prefix}_per_dataset_by_run.png"
    fig_fixed = out_dir / f"{args.out_prefix}_fixed_budget_curves.png"
    fig_tradeoff = out_dir / f"{args.out_prefix}_tradeoff.png"

    plot_overall_score(overall_df, fig_overall)
    plot_grouped_dataset_scores(by_dataset_df, fig_grouped, datasets_order)
    plot_fixed_budget_curves(by_dataset_df, fig_fixed, datasets_order)
    plot_tradeoff(overall_df, fig_tradeoff)

    print(f"wrote_csv: {overall_csv}")
    print(f"wrote_csv: {by_dataset_csv}")
    print(f"wrote_csv: {paper_csv}")
    print(f"wrote_markdown: {report_md}")
    print(f"wrote_figure: {fig_overall}")
    print(f"wrote_figure: {fig_grouped}")
    print(f"wrote_figure: {fig_fixed}")
    print(f"wrote_figure: {fig_tradeoff}")


if __name__ == "__main__":
    main()
