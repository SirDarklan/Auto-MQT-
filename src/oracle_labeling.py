from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean

from data_utils import (
    append_jsonl,
    answers_for_example,
    filter_rows_by_dataset,
    format_prompt,
    load_jsonl,
    parse_dataset_names,
    score_prediction,
)
from mqt_llava_adapter import TimedMqtLlava, build_mqt_llava_backend


DEFAULT_BUDGETS = [2, 4, 8, 16, 36, 64, 144, 256]


def choose_oracle_budget(
    scored_budgets: list[dict],
    full_budget: int,
    tolerance: float = 0.0,
    zero_score_budget: int | None = None,
) -> tuple[int, float]:
    reference_score = max(row["score"] for row in scored_budgets)
    if reference_score <= 0.0:
        if zero_score_budget is not None:
            candidates = [row for row in scored_budgets if int(row["visual_tokens"]) == int(zero_score_budget)]
            if candidates:
                chosen = candidates[0]
                return int(chosen["visual_tokens"]), float(chosen["score"])
        # If every candidate budget scores zero, default to the cheapest budget
        # unless caller requested a specific fallback via `zero_score_budget`.
        cheapest = min(scored_budgets, key=lambda row: row["visual_tokens"])
        return int(cheapest["visual_tokens"]), float(cheapest["score"])

    threshold = max(0.0, reference_score - tolerance)

    for row in sorted(scored_budgets, key=lambda item: item["visual_tokens"]):
        if row["score"] >= threshold:
            return int(row["visual_tokens"]), float(row["score"])
    best = max(scored_budgets, key=lambda item: (item["score"], -item["visual_tokens"]))
    return int(best["visual_tokens"]), float(best["score"])


def example_key(example: dict, index: int) -> str:
    return str(example.get("example_id", index))


def completed_keys(path: str | Path) -> set[str]:
    out_path = Path(path)
    if not out_path.exists():
        return set()
    return {str(row["example_id"]) for row in load_jsonl(out_path) if "example_id" in row}


def label_one_example(
    model: TimedMqtLlava,
    example: dict,
    index: int,
    budgets: list[int],
    tolerance: float,
    max_new_tokens: int,
    score_key: str,
    prompt_style: str,
    zero_score_budget: int | None,
) -> dict:
    full_budget = max(budgets)
    model_prompt = format_prompt(
        example["prompt"],
        task=example.get("task"),
        prompt_style=prompt_style,
    )
    scored_budgets = []

    for budget in budgets:
        result = model.generate(
            image_path=example["image"],
            prompt=model_prompt,
            visual_tokens=budget,
            max_new_tokens=max_new_tokens,
        )
        answers = answers_for_example(example)
        scores = (
            score_prediction(result.text, answers, dataset=example.get("dataset"))
            if answers
            else {
                "exact_match": 0.0,
                "relaxed_match": 0.0,
                "dataset_score": 0.0,
                "dataset_metric_name": "accuracy",
            }
        )
        scored_budgets.append(
            {
                "visual_tokens": budget,
                "prediction": result.text,
                "score": scores[score_key],
                **scores,
                "latency_s": result.latency_s,
            }
        )

    oracle_budget, oracle_score = choose_oracle_budget(
        scored_budgets=scored_budgets,
        full_budget=full_budget,
        tolerance=tolerance,
        zero_score_budget=zero_score_budget,
    )
    return {
        **example,
        "model_prompt": model_prompt,
        "oracle_score_key": score_key,
        "oracle_budget": oracle_budget,
        "oracle_score": oracle_score,
        "budget_results": scored_budgets,
        "example_id": example_key(example, index),
    }


def label_examples_to_jsonl(
    examples: list[dict],
    out: str | Path,
    budgets: list[int],
    tolerance: float,
    max_new_tokens: int,
    score_key: str,
    prompt_style: str,
    zero_score_budget: int | None,
    limit: int | None,
    resume: bool,
) -> list[dict]:
    model = TimedMqtLlava(build_mqt_llava_backend())
    done = completed_keys(out) if resume else set()
    written_rows = []
    processed = 0

    for index, example in enumerate(examples):
        key = example_key(example, index)
        if key in done:
            print(f"skip existing example_id={key}", flush=True)
            continue
        if limit is not None and processed >= limit:
            break

        print(f"label example {processed + 1}: example_id={key}", flush=True)
        try:
            row = label_one_example(
                model=model,
                example=example,
                index=index,
                budgets=budgets,
                tolerance=tolerance,
            max_new_tokens=max_new_tokens,
            score_key=score_key,
            prompt_style=prompt_style,
            zero_score_budget=zero_score_budget,
        )
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                print(
                    "OOM while labeling. Try: restart runtime, ensure sanity-check does not load the model, "
                    "and set MQT_LLAVA_LOAD_8BIT=1 in notebook env cell.",
                    flush=True,
                )
            raise
        append_jsonl(row, out)
        written_rows.append(row)
        processed += 1
        print(
            f"wrote example_id={key} oracle_budget={row['oracle_budget']} "
            f"oracle_score={row['oracle_score']}",
            flush=True,
        )

    return written_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Input JSONL with image, prompt, answer")
    parser.add_argument("--out", default="data/oracle_labels.jsonl")
    parser.add_argument("--budgets", nargs="+", type=int, default=DEFAULT_BUDGETS)
    parser.add_argument("--tolerance", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--score-key", choices=["exact_match", "relaxed_match", "dataset_score"], default="dataset_score")
    parser.add_argument("--prompt-style", choices=["none", "short"], default="short")
    parser.add_argument(
        "--zero-score-budget",
        type=int,
        default=None,
        help=(
            "When all candidate budgets score 0 for an example, optionally assign this budget "
            "(if present in --budgets). If unset, uses the cheapest budget."
        ),
    )
    parser.add_argument("--limit", type=int, default=None, help="Maximum new examples to label this run")
    parser.add_argument("--no-resume", action="store_true", help="Do not skip example_ids already in --out")
    parser.add_argument(
        "--include-datasets",
        nargs="+",
        default=None,
        help="Optional dataset allow-list (space/comma separated), e.g. vqav2 gqa scienceqa_img",
    )
    parser.add_argument(
        "--exclude-datasets",
        nargs="+",
        default=None,
        help="Optional dataset deny-list (space/comma separated), e.g. textvqa",
    )
    args = parser.parse_args()

    include_datasets = parse_dataset_names(args.include_datasets)
    exclude_datasets = parse_dataset_names(args.exclude_datasets)
    examples = filter_rows_by_dataset(
        load_jsonl(args.data),
        include_datasets=include_datasets,
        exclude_datasets=exclude_datasets,
    )
    print(f"examples_after_dataset_filter: {len(examples)}")
    if include_datasets is not None:
        print(f"include_datasets: {sorted(include_datasets)}")
    if exclude_datasets is not None:
        print(f"exclude_datasets: {sorted(exclude_datasets)}")
    rows = label_examples_to_jsonl(
        examples=examples,
        out=args.out,
        budgets=args.budgets,
        tolerance=args.tolerance,
        max_new_tokens=args.max_new_tokens,
        score_key=args.score_key,
        prompt_style=args.prompt_style,
        zero_score_budget=args.zero_score_budget,
        limit=args.limit,
        resume=not args.no_resume,
    )

    print(f"new_examples: {len(rows)}")
    if rows:
        print(f"avg_new_oracle_budget: {mean(row['oracle_budget'] for row in rows):.2f}")
    print(f"output: {args.out}")


if __name__ == "__main__":
    main()
