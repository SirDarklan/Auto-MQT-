from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean

from data_utils import (
    answers_for_example,
    filter_rows_by_dataset,
    format_prompt,
    load_jsonl,
    parse_dataset_names,
    score_prediction,
    write_jsonl,
)
from mqt_llava_adapter import TimedMqtLlava, build_mqt_llava_backend
from task_token_policy import TaskTokenPolicy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="JSONL with image, prompt, answer, optional task")
    parser.add_argument("--policy", default="configs/task_token_policy.yaml")
    parser.add_argument("--fixed-budget", type=int, default=None)
    parser.add_argument("--out", default="results.jsonl")
    parser.add_argument("--prompt-style", choices=["none", "short"], default="short")
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

    policy = TaskTokenPolicy.from_yaml(args.policy)
    model = TimedMqtLlava(build_mqt_llava_backend())
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

    outputs = []
    for ex in examples:
        budget = args.fixed_budget or policy.budget_for(
            prompt=ex["prompt"],
            task=ex.get("task"),
        )
        result = model.generate(
            image_path=ex["image"],
            prompt=format_prompt(ex["prompt"], task=ex.get("task"), prompt_style=args.prompt_style),
            visual_tokens=budget,
        )
        scores = score_prediction(result.text, answers_for_example(ex), dataset=ex.get("dataset"))
        outputs.append(
            {
                **ex,
                "model_prompt": format_prompt(ex["prompt"], task=ex.get("task"), prompt_style=args.prompt_style),
                "prediction": result.text,
                "visual_tokens": result.visual_tokens,
                "latency_s": result.latency_s,
                **scores,
            }
        )

    if not outputs:
        raise ValueError("No examples to evaluate after dataset filtering.")

    write_jsonl(outputs, Path(args.out))

    print(f"examples: {len(outputs)}")
    print(f"exact_accuracy: {mean(row['exact_match'] for row in outputs):.4f}")
    print(f"relaxed_accuracy: {mean(row['relaxed_match'] for row in outputs):.4f}")
    print(f"dataset_score: {mean(row['dataset_score'] for row in outputs):.4f}")
    print(f"avg_visual_tokens: {mean(row['visual_tokens'] for row in outputs):.2f}")
    print(f"avg_latency_s: {mean(row['latency_s'] for row in outputs):.4f}")


if __name__ == "__main__":
    main()
