from __future__ import annotations

import argparse
from statistics import mean

import torch

from data_utils import (
    answers_for_example,
    feature_from_example,
    filter_rows_by_dataset,
    format_prompt,
    load_jsonl,
    parse_dataset_names,
    score_prediction,
    write_jsonl,
)
from mqt_llava_adapter import TimedMqtLlava, build_mqt_llava_backend
from router_model import LateFusionRouter, RouterConfig, predict_budget


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="JSONL with image, prompt, answer")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default="results_router.jsonl")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--fallback-threshold", type=float, default=None)
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

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = RouterConfig(**checkpoint["router_config"])
    checkpoint_threshold = float(
        checkpoint.get(
            "tuned_fallback_threshold",
            checkpoint.get("raw_config", {}).get(
                "confidence_fallback_threshold",
                config.confidence_fallback_threshold,
            ),
        )
    )
    checkpoint_cost_bias = float(
        checkpoint.get(
            "tuned_cost_bias",
            checkpoint.get("raw_config", {}).get("inference_cost_bias", 0.0),
        )
    )
    cost_power = float(checkpoint.get("raw_config", {}).get("cost_power", 1.0))
    threshold = checkpoint_threshold if args.fallback_threshold is None else args.fallback_threshold

    router = LateFusionRouter(config, mode=checkpoint["mode"])
    router.load_state_dict(checkpoint["model_state"])
    router.eval()

    model = TimedMqtLlava(build_mqt_llava_backend())
    rows = []
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

    for example in examples:
        prompt_feature = feature_from_example(
            example,
            field="prompt_embedding",
            fallback_text=example["prompt"],
            dim=config.prompt_dim,
        )
        image_feature = feature_from_example(
            example,
            field="image_embedding",
            fallback_text=example["image"],
            dim=config.image_dim,
        )
        with torch.no_grad():
            logits = router(
                torch.tensor(prompt_feature, dtype=torch.float32).unsqueeze(0),
                torch.tensor(image_feature, dtype=torch.float32).unsqueeze(0),
            ).squeeze(0)
            budget, confidence = predict_budget(
                logits,
                config.budgets,
                threshold,
                cost_bias=checkpoint_cost_bias,
                cost_power=cost_power,
            )

        result = model.generate(
            image_path=example["image"],
            prompt=format_prompt(example["prompt"], task=example.get("task"), prompt_style=args.prompt_style),
            visual_tokens=budget,
            max_new_tokens=args.max_new_tokens,
        )
        scores = score_prediction(result.text, answers_for_example(example), dataset=example.get("dataset"))
        rows.append(
            {
                **example,
                "model_prompt": format_prompt(example["prompt"], task=example.get("task"), prompt_style=args.prompt_style),
                "prediction": result.text,
                "visual_tokens": result.visual_tokens,
                "router_confidence": confidence,
                "latency_s": result.latency_s,
                **scores,
            }
        )

    if not rows:
        raise ValueError("No examples to evaluate after dataset filtering.")

    write_jsonl(rows, args.out)
    print(f"examples: {len(rows)}")
    print(f"exact_accuracy: {mean(row['exact_match'] for row in rows):.4f}")
    print(f"relaxed_accuracy: {mean(row['relaxed_match'] for row in rows):.4f}")
    print(f"dataset_score: {mean(row['dataset_score'] for row in rows):.4f}")
    print(f"avg_visual_tokens: {mean(row['visual_tokens'] for row in rows):.2f}")
    print(f"avg_latency_s: {mean(row['latency_s'] for row in rows):.4f}")


if __name__ == "__main__":
    main()
