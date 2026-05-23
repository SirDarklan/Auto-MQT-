from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from PIL import Image
import torch
from tqdm import tqdm
from transformers import AutoProcessor, CLIPModel

from data_utils import append_jsonl, filter_rows_by_dataset, load_jsonl, parse_dataset_names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract frozen prompt/image embeddings for router training. "
            "Writes JSONL rows with `prompt_embedding` and `image_embedding`."
        )
    )
    parser.add_argument("--data", required=True, help="Input JSONL manifest")
    parser.add_argument("--out", required=True, help="Output JSONL with embeddings")
    parser.add_argument(
        "--clip-model",
        default="openai/clip-vit-large-patch14-336",
        help="Hugging Face CLIP model name",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--device",
        default="auto",
        help="Device for feature extraction (auto, cuda, cpu, cuda:0, ...)",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="auto",
        help="Model dtype (float16 recommended on GPU; float32 on CPU)",
    )
    parser.add_argument("--max-text-tokens", type=int, default=77)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true", help="Ignore existing output and rewrite from scratch")
    parser.add_argument(
        "--prompt-field",
        default="model_prompt",
        help="Preferred prompt field to embed (falls back to `prompt`)",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="L2-normalize embeddings before writing",
    )
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
    return parser.parse_args()


def choose_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def choose_dtype(dtype_arg: str, device: torch.device) -> torch.dtype:
    if dtype_arg == "float32":
        return torch.float32
    if dtype_arg == "float16":
        return torch.float16
    if dtype_arg == "bfloat16":
        return torch.bfloat16
    # auto
    if device.type == "cuda":
        return torch.float16
    return torch.float32


def completed_ids(path: str | Path) -> set[str]:
    out_path = Path(path)
    if not out_path.exists():
        return set()
    return {str(row["example_id"]) for row in load_jsonl(out_path) if "example_id" in row}


def iter_batches(rows: list[dict[str, Any]], batch_size: int):
    for index in range(0, len(rows), batch_size):
        yield rows[index : index + batch_size]


def read_prompt(row: dict[str, Any], preferred_field: str) -> str:
    if preferred_field in row and isinstance(row[preferred_field], str):
        return row[preferred_field]
    if "prompt" not in row:
        raise KeyError("Each row must include `prompt` or the configured --prompt-field.")
    return str(row["prompt"])


def read_image(path_value: str) -> Image.Image:
    path = Path(path_value)
    if not path.exists():
        path = Path(path_value.replace("\\", "/"))
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    with Image.open(path) as image:
        return image.convert("RGB")


def move_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if hasattr(value, "to") else value
    return moved


def main() -> None:
    args = parse_args()

    include_datasets = parse_dataset_names(args.include_datasets)
    exclude_datasets = parse_dataset_names(args.exclude_datasets)
    source_rows = filter_rows_by_dataset(
        load_jsonl(args.data),
        include_datasets=include_datasets,
        exclude_datasets=exclude_datasets,
    )
    if args.limit is not None:
        source_rows = source_rows[: args.limit]

    done = set() if args.overwrite else completed_ids(args.out)
    rows = [row for row in source_rows if str(row.get("example_id", "")) not in done]
    if not rows:
        print("No new rows to process.")
        print(f"output: {args.out}")
        return

    device = choose_device(args.device)
    dtype = choose_dtype(args.dtype, device)

    print(f"rows_total: {len(source_rows)}")
    print(f"rows_pending: {len(rows)}")
    if include_datasets is not None:
        print(f"include_datasets: {sorted(include_datasets)}")
    if exclude_datasets is not None:
        print(f"exclude_datasets: {sorted(exclude_datasets)}")
    print(f"device: {device}")
    print(f"dtype: {dtype}")
    print(f"clip_model: {args.clip_model}")

    processor = AutoProcessor.from_pretrained(args.clip_model)
    model = CLIPModel.from_pretrained(args.clip_model, torch_dtype=dtype).to(device)
    model.eval()

    if args.overwrite:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("", encoding="utf-8")

    written = 0
    total_batches = (len(rows) + args.batch_size - 1) // args.batch_size
    for batch in tqdm(iter_batches(rows, args.batch_size), total=total_batches, desc="Extracting", unit="batch"):
        prompts = [read_prompt(row, args.prompt_field) for row in batch]
        images = [read_image(str(row["image"])) for row in batch]

        text_inputs = processor(
            text=prompts,
            padding=True,
            truncation=True,
            max_length=args.max_text_tokens,
            return_tensors="pt",
        )
        image_inputs = processor(images=images, return_tensors="pt")

        text_inputs = move_to_device(text_inputs, device)
        image_inputs = move_to_device(image_inputs, device)

        with torch.inference_mode():
            text_features = model.get_text_features(**text_inputs)
            image_features = model.get_image_features(**image_inputs)
            if args.normalize:
                text_features = torch.nn.functional.normalize(text_features, dim=-1)
                image_features = torch.nn.functional.normalize(image_features, dim=-1)

        text_vectors = text_features.detach().float().cpu().tolist()
        image_vectors = image_features.detach().float().cpu().tolist()

        for row, prompt_embedding, image_embedding in zip(batch, text_vectors, image_vectors):
            out_row = {
                **row,
                "prompt_embedding": prompt_embedding,
                "image_embedding": image_embedding,
            }
            append_jsonl(out_row, args.out)
            written += 1

    print(f"wrote_rows: {written}")
    print(f"output: {args.out}")


if __name__ == "__main__":
    main()
