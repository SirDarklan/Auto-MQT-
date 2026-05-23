from __future__ import annotations

import argparse
from collections import Counter
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from data_utils import format_prompt, write_jsonl


@dataclass(frozen=True)
class PreparedExample:
    dataset: str
    split: str
    example_id: str
    image: str
    prompt: str
    answer: str
    answers: list[str]
    task: str | None

    def as_row(self) -> dict[str, Any]:
        row = {
            "dataset": self.dataset,
            "split": self.split,
            "example_id": self.example_id,
            "image": self.image,
            "prompt": self.prompt,
            "answer": self.answer,
            "answers": self.answers,
        }
        if self.task:
            row["task"] = self.task
        return row


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    return " ".join(str(value).strip().split())


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return slug.strip("_") or "example"


def to_manifest_path(path: str | Path) -> str:
    # Keep manifest image paths portable across Windows and Linux/Colab.
    return Path(path).as_posix()


def first_existing(row: dict[str, Any], candidates: list[str]) -> Any:
    for key in candidates:
        if key in row and row[key] is not None:
            return row[key]
    return None


def column_candidates(configured: str | list[str] | None, defaults: list[str]) -> list[str]:
    if configured is None:
        return defaults
    if isinstance(configured, list):
        return [str(item) for item in configured] + defaults
    return [str(configured)] + defaults


def normalize_answers(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [clean_text(value)] if clean_text(value) else []
    if isinstance(value, dict):
        for key in ("text", "answer", "answers", "labels"):
            if key in value:
                return normalize_answers(value[key])
        flattened: list[str] = []
        for item in value.values():
            flattened.extend(normalize_answers(item))
        return [answer for answer in flattened if answer]
    if isinstance(value, (list, tuple)):
        answers: list[str] = []
        for item in value:
            answers.extend(normalize_answers(item))
        return [answer for answer in answers if answer]
    return [clean_text(value)] if clean_text(value) else []


def normalize_question(value: Any) -> str:
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                role = clean_text(item.get("role", "")).lower()
                content = clean_text(item.get("content") or item.get("value") or item.get("text"))
                if role in {"user", "human"} and content:
                    return content.replace("<image>", "").strip()
            else:
                text = clean_text(item)
                if text:
                    return text
        return ""
    if isinstance(value, dict):
        for key in ("question", "query", "prompt", "content", "value", "text"):
            if key in value:
                return normalize_question(value[key])
        return clean_text(value)
    text = clean_text(value)
    return text.replace("<image>", "").strip()


def normalize_choices(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, dict):
        if "text" in value:
            return normalize_choices(value["text"])
        return [clean_text(item) for item in value.values() if clean_text(item)]
    if isinstance(value, (list, tuple)):
        return [clean_text(item) for item in value if clean_text(item)]
    return [clean_text(value)] if clean_text(value) else []


def select_answer_from_choices(answer_value: Any, choices: list[str]) -> str:
    if answer_value is None:
        return ""
    if isinstance(answer_value, int) and 0 <= answer_value < len(choices):
        return choices[answer_value]
    text = clean_text(answer_value)
    if text.isdigit():
        index = int(text)
        if 0 <= index < len(choices):
            return choices[index]
    return text


def make_scienceqa_prompt(question: str, choices: list[str]) -> str:
    if not choices:
        return question
    choice_lines = [f"{chr(ord('A') + idx)}. {choice}" for idx, choice in enumerate(choices)]
    return f"{question}\nChoices:\n" + "\n".join(choice_lines)


def parse_scienceqa_problem(problem: str) -> tuple[str, list[str]]:
    text = normalize_question(problem)
    choices = [clean_text(match.group(1)) for match in re.finditer(r"(?m)^[A-Z]\.\s*(.+)$", text)]
    return text, [choice for choice in choices if choice]


def choice_letter_to_index(letter: str) -> int | None:
    if not letter:
        return None
    token = clean_text(letter).upper()
    if len(token) == 1 and "A" <= token <= "Z":
        return ord(token) - ord("A")
    return None


def image_extension(image_obj: Any, source: Any) -> str:
    if isinstance(source, (str, Path)):
        suffix = Path(source).suffix.lower()
        if suffix in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            return suffix
    fmt = getattr(image_obj, "format", None)
    if fmt:
        return f".{fmt.lower().replace('jpeg', 'jpg')}"
    return ".jpg"


def save_image(image_value: Any, out_dir: Path, stem: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(image_value, dict):
        path_value = image_value.get("path")
        bytes_value = image_value.get("bytes")
        if path_value and Path(path_value).exists():
            ext = image_extension(None, path_value)
            out_path = out_dir / f"{stem}{ext}"
            shutil.copyfile(path_value, out_path)
            return out_path
        if bytes_value:
            from io import BytesIO

            image_value = Image.open(BytesIO(bytes_value))

    if isinstance(image_value, (str, Path)):
        source_path = Path(image_value)
        if source_path.exists():
            ext = image_extension(None, source_path)
            out_path = out_dir / f"{stem}{ext}"
            shutil.copyfile(source_path, out_path)
            return out_path

    if isinstance(image_value, Image.Image):
        ext = image_extension(image_value, None)
        out_path = out_dir / f"{stem}{ext}"
        image = image_value.convert("RGB") if image_value.mode not in {"RGB", "L"} else image_value
        image.save(out_path)
        return out_path

    raise ValueError(f"Unsupported image value type: {type(image_value).__name__}")


def pick_indices(length: int, limit: int | None, seed: int) -> list[int]:
    indices = list(range(length))
    if limit is None or limit >= length:
        return indices
    rng = random.Random(seed)
    rng.shuffle(indices)
    return sorted(indices[:limit])


def load_hf_split(hf_id: str, split: str, subset: str | None = None):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("Install Hugging Face datasets with `pip install datasets`.") from exc

    try:
        if subset:
            return load_dataset(hf_id, subset, split=split)
        return load_dataset(hf_id, split=split)
    except ValueError as exc:
        message = str(exc)
        if "Unknown split" not in message:
            raise

        dataset_dict = load_dataset(hf_id, subset) if subset else load_dataset(hf_id)
        if not hasattr(dataset_dict, "keys"):
            raise

        available_splits = list(dataset_dict.keys())
        requested_lower = split.lower()
        if requested_lower in {"validation", "val", "dev"}:
            preferred = ["validation", "val", "dev", "train", "test"]
        elif requested_lower == "test":
            preferred = ["test", "validation", "train"]
        else:
            preferred = [split, "train", "validation", "test"]

        fallback_split = None
        for candidate in preferred:
            if candidate in available_splits:
                fallback_split = candidate
                break
        if fallback_split is None and available_splits:
            fallback_split = available_splits[0]
        if fallback_split is None:
            raise

        print(
            f"[warn] {hf_id}: split `{split}` not found; "
            f"using `{fallback_split}` from {available_splits}",
            flush=True,
        )
        return dataset_dict[fallback_split]


def normalize_vqa_row(
    row: dict[str, Any],
    dataset_name: str,
    split_name: str,
    row_index: int,
    dataset_config: dict[str, Any],
    image_dir: Path,
) -> PreparedExample | None:
    columns = dataset_config.get("columns", {})
    question = normalize_question(
        first_existing(
            row,
            column_candidates(columns.get("question"), ["question", "query", "prompt", "conversations"]),
        )
    )
    answers = normalize_answers(
        first_existing(row, column_candidates(columns.get("answers"), ["answers", "answer", "label"]))
    )
    image_value = first_existing(
        row,
        column_candidates(columns.get("image"), ["image", "img", "picture"]),
    )
    if not question or not answers or image_value is None:
        return None

    raw_id = first_existing(row, ["question_id", "id", "image_id"]) or row_index
    example_id = slugify(f"{dataset_name}_{split_name}_{raw_id}")
    image_path = save_image(image_value, image_dir / dataset_name / split_name, example_id)
    return PreparedExample(
        dataset=dataset_name,
        split=split_name,
        example_id=example_id,
        image=to_manifest_path(image_path),
        prompt=question,
        answer=answers[0],
        answers=answers,
        task=dataset_config.get("task"),
    )


def normalize_gqa_multiqa_rows(
    row: dict[str, Any],
    dataset_name: str,
    split_name: str,
    row_index: int,
    dataset_config: dict[str, Any],
    image_dir: Path,
) -> list[PreparedExample]:
    columns = dataset_config.get("columns", {})
    image_value = first_existing(row, column_candidates(columns.get("image"), ["image", "img", "picture"]))
    qa_items = first_existing(row, column_candidates(columns.get("qa"), ["qa", "qas", "questions"]))
    if image_value is None or not isinstance(qa_items, list) or not qa_items:
        return []

    raw_id = first_existing(row, ["image_id", "id"]) or row_index
    image_stem = slugify(f"{dataset_name}_{split_name}_{raw_id}")
    image_path = save_image(image_value, image_dir / dataset_name / split_name, image_stem)
    image_path_str = to_manifest_path(image_path)

    prepared_rows: list[PreparedExample] = []
    for qa_index, qa_item in enumerate(qa_items):
        if not isinstance(qa_item, dict):
            continue
        question = normalize_question(
            first_existing(
                qa_item,
                column_candidates(columns.get("question"), ["question", "query", "prompt"]),
            )
        )
        short_answer = clean_text(first_existing(qa_item, column_candidates(columns.get("answers"), ["answer"])))
        full_answer = clean_text(qa_item.get("fullAnswer"))
        answers = [answer for answer in [short_answer, full_answer] if answer]
        if not question or not answers:
            continue

        example_id = slugify(f"{dataset_name}_{split_name}_{raw_id}_qa_{qa_index}")
        prepared_rows.append(
            PreparedExample(
                dataset=dataset_name,
                split=split_name,
                example_id=example_id,
                image=image_path_str,
                prompt=question,
                answer=answers[0],
                answers=answers,
                task=dataset_config.get("task"),
            )
        )
    return prepared_rows


def normalize_scienceqa_row(
    row: dict[str, Any],
    dataset_name: str,
    split_name: str,
    row_index: int,
    dataset_config: dict[str, Any],
    image_dir: Path,
) -> PreparedExample | None:
    columns = dataset_config.get("columns", {})
    question_raw = first_existing(
        row,
        column_candidates(columns.get("question"), ["question", "query", "prompt", "problem"]),
    )
    question = normalize_question(question_raw)
    choices = normalize_choices(
        first_existing(row, column_candidates(columns.get("choices"), ["choices", "options"]))
    )
    if not choices and isinstance(question_raw, str):
        _, choices = parse_scienceqa_problem(question_raw)

    answer_value = first_existing(row, column_candidates(columns.get("answer"), ["answer", "label", "answer_idx"]))
    answer_text = select_answer_from_choices(answer_value, choices)
    image_value = first_existing(
        row,
        column_candidates(columns.get("image"), ["image", "img", "picture", "images"]),
    )
    if isinstance(image_value, list) and image_value:
        image_value = image_value[0]

    if not question or not answer_text or image_value is None:
        return None

    raw_id = first_existing(row, ["id", "question_id", "pid"]) or row_index
    example_id = slugify(f"{dataset_name}_{split_name}_{raw_id}")
    image_path = save_image(image_value, image_dir / dataset_name / split_name, example_id)

    answers = [answer_text]
    if isinstance(answer_value, str):
        letter_index = choice_letter_to_index(answer_value)
        if letter_index is not None:
            letter = answer_value.upper()
            answers.append(letter)
            if 0 <= letter_index < len(choices):
                answers.append(choices[letter_index])
        elif answer_value:
            answers.append(clean_text(answer_value))
    answers = [answer for answer in answers if answer]
    # Deduplicate while preserving order.
    deduped_answers: list[str] = []
    seen = set()
    for answer in answers:
        key = answer.lower()
        if key not in seen:
            deduped_answers.append(answer)
            seen.add(key)

    return PreparedExample(
        dataset=dataset_name,
        split=split_name,
        example_id=example_id,
        image=to_manifest_path(image_path),
        prompt=make_scienceqa_prompt(question, choices),
        answer=deduped_answers[0],
        answers=deduped_answers,
        task=dataset_config.get("task"),
    )


def normalize_rows(
    row: dict[str, Any],
    dataset_name: str,
    split_name: str,
    row_index: int,
    dataset_config: dict[str, Any],
    image_dir: Path,
) -> list[PreparedExample]:
    adapter = dataset_config.get("adapter", "vqa")
    if adapter == "scienceqa":
        prepared = normalize_scienceqa_row(row, dataset_name, split_name, row_index, dataset_config, image_dir)
        return [prepared] if prepared else []
    if adapter == "gqa_multiqa":
        return normalize_gqa_multiqa_rows(row, dataset_name, split_name, row_index, dataset_config, image_dir)
    if adapter == "vqa":
        prepared = normalize_vqa_row(row, dataset_name, split_name, row_index, dataset_config, image_dir)
        return [prepared] if prepared else []
    raise ValueError(f"Unknown adapter `{adapter}` for dataset `{dataset_name}`")


def prepare_dataset_split(
    dataset_config: dict[str, Any],
    split_role: str,
    image_dir: Path,
    seed: int,
    prompt_style: str,
) -> list[dict[str, Any]]:
    dataset_name = dataset_config["name"]
    split_name = dataset_config[f"{split_role}_split"]
    limit = dataset_config.get(f"{split_role}_limit")
    hf_id = dataset_config["hf_id"]
    subset = dataset_config.get("subset")

    hf_dataset = load_hf_split(hf_id=hf_id, subset=subset, split=split_name)
    available_columns = set(getattr(hf_dataset, "column_names", []) or [])
    adapter = dataset_config.get("adapter", "vqa")
    columns_cfg = dataset_config.get("columns", {})
    if available_columns and adapter == "vqa":
        image_candidates = column_candidates(columns_cfg.get("image"), ["image", "img", "picture"])
        if not any(candidate in available_columns for candidate in image_candidates):
            raise ValueError(
                f"{dataset_name}:{split_name} is missing image column. "
                f"Tried {image_candidates}, available columns: {sorted(available_columns)}. "
                "Use an image-backed HF mirror for this dataset."
            )

    rows: list[dict[str, Any]] = []
    answer_lengths: list[int] = []
    skipped = 0
    row_errors = 0
    source_indices = list(range(len(hf_dataset)))
    if limit is not None:
        rng = random.Random(seed)
        rng.shuffle(source_indices)

    for row_index in source_indices:
        if limit is not None and len(rows) >= int(limit):
            break
        try:
            prepared_rows = normalize_rows(
                row=dict(hf_dataset[row_index]),
                dataset_name=dataset_name,
                split_name=split_name,
                row_index=row_index,
                dataset_config=dataset_config,
                image_dir=image_dir,
            )
        except Exception as exc:  # noqa: BLE001
            row_errors += 1
            if row_errors <= 5:
                print(
                    f"[warn] {dataset_name}:{split_name} row_index={row_index} parse error: {exc}",
                    flush=True,
                )
            continue

        if not prepared_rows:
            skipped += 1
            continue

        for prepared in prepared_rows:
            if limit is not None and len(rows) >= int(limit):
                break
            row = prepared.as_row()
            row["prompt"] = format_prompt(row["prompt"], task=row.get("task"), prompt_style=prompt_style)
            rows.append(row)
            if isinstance(row.get("answers"), list):
                answer_lengths.append(len(row["answers"]))

    if limit and not rows:
        raise ValueError(
            f"{dataset_name}:{split_name} produced 0 rows for limit={limit}. "
            "Check split names and column mapping."
        )

    print(
        f"{dataset_name}:{split_name} -> kept={len(rows)} skipped={skipped} row_errors={row_errors} "
        f"limit={limit or 'all'}"
    )
    if answer_lengths:
        avg_answers = sum(answer_lengths) / len(answer_lengths)
        print(
            f"{dataset_name}:{split_name} -> avg_answers_per_example={avg_answers:.2f}",
            flush=True,
        )
        if dataset_name in {"vqav2", "textvqa"} and avg_answers < 3.0:
            print(
                f"[warn] {dataset_name}:{split_name} has low answer multiplicity "
                f"(avg={avg_answers:.2f}). VQA-soft scoring quality may be weak.",
                flush=True,
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/datasets.yaml")
    parser.add_argument("--datasets", nargs="*", default=None, help="Optional dataset names to prepare")
    parser.add_argument("--train-limit", type=int, default=None, help="Override train_limit for all datasets")
    parser.add_argument("--eval-limit", type=int, default=None, help="Override eval_limit for all datasets")
    parser.add_argument("--prompt-style", choices=["none", "short"], default="none")
    parser.add_argument(
        "--strict-datasets",
        action="store_true",
        help="Fail immediately if any dataset split cannot be prepared",
    )
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    output = config["output"]
    image_dir = Path(output["image_dir"])
    manifest_dir = Path(output["manifest_dir"])
    seed = int(output.get("seed", 7))
    selected = set(args.datasets or [])

    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []

    for dataset_config in config["datasets"]:
        if not dataset_config.get("enabled", True):
            continue
        if selected and dataset_config["name"] not in selected:
            continue
        if args.train_limit is not None:
            dataset_config["train_limit"] = args.train_limit
        if args.eval_limit is not None:
            dataset_config["eval_limit"] = args.eval_limit

        try:
            train_rows.extend(prepare_dataset_split(dataset_config, "train", image_dir, seed, args.prompt_style))
            eval_rows.extend(prepare_dataset_split(dataset_config, "eval", image_dir, seed + 1, args.prompt_style))
        except Exception as exc:  # noqa: BLE001
            if args.strict_datasets:
                raise
            print(
                f"[warn] skipping dataset `{dataset_config['name']}` due to error: {exc}",
                flush=True,
            )
            continue

    write_jsonl(train_rows, manifest_dir / output.get("train_manifest", "train.jsonl"))
    write_jsonl(eval_rows, manifest_dir / output.get("eval_manifest", "eval.jsonl"))
    print(f"wrote train rows: {len(train_rows)}")
    print(f"wrote eval rows: {len(eval_rows)}")
    train_counts = Counter(row.get("dataset", "unknown") for row in train_rows)
    eval_counts = Counter(row.get("dataset", "unknown") for row in eval_rows)
    print(f"train dataset counts: {dict(train_counts)}")
    print(f"eval dataset counts: {dict(eval_counts)}")


if __name__ == "__main__":
    main()
