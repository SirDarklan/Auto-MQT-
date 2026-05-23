from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: Iterable[dict[str, Any]], path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def append_jsonl(row: dict[str, Any], path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()


def canonical_dataset_name(name: str | None) -> str:
    if name is None:
        return "unknown"
    value = str(name).strip().lower()
    return value or "unknown"


def parse_dataset_names(values: list[str] | None) -> set[str] | None:
    if not values:
        return None
    names: set[str] = set()
    for value in values:
        for part in str(value).split(","):
            normalized = canonical_dataset_name(part)
            if normalized and normalized != "unknown":
                names.add(normalized)
    return names or None


def filter_rows_by_dataset(
    rows: list[dict[str, Any]],
    include_datasets: set[str] | None = None,
    exclude_datasets: set[str] | None = None,
) -> list[dict[str, Any]]:
    if include_datasets is None and exclude_datasets is None:
        return rows

    filtered: list[dict[str, Any]] = []
    for row in rows:
        dataset = canonical_dataset_name(row.get("dataset"))
        if include_datasets is not None and dataset not in include_datasets:
            continue
        if exclude_datasets is not None and dataset in exclude_datasets:
            continue
        filtered.append(row)
    return filtered


def normalize_answer(text: str) -> str:
    return " ".join(text.lower().strip().split())


def exact_match(prediction: str, answer: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(answer))


def exact_match_any(prediction: str, answers: Any) -> float:
    if isinstance(answers, str):
        return exact_match(prediction, answers)
    if isinstance(answers, (list, tuple)):
        return float(any(exact_match(prediction, str(answer)) for answer in answers))
    return exact_match(prediction, str(answers))


def normalize_for_relaxed_match(text: str) -> str:
    normalized = normalize_answer(str(text))
    normalized = re.sub(r"\b([ap])\.?\s*m\.?\b", r"\1m", normalized)
    normalized = re.sub(r"\b(\d{1,2}:\d{2})\s*([ap]m)\b", r"\1 \2", normalized)
    normalized = re.sub(r"[^a-z0-9:]+", " ", normalized)
    return " ".join(normalized.split())


def relaxed_match(prediction: str, answer: str) -> float:
    pred = normalize_for_relaxed_match(prediction)
    target = normalize_for_relaxed_match(answer)
    if not pred or not target:
        return 0.0
    if pred == target:
        return 1.0
    return float(f" {target} " in f" {pred} ")


def relaxed_match_any(prediction: str, answers: Any) -> float:
    if isinstance(answers, str):
        return relaxed_match(prediction, answers)
    if isinstance(answers, (list, tuple)):
        return float(any(relaxed_match(prediction, str(answer)) for answer in answers))
    return relaxed_match(prediction, str(answers))


def vqa_soft_accuracy(prediction: str, answers: Any) -> float:
    if isinstance(answers, str):
        answers_list = [answers]
    elif isinstance(answers, (list, tuple)):
        answers_list = [str(answer) for answer in answers]
    else:
        answers_list = [str(answers)]

    if not answers_list:
        return 0.0
    normalized_prediction = normalize_answer(prediction)
    normalized_answers = [normalize_answer(answer) for answer in answers_list]
    match_count = sum(answer == normalized_prediction for answer in normalized_answers)

    # Match the common VQAv2/TextVQA soft-accuracy formulation used by EvalAI
    # when 10 annotator answers are available.
    if len(normalized_answers) == 10:
        m = float(match_count)
        score = (m * min(1.0, max(0.0, (m - 1.0) / 3.0)) + (10.0 - m) * min(1.0, m / 3.0)) / 10.0
        return float(score)

    if len(normalized_answers) >= 3:
        return float(min(1.0, match_count / 3.0))
    return float(match_count / max(1, len(normalized_answers)))


def dataset_score(prediction: str, answers: Any, dataset: str | None = None) -> tuple[float, str]:
    dataset_name = (dataset or "").lower()
    if dataset_name in {"vqav2", "textvqa", "vizwiz"}:
        return vqa_soft_accuracy(prediction, answers), "vqa_soft_accuracy"
    if dataset_name in {"gqa", "scienceqa_img", "scienceqa"}:
        return exact_match_any(prediction, answers), "accuracy"

    # Fallback heuristic when dataset is unknown.
    if isinstance(answers, (list, tuple)) and len(answers) >= 3:
        return vqa_soft_accuracy(prediction, answers), "vqa_soft_accuracy"
    return exact_match_any(prediction, answers), "accuracy"


def score_prediction(
    prediction: str,
    answers: Any,
    dataset: str | None = None,
) -> dict[str, float]:
    ds_score, metric_name = dataset_score(prediction, answers, dataset=dataset)
    return {
        "exact_match": exact_match_any(prediction, answers),
        "relaxed_match": relaxed_match_any(prediction, answers),
        "dataset_score": ds_score,
        "dataset_metric_name": metric_name,
    }


def answers_for_example(example: dict[str, Any]) -> list[str]:
    if "answers" in example and isinstance(example["answers"], list) and example["answers"]:
        return [str(answer) for answer in example["answers"]]
    if "answer" in example:
        return [str(example["answer"])]
    return []


def format_prompt(prompt: str, task: str | None = None, prompt_style: str = "short") -> str:
    if prompt_style == "none":
        return prompt
    if prompt_style != "short":
        raise ValueError(f"Unknown prompt style: {prompt_style}")

    lowered = prompt.lower()
    if "answer with" in lowered or "do not explain" in lowered:
        return prompt

    if "choices:" in lowered or "options:" in lowered:
        instruction = "Answer with only the correct option letter (A, B, C, ...). Do not explain."
    elif task == "ocr":
        instruction = "Answer with only the text or number. Do not explain."
    else:
        instruction = "Answer with a short phrase only. Do not explain."
    return f"{prompt}\n{instruction}"


def stable_hash_vector(text: str, dim: int) -> np.ndarray:
    """Deterministic feature fallback for pipeline tests before real embeddings exist."""
    vector = np.zeros(dim, dtype=np.float32)
    tokens = normalize_answer(text).split()
    for token in tokens or [text]:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, byteorder="little", signed=False)
        index = value % dim
        sign = 1.0 if ((value >> 8) & 1) else -1.0
        vector[index] += sign
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector /= norm
    return vector


def feature_from_example(
    example: dict[str, Any],
    field: str,
    fallback_text: str,
    dim: int,
) -> np.ndarray:
    if field in example:
        return np.asarray(example[field], dtype=np.float32)
    return stable_hash_vector(fallback_text, dim)
