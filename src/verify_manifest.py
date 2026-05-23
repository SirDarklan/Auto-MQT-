from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from data_utils import load_jsonl


REQUIRED_FIELDS = ["dataset", "split", "example_id", "image", "prompt", "answer"]


def resolve_image_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.exists():
        return path
    # Handle manifests produced on Windows (`\`) when validating on Linux/Colab.
    normalized = Path(raw_path.replace("\\", "/"))
    if normalized.exists():
        return normalized
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--max-errors", type=int, default=20)
    args = parser.parse_args()

    rows = load_jsonl(args.manifest)
    errors = []
    for row_index, row in enumerate(rows):
        for field in REQUIRED_FIELDS:
            if field not in row or row[field] in {None, ""}:
                errors.append(f"row {row_index}: missing `{field}`")

        image_path = resolve_image_path(str(row.get("image", "")))
        if not image_path.exists():
            errors.append(f"row {row_index}: image not found `{image_path}`")
        else:
            try:
                with Image.open(image_path) as image:
                    image.verify()
            except Exception as exc:  # noqa: BLE001
                errors.append(f"row {row_index}: unreadable image `{image_path}` ({exc})")

        if len(errors) >= args.max_errors:
            break

    if errors:
        print(f"manifest: {args.manifest}")
        print(f"rows: {len(rows)}")
        print("errors:")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)

    print(f"manifest ok: {args.manifest}")
    print(f"rows: {len(rows)}")


if __name__ == "__main__":
    main()
