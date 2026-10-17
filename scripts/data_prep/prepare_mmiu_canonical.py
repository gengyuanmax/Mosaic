#!/usr/bin/env python3
"""Convert the downloaded MMIU parquet into MosaicGym canonical JSON files."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare MMIU canonical datasets for MosaicGym.")
    parser.add_argument(
        "--mmiu-root",
        default="data/mmiu",
        help="MMIU root containing all.parquet and extracted image category directories.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Output directory for canonical JSON files. Defaults to <mmiu-root>/datasets.",
    )
    parser.add_argument(
        "--write-all",
        action="store_true",
        help="Also write all.json containing every MMIU sample.",
    )
    return parser


def clean_image_path(value: str) -> str:
    text = str(value or "").strip()
    while text.startswith("./"):
        text = text[2:]
    return text


def normalize_image_paths(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [clean_image_path(item) for item in value if str(item or "").strip()]
    if hasattr(value, "tolist"):
        return normalize_image_paths(value.tolist())
    return [clean_image_path(value)]


def safe_filename(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._+-]+", "_", str(value or "").strip())
    return name.strip("._") or "task"


def build_prompt(row) -> str:
    context = str(row.get("context") or "").strip()
    question = str(row.get("question") or "").strip()
    options = str(row.get("options") or "").strip()
    parts = []
    if context:
        parts.append(context)
    if question and question not in context:
        parts.append("Question: %s" % question)
    if options and options not in context:
        parts.append("Choices:\n%s" % options)
    parts.append("Answer with only the correct option letter.")
    return "\n\n".join(parts)


def to_canonical(row, index: int) -> dict:
    task = str(row.get("task") or "mmiu").strip()
    image_paths = normalize_image_paths(row.get("input_image_path"))
    images = [
        {
            "image_id": "img_%d" % (idx + 1),
            "uri_or_path": image_path,
            "modality": "image",
            "source": str(row.get("source") or ""),
        }
        for idx, image_path in enumerate(image_paths)
    ]
    return {
        "sample_id": "mmiu_%06d" % index,
        "dataset_name": "mmiu",
        "prompt": build_prompt(row),
        "images": images,
        "task_type": task,
        "target": {
            "type": "classification",
            "format": "mc",
            "value": str(row.get("output") or "").strip(),
        },
        "metadata": {
            "task": task,
            "visual_input_component": str(row.get("visual_input_component") or ""),
            "source": str(row.get("source") or ""),
            "question": str(row.get("question") or ""),
            "options": str(row.get("options") or ""),
            "mmiu_index": index,
        },
    }


def write_json(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_json_object(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def main() -> int:
    args = build_parser().parse_args()
    mmiu_root = Path(args.mmiu_root).resolve()
    parquet_path = mmiu_root / "all.parquet"
    if not parquet_path.exists():
        raise SystemExit("MMIU parquet not found: %s" % parquet_path)

    output_dir = Path(args.output_dir).resolve() if args.output_dir else mmiu_root / "datasets"
    frame = pd.read_parquet(parquet_path)
    records = [to_canonical(row, index) for index, row in enumerate(frame.to_dict("records"))]

    by_task: dict[str, list[dict]] = {}
    for record in records:
        by_task.setdefault(str(record["task_type"]), []).append(record)

    manifest = {
        "mmiu_root": str(mmiu_root),
        "parquet_path": str(parquet_path),
        "num_samples": len(records),
        "num_tasks": len(by_task),
        "tasks": {},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for task_name, task_records in sorted(by_task.items()):
        filename = safe_filename(task_name) + ".json"
        write_json(output_dir / filename, task_records)
        manifest["tasks"][task_name] = {
            "path": str(output_dir / filename),
            "num_samples": len(task_records),
        }

    if args.write_all:
        write_json(output_dir / "all.json", records)

    write_json_object(output_dir / "manifest.json", manifest)
    print("wrote %d samples across %d tasks to %s" % (len(records), len(by_task), output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
