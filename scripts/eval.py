"""
Evaluation script for invoice parser experiments.

Modes:
    simple  -> local sample invoices + local ground truth JSON
    full    -> Hugging Face benchmark dataset

Examples:
    uv run python scripts/eval.py \
      --mode simple \
      --experiment exp1_ocr_ner.py

    uv run python scripts/eval.py \
      --mode simple \
      --experiment exp2_vlm.py \
      --experiment-arg --vlm-backend --experiment-arg qwen25vl

    uv run python scripts/eval.py \
      --mode full \
      --experiment exp2_vlm.py \
      --limit 100
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shlex
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

from datasets import load_dataset
from PIL import Image

SCALAR_FIELDS = [
    "totalAmount",
    "taxAmount",
    "dateTime",
    "merchantName",
    "merchantAddress",
    "currencyCode",
    "merchantCountry",
    "merchantState",
    "merchantCity",
    "merchantPostalCode",
    "merchantPhone",
    "merchantEmail",
    "invoiceReceiptNumber",
    "paidAmount",
    "discountAmount",
    "serviceCharge",
]
LINE_ITEM_FIELDS = ["productName", "quantity", "unitPrice", "totalPrice", "productCode"]


@dataclass
class Example:
    example_id: str
    image_path: Path
    ground_truth: dict[str, Any]


@dataclass
class FieldCounter:
    correct: int = 0
    total: int = 0


def _unwrap_value(value: Any) -> Any:
    if isinstance(value, dict) and "value" in value:
        return value["value"]
    return value


def _normalize_gt_payload(payload: dict[str, Any]) -> dict[str, Any]:
    data = payload.get("data", payload)
    normalized: dict[str, Any] = {}

    for field in SCALAR_FIELDS:
        normalized[field] = _unwrap_value(data.get(field))

    line_items = data.get("productLineItems") or []
    normalized_items: list[dict[str, Any]] = []
    for item in line_items:
        out_item: dict[str, Any] = {}
        for field in LINE_ITEM_FIELDS:
            out_item[field] = _unwrap_value(item.get(field))
        normalized_items.append(out_item)

    normalized["productLineItems"] = normalized_items
    return normalized


def _normalize_text(value: str) -> str:
    return " ".join(value.strip().split()).casefold()


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return None
        return float(value)

    if not isinstance(value, str):
        return None

    cleaned = value.strip()
    if not cleaned:
        return None

    # Keep digits, decimal sign, plus/minus.
    cleaned = re.sub(r"[^0-9.\-+]", "", cleaned.replace(",", ""))
    if cleaned in {"", ".", "+", "-", "+.", "-."}:
        return None

    try:
        return float(cleaned)
    except ValueError:
        return None


def _values_match(pred: Any, truth: Any) -> bool:
    if pred is None and truth is None:
        return True
    if pred is None or truth is None:
        return False

    pred_num = _to_float(pred)
    truth_num = _to_float(truth)
    if pred_num is not None and truth_num is not None:
        return abs(pred_num - truth_num) <= 1e-2

    return _normalize_text(str(pred)) == _normalize_text(str(truth))


def _score_example(pred: dict[str, Any], truth: dict[str, Any]) -> dict[str, Any]:
    scalar_correct = 0
    scalar_total = len(SCALAR_FIELDS)

    per_field: dict[str, bool] = {}
    for field in SCALAR_FIELDS:
        ok = _values_match(pred.get(field), truth.get(field))
        per_field[field] = ok
        scalar_correct += int(ok)

    pred_items = pred.get("productLineItems") or []
    truth_items = truth.get("productLineItems") or []

    li_correct = 0
    li_total = max(len(pred_items), len(truth_items)) * len(LINE_ITEM_FIELDS)
    if li_total == 0:
        li_total = 0

    for idx in range(max(len(pred_items), len(truth_items))):
        pred_item = pred_items[idx] if idx < len(pred_items) else {}
        truth_item = truth_items[idx] if idx < len(truth_items) else {}
        for field in LINE_ITEM_FIELDS:
            li_correct += int(_values_match(pred_item.get(field), truth_item.get(field)))

    total_correct = scalar_correct + li_correct
    total_fields = scalar_total + li_total

    return {
        "scalar_correct": scalar_correct,
        "scalar_total": scalar_total,
        "line_item_correct": li_correct,
        "line_item_total": li_total,
        "total_correct": total_correct,
        "total_fields": total_fields,
        "per_field": per_field,
    }


def _parse_json_maybe(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _extract_dataset_truth(row: dict[str, Any]) -> dict[str, Any]:
    for key in ("parsed_data", "ground_truth", "annotation", "label"):
        if key in row:
            parsed = _parse_json_maybe(row[key])
            if isinstance(parsed, dict):
                return _normalize_gt_payload(parsed)

    raise ValueError(f"Could not locate ground-truth payload in dataset row keys: {list(row.keys())}")


def _dataset_image_to_path(image_obj: Any, tmp_dir: Path, example_id: str) -> Path:
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", example_id)
    out_path = tmp_dir / f"{safe_id}.png"

    if isinstance(image_obj, Image.Image):
        image_obj.save(out_path)
        return out_path

    if isinstance(image_obj, dict):
        if image_obj.get("path"):
            p = Path(image_obj["path"])
            if p.exists():
                return p
        if image_obj.get("bytes"):
            img = Image.open(BytesIO(image_obj["bytes"]))
            img.save(out_path)
            return out_path

    if isinstance(image_obj, str):
        p = Path(image_obj)
        if p.exists():
            return p

    raise ValueError(f"Unsupported image format from dataset: {type(image_obj).__name__}")


def load_examples_simple(sample_dir: Path, ground_truth_path: Path) -> list[Example]:
    payload = json.loads(ground_truth_path.read_text())
    rows = payload.get("test_images", [])
    examples: list[Example] = []

    for row in rows:
        ex_id = str(row.get("image_id"))
        gt = _normalize_gt_payload(row.get("ground_truth", {}))
        raw_image_path = row.get("image_path", "")
        image_name = Path(raw_image_path).name
        image_path = sample_dir / image_name
        if not image_path.exists():
            raise FileNotFoundError(f"Sample image missing for example {ex_id}: {image_path}")
        examples.append(Example(example_id=ex_id, image_path=image_path, ground_truth=gt))

    return examples


def load_examples_full(dataset_name: str, split: str, limit: int | None) -> list[Example]:
    ds = load_dataset(dataset_name, split=split, trust_remote_code=True)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    examples: list[Example] = []
    with tempfile.TemporaryDirectory(prefix="invoice_eval_images_") as tmp:
        tmp_dir = Path(tmp)
        for idx, row in enumerate(ds):
            ex_id = str(row.get("id", idx))
            truth = _extract_dataset_truth(row)
            image_key = "image" if "image" in row else "raw_data"
            if image_key not in row:
                raise ValueError(f"Could not find image key in row {ex_id}. Keys: {list(row.keys())}")
            image_path = _dataset_image_to_path(row[image_key], tmp_dir, ex_id)

            # Persist image into stable temp file under workspace temp for the subprocess call.
            safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", ex_id)
            stable_tmp = Path(tempfile.gettempdir()) / f"invoice_eval_{safe_id}.png"
            if image_path != stable_tmp:
                with Image.open(image_path) as img:
                    img.save(stable_tmp)
            examples.append(Example(example_id=ex_id, image_path=stable_tmp, ground_truth=truth))

    return examples


def run_experiment(
    experiment_path: Path,
    image_path: Path,
    extra_args: list[str],
    cwd: Path,
) -> dict[str, Any]:
    with tempfile.NamedTemporaryFile(prefix="invoice_pred_", suffix=".json", delete=False) as f:
        out_file = Path(f.name)

    cmd = [
        sys.executable,
        str(experiment_path),
        "--image",
        str(image_path),
        "--output",
        str(out_file),
        *extra_args,
    ]

    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "Experiment failed\n"
            f"Command: {' '.join(shlex.quote(x) for x in cmd)}\n"
            f"Exit code: {proc.returncode}\n"
            f"Stdout:\n{proc.stdout}\n"
            f"Stderr:\n{proc.stderr}"
        )

    payload = json.loads(out_file.read_text())
    out_file.unlink(missing_ok=True)

    receipt = payload.get("receipt")
    if not isinstance(receipt, dict):
        raise ValueError(f"Experiment output missing 'receipt' dict: {out_file}")
    return receipt


def evaluate(
    mode: str,
    experiment_path: Path,
    experiment_args: list[str],
    sample_dir: Path,
    sample_ground_truth: Path,
    dataset_name: str,
    split: str,
    limit: int | None,
) -> dict[str, Any]:
    if mode == "simple":
        examples = load_examples_simple(sample_dir=sample_dir, ground_truth_path=sample_ground_truth)
    else:
        examples = load_examples_full(dataset_name=dataset_name, split=split, limit=limit)

    totals = {
        "scalar_correct": 0,
        "scalar_total": 0,
        "line_item_correct": 0,
        "line_item_total": 0,
        "total_correct": 0,
        "total_fields": 0,
    }
    per_field_totals = {field: FieldCounter() for field in SCALAR_FIELDS}
    per_example: list[dict[str, Any]] = []

    repo_root = Path(__file__).resolve().parent.parent

    for i, ex in enumerate(examples, start=1):
        print(f"[{i}/{len(examples)}] Evaluating example {ex.example_id} ...")
        pred = run_experiment(
            experiment_path=experiment_path,
            image_path=ex.image_path,
            extra_args=experiment_args,
            cwd=repo_root,
        )
        score = _score_example(pred=pred, truth=ex.ground_truth)

        for k in totals:
            totals[k] += score[k]

        for field, ok in score["per_field"].items():
            per_field_totals[field].total += 1
            per_field_totals[field].correct += int(ok)

        per_example.append(
            {
                "id": ex.example_id,
                "image_path": str(ex.image_path),
                "scalar_accuracy": round(score["scalar_correct"] / max(score["scalar_total"], 1), 4),
                "line_item_accuracy": (
                    round(score["line_item_correct"] / score["line_item_total"], 4)
                    if score["line_item_total"] > 0
                    else None
                ),
                "overall_accuracy": round(score["total_correct"] / max(score["total_fields"], 1), 4),
            }
        )

    result = {
        "mode": mode,
        "experiment": str(experiment_path),
        "num_examples": len(examples),
        "metrics": {
            "scalar_accuracy": round(totals["scalar_correct"] / max(totals["scalar_total"], 1), 4),
            "line_item_accuracy": (
                round(totals["line_item_correct"] / totals["line_item_total"], 4)
                if totals["line_item_total"] > 0
                else None
            ),
            "overall_accuracy": round(totals["total_correct"] / max(totals["total_fields"], 1), 4),
            "counts": totals,
            "per_field_accuracy": {
                field: round(counter.correct / max(counter.total, 1), 4)
                for field, counter in per_field_totals.items()
            },
        },
        "per_example": per_example,
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate invoice extraction experiments")
    parser.add_argument("--mode", choices=["simple", "full"], required=True, help="Evaluation mode")
    parser.add_argument(
        "--experiment",
        type=Path,
        required=True,
        help="Path to experiment script (e.g., exp1_ocr_ner.py or exp2_vlm.py)",
    )
    parser.add_argument(
        "--experiment-arg",
        action="append",
        default=[],
        help="Pass-through arg to experiment script (repeat flag for multiple args)",
    )

    parser.add_argument("--sample-dir", type=Path, default=Path("data/sample-invoices"))
    parser.add_argument("--sample-ground-truth", type=Path, default=Path("data/sample-invoices/ground_truth.json"))

    parser.add_argument("--dataset", type=str, default="mychen76/invoices-and-receipts_ocr_v1")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--limit", type=int, default=None, help="Max examples (mainly for full mode)")

    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output file path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_path = args.experiment.resolve()
    if not experiment_path.exists():
        raise FileNotFoundError(f"Experiment script not found: {experiment_path}")

    result = evaluate(
        mode=args.mode,
        experiment_path=experiment_path,
        experiment_args=args.experiment_arg,
        sample_dir=args.sample_dir,
        sample_ground_truth=args.sample_ground_truth,
        dataset_name=args.dataset,
        split=args.split,
        limit=args.limit,
    )

    metrics = result["metrics"]
    print("\n=== Evaluation Summary ===")
    print(f"Mode: {result['mode']}")
    print(f"Experiment: {result['experiment']}")
    print(f"Examples: {result['num_examples']}")
    print(f"Scalar accuracy   : {metrics['scalar_accuracy']:.4f}")
    if metrics["line_item_accuracy"] is None:
        print("Line-item accuracy: n/a")
    else:
        print(f"Line-item accuracy: {metrics['line_item_accuracy']:.4f}")
    print(f"Overall accuracy  : {metrics['overall_accuracy']:.4f}")

    print("\nPer-field accuracy:")
    for field, score in metrics["per_field_accuracy"].items():
        print(f"  {field:>20}: {score:.4f}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2))
        print(f"\nSaved report to {args.output}")


if __name__ == "__main__":
    main()
