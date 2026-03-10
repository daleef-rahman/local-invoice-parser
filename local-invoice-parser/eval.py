"""
Evaluation script for invoice parser experiments.

Modes:
    simple  -> local sample invoices + local ground truth JSON
    full    -> Hugging Face benchmark dataset

Examples:
    uv run python local-invoice-parser/eval.py \
      --mode simple \
      --experiment exp1_ocr_ner_gliner2

    uv run python local-invoice-parser/eval.py \
      --mode simple \
      --experiment exp2_ocr_ner_qwen3

    uv run python local-invoice-parser/eval.py \
      --mode full \
      --experiment exp3_vlm_qwen25vl \
      --limit 100
"""

from __future__ import annotations

import logging
import os
import warnings

# Suppress noisy third-party logs before any imports that trigger them.
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
warnings.filterwarnings("ignore", category=Warning, module="requests")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

import argparse
import json
import math
import re
from difflib import SequenceMatcher
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

from datasets import load_dataset
from PIL import Image
from experiments.catalog import EXPERIMENT_SPECS, resolve_experiment_id
from pipeline import (
    PreparedPipeline,
    close_pipeline,
    prepare_pipeline,
    run_pipeline,
)
from runtime import managed_experiment_runtime

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
class FieldAccumulator:
    score: float = 0.0
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


# ── field-type sets ────────────────────────────────────────────────────────────

_NUMERIC_FIELDS = frozenset({"totalAmount", "taxAmount", "paidAmount", "discountAmount", "serviceCharge"})
_OPTIONAL_ZERO_FIELDS = frozenset({"discountAmount", "serviceCharge"})
_NULL_EQUIVALENTS = frozenset({"na", "n/a", "n.a.", "none", "null", "-", "--", ""})

_COUNTRY_ALIASES: dict[str, str] = {
    "usa": "united states", "us": "united states", "u.s.": "united states",
    "u.s.a.": "united states", "uk": "united kingdom", "u.k.": "united kingdom",
    "gb": "united kingdom", "great britain": "united kingdom",
}

_STATE_ALIASES: dict[str, str] = {
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas",
    "ca": "california", "co": "colorado", "ct": "connecticut", "de": "delaware",
    "fl": "florida", "ga": "georgia", "hi": "hawaii", "id": "idaho",
    "il": "illinois", "in": "indiana", "ia": "iowa", "ks": "kansas",
    "ky": "kentucky", "la": "louisiana", "me": "maine", "md": "maryland",
    "ma": "massachusetts", "mi": "michigan", "mn": "minnesota", "ms": "mississippi",
    "mo": "missouri", "mt": "montana", "ne": "nebraska", "nv": "nevada",
    "nh": "new hampshire", "nj": "new jersey", "nm": "new mexico", "ny": "new york",
    "nc": "north carolina", "nd": "north dakota", "oh": "ohio", "ok": "oklahoma",
    "or": "oregon", "pa": "pennsylvania", "ri": "rhode island", "sc": "south carolina",
    "sd": "south dakota", "tn": "tennessee", "tx": "texas", "ut": "utah",
    "vt": "vermont", "va": "virginia", "wa": "washington", "wv": "west virginia",
    "wi": "wisconsin", "wy": "wyoming", "dc": "district of columbia",
}

# ── per-type scorers ───────────────────────────────────────────────────────────


def _is_null_equivalent(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return _normalize_text(value) in _NULL_EQUIVALENTS
    return False


def _score_numeric(pred: Any, truth: Any, optional_zero: bool = False) -> float:
    """1.0 = exact (≤0.01 diff), 0.5 = within 5%, 0.0 = otherwise."""
    pred_f, truth_f = _to_float(pred), _to_float(truth)

    if optional_zero:
        pred_zero = pred is None or (pred_f is not None and abs(pred_f) < 1e-2)
        truth_zero = truth is None or (truth_f is not None and abs(truth_f) < 1e-2)
        if pred_zero and truth_zero:
            return 1.0

    if pred is None and truth is None:
        return 1.0
    if pred is None or truth is None:
        return 0.0
    if pred_f is None or truth_f is None:
        return 0.0

    diff = abs(pred_f - truth_f)
    if diff <= 1e-2:
        return 1.0
    if truth_f != 0 and diff / abs(truth_f) <= 0.05:
        return 0.5
    return 0.0


def _score_string(pred: Any, truth: Any, field_type: str | None = None) -> float:
    """1.0 = high similarity (≥0.95), 0.5 = partial (≥0.70), 0.0 = low."""
    if _is_null_equivalent(pred) and _is_null_equivalent(truth):
        return 1.0
    if _is_null_equivalent(pred) or _is_null_equivalent(truth):
        return 0.0

    p = _normalize_text(str(pred))
    t = _normalize_text(str(truth))

    if field_type == "country":
        p, t = _COUNTRY_ALIASES.get(p, p), _COUNTRY_ALIASES.get(t, t)
    elif field_type == "state":
        p, t = _STATE_ALIASES.get(p, p), _STATE_ALIASES.get(t, t)

    if p == t:
        return 1.0

    sim = SequenceMatcher(None, p, t).ratio()
    if sim >= 0.95:
        return 1.0
    if sim >= 0.70:
        return 0.5
    return 0.0


def _score_datetime(pred: Any, truth: Any) -> float:
    """1.0 = date+time match (≤1 min), 0.5 = date only, 0.0 = different date."""
    if pred is None and truth is None:
        return 1.0
    if pred is None or truth is None:
        return 0.0

    _fmts = [
        "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M",
        "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d",
    ]

    pred_dt = truth_dt = None
    for fmt in _fmts:
        try:
            pred_dt = datetime.strptime(str(pred).strip(), fmt)
            break
        except ValueError:
            pass
    for fmt in _fmts:
        try:
            truth_dt = datetime.strptime(str(truth).strip(), fmt)
            break
        except ValueError:
            pass

    if pred_dt is None or truth_dt is None:
        return _score_string(pred, truth)

    if pred_dt.date() != truth_dt.date():
        return 0.0
    return 1.0 if abs((pred_dt - truth_dt).total_seconds()) <= 60 else 0.5


def _score_field(field: str, pred_val: Any, truth_val: Any) -> float:
    if field in _NUMERIC_FIELDS:
        return _score_numeric(pred_val, truth_val, optional_zero=field in _OPTIONAL_ZERO_FIELDS)
    if field == "dateTime":
        return _score_datetime(pred_val, truth_val)
    field_type = "country" if field == "merchantCountry" else ("state" if field == "merchantState" else None)
    return _score_string(pred_val, truth_val, field_type=field_type)


def _score_line_items(pred_items: list, truth_items: list) -> tuple[float, float]:
    """Best-match pairing of line items. Returns (total_score, max_possible)."""
    if not pred_items and not truth_items:
        return 0.0, 0.0

    n_fields = len(LINE_ITEM_FIELDS)
    max_count = max(len(pred_items), len(truth_items))

    if not pred_items or not truth_items:
        return 0.0, float(max_count * n_fields)

    used: set[int] = set()
    total_score = 0.0

    for t_item in truth_items:
        t_name = _normalize_text(str(t_item.get("productName") or ""))
        best_idx, best_sim = -1, -1.0
        for i, p_item in enumerate(pred_items):
            if i in used:
                continue
            sim = SequenceMatcher(None, t_name, _normalize_text(str(p_item.get("productName") or ""))).ratio()
            if sim > best_sim:
                best_sim, best_idx = sim, i

        if best_idx >= 0:
            used.add(best_idx)
            p_item = pred_items[best_idx]
            for f in LINE_ITEM_FIELDS:
                total_score += _score_field(f, p_item.get(f), t_item.get(f))

    return total_score, float(max_count * n_fields)


def _score_example(pred: dict[str, Any], truth: dict[str, Any]) -> dict[str, Any]:
    scalar_score = 0.0
    scalar_total = float(len(SCALAR_FIELDS))
    per_field: dict[str, float] = {}

    for field in SCALAR_FIELDS:
        s = _score_field(field, pred.get(field), truth.get(field))
        per_field[field] = s
        scalar_score += s

    li_score, li_total = _score_line_items(
        pred.get("productLineItems") or [],
        truth.get("productLineItems") or [],
    )

    return {
        "scalar_score": scalar_score,
        "scalar_total": scalar_total,
        "line_item_score": li_score,
        "line_item_total": li_total,
        "total_score": scalar_score + li_score,
        "total_fields": scalar_total + li_total,
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

            # Persist image into stable temp file; source temporary directory is closed on return.
            safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", ex_id)
            stable_tmp = Path(tempfile.gettempdir()) / f"invoice_eval_{safe_id}.png"
            if image_path != stable_tmp:
                with Image.open(image_path) as img:
                    img.save(stable_tmp)
            examples.append(Example(example_id=ex_id, image_path=stable_tmp, ground_truth=truth))

    return examples


def evaluate(
    mode: str,
    experiment_name: str,
    prepared: PreparedPipeline,
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
        "scalar_score": 0.0,
        "scalar_total": 0.0,
        "line_item_score": 0.0,
        "line_item_total": 0.0,
        "total_score": 0.0,
        "total_fields": 0.0,
    }
    per_field_totals = {field: FieldAccumulator() for field in SCALAR_FIELDS}
    per_example: list[dict[str, Any]] = []

    total_wall_s = 0.0
    for i, ex in enumerate(examples, start=1):
        print(f"[{i}/{len(examples)}] Evaluating example {ex.example_id} ...")
        t_start = time.perf_counter()
        run_result = run_pipeline(prepared, str(ex.image_path))
        pred = run_result.receipt.model_dump()
        timings = run_result.timings
        wall_s = round(time.perf_counter() - t_start, 3)
        total_wall_s += wall_s

        score = _score_example(pred=pred, truth=ex.ground_truth)

        for k in totals:
            totals[k] += score[k]

        for field, s in score["per_field"].items():
            per_field_totals[field].total += 1
            per_field_totals[field].score += s

        per_example.append(
            {
                "id": ex.example_id,
                "image_path": str(ex.image_path),
                "scalar_accuracy": round(score["scalar_score"] / max(score["scalar_total"], 1), 4),
                "line_item_accuracy": (
                    round(score["line_item_score"] / score["line_item_total"], 4)
                    if score["line_item_total"] > 0
                    else None
                ),
                "overall_accuracy": round(score["total_score"] / max(score["total_fields"], 1), 4),
                "timings_s": {**timings, "wall": wall_s},
            }
        )

    n = len(examples)
    result = {
        "mode": mode,
        "experiment": experiment_name,
        "num_examples": n,
        "timing_s": {
            "total_wall": round(total_wall_s, 3),
            "avg_wall_per_invoice": round(total_wall_s / n, 3) if n else None,
        },
        "metrics": {
            "scalar_accuracy": round(totals["scalar_score"] / max(totals["scalar_total"], 1), 4),
            "line_item_accuracy": (
                round(totals["line_item_score"] / totals["line_item_total"], 4)
                if totals["line_item_total"] > 0
                else None
            ),
            "overall_accuracy": round(totals["total_score"] / max(totals["total_fields"], 1), 4),
            "counts": totals,
            "per_field_accuracy": {
                field: round(acc.score / max(acc.total, 1), 4)
                for field, acc in per_field_totals.items()
            },
        },
        "per_example": per_example,
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate invoice extraction experiments")
    parser.add_argument("--mode", choices=["simple", "full"], required=True, help="Evaluation mode")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--experiment",
        type=str,
        help="Experiment id from the catalog (for example: exp1_ocr_ner_gliner2)",
    )
    selection.add_argument("--all", action="store_true", help="Run all experiments in the catalog")

    parser.add_argument("--sample-dir", type=Path, default=Path("data/sample-invoices"))
    parser.add_argument("--sample-ground-truth", type=Path, default=Path("data/sample-invoices/ground_truth.json"))

    parser.add_argument("--dataset", type=str, default="mychen76/invoices-and-receipts_ocr_v1")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--limit", type=int, default=None, help="Max examples (mainly for full mode)")

    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output file path")
    return parser.parse_args()


def _print_summary(result: dict[str, Any]) -> None:
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


def _default_output_path(selection_name: str, mode: str) -> Path:
    repo_root = Path(__file__).resolve().parent.parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return repo_root / "reports" / f"{selection_name}_{mode}_{timestamp}.json"


def _save_report(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))


def _run_experiment(
    *,
    mode: str,
    experiment_id: str,
    sample_dir: Path,
    sample_ground_truth: Path,
    dataset_name: str,
    split: str,
    limit: int | None,
) -> dict[str, Any]:
    spec = EXPERIMENT_SPECS[experiment_id]
    print(f"\n=== Running {experiment_id} ===")
    with managed_experiment_runtime(spec):
        loaded = prepare_pipeline(experiment_id)
        try:
            return evaluate(
                mode=mode,
                experiment_name=experiment_id,
                prepared=loaded,
                sample_dir=sample_dir,
                sample_ground_truth=sample_ground_truth,
                dataset_name=dataset_name,
                split=split,
                limit=limit,
            )
        finally:
            close_pipeline(loaded)


def main() -> None:
    args = parse_args()
    experiment_ids = (
        [resolve_experiment_id(args.experiment)]
        if args.experiment
        else sorted(EXPERIMENT_SPECS.keys())
    )

    results = []
    for experiment_id in experiment_ids:
        result = _run_experiment(
            mode=args.mode,
            experiment_id=experiment_id,
            sample_dir=args.sample_dir,
            sample_ground_truth=args.sample_ground_truth,
            dataset_name=args.dataset,
            split=args.split,
            limit=args.limit,
        )
        results.append(result)
        _print_summary(result)

    if len(results) == 1:
        payload: dict[str, Any] = results[0]
        selection_name = experiment_ids[0]
    else:
        payload = {
            "mode": args.mode,
            "experiments": results,
            "summary": [
                {
                    "experiment": result["experiment"],
                    "num_examples": result["num_examples"],
                    "scalar_accuracy": result["metrics"]["scalar_accuracy"],
                    "line_item_accuracy": result["metrics"]["line_item_accuracy"],
                    "overall_accuracy": result["metrics"]["overall_accuracy"],
                    "avg_wall_per_invoice": result["timing_s"]["avg_wall_per_invoice"],
                }
                for result in results
            ],
        }
        selection_name = "all_experiments"
        print("\n=== Combined Summary ===")
        for result in results:
            metrics = result["metrics"]
            print(
                f"{result['experiment']}: overall={metrics['overall_accuracy']:.4f}, "
                f"scalar={metrics['scalar_accuracy']:.4f}, "
                f"line_item={metrics['line_item_accuracy'] if metrics['line_item_accuracy'] is not None else 'n/a'}"
            )

    output_path = args.output or _default_output_path(selection_name, args.mode)
    _save_report(payload, output_path)
    print(f"\nSaved report to {output_path}")


if __name__ == "__main__":
    main()
