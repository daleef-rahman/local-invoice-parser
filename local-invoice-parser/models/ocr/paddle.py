from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from paddleocr import PaddleOCR
from PIL import Image


def load_backend(*, lang: str = "en", use_textline_orientation: bool = True) -> PaddleOCR:
    return PaddleOCR(
        use_textline_orientation=use_textline_orientation,
        lang=lang,
    )


def extract_text(
    ocr: PaddleOCR,
    image_path: str,
    *,
    max_image_side: int | None = 1600,
) -> tuple[str, list[dict[str, Any]]]:
    ocr_input_path, temp_image_path, original_size, resized_size = _prepare_image_for_ocr(
        image_path,
        max_image_side=max_image_side,
    )
    try:
        result = ocr.predict(ocr_input_path)
    finally:
        if temp_image_path is not None:
            temp_image_path.unlink(missing_ok=True)

    lines: list[str] = []
    raw: list[dict[str, Any]] = []
    if resized_size is not None:
        raw.append({
            "meta": {
                "source_size": list(original_size),
                "ocr_input_size": list(resized_size),
            }
        })

    for page in result:
        for text, score, box in zip(page["rec_texts"], page["rec_scores"], page["rec_polys"]):
            lines.append(text)
            raw.append({
                "text": text,
                "confidence": round(float(score), 4),
                "box": box.tolist(),
            })
    return "\n".join(lines), raw


def _prepare_image_for_ocr(
    image_path: str,
    *,
    max_image_side: int | None,
) -> tuple[str, Path | None, tuple[int, int] | None, tuple[int, int] | None]:
    if max_image_side is None:
        return image_path, None, None, None

    src_path = Path(image_path)
    with Image.open(src_path) as img:
        original_size = img.size
        if max(original_size) <= max_image_side:
            return image_path, None, original_size, None

        resized = img.convert("RGB")
        resized.thumbnail((max_image_side, max_image_side), Image.LANCZOS)
        resized_size = resized.size

        tmp = tempfile.NamedTemporaryFile(
            prefix="ocr_input_",
            suffix=src_path.suffix or ".png",
            delete=False,
        )
        tmp_path = Path(tmp.name)
        tmp.close()
        resized.save(tmp_path)

    return str(tmp_path), tmp_path, original_size, resized_size
