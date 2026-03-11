from __future__ import annotations

from typing import Any

from paddleocr import PaddleOCR


def load_backend(*, lang: str = "en", use_textline_orientation: bool = True) -> PaddleOCR:
    return PaddleOCR(
        use_textline_orientation=use_textline_orientation,
        lang=lang,
    )


def extract_text(
    ocr: PaddleOCR,
    image_path: str,
) -> tuple[str, list[dict[str, Any]]]:
    result = ocr.predict(image_path)

    lines: list[str] = []
    raw: list[dict[str, Any]] = []

    for page in result:
        for text, score, box in zip(page["rec_texts"], page["rec_scores"], page["rec_polys"]):
            lines.append(text)
            raw.append({
                "text": text,
                "confidence": round(float(score), 4),
                "box": box.tolist(),
            })
    return "\n".join(lines), raw
