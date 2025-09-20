"""PaddleOCR命令行工具的核心API。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence


@dataclass
class OCRLine:
    """表示一条OCR识别结果。"""

    text: str
    confidence: float
    bbox: List[tuple[float, float]]

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "confidence": self.confidence,
            "bbox": [[float(x), float(y)] for x, y in self.bbox],
        }


@dataclass
class OCRResult:
    """单张图片的识别结果。"""

    image_path: Path
    lines: List[OCRLine]

    def to_dict(self) -> dict:
        return {
            "image": str(self.image_path),
            "lines": [line.to_dict() for line in self.lines],
        }

    def to_text(self, joiner: str = "\n") -> str:
        return joiner.join(line.text for line in self.lines)

    def filtered(self, min_confidence: float) -> "OCRResult":
        if min_confidence <= 0:
            return self
        filtered_lines = [line for line in self.lines if line.confidence >= min_confidence]
        if len(filtered_lines) == len(self.lines):
            return self
        return OCRResult(image_path=self.image_path, lines=filtered_lines)


class PaddleOCRService:
    """封装PaddleOCR识别流程。"""

    def __init__(
        self,
        *,
        lang: str = "ch",
        use_gpu: bool = False,
        use_angle_cls: bool = True,
        show_log: bool = False,
        **paddle_kwargs,
    ) -> None:
        try:
            from paddleocr import PaddleOCR  # type: ignore
        except ImportError as exc:  # pragma: no cover - 依赖缺失时提示
            raise RuntimeError(
                "PaddleOCRService需要'paddleocr'库，请先运行: pip install paddleocr"
            ) from exc

        if "use_angle_cls" not in paddle_kwargs:
            paddle_kwargs["use_angle_cls"] = use_angle_cls
        if "show_log" not in paddle_kwargs:
            paddle_kwargs["show_log"] = show_log

        self._use_angle_cls = bool(paddle_kwargs.get("use_angle_cls", use_angle_cls))
        self._ocr = PaddleOCR(lang=lang, use_gpu=use_gpu, **paddle_kwargs)

    def run(self, image_path: Path) -> OCRResult:
        if not image_path.exists():
            raise FileNotFoundError(f"未找到图片文件: {image_path}")
        if not image_path.is_file():
            raise FileNotFoundError(f"路径不是文件: {image_path}")

        raw_result = self._ocr.ocr(str(image_path), cls=self._use_angle_cls)

        def _is_line(entry: object) -> bool:
            if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                return False
            points, payload = entry
            return isinstance(points, (list, tuple)) and isinstance(payload, (list, tuple))

        if raw_result and isinstance(raw_result[0], list) and not _is_line(raw_result[0]) and raw_result[0]:
            first_item = raw_result[0][0]
            if _is_line(first_item):
                raw_result = raw_result[0]

        lines: List[OCRLine] = []
        for item in raw_result or []:
            if not item or len(item) < 2:
                continue
            points = item[0]
            text, confidence = item[1]
            bbox = [(float(x), float(y)) for x, y in points]
            lines.append(OCRLine(text=text, confidence=float(confidence), bbox=bbox))
        return OCRResult(image_path=image_path, lines=lines)

    def run_batch(self, image_paths: Sequence[Path]) -> List[OCRResult]:
        return [self.run(path) for path in image_paths]
