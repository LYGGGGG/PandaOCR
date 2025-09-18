"""Command line utility for table recognition based on OCR.

This module provides a thin wrapper around OpenCV + Tesseract style
processing so that PandaOCR users can batch convert table screenshots into
structured CSV/JSON outputs.  The implementation focuses on being easy to
understand and customise rather than being a drop-in replacement for heavy
frameworks.

Typical usage::

    python table_ocr_tool.py --image demo.png --csv demo.csv

The script detects horizontal/vertical ruling lines, segments the image into
cells and runs OCR for each cell.  Text recognition requires either
``pytesseract`` or ``rapidocr-onnxruntime``.  When both are available,
RapidOCR is used because it performs better on multilingual text.

The module intentionally keeps dependencies optional.  Each backend is only
imported when requested at runtime.  When neither backend can be imported the
script raises a :class:`RuntimeError` with installation hints.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

try:
    import cv2  # type: ignore
    import numpy as np  # type: ignore
except ImportError as exc:  # pragma: no cover - optional dependency guard
    raise RuntimeError(
        "table_ocr_tool requires 'opencv-python' and 'numpy'.\n"
        "Install them via 'pip install opencv-python numpy'."
    ) from exc

_LOGGER = logging.getLogger(__name__)


@dataclass
class Cell:
    """Represents a detected table cell."""

    row: int
    column: int
    bbox: Tuple[int, int, int, int]
    text: str = ""

    def to_dict(self) -> dict:
        return {
            "row": self.row,
            "column": self.column,
            "bbox": self.bbox,
            "text": self.text,
        }


class OCRBackend:
    """Abstract base class for OCR engines."""

    def recognise(self, image: "np.ndarray") -> str:
        raise NotImplementedError


class RapidOCRBackend(OCRBackend):
    """OCR backend using rapidocr_onnxruntime."""

    def __init__(self) -> None:
        from rapidocr_onnxruntime import RapidOCR  # type: ignore

        self._ocr = RapidOCR()

    def recognise(self, image: "np.ndarray") -> str:
        result, _ = self._ocr(image)
        if not result:
            return ""
        return " ".join(item[1] for item in result)


class TesseractBackend(OCRBackend):
    """OCR backend using pytesseract."""

    def __init__(self) -> None:
        import pytesseract  # type: ignore

        self._tesseract = pytesseract

    def recognise(self, image: "np.ndarray") -> str:
        return self._tesseract.image_to_string(image, config="--psm 6").strip()


def _load_ocr_backend(preferred: Sequence[str]) -> OCRBackend:
    last_error = None
    for name in preferred:
        try:
            if name == "rapidocr":
                return RapidOCRBackend()
            if name == "tesseract":
                return TesseractBackend()
        except Exception as exc:  # pragma: no cover - runtime import errors
            last_error = exc
            _LOGGER.debug("Failed to initialise %s backend: %s", name, exc)
    raise RuntimeError(
        "No OCR backend is available.\n"
        "Install 'rapidocr-onnxruntime' or 'pytesseract'."
    ) from last_error


def _binarise(image: "np.ndarray") -> "np.ndarray":
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 10
    )


def _extract_table_mask(binary: "np.ndarray") -> Tuple["np.ndarray", "np.ndarray"]:
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (binary.shape[1] // 30, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, binary.shape[0] // 30))

    horizontal_lines = cv2.erode(binary, horizontal_kernel, iterations=1)
    horizontal_lines = cv2.dilate(horizontal_lines, horizontal_kernel, iterations=1)

    vertical_lines = cv2.erode(binary, vertical_kernel, iterations=1)
    vertical_lines = cv2.dilate(vertical_lines, vertical_kernel, iterations=1)

    return horizontal_lines, vertical_lines


def _detect_cells(binary: "np.ndarray") -> List[Tuple[int, int, int, int]]:
    horizontal_lines, vertical_lines = _extract_table_mask(binary)
    table_mask = cv2.add(horizontal_lines, vertical_lines)
    contours, _ = cv2.findContours(table_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    boxes: List[Tuple[int, int, int, int]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < 10 or h < 10:
            continue
        boxes.append((x, y, w, h))

    boxes.sort(key=lambda box: (box[1], box[0]))
    return boxes


def _cluster_rows(boxes: Sequence[Tuple[int, int, int, int]], tolerance: int = 10) -> List[List[Tuple[int, int, int, int]]]:
    rows: List[List[Tuple[int, int, int, int]]] = []
    for box in boxes:
        x, y, w, h = box
        if not rows:
            rows.append([box])
            continue
        last_row = rows[-1]
        _, last_y, _, last_h = last_row[0]
        if abs(y - last_y) <= tolerance or abs((y + h) - (last_y + last_h)) <= tolerance:
            last_row.append(box)
        else:
            rows.append([box])

    for row in rows:
        row.sort(key=lambda item: item[0])
    return rows


def recognise_table(image_path: Path, backend_priority: Sequence[str] | None = None) -> List[Cell]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Unable to load image: {image_path}")

    binary = _binarise(image)
    boxes = _detect_cells(binary)
    rows = _cluster_rows(boxes)

    backend = _load_ocr_backend(backend_priority or ("rapidocr", "tesseract"))

    cells: List[Cell] = []
    for row_idx, row in enumerate(rows):
        for col_idx, (x, y, w, h) in enumerate(row):
            cell_image = image[y : y + h, x : x + w]
            text = backend.recognise(cell_image)
            cells.append(Cell(row=row_idx, column=col_idx, bbox=(x, y, w, h), text=text))
    return cells


def _write_csv(cells: Sequence[Cell], output_path: Path) -> None:
    if not cells:
        output_path.write_text("")
        return

    max_row = max(cell.row for cell in cells)
    max_col = max(cell.column for cell in cells)

    table: List[List[str]] = [["" for _ in range(max_col + 1)] for _ in range(max_row + 1)]
    for cell in cells:
        table[cell.row][cell.column] = cell.text

    with output_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerows(table)


def _write_json(cells: Sequence[Cell], output_path: Path) -> None:
    data = [cell.to_dict() for cell in cells]
    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Table recognition helper for PandaOCR")
    parser.add_argument("--image", required=True, type=Path, help="Path to the input table screenshot")
    parser.add_argument("--csv", type=Path, help="Optional CSV output path")
    parser.add_argument("--json", type=Path, help="Optional JSON output path")
    parser.add_argument(
        "--backend",
        choices=["rapidocr", "tesseract"],
        action="append",
        help=(
            "Preferred OCR backend(s).  Can be supplied multiple times.  When omitted, "
            "the tool tries rapidocr first and falls back to pytesseract."
        ),
    )
    parser.add_argument("--debug", action="store_true", help="Enable verbose logging")
    return parser.parse_args(args)


def main(cli_args: Sequence[str] | None = None) -> int:
    args = parse_args(cli_args)
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

    cells = recognise_table(args.image, backend_priority=args.backend)

    if args.csv:
        _write_csv(cells, args.csv)
        _LOGGER.info("CSV exported to %s", args.csv)
    if args.json:
        _write_json(cells, args.json)
        _LOGGER.info("JSON exported to %s", args.json)

    for cell in cells:
        _LOGGER.debug("Cell r%d c%d: %s", cell.row, cell.column, cell.text)

    if not (args.csv or args.json):
        for cell in cells:
            print(f"[{cell.row}, {cell.column}] {cell.text}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
