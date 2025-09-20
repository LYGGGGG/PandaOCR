"""允许通过 `python -m paddle_ocr_app` 直接运行命令行工具。"""
from __future__ import annotations

from .cli import main

if __name__ == "__main__":  # pragma: no cover - 入口点
    raise SystemExit(main())
