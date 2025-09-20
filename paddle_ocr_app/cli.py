"""PaddleOCR命令行工具。"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable, List, Sequence

from . import OCRResult, PaddleOCRService

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _confidence(value: str) -> float:
    try:
        result = float(value)
    except ValueError as exc:  # pragma: no cover - argparse会处理
        raise argparse.ArgumentTypeError("请输入合法的浮点数") from exc
    if not 0.0 <= result <= 1.0:
        raise argparse.ArgumentTypeError("置信度阈值需要在0~1之间")
    return result


def _collect_image_paths(inputs: Sequence[Path], recursive: bool) -> List[Path]:
    images: List[Path] = []
    seen = set()
    for input_path in inputs:
        if not input_path.exists():
            raise FileNotFoundError(f"未找到路径: {input_path}")
        if input_path.is_file():
            if input_path.suffix.lower() in IMAGE_EXTENSIONS:
                if input_path not in seen:
                    seen.add(input_path)
                    images.append(input_path)
            else:
                logging.warning("忽略不支持的文件: %s", input_path)
            continue

        iterator: Iterable[Path]
        iterator = input_path.rglob("*") if recursive else input_path.glob("*")
        for candidate in iterator:
            if candidate.is_file() and candidate.suffix.lower() in IMAGE_EXTENSIONS:
                if candidate not in seen:
                    seen.add(candidate)
                    images.append(candidate)
    return images


def _print_result(result: OCRResult) -> None:
    header = f"===== {result.image_path} ====="
    print(header)
    if not result.lines:
        print("未识别到文本。\n")
        return
    for line in result.lines:
        print(f"[{line.confidence:.0%}] {line.text}")
    print()


def _write_text(results: Sequence[OCRResult], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as fp:
        for index, result in enumerate(results):
            fp.write(f"# {result.image_path}\n")
            if result.lines:
                fp.write(result.to_text("\n"))
                fp.write("\n")
            if index != len(results) - 1:
                fp.write("\n")


def _write_json(results: Sequence[OCRResult], output_path: Path) -> None:
    payload = [result.to_dict() for result in results]
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _visualize_result(result: OCRResult, output_dir: Path, font_path: Path | None) -> Path:
    if not result.lines:
        raise ValueError("当前图片没有识别结果，无法生成可视化。")

    try:
        from paddleocr import draw_ocr  # type: ignore
        from PIL import Image  # type: ignore
    except ImportError as exc:  # pragma: no cover - 依赖缺失
        raise RuntimeError("生成可视化需要安装'pillow'依赖: pip install pillow") from exc

    image = Image.open(result.image_path).convert("RGB")
    boxes = [line.bbox for line in result.lines]
    texts = [line.text for line in result.lines]
    scores = [line.confidence for line in result.lines]
    im_show = draw_ocr(image, boxes, texts, scores, font_path=str(font_path) if font_path else None)
    annotated = Image.fromarray(im_show)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{result.image_path.stem}_ocr.png"
    annotated.save(output_path)
    return output_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="基于PaddleOCR的轻量级识别工具")
    parser.add_argument("inputs", nargs="+", type=Path, help="图片文件或文件夹，可以一次传入多个")
    parser.add_argument("--lang", default="ch", help="PaddleOCR的语言代码，例如ch、en")
    parser.add_argument("--use-gpu", action="store_true", help="使用GPU进行识别")
    parser.add_argument("--recursive", action="store_true", help="递归遍历文件夹中的图片")
    parser.add_argument("--min-confidence", type=_confidence, default=0.0, help="过滤低于阈值的识别结果")
    parser.add_argument("--limit", type=int, help="仅处理指定数量的图片")
    parser.add_argument("--output", type=Path, help="将纯文本结果写入指定文件")
    parser.add_argument("--json", dest="json_path", type=Path, help="将结构化结果保存为JSON文件")
    parser.add_argument("--visualize", type=Path, help="输出识别标注图像的目录")
    parser.add_argument("--font", type=Path, help="可视化文字使用的字体文件路径")
    parser.add_argument("--det-model-dir", type=Path, help="自定义检测模型目录")
    parser.add_argument("--rec-model-dir", type=Path, help="自定义识别模型目录")
    parser.add_argument("--cls-model-dir", type=Path, help="自定义方向分类模型目录")
    parser.add_argument("--debug", action="store_true", help="显示调试日志")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, format="%(levelname)s: %(message)s")

    try:
        image_paths = _collect_image_paths(args.inputs, args.recursive)
    except FileNotFoundError as exc:
        logging.error("%s", exc)
        return 1

    if not image_paths:
        logging.error("未找到任何可识别的图片。")
        return 1

    if args.limit is not None and args.limit >= 0:
        image_paths = image_paths[: args.limit]

    paddle_kwargs = {}
    if args.det_model_dir:
        paddle_kwargs["det_model_dir"] = str(args.det_model_dir)
    if args.rec_model_dir:
        paddle_kwargs["rec_model_dir"] = str(args.rec_model_dir)
    if args.cls_model_dir:
        paddle_kwargs["cls_model_dir"] = str(args.cls_model_dir)

    try:
        service = PaddleOCRService(
            lang=args.lang,
            use_gpu=args.use_gpu,
            show_log=args.debug,
            **paddle_kwargs,
        )
    except RuntimeError as exc:
        logging.error("%s", exc)
        return 1

    results: List[OCRResult] = []
    for path in image_paths:
        logging.info("开始识别: %s", path)
        try:
            result = service.run(path)
        except Exception as exc:  # pragma: no cover - 运行时错误
            logging.error("识别失败 %s: %s", path, exc)
            continue
        filtered_result = result.filtered(args.min_confidence)
        results.append(filtered_result)
        _print_result(filtered_result)

        if args.visualize:
            try:
                saved_path = _visualize_result(filtered_result, args.visualize, args.font)
            except Exception as exc:  # pragma: no cover - 可视化失败
                logging.warning("生成可视化失败 %s: %s", path, exc)
            else:
                logging.info("可视化结果已保存: %s", saved_path)

    if args.output and results:
        _write_text(results, args.output)
        logging.info("文本结果已写入: %s", args.output)

    if args.json_path and results:
        _write_json(results, args.json_path)
        logging.info("JSON结果已写入: %s", args.json_path)

    return 0 if results else 2
