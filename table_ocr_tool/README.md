# 表格识别工具

该目录提供一个独立的 Python 脚本 `table_ocr_tool.py`，用于在 PandaOCR 项目中快速体验表格识别功能。脚本基于「线框检测 + OCR」思路，将表格截图拆分成单元格并输出 CSV 或 JSON 结果，方便后续导入到 Excel、Notion 等工具。

## 功能特性
- **自动检测表格结构**：使用 OpenCV 对横线和竖线进行形态学处理，推断表格单元格布局。
- **多 OCR 引擎支持**：优先调用 [`rapidocr-onnxruntime`](https://github.com/RapidAI/RapidOCR)，若未安装则回退到 [`pytesseract`](https://github.com/madmaze/pytesseract)。
- **多种输出格式**：支持 CSV（二维表格）和 JSON（包含坐标与文本的结构化信息）。
- **无侵入集成**：脚本与 PandaOCR 主程序解耦，可单独运行或与已有工作流组合。

## 环境依赖
- Python 3.9+
- [opencv-python](https://pypi.org/project/opencv-python/)
- [numpy](https://pypi.org/project/numpy/)
- 以下 OCR 后端二选一：
  - [rapidocr-onnxruntime](https://pypi.org/project/rapidocr-onnxruntime/)（默认优先）
  - [pytesseract](https://pypi.org/project/pytesseract/) + [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)

```bash
pip install opencv-python numpy rapidocr-onnxruntime
# 或者
pip install opencv-python numpy pytesseract
```

> 若使用 `pytesseract`，请根据操作系统安装 Tesseract 主程序，并确保其在系统 PATH 中。

## 快速上手
```bash
python table_ocr_tool.py --image ./samples/table.png --csv table.csv --json table.json
```

- `--image`：表格截图路径。
- `--csv`：输出 CSV 文件路径（可选）。
- `--json`：输出 JSON 文件路径（可选）。
- `--backend`：指定 OCR 后端，可多次传入（如 `--backend rapidocr --backend tesseract`）。
- `--debug`：输出调试日志，观察单元格识别细节。

当不指定 `--csv`/`--json` 时，脚本会在控制台逐行打印 `[row, col] 文本` 结果，适合快速验证。

## 在代码中调用

如果不想通过命令行参数运行，可在其他脚本中直接调用 `run_table_recognition`：

```python
from table_ocr_tool.table_ocr_tool import run_table_recognition

cells = run_table_recognition("./samples/table.png")
for cell in cells:
    print(cell.row, cell.column, cell.text)
```

`run_table_recognition` 会读取 `image` 路径指向的图片（或直接接受 `numpy.ndarray` 图像），返回包含行列、坐标与文本的 `Cell` 对象列表。根据需要还可以同时传入 `csv_path`/`json_path`，在不依赖命令行的情况下导出识别结果。

## 工作流程
1. 读入图像并转为灰度图，应用自适应阈值生成二值图。
2. 使用形态学腐蚀/膨胀分别提取横线与竖线，合并获得表格线框。
3. 通过轮廓检测获取单元格候选框，按行列排序。
4. 对每个单元格裁剪图像，调用 OCR 引擎识别文本。
5. 根据需要导出 CSV 或 JSON。

## 与 PandaOCR 联动思路
- 可将脚本输出的 CSV 作为 PandaOCR 识别后的后处理步骤，例如在监听剪贴板模式下保存截图，再批处理识别。
- 针对特定接口（如百度/腾讯表格识别 API），可改写 `OCRBackend` 以对接 HTTP 服务，实现更高精度或批量处理。

欢迎根据自身需求扩展，例如：
- 加入自定义阈值、行列合并逻辑。
- 对接更多 OCR 引擎或自训练模型。
- 集成 GUI 以便在 Windows 上一键处理表格截图。
