# 象棋棋子光电目标识别

本项目完成赛题一任务1：从赛方 `象棋样本.docx` 提取红黑共 14 类棋子，使用合成扩增训练轻量 CNN，再通过 OpenCV 定位圆形棋子并输出阵营、类别和分值。

完整的程序流程图、代码结构和核心技术说明见
[《程序流程与核心技术解析》](docs/程序流程与核心技术解析.md)。

识别路线为：

1. DOCX OOXML 裁剪参数→ 14 张原始棋子。
2. 透视、曝光、阴影、模糊、噪声、压缩等扩增→ CNN 训练数据。
3. 纸张透视校正＋Hough 圆候选→ CNN 14 类分类。
4. 图片、视频和摄像头中用矩形框与中文标签输出结果。

## 环境

已创建的 Conda 环境为 `chess_cv`：

```bash
conda activate chess_cv
cd ~/桌面/程序设计大赛/chess_recognition
```

验证 GPU：

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 从头生成数据和模型

```bash
python prepare_data.py \
  --docx ../赛题一所需资料/象棋样本.docx \
  --output data

python train.py \
  --data data \
  --config config.json \
  --output models \
  --device cuda
```

主要输出：

- `data/base/`：14 张从 DOCX 精确还原的棋子。
- `data/augmentation_preview.jpg`：扩增效果预览。
- `models/best_model.pt`：Python 推理和继续训练检查点。
- `models/chess_cnn.ts`：TorchScript 推理模型。
- `models/training_report.json`：训练与 14 张原图评估结果。

## 静态评估

```bash
python evaluate.py \
  --model models/best_model.pt \
  --data data \
  --image ../象棋图片.jpg \
  --output outputs/evaluation \
  --device cuda
```

标注图片保存在 `outputs/evaluation/annotated_test.jpg`。

## 检测图片

```bash
python detect.py \
  --source ../象棋图片.jpg \
  --model models/best_model.pt \
  --output outputs/象棋图片_检测结果.jpg \
  --device cuda
```

## 检测录制视频

```bash
python detect.py \
  --source /绝对路径/录制视频.mp4 \
  --model models/best_model.pt \
  --output outputs/视频检测结果.mp4 \
  --device cuda
```

终端会输出“红-帅-10分”等结果，同时生成同名 `.jsonl` 逐帧记录。
程序结束时会自动调用 FFmpeg，将视频转为兼容性更好的 H.264 High Profile、`yuv420p` 像素格式，并启用 MP4 faststart。如果仅需保留 OpenCV 原始 MP4V 视频，可添加 `--keep-mp4v`。

## 实时摄像头

```bash
python detect.py \
  --camera 0 \
  --model models/best_model.pt \
  --output outputs/摄像头检测结果.mp4 \
  --device cuda \
  --show
```

显示窗口中按 `q` 退出。

## 打印与录制要求

- 使用原始 `象棋样本.docx`。
- 打印选择“实际大小”或 100%，禁用“适应纸张”。
- 棋子尺寸不得改动。
- 摄像头与棋子距离不小于 20 cm。
- 尽量让纸张四角进入画面，有利于透视校正。
- 保证 14 枚棋子完整可见，避免严重反光、失焦和遮挡。

## 测试

```bash
python -m pytest -q
```

程序对视频无法打开、摄像头不可用和画面无棋子等情况会输出明确提示，不会将空结果当作程序异常。

## 仓库内容

- `chessvision/`：数据增强、CNN、圆检测、渲染和时序跟踪等核心模块。
- `data/base/`：从赛方 DOCX 提取的红黑 14 类基础训练样本；训练时在线随机扩增。
- `models/`：最佳 PyTorch 检查点、TorchScript 模型以及训练曲线和评估记录。
- `docs/`：程序整体流程图和核心技术解析。
- `tests/`：核心流程自动化测试。

`outputs/`、测试录像、逐帧日志、Python 缓存和本地编辑器配置已通过 `.gitignore` 排除，不会上传到 GitHub。
