#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from chessvision.augment import augment_array
from chessvision.dataset import load_manifest
from chessvision.docx_extract import extract_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从 DOCX 提取 14 类象棋棋子并生成扩增预览")
    parser.add_argument("--docx", required=True, help="象棋样本.docx 路径")
    parser.add_argument("--output", default="data", help="数据输出目录")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def find_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def write_augmentation_preview(data_dir: Path, seed: int) -> None:
    manifest = load_manifest(data_dir)
    tile, columns, variants = 112, 8, 7
    canvas = Image.new("RGB", (columns * tile, len(manifest) * tile), "white")
    draw = ImageDraw.Draw(canvas)
    font = find_font(15)
    for row, item in enumerate(manifest):
        source = Image.open(data_dir / item["file"]).convert("RGB")
        label_tile = Image.new("RGB", (tile, tile), "#f2f2f2")
        label_draw = ImageDraw.Draw(label_tile)
        label_draw.text((6, 35), item["display"], fill="black", font=font)
        canvas.paste(label_tile, (0, row * tile))
        for column in range(variants):
            rng = np.random.default_rng(seed + row * 1000 + column)
            image = Image.fromarray(augment_array(source, rng, input_size=96))
            canvas.paste(image, (column * tile + tile + 8, row * tile + 8))
    canvas.save(data_dir / "augmentation_preview.jpg", quality=92)


def main() -> int:
    args = parse_args()
    docx = Path(args.docx).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if not docx.exists():
        print(f"错误：DOCX 不存在: {docx}", file=sys.stderr)
        return 2
    manifest = extract_dataset(docx, output)
    write_augmentation_preview(output, args.seed)
    summary = {
        "docx": str(docx),
        "output": str(output),
        "classes": len(manifest),
        "files": [item["file"] for item in manifest],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"已生成参考图: {output / 'reference_sheet.jpg'}")
    print(f"已生成扩增预览: {output / 'augmentation_preview.jpg'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
