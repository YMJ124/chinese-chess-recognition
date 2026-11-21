from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .detection import Detection


def find_chinese_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    )
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def draw_detections(frame_bgr: np.ndarray, detections: list[Detection], show_confidence: bool = True) -> np.ndarray:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    font_size = max(18, round(min(frame_bgr.shape[:2]) * 0.023))
    font = find_chinese_font(font_size)
    header_font = find_chinese_font(max(20, round(font_size * 1.1)))

    for detection in detections:
        color = (220, 32, 32) if detection.piece.faction == "红" else (25, 25, 25)
        line_width = max(2, font_size // 8)
        draw.rectangle((detection.x1, detection.y1, detection.x2, detection.y2), outline=color, width=line_width)
        text = detection.piece.display
        if show_confidence:
            text += f" {detection.confidence:.2f}"
        text_bbox = draw.textbbox((0, 0), text, font=font, stroke_width=0)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        tx = min(max(0, detection.x1), max(0, image.width - text_width - 8))
        ty = detection.y1 - text_height - 10
        if ty < 0:
            ty = min(image.height - text_height - 8, detection.y2 + 4)
        draw.rectangle((tx, ty, tx + text_width + 8, ty + text_height + 7), fill=color)
        draw.text((tx + 4, ty + 2 - text_bbox[1]), text, font=font, fill="white")

    header = f"棋子数量: {len(detections)}"
    header_bbox = draw.textbbox((0, 0), header, font=header_font)
    draw.rectangle((8, 8, header_bbox[2] + 24, header_bbox[3] + 20), fill=(0, 80, 150))
    draw.text((16, 10 - header_bbox[1]), header, font=header_font, fill="white")
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
