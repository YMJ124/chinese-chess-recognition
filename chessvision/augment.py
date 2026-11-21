from __future__ import annotations

from io import BytesIO

import cv2
import numpy as np
import torch
from PIL import Image


def _jpeg_roundtrip(image: np.ndarray, quality: int) -> np.ndarray:
    ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return image
    return cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def canonical_array(image: Image.Image, input_size: int = 96) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"))
    interpolation = cv2.INTER_AREA if min(rgb.shape[:2]) >= input_size else cv2.INTER_CUBIC
    return cv2.resize(rgb, (input_size, input_size), interpolation=interpolation)


def augment_array(image: Image.Image, rng: np.random.Generator, input_size: int = 96, severity: float = 1.0) -> np.ndarray:
    source = np.asarray(image.convert("RGB"))
    canvas_size = 128
    base_level = int(rng.integers(232, 256))
    yy, xx = np.mgrid[0:canvas_size, 0:canvas_size]
    gradient = rng.uniform(-10, 10) * (xx / max(canvas_size - 1, 1)) + rng.uniform(-8, 8) * (yy / max(canvas_size - 1, 1))
    canvas = np.clip(base_level + gradient[..., None], 0, 255).astype(np.uint8)
    canvas = np.repeat(canvas, 3, axis=2)

    scale = rng.uniform(0.72, 0.96)
    target_side = max(40, round(canvas_size * scale))
    resized = cv2.resize(source, (target_side, target_side), interpolation=cv2.INTER_AREA if target_side < source.shape[0] else cv2.INTER_CUBIC)
    dx = round(rng.uniform(-8, 8) * severity)
    dy = round(rng.uniform(-8, 8) * severity)
    x0 = (canvas_size - target_side) // 2 + dx
    y0 = (canvas_size - target_side) // 2 + dy
    x1, y1 = max(0, x0), max(0, y0)
    x2, y2 = min(canvas_size, x0 + target_side), min(canvas_size, y0 + target_side)
    if x2 > x1 and y2 > y1:
        canvas[y1:y2, x1:x2] = resized[y1 - y0:y2 - y0, x1 - x0:x2 - x0]

    center = (canvas_size / 2, canvas_size / 2)
    angle = rng.uniform(-22, 22) * severity
    affine_scale = rng.uniform(0.90, 1.08)
    matrix = cv2.getRotationMatrix2D(center, angle, affine_scale)
    border = int(rng.integers(232, 256))
    transformed = cv2.warpAffine(canvas, matrix, (canvas_size, canvas_size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(border,) * 3)

    jitter = 10.0 * severity
    src = np.float32([[0, 0], [canvas_size - 1, 0], [canvas_size - 1, canvas_size - 1], [0, canvas_size - 1]])
    dst = src + rng.uniform(-jitter, jitter, size=(4, 2)).astype(np.float32)
    perspective = cv2.getPerspectiveTransform(src, dst)
    transformed = cv2.warpPerspective(transformed, perspective, (canvas_size, canvas_size), flags=cv2.INTER_LINEAR, borderValue=(border,) * 3)

    work = transformed.astype(np.float32) / 255.0
    contrast = rng.uniform(0.65, 1.35)
    brightness = rng.uniform(-0.15, 0.15) * severity
    work = (work - 0.5) * contrast + 0.5 + brightness
    gamma = rng.uniform(0.68, 1.45)
    work = np.clip(work, 0, 1) ** gamma
    color_gain = np.array([rng.uniform(0.88, 1.12), rng.uniform(0.90, 1.08), rng.uniform(0.88, 1.12)], dtype=np.float32)
    work *= color_gain

    if rng.random() < 0.65:
        direction = rng.uniform(0, np.pi * 2)
        shadow = (np.cos(direction) * xx + np.sin(direction) * yy) / canvas_size
        shadow = shadow - shadow.min()
        shadow /= max(shadow.max(), 1e-6)
        shadow_strength = rng.uniform(0.05, 0.28) * severity
        work *= (1.0 - shadow_strength * shadow[..., None])
    if rng.random() < 0.35:
        gx, gy = rng.uniform(20, 108, size=2)
        sigma = rng.uniform(18, 40)
        glare = np.exp(-((xx - gx) ** 2 + (yy - gy) ** 2) / (2 * sigma * sigma))
        work += glare[..., None] * rng.uniform(0.05, 0.25) * severity

    work = np.clip(work * 255.0, 0, 255).astype(np.uint8)
    if rng.random() < 0.70:
        sigma = rng.uniform(0.15, 1.6) * severity
        kernel = max(1, int(round(sigma * 4)) | 1)
        work = cv2.GaussianBlur(work, (kernel, kernel), sigmaX=sigma)
    if rng.random() < 0.65:
        noise_sigma = rng.uniform(1.0, 10.0) * severity
        noise = rng.normal(0, noise_sigma, size=work.shape)
        work = np.clip(work.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if rng.random() < 0.55:
        work = _jpeg_roundtrip(work, int(rng.integers(45, 96)))
    return cv2.resize(work, (input_size, input_size), interpolation=cv2.INTER_AREA)


def array_to_tensor(array: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1))).float().div_(255.0)
    return tensor.sub_(0.5).div_(0.5)


def tensor_from_image(image: Image.Image, input_size: int = 96) -> torch.Tensor:
    return array_to_tensor(canonical_array(image, input_size=input_size))


def encode_preview_jpeg(array: np.ndarray, quality: int = 92) -> bytes:
    output = BytesIO()
    Image.fromarray(array).save(output, format="JPEG", quality=quality)
    return output.getvalue()
