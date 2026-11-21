from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from .augment import tensor_from_image
from .labels import CLASS_BY_INDEX, PIECE_CLASSES
from .model import load_checkpoint


@dataclass
class CircleCandidate:
    x: float
    y: float
    radius: float
    quality: float


@dataclass
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    center_x: float
    center_y: float
    radius: float
    class_index: int
    confidence: float
    circle_quality: float

    @property
    def piece(self):
        return CLASS_BY_INDEX[self.class_index]

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            "key": self.piece.key,
            "faction": self.piece.faction,
            "name": self.piece.name,
            "score": self.piece.score,
            "display": self.piece.display,
        }


def _order_quad(points: np.ndarray) -> np.ndarray:
    points = points.astype(np.float32)
    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = points.sum(axis=1)
    differences = np.diff(points, axis=1).reshape(-1)
    ordered[0] = points[np.argmin(sums)]  # 左上
    ordered[2] = points[np.argmax(sums)]  # 右下
    ordered[1] = points[np.argmin(differences)]  # 右上
    ordered[3] = points[np.argmax(differences)]  # 左下
    return ordered


def rectify_printed_page(frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """尝试将拍摄到的纸张透视校正，返回校正图和从校正图到原图的反变换。"""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blurred, 45, 135)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    image_area = frame_bgr.shape[0] * frame_bgr.shape[1]
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:20]:
        area = cv2.contourArea(contour)
        if area < image_area * 0.20:
            break
        perimeter = cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(polygon) != 4 or not cv2.isContourConvex(polygon):
            continue
        source = _order_quad(polygon.reshape(4, 2))
        width_top = np.linalg.norm(source[1] - source[0])
        width_bottom = np.linalg.norm(source[2] - source[3])
        height_left = np.linalg.norm(source[3] - source[0])
        height_right = np.linalg.norm(source[2] - source[1])
        target_width = int(round(max(width_top, width_bottom)))
        target_height = int(round(max(height_left, height_right)))
        if min(target_width, target_height) < 300:
            continue
        aspect = min(target_width, target_height) / max(target_width, target_height)
        if not 0.45 <= aspect <= 0.90:
            continue
        destination = np.float32(
            [[0, 0], [target_width - 1, 0], [target_width - 1, target_height - 1], [0, target_height - 1]]
        )
        transform = cv2.getPerspectiveTransform(source, destination)
        inverse = cv2.getPerspectiveTransform(destination, source)
        rectified = cv2.warpPerspective(frame_bgr, transform, (target_width, target_height), borderValue=(255, 255, 255))
        return rectified, inverse
    return None


def _circle_quality(gray: np.ndarray, x: float, y: float, radius: float) -> float:
    height, width = gray.shape
    half = int(max(8, round(radius * 1.12)))
    x0, x1 = max(0, int(round(x)) - half), min(width, int(round(x)) + half + 1)
    y0, y1 = max(0, int(round(y)) - half), min(height, int(round(y)) + half + 1)
    crop = gray[y0:y1, x0:x1]
    if crop.size == 0 or min(crop.shape) < 16:
        return 0.0
    yy, xx = np.mgrid[y0:y1, x0:x1]
    normalized_radius = np.sqrt((xx - x) ** 2 + (yy - y) ** 2) / max(radius, 1.0)
    dark = crop < 190
    ring_mask = (normalized_radius >= 0.72) & (normalized_radius <= 1.02)
    inner_mask = normalized_radius <= 0.62
    outer_mask = normalized_radius >= 1.05
    if not ring_mask.any() or not inner_mask.any():
        return 0.0
    ring_dark = float(dark[ring_mask].mean())
    inner_dark = float(dark[inner_mask].mean())
    outer_light = float((crop[outer_mask] > 180).mean()) if outer_mask.any() else 0.5
    # 棋子外圈应连续较黑，内部应有字，外侧通常为纸张浅色背景。
    return 0.55 * min(ring_dark / 0.35, 1.0) + 0.25 * min(inner_dark / 0.22, 1.0) + 0.20 * outer_light


def find_circle_candidates(frame_bgr: np.ndarray, max_dimension: int = 1600) -> list[CircleCandidate]:
    height, width = frame_bgr.shape[:2]
    scale = min(1.0, max_dimension / max(height, width))
    if scale < 1.0:
        resized = cv2.resize(frame_bgr, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    else:
        resized = frame_bgr
    base_gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(base_gray, (7, 7), 1.4)
    minimum = min(gray.shape)
    # 整页拍摄时棋子半径约占页面短边 8%~10%。下限过小会把汉字局部笔画
    # 误当成圆，上限过大会生成跨越相邻棋子的假圆。
    # 720p 视频拉远时棋子半径会降至短边的 3.5%~5%，原 5.5% 下限会漏掉远处棋子。
    min_radius = max(12, round(minimum * 0.035))
    max_radius = max(min_radius + 2, round(minimum * 0.12))
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(20, round(minimum * 0.075)),
        param1=120,
        param2=42,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    original_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    def select_candidates(circle_array: np.ndarray | None) -> list[CircleCandidate]:
        if circle_array is None:
            return []
        candidates: list[CircleCandidate] = []
        for cx, cy, radius in circle_array[0]:
            x, y, r = float(cx / scale), float(cy / scale), float(radius / scale)
            quality = _circle_quality(original_gray, x, y, r)
            if quality >= 0.48:
                candidates.append(CircleCandidate(x, y, r, quality))

        if len(candidates) >= 4:
            median_radius = float(np.median([item.radius for item in candidates]))
            # 斜拍与手机电子防抖会让同页上下棋子半径相差约 40%；保留较宽范围，
            # 再由圆心去重和 3/3/3/3/2 版面结构排除异常大圆。
            candidates = [item for item in candidates if 0.65 * median_radius <= item.radius <= 1.45 * median_radius]

        candidates.sort(key=lambda item: item.quality, reverse=True)
        selected: list[CircleCandidate] = []
        for candidate in candidates:
            duplicate = False
            for kept in selected:
                distance = float(np.hypot(candidate.x - kept.x, candidate.y - kept.y))
                if distance < 0.55 * min(candidate.radius, kept.radius):
                    duplicate = True
                    break
            if not duplicate:
                selected.append(candidate)
        return selected

    selected = select_candidates(circles)
    # 质量筛选后候选不足时才启用 CLAHE，避免正常帧重复计算。CLAHE 仅服务于圆检测，
    # 不改动分类用的彩色原图，因此不会破坏偏暗的红墨颜色。
    if len(selected) < 14:
        enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(base_gray)
        enhanced = cv2.GaussianBlur(enhanced, (7, 7), 1.4)
        enhanced_circles = cv2.HoughCircles(
            enhanced,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=max(20, round(minimum * 0.075)),
            param1=120,
            # 回退阶段适当降低累加器阈值以召回远距离小圆；后续版面约束会排除假圆。
            param2=38,
            minRadius=min_radius,
            maxRadius=max_radius,
        )
        if circles is None:
            circles = enhanced_circles
        elif enhanced_circles is not None:
            circles = np.concatenate((circles, enhanced_circles), axis=1)
        selected = select_candidates(circles)
    return selected


def _candidate_crop(frame_bgr: np.ndarray, candidate: CircleCandidate, factor: float | None = None) -> Image.Image | None:
    height, width = frame_bgr.shape[:2]
    if factor is None:
        # 720p 远距离小目标需要更多圆外留白以匹配训练尺度；高分辨率照片保留紧裁剪，
        # 避免过多背景压缩汉字细节。80~140 像素之间线性过渡。
        factor = float(np.interp(candidate.radius, (80.0, 140.0), (1.48, 1.13)))
    half = int(round(candidate.radius * factor))
    center_x, center_y = int(round(candidate.x)), int(round(candidate.y))
    x1, x2 = max(0, center_x - half), min(width, center_x + half)
    y1, y2 = max(0, center_y - half), min(height, center_y + half)
    if x2 - x1 < 16 or y2 - y1 < 16:
        return None
    crop = cv2.cvtColor(frame_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
    return Image.fromarray(crop)


def _red_ink_score(image: Image.Image) -> float:
    """计算墨迹相对纸张背景的红色增量，抵消暖色灯光和白平衡偏色。"""
    rgb = np.asarray(image, dtype=np.float32)
    height, width = rgb.shape[:2]
    yy, xx = np.mgrid[:height, :width]
    inside = np.hypot(xx - width / 2, yy - height / 2) < min(height, width) * 0.46
    pixels = rgb[inside]
    if not pixels.size:
        return 0.0
    luminance = pixels.mean(axis=1)
    chroma = pixels[:, 0] - (pixels[:, 1] + pixels[:, 2]) / 2
    dark_limit = np.percentile(luminance, 40)
    light_limit = np.percentile(luminance, 65)
    foreground = chroma[luminance < dark_limit]
    background = chroma[luminance > light_limit]
    if not foreground.size or not background.size:
        return 0.0
    return float(foreground.mean() - background.mean())


def _ink_is_red(image: Image.Image) -> bool:
    # 单棋子通用模式使用保守阈值；完整样本页会由版面结构提供更可靠的阵营先验。
    return _red_ink_score(image) > 3.0


def _normalize_piece_crop(image: Image.Image, force_red: bool | None = None) -> Image.Image:
    """将光照、曝光和褪色后的棋子恢复为训练样本的白底标准墨色。"""
    rgb = np.asarray(image, dtype=np.float32)
    gray = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    background = float(np.percentile(gray, 82))
    darkest = float(np.percentile(gray, 4))
    alpha = np.clip((background - gray) / max(background - darkest, 25.0), 0.0, 1.0)
    # 略微增强浅色抗锯齿边缘，避免曝光后细笔画消失。
    alpha = np.power(alpha, 0.82)
    is_red = _ink_is_red(image) if force_red is None else force_red
    ink = np.array([205, 8, 8] if is_red else [15, 15, 15], dtype=np.float32)
    normalized = 250.0 * (1.0 - alpha[..., None]) + ink * alpha[..., None]
    return Image.fromarray(np.clip(normalized, 0, 255).astype(np.uint8))


def _sample_page_rows(candidates: list[CircleCandidate]) -> list[list[int]] | None:
    """识别赛方固定 3/3/3/3/2 样本页布局，返回从上到下、从左到右的索引。"""
    if len(candidates) != 14:
        return None
    median_radius = float(np.median([item.radius for item in candidates]))
    rows: list[list[int]] = []
    row_centers: list[float] = []
    for index in sorted(range(len(candidates)), key=lambda item: candidates[item].y):
        y = candidates[index].y
        if rows and abs(y - row_centers[-1]) <= 0.70 * median_radius:
            rows[-1].append(index)
            row_centers[-1] = float(np.mean([candidates[item].y for item in rows[-1]]))
        else:
            rows.append([index])
            row_centers.append(y)
    if [len(row) for row in rows] != [3, 3, 3, 3, 2]:
        return None
    return [sorted(row, key=lambda item: candidates[item].x) for row in rows]


def _sample_page_factions(candidates: list[CircleCandidate]) -> list[bool] | None:
    """识别赛方固定 3/3/3/3/2 样本页布局，返回每个候选是否为红方。"""
    ordered_rows = _sample_page_rows(candidates)
    if ordered_rows is None:
        return None
    factions = [False] * len(candidates)
    # 前两行均为黑方；第三行左侧为黑卒、其余为红方；后两行均为红方。
    factions[ordered_rows[2][1]] = True
    factions[ordered_rows[2][2]] = True
    for row in ordered_rows[3:]:
        for index in row:
            factions[index] = True
    return factions


def _select_sample_page_candidates(candidates: list[CircleCandidate]) -> list[CircleCandidate]:
    """候选略多于 14 时，找出最符合固定样本页结构的 14 个圆。"""
    if len(candidates) < 14 or len(candidates) > 17:
        return candidates
    if len(candidates) == 14:
        return candidates
    best: tuple[float, list[CircleCandidate]] | None = None
    for indices in combinations(range(len(candidates)), 14):
        subset = [candidates[index] for index in indices]
        rows = _sample_page_rows(subset)
        if rows is None:
            continue
        radii = np.array([item.radius for item in subset], dtype=np.float32)
        radius_consistency = float(radii.std() / max(radii.mean(), 1.0))
        # 每个三枚棋子的印刷行均近似等间距。这个约束可排除横跨相邻棋子的
        # 大假圆；只看圆质量和半径时，这类假圆可能比真实外圈更“规则”。
        spacing_imbalance = float(
            np.mean(
                [
                    abs((subset[row[1]].x - subset[row[0]].x) - (subset[row[2]].x - subset[row[1]].x))
                    / max(subset[row[2]].x - subset[row[0]].x, 1.0)
                    for row in rows
                    if len(row) == 3
                ]
            )
        )
        score = float(sum(item.quality for item in subset)) - radius_consistency - 2.0 * spacing_imbalance
        if best is None or score > best[0]:
            best = (score, subset)
    return best[1] if best is not None else candidates


def _unique_class_assignment(probabilities: np.ndarray, factions: list[bool]) -> list[int]:
    """在已确认的样本页内，用动态规划实现每方七类各一枚的最大似然分配。"""
    assignments = [-1] * len(factions)
    for is_red, class_offset in ((False, 0), (True, 7)):
        candidate_indices = [index for index, value in enumerate(factions) if value == is_red]
        # state: 已使用类别位图 -> (累计对数概率, 已选局部类别序列)
        states: dict[int, tuple[float, list[int]]] = {0: (0.0, [])}
        for candidate_index in candidate_indices:
            next_states: dict[int, tuple[float, list[int]]] = {}
            for mask, (score, chosen) in states.items():
                for local_class in range(7):
                    bit = 1 << local_class
                    if mask & bit:
                        continue
                    probability = max(float(probabilities[candidate_index, class_offset + local_class]), 1e-9)
                    new_mask = mask | bit
                    option = (score + float(np.log(probability)), chosen + [local_class])
                    if new_mask not in next_states or option[0] > next_states[new_mask][0]:
                        next_states[new_mask] = option
            states = next_states
        _, chosen = states[(1 << 7) - 1]
        for candidate_index, local_class in zip(candidate_indices, chosen, strict=True):
            assignments[candidate_index] = class_offset + local_class
    return assignments


class ChessRecognizer:
    def __init__(self, model_path: str | Path, device: str = "auto", confidence_threshold: float = 0.65) -> None:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("请求使用 CUDA，但当前环境不可用")
        self.device = torch.device(device)
        self.model, self.checkpoint = load_checkpoint(str(model_path), device=self.device)
        self.input_size = int(self.checkpoint.get("input_size", 96))
        self.confidence_threshold = confidence_threshold

    @torch.inference_mode()
    def classify_candidates(self, frame_bgr: np.ndarray, candidates: list[CircleCandidate]) -> tuple[list[Detection], np.ndarray]:
        candidates = _select_sample_page_candidates(candidates)
        crops: list[Image.Image] = []
        valid_candidates: list[CircleCandidate] = []
        for candidate in candidates:
            crop = _candidate_crop(frame_bgr, candidate)
            if crop is not None:
                crops.append(crop)
                valid_candidates.append(candidate)
        if not crops:
            return [], np.empty((0, len(PIECE_CLASSES)), dtype=np.float32)
        page_factions = _sample_page_factions(valid_candidates)
        tensors = [
            tensor_from_image(
                _normalize_piece_crop(crop, force_red=page_factions[index] if page_factions is not None else None),
                input_size=self.input_size,
            )
            for index, crop in enumerate(crops)
        ]
        inputs = torch.stack(tensors).to(self.device)
        probabilities = torch.softmax(self.model(inputs), dim=1).cpu().numpy()
        assigned_classes = _unique_class_assignment(probabilities, page_factions) if page_factions is not None else None
        # 远距离视频中炮类在带较多留白的 1.48× 裁剪下最稳定，但“車”的细密笔画
        # 会被压缩得过小。首轮版面分配找到红車后，只对这一枚小目标以 1.24×
        # 紧裁剪复核；每帧仅增加一个 96×96 样本，远轻于全页多尺度 TTA。
        if assigned_classes is not None:
            refined_indices = [
                index
                for index, class_index in enumerate(assigned_classes)
                if class_index == 12 and valid_candidates[index].radius < 140.0
            ]
            if refined_indices:
                refined_tensors = []
                for index in refined_indices:
                    refined_crop = _candidate_crop(frame_bgr, valid_candidates[index], factor=1.24)
                    if refined_crop is None:
                        continue
                    refined_tensors.append(
                        tensor_from_image(
                            _normalize_piece_crop(refined_crop, force_red=True),
                            input_size=self.input_size,
                        )
                    )
                if len(refined_tensors) == len(refined_indices):
                    refined_inputs = torch.stack(refined_tensors).to(self.device)
                    refined_probabilities = torch.softmax(self.model(refined_inputs), dim=1).cpu().numpy()
                    for index, refined_probability in zip(refined_indices, refined_probabilities, strict=True):
                        probabilities[index] = refined_probability
                    assigned_classes = _unique_class_assignment(probabilities, page_factions)
        detections: list[Detection] = []
        height, width = frame_bgr.shape[:2]
        for index, (candidate, probability) in enumerate(zip(valid_candidates, probabilities, strict=True)):
            class_index = assigned_classes[index] if assigned_classes is not None else int(probability.argmax())
            confidence = float(probability[class_index])
            # 完整样本页同时满足圆形、版面和每类唯一约束，可接纳因低分辨率而降低的概率。
            threshold = 0.0 if assigned_classes is not None else self.confidence_threshold
            if confidence < threshold:
                continue
            half = int(round(candidate.radius * 1.10))
            detections.append(
                Detection(
                    x1=max(0, int(round(candidate.x)) - half),
                    y1=max(0, int(round(candidate.y)) - half),
                    x2=min(width - 1, int(round(candidate.x)) + half),
                    y2=min(height - 1, int(round(candidate.y)) + half),
                    center_x=candidate.x,
                    center_y=candidate.y,
                    radius=candidate.radius,
                    class_index=class_index,
                    confidence=confidence,
                    circle_quality=candidate.quality,
                )
            )
        # 赛方样本每类一枚：同类重复候选仅保留综合分最高者。
        best_by_class: dict[int, Detection] = {}
        for detection in detections:
            score = detection.confidence * (0.75 + 0.25 * detection.circle_quality)
            previous = best_by_class.get(detection.class_index)
            previous_score = previous.confidence * (0.75 + 0.25 * previous.circle_quality) if previous else -1.0
            if score > previous_score:
                best_by_class[detection.class_index] = detection
        return sorted(best_by_class.values(), key=lambda item: (item.center_y, item.center_x)), probabilities

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        candidates = find_circle_candidates(frame_bgr)
        direct_detections, _ = self.classify_candidates(frame_bgr, candidates)
        if len(direct_detections) >= 12:
            return direct_detections

        rectification = rectify_printed_page(frame_bgr)
        if rectification is None:
            return direct_detections
        rectified, inverse = rectification
        rectified_candidates = find_circle_candidates(rectified)
        rectified_detections, _ = self.classify_candidates(rectified, rectified_candidates)
        if len(rectified_detections) <= len(direct_detections):
            return direct_detections

        height, width = frame_bgr.shape[:2]
        mapped: list[Detection] = []
        for detection in rectified_detections:
            corners = np.float32(
                [[[detection.x1, detection.y1], [detection.x2, detection.y1], [detection.x2, detection.y2], [detection.x1, detection.y2]]]
            )
            projected = cv2.perspectiveTransform(corners, inverse)[0]
            x1 = max(0, int(np.floor(projected[:, 0].min())))
            y1 = max(0, int(np.floor(projected[:, 1].min())))
            x2 = min(width - 1, int(np.ceil(projected[:, 0].max())))
            y2 = min(height - 1, int(np.ceil(projected[:, 1].max())))
            mapped.append(
                Detection(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    center_x=(x1 + x2) / 2,
                    center_y=(y1 + y2) / 2,
                    radius=(x2 - x1 + y2 - y1) / 4,
                    class_index=detection.class_index,
                    confidence=detection.confidence,
                    circle_quality=detection.circle_quality,
                )
            )
        return sorted(mapped, key=lambda item: (item.center_y, item.center_x))
