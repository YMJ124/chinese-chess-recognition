from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .detection import Detection


@dataclass
class _Track:
    detection: Detection
    hits: int = 1
    misses: int = 0
    confidence_history: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.confidence_history.append(self.detection.confidence)


class TemporalTracker:
    def __init__(
        self,
        smoothing: float = 0.35,
        max_misses: int = 5,
        min_hits: int = 2,
        confidence_window: int = 15,
        confidence_smoothing: float = 0.25,
    ) -> None:
        self.smoothing = smoothing
        self.max_misses = max_misses
        self.min_hits = min_hits
        self.confidence_window = confidence_window
        self.confidence_smoothing = confidence_smoothing
        self.tracks: list[_Track] = []

    def update(self, detections: list[Detection]) -> list[Detection]:
        unmatched = set(range(len(detections)))
        for track in self.tracks:
            best_index = None
            best_distance = float("inf")
            for index in unmatched:
                detection = detections[index]
                if detection.class_index != track.detection.class_index:
                    continue
                distance = float(np.hypot(detection.center_x - track.detection.center_x, detection.center_y - track.detection.center_y))
                if distance < max(detection.radius, track.detection.radius) * 1.4 and distance < best_distance:
                    best_index = index
                    best_distance = distance
            if best_index is None:
                track.misses += 1
                continue
            new = detections[best_index]
            old = track.detection
            alpha = self.smoothing
            for field in ("x1", "y1", "x2", "y2"):
                setattr(old, field, round((1 - alpha) * getattr(old, field) + alpha * getattr(new, field)))
            for field in ("center_x", "center_y", "radius", "circle_quality"):
                setattr(old, field, (1 - alpha) * getattr(old, field) + alpha * getattr(new, field))
            # 概率比位置更容易受单帧压缩和运动模糊影响；15 帧中位数可覆盖
            # 手机快速移动时约 0.3 秒的模糊段，检测框的位置仍按当前帧快速跟随。
            track.confidence_history.append(new.confidence)
            track.confidence_history = track.confidence_history[-self.confidence_window :]
            median_confidence = float(np.median(track.confidence_history))
            old.confidence = (1.0 - self.confidence_smoothing) * old.confidence + self.confidence_smoothing * median_confidence
            track.hits += 1
            track.misses = 0
            unmatched.remove(best_index)
        for index in unmatched:
            self.tracks.append(_Track(detections[index]))
        self.tracks = [track for track in self.tracks if track.misses <= self.max_misses]
        return [track.detection for track in self.tracks if track.hits >= self.min_hits and track.misses <= 2]
