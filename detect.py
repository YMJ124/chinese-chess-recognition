#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2

from chessvision.detection import ChessRecognizer
from chessvision.render import draw_detections
from chessvision.tracker import TemporalTracker


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="象棋棋子图片、视频与摄像头检测")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--source", help="图片或视频路径")
    group.add_argument("--camera", type=int, help="摄像头编号，如 0")
    parser.add_argument("--model", default="models/best_model.pt")
    parser.add_argument("--output", default=None, help="标注图片或视频路径")
    parser.add_argument("--log", default=None, help="JSONL 检测记录路径")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--confidence", type=float, default=0.65)
    parser.add_argument("--show", action="store_true", help="显示检测窗口，按 q 退出")
    parser.add_argument("--keep-mp4v", action="store_true", help="不转码 H.264，直接保留 OpenCV MP4V 视频")
    return parser.parse_args()


def detection_signature(detections) -> tuple:
    # 置信度会逐帧微幅波动，不应因此每帧刷新终端。
    return tuple(sorted(item.class_index for item in detections))


def print_detections(prefix: str, detections) -> None:
    # 按赛题文档规定的输出示例显示：红 - 帅 - 10 分 / 黑 - 将 - 10 分。
    # 置信度仍保存在 JSON/JSONL 中，终端仅输出规定格式的类别与分值。
    labels = " / ".join(
        f"{item.piece.faction} - {item.piece.name} - {item.piece.score} 分"
        for item in detections
    )
    print(f"{prefix} 棋子数量={len(detections)}" + (f" | {labels}" if labels else ""), flush=True)


def process_image(args: argparse.Namespace, recognizer: ChessRecognizer, source: Path) -> int:
    frame = cv2.imread(str(source))
    if frame is None:
        print(f"错误：无法读取图片 {source}", file=sys.stderr)
        return 2
    detections = recognizer.detect(frame)
    annotated = draw_detections(frame, detections)
    output = Path(args.output) if args.output else Path("outputs") / f"{source.stem}_detected.jpg"
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), annotated)
    log_path = Path(args.log) if args.log else output.with_suffix(".json")
    log_path.write_text(json.dumps([item.to_dict() for item in detections], ensure_ascii=False, indent=2), encoding="utf-8")
    print_detections("[图片]", detections)
    print(f"标注图片: {output.resolve()}")
    print(f"检测记录: {log_path.resolve()}")
    if args.show:
        cv2.imshow("象棋棋子识别", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0


def transcode_to_h264(intermediate_path: Path, output_path: Path) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        print("警告：未找到 FFmpeg，将保留 MP4V 格式视频。", file=sys.stderr)
        return False
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(intermediate_path),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-profile:v",
        "high",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    try:
        subprocess.run(command, check=True)
        return True
    except subprocess.CalledProcessError as error:
        print(f"警告：H.264 转码失败（退出码 {error.returncode}），将保留 MP4V 视频。", file=sys.stderr)
        return False


def process_video(args: argparse.Namespace, recognizer: ChessRecognizer) -> int:
    source = args.camera if args.camera is not None else str(Path(args.source).resolve())
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        print(f"错误：无法打开视频源 {source}", file=sys.stderr)
        return 2
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1:
        fps = 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path = Path(args.output) if args.output else Path("outputs") / "detected_video.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    intermediate_path = output_path if args.keep_mp4v else output_path.with_name(f".{output_path.stem}.mp4v-temporary.mp4")
    writer = cv2.VideoWriter(str(intermediate_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        print(f"错误：无法创建输出视频 {intermediate_path}", file=sys.stderr)
        return 2
    log_path = Path(args.log) if args.log else output_path.with_suffix(".jsonl")
    tracker = TemporalTracker()
    frame_index = 0
    last_print = 0.0
    previous_signature = None
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log_file:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            raw = recognizer.detect(frame)
            stable = tracker.update(raw)
            annotated = draw_detections(frame, stable)
            writer.write(annotated)
            timestamp = frame_index / fps
            log_file.write(json.dumps({"frame": frame_index, "time": timestamp, "detections": [item.to_dict() for item in stable]}, ensure_ascii=False) + "\n")
            now = time.perf_counter()
            signature = detection_signature(stable)
            if signature != previous_signature or now - last_print >= 1.0:
                print_detections(f"[帧 {frame_index} / {timestamp:.2f}s]", stable)
                previous_signature = signature
                last_print = now
            if args.show:
                cv2.imshow("象棋棋子识别", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            frame_index += 1
    elapsed = max(time.perf_counter() - started, 1e-6)
    capture.release()
    writer.release()
    if args.show:
        cv2.destroyAllWindows()
    codec = "MP4V"
    if not args.keep_mp4v:
        print("正在转码为 H.264（libx264 / yuv420p）…")
        if transcode_to_h264(intermediate_path, output_path):
            codec = "H.264"
            intermediate_path.unlink(missing_ok=True)
        else:
            shutil.move(str(intermediate_path), str(output_path))
    print(f"处理完成: {frame_index} 帧, 平均 {frame_index / elapsed:.2f} FPS")
    print(f"标注视频: {output_path.resolve()} ({codec})")
    print(f"检测记录: {log_path.resolve()}")
    return 0


def main() -> int:
    args = parse_args()
    recognizer = ChessRecognizer(args.model, device=args.device, confidence_threshold=args.confidence)
    print(f"推理设备: {recognizer.device}")
    if args.source and Path(args.source).suffix.lower() in IMAGE_EXTENSIONS:
        return process_image(args, recognizer, Path(args.source).resolve())
    return process_video(args, recognizer)


if __name__ == "__main__":
    raise SystemExit(main())
