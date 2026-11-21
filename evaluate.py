#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import torch
from PIL import Image

from chessvision.augment import tensor_from_image
from chessvision.dataset import load_manifest
from chessvision.detection import ChessRecognizer
from chessvision.labels import PIECE_CLASSES
from chessvision.model import load_checkpoint
from chessvision.render import draw_detections


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估棋子分类模型和静态目标检测")
    parser.add_argument("--model", default="models/best_model.pt")
    parser.add_argument("--data", default="data")
    parser.add_argument("--image", default="../象棋图片.jpg")
    parser.add_argument("--output", default="outputs/evaluation")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--confidence", type=float, default=0.65)
    return parser.parse_args()


@torch.inference_mode()
def exact_classification(model_path: Path, data_dir: Path, device_name: str) -> list[dict]:
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    model, checkpoint = load_checkpoint(str(model_path), device=device)
    manifest = load_manifest(data_dir)
    results: list[dict] = []
    for item in manifest:
        image = Image.open(data_dir / item["file"]).convert("RGB")
        tensor = tensor_from_image(image, int(checkpoint.get("input_size", 96))).unsqueeze(0).to(device)
        probabilities = torch.softmax(model(tensor), dim=1)[0]
        prediction = int(probabilities.argmax().item())
        results.append(
            {
                "truth": item["index"],
                "truth_display": item["display"],
                "prediction": prediction,
                "prediction_display": PIECE_CLASSES[prediction].display,
                "confidence": float(probabilities[prediction].item()),
                "correct": prediction == item["index"],
            }
        )
    return results


def main() -> int:
    args = parse_args()
    model_path, data_dir, image_path = Path(args.model).resolve(), Path(args.data).resolve(), Path(args.image).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    exact_results = exact_classification(model_path, data_dir, args.device)
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise FileNotFoundError(f"无法读取测试图片: {image_path}")
    recognizer = ChessRecognizer(model_path, device=args.device, confidence_threshold=args.confidence)
    detections = recognizer.detect(frame)
    annotated = draw_detections(frame, detections)
    annotated_path = output_dir / "annotated_test.jpg"
    cv2.imwrite(str(annotated_path), annotated)

    found_classes = {item.class_index for item in detections}
    report = {
        "model": str(model_path),
        "image": str(image_path),
        "device": str(recognizer.device),
        "exact_classification": {
            "correct": sum(item["correct"] for item in exact_results),
            "total": len(exact_results),
            "accuracy": sum(item["correct"] for item in exact_results) / len(exact_results),
            "items": exact_results,
        },
        "image_detection": {
            "count": len(detections),
            "unique_classes": len(found_classes),
            "missing_classes": [item.display for item in PIECE_CLASSES if item.index not in found_classes],
            "detections": [item.to_dict() for item in detections],
            "annotated_image": str(annotated_path),
        },
    }
    (output_dir / "evaluation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if len(detections) == 14 and all(item["correct"] for item in exact_results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
