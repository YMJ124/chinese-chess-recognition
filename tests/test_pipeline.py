from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import torch

from chessvision.docx_extract import extract_dataset
from chessvision.labels import PIECE_CLASSES
from chessvision.model import ChessCNN


PROJECT_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_DIR = PROJECT_DIR.parent


class PipelineTests(unittest.TestCase):
    def test_label_contract(self) -> None:
        self.assertEqual(len(PIECE_CLASSES), 14)
        self.assertEqual(len({item.key for item in PIECE_CLASSES}), 14)
        self.assertEqual({item.faction for item in PIECE_CLASSES}, {"红", "黑"})
        self.assertEqual(PIECE_CLASSES[0].display, "黑-将-10分")
        self.assertEqual(PIECE_CLASSES[-1].display, "红-兵-1分")

    def test_model_output_shape(self) -> None:
        model = ChessCNN()
        output = model(torch.zeros(2, 3, 96, 96))
        self.assertEqual(tuple(output.shape), (2, 14))

    def test_docx_extraction(self) -> None:
        docx = WORKSPACE_DIR / "赛题一所需资料" / "象棋样本.docx"
        with tempfile.TemporaryDirectory() as temporary:
            manifest = extract_dataset(docx, temporary)
            self.assertEqual(len(manifest), 14)
            for index, item in enumerate(manifest):
                self.assertEqual(item["index"], index)
                self.assertTrue((Path(temporary) / item["file"]).exists())
                self.assertGreater(item["width"], 150)
                self.assertGreater(item["height"], 150)

    def test_trained_static_detection(self) -> None:
        model = PROJECT_DIR / "models" / "best_model.pt"
        image_path = WORKSPACE_DIR / "象棋图片.jpg"
        if not model.exists() or not image_path.exists():
            self.skipTest("训练模型或静态测试图尚未生成")
        from chessvision.detection import ChessRecognizer

        frame = cv2.imread(str(image_path))
        recognizer = ChessRecognizer(model, device="cpu", confidence_threshold=0.65)
        detections = recognizer.detect(frame)
        self.assertEqual(len(detections), 14)
        self.assertEqual({item.class_index for item in detections}, set(range(14)))


if __name__ == "__main__":
    unittest.main()
