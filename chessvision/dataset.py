from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .augment import array_to_tensor, augment_array, tensor_from_image


def load_manifest(data_dir: str | Path) -> list[dict]:
    data_dir = Path(data_dir)
    return json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))


class AugmentedChessDataset(Dataset):
    def __init__(
        self,
        data_dir: str | Path,
        samples_per_class: int,
        input_size: int = 96,
        seed: int = 42,
        training: bool = True,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.manifest = load_manifest(self.data_dir)
        self.images = [Image.open(self.data_dir / item["file"]).convert("RGB") for item in self.manifest]
        self.samples_per_class = samples_per_class
        self.input_size = input_size
        self.seed = seed
        self.training = training
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.images) * self.samples_per_class

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        label = index % len(self.images)
        epoch_component = self.epoch if self.training else 0
        rng = np.random.default_rng(self.seed + index * 104729 + epoch_component * 1000003)
        severity = 1.0 if self.training else 0.8
        array = augment_array(self.images[label], rng, input_size=self.input_size, severity=severity)
        return array_to_tensor(array), label


class ExactChessDataset(Dataset):
    def __init__(self, data_dir: str | Path, input_size: int = 96) -> None:
        self.data_dir = Path(data_dir)
        self.manifest = load_manifest(self.data_dir)
        self.input_size = input_size

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image = Image.open(self.data_dir / self.manifest[index]["file"]).convert("RGB")
        return tensor_from_image(image, input_size=self.input_size), index
