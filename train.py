#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/chess_matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from chessvision.dataset import AugmentedChessDataset, ExactChessDataset, load_manifest
from chessvision.labels import PIECE_CLASSES
from chessvision.metrics import classification_metrics
from chessvision.model import ChessCNN, load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练 14 类象棋棋子 CNN")
    parser.add_argument("--data", default="data", help="prepare_data.py 生成的目录")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--output", default="models")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--epochs", type=int, default=None, help="覆盖配置文件中的训练轮数")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求使用 CUDA，但 torch.cuda.is_available() 为 False")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def evaluate_loader(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module) -> tuple[float, dict, np.ndarray]:
    model.eval()
    confusion = np.zeros((len(PIECE_CLASSES), len(PIECE_CLASSES)), dtype=np.int64)
    loss_sum = 0.0
    count = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss_sum += float(loss.item()) * targets.numel()
        count += targets.numel()
        predictions = logits.argmax(dim=1)
        for truth, prediction in zip(targets.cpu().numpy(), predictions.cpu().numpy(), strict=True):
            confusion[int(truth), int(prediction)] += 1
    return loss_sum / max(count, 1), classification_metrics(confusion), confusion


def plot_history(history: list[dict], output: Path) -> None:
    epochs = [item["epoch"] for item in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(epochs, [item["train_loss"] for item in history], label="train")
    axes[0].plot(epochs, [item["val_loss"] for item in history], label="validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(epochs, [item["val_accuracy"] for item in history], label="accuracy")
    axes[1].plot(epochs, [item["val_macro_f1"] for item in history], label="macro F1")
    axes[1].set_ylim(0, 1.02)
    axes[1].set_title("Validation metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data).resolve()
    config_path = Path(args.config).resolve()
    output_dir = Path(args.output).resolve()
    if not (data_dir / "manifest.json").exists():
        print("错误：未找到 data/manifest.json，请先运行 prepare_data.py", file=sys.stderr)
        return 2
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if args.epochs is not None:
        config["epochs"] = args.epochs
    seed_everything(int(config["seed"]))
    device = choose_device(args.device)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = AugmentedChessDataset(
        data_dir,
        samples_per_class=int(config["train_samples_per_class"]),
        input_size=int(config["input_size"]),
        seed=int(config["seed"]),
        training=True,
    )
    val_dataset = AugmentedChessDataset(
        data_dir,
        samples_per_class=int(config["val_samples_per_class"]),
        input_size=int(config["input_size"]),
        seed=int(config["seed"]) + 900001,
        training=False,
    )
    exact_dataset = ExactChessDataset(data_dir, input_size=int(config["input_size"]))
    pin_memory = device.type == "cuda"
    workers = min(int(config["num_workers"]), max((os.cpu_count() or 2) // 2, 1))
    loader_args = {
        "batch_size": int(config["batch_size"]),
        "num_workers": workers,
        "pin_memory": pin_memory,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_args)
    exact_loader = DataLoader(exact_dataset, shuffle=False, **loader_args)

    model = ChessCNN(num_classes=len(PIECE_CLASSES), dropout=float(config["dropout"])).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.02)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(config["epochs"]), eta_min=1e-5)
    scaler = GradScaler("cuda", enabled=device.type == "cuda")

    print(f"训练设备: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"训练样本/轮: {len(train_dataset)}, 验证样本: {len(val_dataset)}, 批大小: {config['batch_size']}")

    best_f1 = -1.0
    best_epoch = 0
    stale_epochs = 0
    history: list[dict] = []
    started = time.time()
    checkpoint_path = output_dir / "best_model.pt"

    for epoch in range(1, int(config["epochs"]) + 1):
        train_dataset.set_epoch(epoch)
        model.train()
        running_loss = 0.0
        seen = 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch:02d}", leave=False)
        for inputs, targets in progress:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(inputs)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.item()) * targets.numel()
            seen += targets.numel()
            progress.set_postfix(loss=f"{running_loss / seen:.4f}")
        scheduler.step()

        train_loss = running_loss / max(seen, 1)
        val_loss, val_metrics, confusion = evaluate_loader(model, val_loader, device, criterion)
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(record)
        print(
            f"Epoch {epoch:02d}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} val_f1={val_metrics['macro_f1']:.4f}"
        )
        if val_metrics["macro_f1"] > best_f1 + 1e-5:
            best_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "num_classes": len(PIECE_CLASSES),
                    "dropout": float(config["dropout"]),
                    "input_size": int(config["input_size"]),
                    "classes": [item.to_dict() for item in PIECE_CLASSES],
                    "config": config,
                    "best_epoch": best_epoch,
                    "best_val_macro_f1": best_f1,
                },
                checkpoint_path,
            )
            np.savetxt(output_dir / "best_confusion_matrix.csv", confusion, fmt="%d", delimiter=",")
        else:
            stale_epochs += 1
            if stale_epochs >= int(config["early_stopping_patience"]):
                print(f"早停：连续 {stale_epochs} 轮未提升")
                break

    (output_dir / "training_history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output_dir / "training_history.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    plot_history(history, output_dir / "training_curves.png")

    best_model, checkpoint = load_checkpoint(str(checkpoint_path), device=device)
    exact_loss, exact_metrics, exact_confusion = evaluate_loader(best_model, exact_loader, device, criterion)
    np.savetxt(output_dir / "exact_confusion_matrix.csv", exact_confusion, fmt="%d", delimiter=",")
    scripted = torch.jit.script(best_model.cpu())
    scripted.save(str(output_dir / "chess_cnn.ts"))
    report = {
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "best_epoch": checkpoint["best_epoch"],
        "best_val_macro_f1": checkpoint["best_val_macro_f1"],
        "exact_loss": exact_loss,
        "exact_metrics": exact_metrics,
        "elapsed_seconds": round(time.time() - started, 2),
    }
    (output_dir / "training_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
