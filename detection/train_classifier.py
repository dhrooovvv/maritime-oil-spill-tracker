"""Fine-tune EfficientNetB0 on labelled oil/look-alike candidate crops.

Expected directory structure:

data/classifier/
    train/look_alike/
    train/oil/
    val/look_alike/
    val/oil/

This script is deliberately separate from Streamlit. It does not fabricate
data or report accuracy until labelled images are supplied and training runs.
"""

import argparse
from pathlib import Path

try:
    import numpy as np
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from torchvision import datasets, models, transforms
except Exception as exc:  # pragma: no cover - depends on the local environment
    raise SystemExit(
        "Training requires torch, torchvision, and their dependencies. "
        f"Import error: {exc}"
    ) from exc

try:
    from .classifier import CLASS_NAMES, build_model, select_device
except ImportError:
    from classifier import CLASS_NAMES, build_model, select_device


def parse_args():
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=project_root / "data" / "classifier",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "models" / "efficientnet_b0_oil_classifier.pth",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def make_datasets(data_dir):
    weights = models.EfficientNet_B0_Weights.DEFAULT
    base_transform = weights.transforms()
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            base_transform,
        ]
    )
    train_dataset = datasets.ImageFolder(data_dir / "train", transform=train_transform)
    val_dataset = datasets.ImageFolder(data_dir / "val", transform=base_transform)

    expected = {"look_alike": 0, "oil": 1}
    if train_dataset.class_to_idx != expected or val_dataset.class_to_idx != expected:
        raise ValueError(
            "Dataset folders must be named look_alike and oil so the class mapping is "
            f"{dict(enumerate(CLASS_NAMES))}. Found train={train_dataset.class_to_idx}, "
            f"val={val_dataset.class_to_idx}."
        )
    return train_dataset, val_dataset


def calculate_metrics(targets, predictions):
    confusion = np.zeros((2, 2), dtype=np.int64)
    for target, prediction in zip(targets, predictions):
        confusion[int(target), int(prediction)] += 1

    true_positive = confusion[1, 1]
    false_positive = confusion[0, 1]
    false_negative = confusion[1, 0]
    total = confusion.sum()
    accuracy = float(np.trace(confusion) / total) if total else 0.0
    precision = float(true_positive / (true_positive + false_positive)) if true_positive + false_positive else 0.0
    recall = float(true_positive / (true_positive + false_negative)) if true_positive + false_negative else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return {
        "accuracy": accuracy,
        "precision_oil": precision,
        "recall_oil": recall,
        "f1_oil": f1,
        "confusion_matrix": confusion.tolist(),
    }


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    targets = []
    predictions = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        if training:
            optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)
        if training:
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item()) * labels.size(0)
        targets.extend(labels.detach().cpu().numpy().tolist())
        predictions.extend(logits.argmax(dim=1).detach().cpu().numpy().tolist())

    metrics = calculate_metrics(targets, predictions)
    dataset_size = len(loader.dataset)
    metrics["loss"] = total_loss / dataset_size if dataset_size else 0.0
    return metrics


def main():
    args = parse_args()
    train_dataset, val_dataset = make_datasets(args.data_dir)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = select_device()
    model = build_model(use_imagenet_weights=not args.no_pretrained)
    if args.freeze_backbone:
        for parameter in model.features.parameters():
            parameter.requires_grad = False
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
    )

    best_f1 = -1.0
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        val_metrics = run_epoch(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train loss={train_metrics['loss']:.4f} | "
            f"val loss={val_metrics['loss']:.4f} | "
            f"val accuracy={val_metrics['accuracy']:.3f} | "
            f"val precision={val_metrics['precision_oil']:.3f} | "
            f"val recall={val_metrics['recall_oil']:.3f} | "
            f"val F1={val_metrics['f1_oil']:.3f}"
        )

        if val_metrics["f1_oil"] >= best_f1:
            best_f1 = val_metrics["f1_oil"]
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": list(CLASS_NAMES),
                    "epoch": epoch,
                    "validation_metrics": val_metrics,
                },
                args.output,
            )

    print(f"Saved best checkpoint to {args.output}")


if __name__ == "__main__":
    main()
