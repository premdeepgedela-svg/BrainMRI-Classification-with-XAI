from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from src.datasets.dataloader import create_dataloaders
from src.models.resnet18 import BrainMRIResNet18
from src.training.trainer import Trainer


def main():

    # ==========================================
    # Device
    # ==========================================

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    # ==========================================
    # Dataset
    # ==========================================

    DATASET_PATH = Path(
        r"C:\AI_Research\BrainMRI-XAI-Robustness\data\raw"
        r"\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
        r"\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
    )

    train_loader, val_loader = create_dataloaders(
        DATASET_PATH,
        batch_size=16,
        num_workers=4,
    )

    # ==========================================
    # Model
    # ==========================================

    model = BrainMRIResNet18()

    # ==========================================
    # Loss Function
    # ==========================================

    criterion = nn.CrossEntropyLoss()

    # ==========================================
    # Optimizer
    # ==========================================

    optimizer = optim.Adam(
        filter(
            lambda p: p.requires_grad,
            model.parameters(),
        ),
        lr=1e-4,
    )

    # ==========================================
    # Trainer
    # ==========================================

    trainer = Trainer(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        checkpoint_dir="results/resnet18/checkpoints",
    )

    # ==========================================
    # GPU Verification
    # ==========================================

    print("\n" + "=" * 60)
    print("GPU VERIFICATION")
    print("=" * 60)

    print(f"Selected Device      : {device}")
    print(f"CUDA Available       : {torch.cuda.is_available()}")
    print(f"CUDA Version         : {torch.version.cuda}")
    print(f"GPU Count            : {torch.cuda.device_count()}")

    if torch.cuda.is_available():
        print(f"Current GPU          : {torch.cuda.current_device()}")
        print(f"GPU Name             : {torch.cuda.get_device_name(0)}")
        print(f"Model Device         : {next(model.parameters()).device}")
        print(f"GPU Memory Allocated : {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"GPU Memory Reserved  : {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

    print("=" * 60)

    # ==========================================
    # Training
    # ==========================================

    epochs = 5

    best_accuracy = 0.0

    for epoch in range(epochs):

        print("\n" + "=" * 60)
        print(f"Epoch {epoch + 1}/{epochs}")
        print("=" * 60)

        train_loss, train_acc = trainer.train_one_epoch()

        val_loss, val_acc = trainer.validate()

        print(f"Train Loss : {train_loss:.4f}")
        print(f"Train Acc  : {train_acc:.2f}%")

        print(f"Val Loss   : {val_loss:.4f}")
        print(f"Val Acc    : {val_acc:.2f}%")

        # Save latest checkpoint
        trainer.save_checkpoint(
            epoch + 1,
            best_accuracy,
            "latest.pth",
        )

        # Save epoch checkpoint
        trainer.save_checkpoint(
            epoch + 1,
            best_accuracy,
            f"epoch_{epoch + 1}.pth",
        )

        # Save best model
        if val_acc > best_accuracy:

            best_accuracy = val_acc

            trainer.save_checkpoint(
                epoch + 1,
                best_accuracy,
                "best_model.pth",
            )

            print("✅ Best model updated.")

    print("\n" + "=" * 60)
    print("ResNet-18 Training Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()