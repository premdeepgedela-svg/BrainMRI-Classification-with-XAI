from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from src.datasets.dataloader import create_dataloaders
from src.models.cnn import BrainMRICNN
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
        num_workers=0,
    )

    # ==========================================
    # Model
    # ==========================================

    model = BrainMRICNN()

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001,
    )

    trainer = Trainer(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
    )

    # ==========================================
    # Resume Training
    # ==========================================

    epochs = 5

    checkpoint_path = Path("results/checkpoints/latest.pth")

    start_epoch = 0
    best_accuracy = 0.0

    if checkpoint_path.exists():

        print("\nLoading latest checkpoint...")

        start_epoch, best_accuracy = trainer.load_checkpoint(
            checkpoint_path
        )

        print(f"Resuming from Epoch {start_epoch}")

    # ==========================================
    # Training
    # ==========================================

    for epoch in range(start_epoch, epochs):

        print(f"\nEpoch {epoch + 1}/{epochs}")

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

        # Save every epoch

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

    print("\nTraining Complete.")


if __name__ == "__main__":
    main()