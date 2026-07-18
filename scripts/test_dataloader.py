from pathlib import Path

from src.datasets.dataloader import create_dataloaders


def main():

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

    print("=" * 60)
    print("DataLoader Test")
    print("=" * 60)

    print("Training batches:", len(train_loader))
    print("Validation batches:", len(val_loader))

    batch = next(iter(train_loader))

    print("\nBatch Image Shape :", batch["image"].shape)
    print("Batch Label Shape :", batch["label"].shape)


if __name__ == "__main__":
    main()