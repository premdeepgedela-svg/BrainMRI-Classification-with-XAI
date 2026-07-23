from pathlib import Path
import json

import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score,
)

import torch.nn as nn
import torch.optim as optim

from src.datasets.dataloader import create_dataloaders
from src.models.cnn import BrainMRICNN
from src.training.trainer import Trainer


DATASET_PATH = Path(
    r"C:\AI_Research\BrainMRI-XAI-Robustness\data\raw"
    r"\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
    r"\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
)

RESULTS_DIR = Path("results/evaluation")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    train_loader, val_loader = create_dataloaders(
        DATASET_PATH,
        batch_size=16,
        num_workers=0,
    )

    model = BrainMRICNN()

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters())

    trainer = Trainer(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
    )

    checkpoint = Path("results/checkpoints/best_model.pth")

    trainer.load_checkpoint(checkpoint)

    model.eval()

    predictions = []
    labels = []

    with torch.no_grad():

        for batch in val_loader:

            images = batch["image"].to(device)

            outputs = model(images)

            preds = outputs.argmax(dim=1).cpu().numpy()

            predictions.extend(preds)

            labels.extend(batch["label"].numpy())

    accuracy = accuracy_score(labels, predictions)

    precision = precision_score(labels, predictions)

    recall = recall_score(labels, predictions)

    f1 = f1_score(labels, predictions)

    report = classification_report(labels, predictions)

    matrix = confusion_matrix(labels, predictions)

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
    }

    with open(
        RESULTS_DIR / "metrics.json",
        "w",
    ) as f:

        json.dump(metrics, f, indent=4)

    with open(
        RESULTS_DIR / "classification_report.txt",
        "w",
    ) as f:

        f.write(report)

    pd.DataFrame(
        {
            "True": labels,
            "Predicted": predictions,
        }
    ).to_csv(
        RESULTS_DIR / "predictions.csv",
        index=False,
    )

    disp = ConfusionMatrixDisplay(matrix)

    disp.plot()

    plt.savefig(
        RESULTS_DIR / "confusion_matrix.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print("=" * 60)
    print("Evaluation Complete")
    print("=" * 60)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print()

    print("Results saved to:")

    print(RESULTS_DIR)


if __name__ == "__main__":
    main()