from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.datasets.dataloader import create_dataloaders
from src.models.resnet18 import BrainMRIResNet18


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

    _, val_loader = create_dataloaders(
        DATASET_PATH,
        batch_size=16,
        num_workers=0,
    )

    # ==========================================
    # Model
    # ==========================================

    model = BrainMRIResNet18()

    checkpoint = torch.load(
        "results/resnet18/checkpoints/best_model.pth",
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    # ==========================================
    # Evaluation
    # ==========================================

    y_true = []
    y_pred = []

    with torch.no_grad():

        for batch in val_loader:

            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            outputs = model(images)

            predictions = outputs.argmax(dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predictions.cpu().numpy())

    # ==========================================
    # Metrics
    # ==========================================

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print("=" * 60)
    print("ResNet-18 Evaluation")
    print("=" * 60)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\nClassification Report\n")

    print(classification_report(y_true, y_pred))

    # ==========================================
    # Save Results
    # ==========================================

    output_dir = Path(
        "results/resnet18/evaluation"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_dir / "classification_report.txt",
        "w",
    ) as f:

        f.write(classification_report(y_true, y_pred))

    # ==========================================
    # Confusion Matrix
    # ==========================================

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Normal", "Tumor"],
        yticklabels=["Normal", "Tumor"],
    )

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("ResNet-18 Confusion Matrix")

    plt.tight_layout()

    plt.savefig(
        output_dir / "confusion_matrix.png",
        dpi=300,
    )

    plt.close()

    print("\nResults saved to:")

    print(output_dir)


if __name__ == "__main__":
    main()