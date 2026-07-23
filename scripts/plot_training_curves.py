from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():

    history_file = Path("results/resnet18/training_history.csv")

    if not history_file.exists():
        print("Training history file not found.")
        print("Expected:", history_file)
        return

    history = pd.read_csv(history_file)

    output_dir = Path("results/resnet18/plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # Loss Curve
    # ----------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        history["epoch"],
        history["train_loss"],
        marker="o",
        label="Train Loss",
    )

    plt.plot(
        history["epoch"],
        history["val_loss"],
        marker="o",
        label="Validation Loss",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    plt.savefig(output_dir / "loss_curve.png", dpi=300)

    plt.close()

    # ----------------------------
    # Accuracy Curve
    # ----------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        history["epoch"],
        history["train_acc"],
        marker="o",
        label="Train Accuracy",
    )

    plt.plot(
        history["epoch"],
        history["val_acc"],
        marker="o",
        label="Validation Accuracy",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training and Validation Accuracy")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    plt.savefig(output_dir / "accuracy_curve.png", dpi=300)

    plt.close()

    print("=" * 60)
    print("Training Curves Saved")
    print("=" * 60)
    print(output_dir)


if __name__ == "__main__":
    main()