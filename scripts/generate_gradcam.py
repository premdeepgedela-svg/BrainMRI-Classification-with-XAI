from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.datasets.dataloader import create_dataloaders
from src.explainability.gradcam import GradCAM
from src.models.resnet18 import BrainMRIResNet18


def overlay_heatmap(image, heatmap):

    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET,
    )

    image = image - image.min()

    image = image / (image.max() + 1e-8)

    image = np.uint8(image * 255)

    image = cv2.cvtColor(
        image,
        cv2.COLOR_GRAY2BGR,
    )

    overlay = cv2.addWeighted(
        image,
        0.6,
        heatmap,
        0.4,
        0,
    )

    return overlay


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    DATASET_PATH = Path(
        r"C:\AI_Research\BrainMRI-XAI-Robustness\data\raw"
        r"\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
        r"\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
    )

    _, val_loader = create_dataloaders(
        DATASET_PATH,
        batch_size=1,
        num_workers=0,
    )

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

    gradcam = GradCAM(
        model,
        model.model.layer4[-1],
    )

    output_dir = Path(
        "results/gradcam"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    saved = 0

    for batch in val_loader:

        image = batch["image"].to(device)

        label = batch["label"].item()

        prediction = (
            model(image)
            .argmax(dim=1)
            .item()
        )

        if prediction != label:
            continue

        heatmap = gradcam.generate(image)

        flair = image[0, 3].cpu().numpy()

        overlay = overlay_heatmap(
            flair,
            heatmap,
        )

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.imshow(flair, cmap="gray")
        plt.axis("off")
        plt.title("FLAIR")

        plt.subplot(1, 3, 2)
        plt.imshow(heatmap, cmap="jet")
        plt.axis("off")
        plt.title("Grad-CAM")

        plt.subplot(1, 3, 3)
        plt.imshow(
            cv2.cvtColor(
                overlay,
                cv2.COLOR_BGR2RGB,
            )
        )
        plt.axis("off")
        plt.title("Overlay")

        plt.tight_layout()

        plt.savefig(
            output_dir / f"sample_{saved}.png",
            dpi=300,
        )

        plt.close()

        saved += 1

        if saved == 20:
            break

    gradcam.remove_hooks()

    print("=" * 60)
    print("Grad-CAM Generation Complete")
    print("=" * 60)
    print(f"Saved {saved} visualizations.")
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()