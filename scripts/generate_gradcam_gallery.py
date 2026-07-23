from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

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

    return cv2.addWeighted(
        image,
        0.6,
        heatmap,
        0.4,
        0,
    )


def save_result(
    flair,
    heatmap,
    overlay,
    label,
    prediction,
    confidence,
    filename,
):

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(flair, cmap="gray")
    plt.title("FLAIR")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(heatmap, cmap="jet")
    plt.title("Grad-CAM")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(
        cv2.cvtColor(
            overlay,
            cv2.COLOR_BGR2RGB,
        )
    )

    plt.title(
        f"GT={label}  Pred={prediction}\nConfidence={confidence:.3f}"
    )

    plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


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
        num_workers=4,
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
        "results/gradcam_gallery"
    )

    tumor_dir = output_dir / "tumor"
    normal_dir = output_dir / "normal"

    tumor_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    normal_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    tumor_saved = 0
    normal_saved = 0

    MAX_IMAGES = 20

    for batch in val_loader:

        image = batch["image"].to(device)

        label = batch["label"].item()

        with torch.no_grad():

            output = model(image)

            probs = F.softmax(
                output,
                dim=1,
            )

            confidence = probs.max().item()

            prediction = probs.argmax(dim=1).item()

        if prediction != label:
            continue

        heatmap = gradcam.generate(image)

        flair = image[0, 3].cpu().numpy()

        overlay = overlay_heatmap(
            flair,
            heatmap,
        )

        if label == 1 and tumor_saved < MAX_IMAGES:

            save_result(
                flair,
                heatmap,
                overlay,
                label,
                prediction,
                confidence,
                tumor_dir / f"tumor_{tumor_saved}.png",
            )

            tumor_saved += 1

        if label == 0 and normal_saved < MAX_IMAGES:

            save_result(
                flair,
                heatmap,
                overlay,
                label,
                prediction,
                confidence,
                normal_dir / f"normal_{normal_saved}.png",
            )

            normal_saved += 1

        if (
            tumor_saved >= MAX_IMAGES
            and normal_saved >= MAX_IMAGES
        ):
            break

    gradcam.remove_hooks()

    print("=" * 60)
    print("Grad-CAM Gallery Created")
    print("=" * 60)
    print(f"Tumor Images : {tumor_saved}")
    print(f"Normal Images: {normal_saved}")
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    main()