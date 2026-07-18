from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib


DATASET_PATH = Path(
    r"C:\AI_Research\BrainMRI-XAI-Robustness\data\raw\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
)

# Load first patient
patient = sorted([p for p in DATASET_PATH.iterdir() if p.is_dir()])[0]

modalities = {
    "T1": "t1n",
    "T1ce": "t1c",
    "T2": "t2w",
    "FLAIR": "t2f",
    "Mask": "seg",
}

images = {}

for title, suffix in modalities.items():
    file = next(patient.glob(f"*{suffix}.nii.gz"))
    images[title] = nib.load(str(file)).get_fdata()

slice_idx = images["T1"].shape[2] // 2

fig, axes = plt.subplots(1, 5, figsize=(18, 4))

for ax, (title, volume) in zip(axes, images.items()):
    cmap = "gray" if title != "Mask" else "viridis"

    ax.imshow(volume[:, :, slice_idx], cmap=cmap)
    ax.set_title(title)
    ax.axis("off")

plt.tight_layout()
plt.show()