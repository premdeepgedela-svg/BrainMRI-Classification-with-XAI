from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


DATASET_PATH = Path(
    r"C:\AI_Research\BrainMRI-XAI-Robustness\data\raw"
    r"\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
    r"\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
)

pixel_counts = []

patients = sorted(
    [p for p in DATASET_PATH.iterdir() if p.is_dir()]
)

print(f"Scanning {len(patients)} patients...")

for patient in patients:

    patient_id = patient.name

    mask = nib.load(
        str(patient / f"{patient_id}-seg.nii.gz")
    ).get_fdata()

    for z in range(mask.shape[2]):

        tumor_pixels = np.sum(mask[:, :, z] > 0)

        if tumor_pixels > 0:
            pixel_counts.append(tumor_pixels)

pixel_counts = np.array(pixel_counts)

print("=" * 60)
print("Tumor Pixel Statistics")
print("=" * 60)

print(f"Total Tumor Slices : {len(pixel_counts)}")
print(f"Minimum            : {pixel_counts.min()}")
print(f"Maximum            : {pixel_counts.max()}")
print(f"Mean               : {pixel_counts.mean():.2f}")
print(f"Median             : {np.median(pixel_counts):.2f}")

print("\nPercentiles")

for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
    print(f"{p:>2}% : {np.percentile(pixel_counts, p):.1f}")

plt.figure(figsize=(10,5))

plt.hist(
    pixel_counts,
    bins=100,
)

plt.xlabel("Tumor Pixels per Slice")
plt.ylabel("Number of Slices")
plt.title("Tumor Pixel Distribution")

plt.tight_layout()

plt.savefig(
    "results/tumor_pixel_distribution.png",
    dpi=300,
)

print("\nSaved histogram to results/tumor_pixel_distribution.png")