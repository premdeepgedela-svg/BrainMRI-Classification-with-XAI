from pathlib import Path

from src.datasets.brats_slice_dataset import BraTSSliceDataset

DATASET_PATH = Path(
    r"C:\AI_Research\BrainMRI-XAI-Robustness\data\raw"
    r"\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
    r"\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
)

dataset = BraTSSliceDataset(DATASET_PATH)

print("=" * 60)
print("BraTS Slice Dataset")
print("=" * 60)

print("Total Samples:", len(dataset))

sample = dataset[0]

print("\nPatient ID :", sample["patient_id"])
print("Slice Index:", sample["slice_index"])
print("Image Shape:", sample["image"].shape)
print("Label      :", sample["label"].item())