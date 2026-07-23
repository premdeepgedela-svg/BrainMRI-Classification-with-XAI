from pathlib import Path

from src.datasets.brats_slice_dataset import BraTSSliceDataset

DATASET_PATH = Path(
    r"C:\AI_Research\BrainMRI-XAI-Robustness\data\raw"
    r"\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
    r"\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
)

dataset = BraTSSliceDataset(DATASET_PATH)

sample = dataset[0]

print("=" * 60)

print("Image Shape :", sample["image"].shape)

print("Label       :", sample["label"])

print("Patient     :", sample["patient_id"])

print("Slice       :", sample["slice_index"])