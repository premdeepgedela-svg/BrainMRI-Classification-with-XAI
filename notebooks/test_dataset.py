from src.datasets.brats_dataset import BraTSDataset

DATASET_PATH = (
    r"C:\AI_Research\BrainMRI-XAI-Robustness\data\raw"
    r"\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
    r"\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
)

dataset = BraTSDataset(DATASET_PATH)

print("=" * 60)
print("BraTS Dataset")
print("=" * 60)

print(f"Total Patients : {len(dataset)}")

sample = dataset[0]

print("\nPatient ID:", sample["patient_id"])
print()

for key in ["t1", "t1ce", "t2", "flair", "mask"]:
    volume = sample[key]

    print(f"{key}")
    print(f"Shape : {volume.shape}")
    print(f"Dtype : {volume.dtype}")
    print(f"Min   : {volume.min():.2f}")
    print(f"Max   : {volume.max():.2f}")
    print("-" * 40)