from pathlib import Path

# Update this path if your dataset location changes
DATASET_PATH = Path(
    r"C:\AI_Research\BrainMRI-XAI-Robustness\data\raw\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
)

# Get all patient folders
patients = sorted([p for p in DATASET_PATH.iterdir() if p.is_dir()])

print("=" * 60)
print(f"Total patients: {len(patients)}")
print("=" * 60)

first_patient = patients[0]

print("\nFirst patient folder:")
print(first_patient.name)

print("\nFiles:")

for file in sorted(first_patient.iterdir()):
    print(file.name)