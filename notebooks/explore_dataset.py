from pathlib import Path
import nibabel as nib

DATASET_PATH = Path(
    r"C:\AI_Research\BrainMRI-XAI-Robustness\data\raw\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
)

patients = sorted([p for p in DATASET_PATH.iterdir() if p.is_dir()])

print("=" * 60)
print(f"Total Patients : {len(patients)}")
print("=" * 60)

patient = patients[0]

print(f"\nPatient : {patient.name}\n")

for file in sorted(patient.glob("*.nii.gz")):
    image = nib.load(str(file))

    print(file.name)
    print(f"Shape : {image.shape}")
    print(f"Datatype : {image.get_data_dtype()}")
    print("-" * 50)