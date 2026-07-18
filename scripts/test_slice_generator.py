from pathlib import Path

from src.preprocessing.slice_generator import SliceGenerator

DATASET = Path(
    r"C:\AI_Research\BrainMRI-XAI-Robustness\data\raw"
    r"\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
    r"\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
)

patient = sorted(DATASET.iterdir())[0]

generator = SliceGenerator(patient)

samples = generator.generate()

print("=" * 60)
print("Slice Generator")
print("=" * 60)

print("Patient:", patient.name)
print("Total slices:", len(samples))

tumor = sum(s["label"] for s in samples)

print("Tumor slices:", tumor)
print("Normal slices:", len(samples) - tumor)