from pathlib import Path

import torch
import nibabel as nib
import numpy as np
from tqdm import tqdm

from src.preprocessing.brain_filter import has_brain_tissue


RAW_DATASET = Path(
    r"C:\AI_Research\BrainMRI-XAI-Robustness\data\raw"
    r"\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
    r"\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
)

OUTPUT_DIR = Path(
    r"C:\AI_Research\BrainMRI-XAI-Robustness\data\processed"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def load_volume(patient, suffix):

    patient_id = patient.name

    file = patient / f"{patient_id}-{suffix}.nii.gz"

    return nib.load(str(file)).get_fdata().astype(np.float32)


def main():

    patients = sorted(
        [p for p in RAW_DATASET.iterdir() if p.is_dir()]
    )

    print("=" * 60)
    print(f"Patients : {len(patients)}")
    print("=" * 60)

    total_saved = 0

    for patient in tqdm(patients):

        patient_id = patient.name

        t1 = load_volume(patient, "t1n")
        t1ce = load_volume(patient, "t1c")
        t2 = load_volume(patient, "t2w")
        flair = load_volume(patient, "t2f")
        mask = load_volume(patient, "seg")

        patient_output = OUTPUT_DIR / patient_id
        patient_output.mkdir(
            parents=True,
            exist_ok=True,
        )

        for z in range(flair.shape[2]):

            flair_slice = flair[:, :, z]

            if not has_brain_tissue(flair_slice):
                continue

            image = np.stack(
                [
                    t1[:, :, z],
                    t1ce[:, :, z],
                    t2[:, :, z],
                    flair[:, :, z],
                ],
                axis=0,
            )

            label = int(np.any(mask[:, :, z] > 0))

            sample = {
                "image": torch.from_numpy(image),
                "label": label,
                "patient_id": patient_id,
                "slice_index": z,
            }

            torch.save(
                sample,
                patient_output / f"{z:03d}.pt",
            )

            total_saved += 1

    print("\nDone.")
    print(f"Saved slices : {total_saved}")


if __name__ == "__main__":
    main()