from pathlib import Path

from src.datasets.brats_slice_dataset import BraTSSliceDataset


DATASET_PATH = Path(
    r"C:\AI_Research\BrainMRI-XAI-Robustness\data\raw"
    r"\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
    r"\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
)


def main():

    dataset = BraTSSliceDataset(DATASET_PATH)

    total = len(dataset)

    tumor = 0
    normal = 0

    patient_ids = set()

    for sample in dataset.samples:

        patient_ids.add(sample["patient_id"])

        if sample["label"] == 1:
            tumor += 1
        else:
            normal += 1

    print("=" * 60)
    print("BraTS Dataset Statistics")
    print("=" * 60)

    print(f"Patients      : {len(patient_ids)}")
    print(f"Total Slices  : {total}")
    print()

    print(f"Tumor Slices  : {tumor}")
    print(f"Normal Slices : {normal}")
    print()

    print(f"Tumor %       : {100*tumor/total:.2f}%")
    print(f"Normal %      : {100*normal/total:.2f}%")

    print("=" * 60)


if __name__ == "__main__":
    main()