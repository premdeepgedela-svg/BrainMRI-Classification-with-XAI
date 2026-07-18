from pathlib import Path

import nibabel as nib
import torch
from torch.utils.data import Dataset

from src.preprocessing.slice_generator import SliceGenerator


class BraTSSliceDataset(Dataset):
    """
    Lazy-loading BraTS slice dataset.
    """

    def __init__(self, dataset_path, transform=None):

        self.dataset_path = Path(dataset_path)
        self.transform = transform

        self.samples = []

        patients = sorted(
            [p for p in self.dataset_path.iterdir() if p.is_dir()]
        )

        print(f"Scanning {len(patients)} patients...")

        for patient in patients:

            generator = SliceGenerator(patient)

            self.samples.extend(generator.generate())

        print(f"Indexed {len(self.samples)} slices.")

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, idx):

        sample = self.samples[idx]

        patient = sample["patient_folder"]
        patient_id = sample["patient_id"]
        slice_idx = sample["slice_index"]

        flair = nib.load(
            str(patient / f"{patient_id}-t2f.nii.gz")
        ).get_fdata()

        image = flair[:, :, slice_idx]

        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(sample["label"], dtype=torch.long)

        return {
            "image": image,
            "label": label,
            "patient_id": patient_id,
            "slice_index": slice_idx,
        }