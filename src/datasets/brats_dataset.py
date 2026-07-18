from pathlib import Path

import nibabel as nib
import numpy as np
from torch.utils.data import Dataset


class BraTSDataset(Dataset):
    """
    BraTS 2023 Dataset Loader

    Returns:
        patient_id
        t1
        t1ce
        t2
        flair
        mask
    """

    def __init__(self, dataset_path, transform=None):
        self.dataset_path = Path(dataset_path)
        self.transform = transform

        self.patients = sorted(
            [p for p in self.dataset_path.iterdir() if p.is_dir()]
        )

    def __len__(self):
        return len(self.patients)

    def load_volume(self, file_path):
        volume = nib.load(str(file_path))
        return volume.get_fdata().astype(np.float32)

    def __getitem__(self, idx):

        patient = self.patients[idx]

        patient_id = patient.name

        sample = {
            "patient_id": patient_id,
            "t1": self.load_volume(patient / f"{patient_id}-t1n.nii.gz"),
            "t1ce": self.load_volume(patient / f"{patient_id}-t1c.nii.gz"),
            "t2": self.load_volume(patient / f"{patient_id}-t2w.nii.gz"),
            "flair": self.load_volume(patient / f"{patient_id}-t2f.nii.gz"),
            "mask": self.load_volume(patient / f"{patient_id}-seg.nii.gz"),
        }

        if self.transform:
            sample = self.transform(sample)

        return sample