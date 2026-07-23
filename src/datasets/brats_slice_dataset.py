from pathlib import Path
from collections import OrderedDict

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

from src.preprocessing.slice_generator import SliceGenerator


class BraTSSliceDataset(Dataset):
    """
    BraTS Dataset with LRU patient cache.
    """

    def __init__(
        self,
        dataset_path,
        patient_list=None,
        transform=None,
        max_cache_size=2,
    ):

        self.dataset_path = Path(dataset_path)
        self.transform = transform

        self.samples = []

        # LRU cache
        self.cache = OrderedDict()
        self.max_cache_size = max_cache_size

        if patient_list is None:

            patient_list = sorted(
                [
                    p
                    for p in self.dataset_path.iterdir()
                    if p.is_dir()
                ]
            )

        print(f"Scanning {len(patient_list)} patients...")

        for patient in patient_list:

            generator = SliceGenerator(patient)

            self.samples.extend(generator.generate())

        print(f"Indexed {len(self.samples)} slices.")

    def __len__(self):

        return len(self.samples)

    def load_patient(self, patient, patient_id):

        # -----------------------------
        # Return cached patient
        # -----------------------------

        if patient_id in self.cache:

            self.cache.move_to_end(patient_id)

            return self.cache[patient_id]

        # -----------------------------
        # Load MRI volumes
        # -----------------------------

        t1 = nib.load(
            str(patient / f"{patient_id}-t1n.nii.gz")
        ).get_fdata(dtype=np.float32)

        t1ce = nib.load(
            str(patient / f"{patient_id}-t1c.nii.gz")
        ).get_fdata(dtype=np.float32)

        t2 = nib.load(
            str(patient / f"{patient_id}-t2w.nii.gz")
        ).get_fdata(dtype=np.float32)

        flair = nib.load(
            str(patient / f"{patient_id}-t2f.nii.gz")
        ).get_fdata(dtype=np.float32)

        # -----------------------------
        # Save into cache
        # -----------------------------

        self.cache[patient_id] = (
            t1,
            t1ce,
            t2,
            flair,
        )

        self.cache.move_to_end(patient_id)

        # -----------------------------
        # Remove oldest patient
        # -----------------------------

        if len(self.cache) > self.max_cache_size:

            self.cache.popitem(last=False)

        return self.cache[patient_id]

    def __getitem__(self, idx):

        sample = self.samples[idx]

        patient = sample["patient_folder"]
        patient_id = sample["patient_id"]
        slice_idx = sample["slice_index"]

        t1, t1ce, t2, flair = self.load_patient(
            patient,
            patient_id,
        )

        image = np.stack(
            [
                t1[:, :, slice_idx],
                t1ce[:, :, slice_idx],
                t2[:, :, slice_idx],
                flair[:, :, slice_idx],
            ],
            axis=0,
        )

        image = torch.from_numpy(image)

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(
            sample["label"],
            dtype=torch.long,
        )

        return {
            "image": image,
            "label": label,
            "patient_id": patient_id,
            "slice_index": slice_idx,
        }