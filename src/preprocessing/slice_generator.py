from pathlib import Path

import nibabel as nib
import numpy as np

from src.preprocessing.brain_filter import has_brain_tissue


class SliceGenerator:
    """
    Generates metadata for valid MRI slices.

    No MRI images are stored in memory.
    """

    def __init__(self, patient_folder):
        self.patient_folder = Path(patient_folder)
        self.patient_id = self.patient_folder.name

    def load_volume(self, suffix):

        file = self.patient_folder / f"{self.patient_id}-{suffix}.nii.gz"

        return nib.load(str(file)).get_fdata()

    def generate(self):

        flair = self.load_volume("t2f")
        mask = self.load_volume("seg")

        metadata = []

        for z in range(flair.shape[2]):

            image = flair[:, :, z]

            if not has_brain_tissue(image):
                continue

            label = int(np.any(mask[:, :, z] > 0))

            metadata.append(
                {
                    "patient_folder": self.patient_folder,
                    "patient_id": self.patient_id,
                    "slice_index": z,
                    "label": label,
                }
            )

        return metadata