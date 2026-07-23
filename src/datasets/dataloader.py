from pathlib import Path
import random

from torch.utils.data import DataLoader

from src.datasets.brats_slice_dataset import BraTSSliceDataset


def create_dataloaders(
    dataset_path,
    batch_size=16,
    train_ratio=0.8,
    num_workers=0,
    seed=42,
):
    """
    Create train and validation DataLoaders
    using patient-level splitting.
    """

    dataset_path = Path(dataset_path)

    # =====================================
    # Get all patients
    # =====================================

    patients = sorted(
        [p for p in dataset_path.iterdir() if p.is_dir()]
    )

    print(f"Found {len(patients)} patients.")

    # =====================================
    # Shuffle patients
    # =====================================

    random.seed(seed)
    random.shuffle(patients)

    # =====================================
    # Split patients
    # =====================================

    train_size = int(len(patients) * train_ratio)

    train_patients = patients[:train_size]
    val_patients = patients[train_size:]

    print(f"Training Patients   : {len(train_patients)}")
    print(f"Validation Patients : {len(val_patients)}")

    # =====================================
    # Create datasets
    # =====================================

    train_dataset = BraTSSliceDataset(
        dataset_path,
        patient_list=train_patients,
    )

    val_dataset = BraTSSliceDataset(
        dataset_path,
        patient_list=val_patients,
    )

    # =====================================
    # Train DataLoader
    # =====================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    # =====================================
    # Validation DataLoader
    # =====================================

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader