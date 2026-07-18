from pathlib import Path

from torch.utils.data import DataLoader, random_split

from src.datasets.brats_slice_dataset import BraTSSliceDataset


def create_dataloaders(
    dataset_path,
    batch_size=16,
    train_ratio=0.8,
    num_workers=0,
):
    """
    Create PyTorch train and validation dataloaders.
    """

    dataset = BraTSSliceDataset(dataset_path)

    train_size = int(len(dataset) * train_ratio)
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    return train_loader, val_loader