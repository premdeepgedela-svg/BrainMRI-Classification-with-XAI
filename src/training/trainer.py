from pathlib import Path

import torch
from tqdm import tqdm


class Trainer:

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        checkpoint_dir="results/checkpoints",
    ):

        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def train_one_epoch(self):

        self.model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        progress = tqdm(
            self.train_loader,
            desc="Training",
        )

        for batch in progress:

            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(images)

            loss = self.criterion(outputs, labels)

            loss.backward()

            self.optimizer.step()

            running_loss += loss.item()

            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()

            total += labels.size(0)

            progress.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        epoch_loss = running_loss / len(self.train_loader)

        epoch_acc = 100 * correct / total

        return epoch_loss, epoch_acc

    @torch.no_grad()
    def validate(self):

        self.model.eval()

        running_loss = 0.0
        correct = 0
        total = 0

        progress = tqdm(
            self.val_loader,
            desc="Validation",
        )

        for batch in progress:

            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            outputs = self.model(images)

            loss = self.criterion(outputs, labels)

            running_loss += loss.item()

            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()

            total += labels.size(0)

            progress.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        epoch_loss = running_loss / len(self.val_loader)

        epoch_acc = 100 * correct / total

        return epoch_loss, epoch_acc

    def save_checkpoint(
        self,
        epoch,
        best_accuracy,
        filename="latest.pth",
    ):

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_accuracy": best_accuracy,
        }

        torch.save(
            checkpoint,
            self.checkpoint_dir / filename,
        )

    def load_checkpoint(self, filename):

        checkpoint = torch.load(
            filename,
            map_location=self.device,
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        return (
            checkpoint["epoch"],
            checkpoint["best_accuracy"],
        )