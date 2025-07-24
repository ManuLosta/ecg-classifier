import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

import torch.nn as nn
from tqdm import tqdm
from typing import Tuple
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
import os
from src.data.ptbxl_loader import PTBXLDataset
from src.utils.preprocessing import ECGPreprocessor, ECGDataModule
import torch.optim as optim
from src.models.cnn1d import ResNet1D


class ECGTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        test_loader,
        optimizer,
        criterion,
        checkpoint_dir="checkpoints",
        device="cuda",
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

        self.best_val_score = 0.0
        self.train_losses = []
        self.val_losses = []
        self.val_scores = []
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0

        for batch_idx, (data, target) in enumerate(
            tqdm(self.train_loader, desc="Training")
        ):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def validate_epoch(self) -> Tuple[float, float, float]:
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for data, target in tqdm(self.val_loader, desc="Validation"):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                total_loss += loss.item()

                all_preds.append(output.cpu())
                all_targets.append(target.cpu())

            all_preds = np.vstack(all_preds)
            all_targets = np.vstack(all_targets)

            auc_score = float(roc_auc_score(all_targets, all_preds, average="macro"))
            accuracy = float(
                (all_preds.argmax(axis=1) == all_targets.argmax(axis=1)).mean()
            )

            return float(total_loss / len(self.val_loader)), auc_score, accuracy

    def save_checkpoint(self, epoch: int):
        checkpoint_path = os.path.join(
            self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pth"
        )
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_val_score": self.best_val_score,
            },
            checkpoint_path,
        )
        print(f"Checkpoint saved at {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_val_score = checkpoint["best_val_score"]
        print(
            f"Checkpoint loaded from {checkpoint_path}, resuming from epoch {checkpoint['epoch']}"
        )
        return checkpoint["epoch"]

    def train(self, num_epochs: int, save_every: int = 5):
        print(f"Starting training for {num_epochs} epochs...")

        for epoch in range(num_epochs):
            train_loss = self.train_epoch()
            val_loss, val_score, _ = self.validate_epoch()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_scores.append(val_score)

            print(f"Epoch {epoch+1}/{num_epochs}")
            print(
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val AUC: {val_score:.4f}"
            )

            if val_score > self.best_val_score:
                self.best_val_score = val_score
                print(
                    f"New best validation score: {self.best_val_score:.4f}. Saving model..."
                )
                torch.save(self.model.state_dict(), "best_model.pth")

            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(epoch + 1)

    def evaluate_test(self) -> Tuple[float, float, float]:
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for data, target in tqdm(self.test_loader, desc="Testing"):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                total_loss += loss.item()
                all_preds.append(output.cpu())
                all_targets.append(target.cpu())

            all_preds = np.vstack(all_preds)
            all_targets = np.vstack(all_targets)

            auc_score = float(roc_auc_score(all_targets, all_preds, average="macro"))
            accuracy = float(
                (all_preds.argmax(axis=1) == all_targets.argmax(axis=1)).mean()
            )

            return float(total_loss / len(self.test_loader)), auc_score, accuracy


def main():
    data_path = "dataset"
    ptbxl_dataset = PTBXLDataset(data_path)

    (X_train, y_train), (X_val, y_val), (X_test, y_test), class_names = (
        ptbxl_dataset.get_dataset()
    )

    print(f"Dataset loaded")
    print(f"Train set shape: {X_train.shape}, {y_train.shape}")
    print(f"Validation set shape: {X_val.shape}, {y_val.shape}")
    print(f"Test set shape: {X_test.shape}, {y_test.shape}")
    print(f"Class names: {class_names}")

    preprocessor = ECGPreprocessor(sampling_rate=100, target_length=1000)
    X_train = preprocessor.preprocess(X_train, apply_filters=True)
    X_val = preprocessor.preprocess(X_val, apply_filters=True)
    X_test = preprocessor.preprocess(X_test, apply_filters=True)

    data_module = ECGDataModule(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        batch_size=32,
    )
    train_loader, val_loader, test_loader = data_module.get_dataloaders()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ResNet1D(input_channels=X_train.shape[1], num_classes=len(class_names))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    trainer = ECGTrainer(
        model, train_loader, val_loader, test_loader, optimizer, criterion, device=str(device)
    )

    trainer.train(num_epochs=20, save_every=5)

    test_loss, test_auc, test_accuracy = trainer.evaluate_test()
    print(
        f"Test Loss: {test_loss:.4f}, Test AUC: {test_auc:.4f}, Test Accuracy: {test_accuracy:.4f}"
    )


if __name__ == "__main__":
    main()
