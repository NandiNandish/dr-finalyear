"""
Training engine for Diabetic Retinopathy screening heads.
Supports frozen backbone cached feature training, early stopping on validation QWK,
learning rate scheduling, and multi-seed checkpoints.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.training.metrics import compute_screening_metrics


class DRTrainer:
    """
    Standardized trainer for classical, matched-dense, and hybrid quantum heads.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        scheduler: Optional[Any] = None,
        device: str = "cpu",
        num_epochs: int = 15,
        early_stopping_patience: int = 5,
        save_dir: str = "checkpoints",
        checkpoint_name: str = "best_model.pt",
        verbose: bool = True,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device
        self.num_epochs = num_epochs
        self.early_stopping_patience = early_stopping_patience
        self.save_dir = Path(save_dir)
        self.checkpoint_name = checkpoint_name
        self.verbose = verbose

        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.best_checkpoint_path = self.save_dir / self.checkpoint_name

        self.best_val_qwk = -1.0
        self.history = []

    def _train_epoch(self) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        for batch in self.train_loader:
            if "features" in batch:
                inputs = batch["features"].to(self.device)
            else:
                inputs = batch["image"].to(self.device)

            targets = batch["label"].to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(inputs)
            loss = self.criterion(logits, targets)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * targets.size(0)
            preds = torch.argmax(logits, dim=-1).detach().cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.detach().cpu().numpy())

        epoch_loss = total_loss / len(self.train_loader.dataset)
        acc = float(np.mean(np.array(all_preds) == np.array(all_targets)))
        return epoch_loss, acc

    def _eval_epoch(self) -> Dict[str, Any]:
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_probs = []
        all_targets = []

        with torch.no_grad():
            for batch in self.val_loader:
                if "features" in batch:
                    inputs = batch["features"].to(self.device)
                else:
                    inputs = batch["image"].to(self.device)

                targets = batch["label"].to(self.device)

                logits = self.model(inputs)
                loss = self.criterion(logits, targets)
                total_loss += loss.item() * targets.size(0)

                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                preds = np.argmax(probs, axis=-1)

                all_probs.extend(probs)
                all_preds.extend(preds)
                all_targets.extend(targets.cpu().numpy())

        epoch_loss = total_loss / len(self.val_loader.dataset)
        metrics = compute_screening_metrics(
            y_true=np.array(all_targets),
            y_pred=np.array(all_preds),
            y_probs=np.array(all_probs),
        )
        metrics["loss"] = round(epoch_loss, 4)
        return metrics

    def fit(self) -> Dict[str, Any]:
        """
        Executes complete training loop with validation early stopping on QWK.
        """
        patience_counter = 0

        for epoch in range(1, self.num_epochs + 1):
            train_loss, train_acc = self._train_epoch()
            val_metrics = self._eval_epoch()

            if self.scheduler is not None:
                self.scheduler.step()

            val_qwk = val_metrics["quadratic_weighted_kappa"]
            val_loss = val_metrics["loss"]
            val_acc = val_metrics["accuracy"]

            epoch_record = {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "train_acc": round(train_acc, 4),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_qwk": val_qwk,
                "val_macro_f1": val_metrics["macro_f1"],
                "val_referable_sensitivity": val_metrics["referable_dr"]["sensitivity"],
                "val_referable_specificity": val_metrics["referable_dr"]["specificity"],
            }
            self.history.append(epoch_record)

            if self.verbose:
                print(
                    f"Epoch {epoch:02d}/{self.num_epochs:02d} | "
                    f"Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | "
                    f"Val Loss: {val_loss:.4f} Acc: {val_acc:.3f} QWK: {val_qwk:.3f}"
                )

            # Checkpoint on best validation Quadratic Weighted Kappa
            if val_qwk > self.best_val_qwk:
                self.best_val_qwk = val_qwk
                patience_counter = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "best_val_qwk": self.best_val_qwk,
                        "metrics": val_metrics,
                    },
                    self.best_checkpoint_path,
                )
                if self.verbose:
                    print(f"  --> Saved new best checkpoint (Val QWK: {val_qwk:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    if self.verbose:
                        print(f"Early stopping triggered after {epoch} epochs (patience {self.early_stopping_patience}).")
                    break

        # Load best checkpoint before returning
        if self.best_checkpoint_path.exists():
            checkpoint = torch.load(self.best_checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])

        return {
            "best_val_qwk": self.best_val_qwk,
            "checkpoint_path": str(self.best_checkpoint_path),
            "history": self.history,
        }
