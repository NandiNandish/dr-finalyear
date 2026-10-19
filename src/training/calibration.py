"""
Probability calibration via Temperature Scaling.
Calibrated strictly on VALIDATION split data to optimize probability reliability
prior to single-pass evaluation on the locked test set.
"""

from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class ModelWithTemperature(nn.Module):
    """
    Wraps a trained model with a learned scalar temperature T > 0:
    p = softmax(logits / T)
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(x)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits: torch.Tensor) -> torch.Tensor:
        temperature = self.temperature.clamp(min=0.01)
        return logits / temperature

    def calibrate(self, val_loader, device: str = "cpu", max_iter: int = 50, lr: float = 0.01) -> float:
        """
        Tunes temperature parameter on validation set via NLL loss minimization.
        """
        self.model.eval()
        nll_criterion = nn.CrossEntropyLoss()

        # Collect validation logits and labels
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["features"].to(device) if "features" in batch else batch["image"].to(device)
                labels = batch["label"].to(device)
                logits = self.model(inputs)
                logits_list.append(logits)
                labels_list.append(labels)

        logits = torch.cat(logits_list).to(device)
        labels = torch.cat(labels_list).to(device)

        # Optimize temperature
        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def eval_loss():
            optimizer.zero_grad()
            scaled_logits = self.temperature_scale(logits)
            loss = nll_criterion(scaled_logits, labels)
            loss.backward()
            return loss

        optimizer.step(eval_loss)
        t_val = float(self.temperature.item())
        print(f"Learned optimal temperature on validation set: T = {t_val:.4f}")
        return t_val
