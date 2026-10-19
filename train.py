"""
Main CLI training runner for Diabetic Retinopathy screening heads.
Supports classical, matched-dense, and hybrid quantum architectures.
"""

import os
import argparse
from pathlib import Path
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import FundusDataset
from src.models.full_model import DRScreeningModel
from src.training.loss import get_loss_function
from src.training.trainer import DRTrainer


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_single_run(args):
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"==================================================")
    print(f"Training Head: {args.head.upper()} | Seed: {args.seed} | Device: {device}")
    print(f"==================================================")

    # 1. Datasets and Loaders
    batch_size = 4 if args.fast else args.batch_size
    num_epochs = 3 if args.fast else args.epochs

    train_ds = FundusDataset(
        manifest=args.manifest,
        split="train",
        feature_cache_dir=args.cache_dir,
        oversample=True,  # Crucial: Oversample ONLY train split post-split
        seed=args.seed,
    )
    val_ds = FundusDataset(
        manifest=args.manifest,
        split="val",
        feature_cache_dir=args.cache_dir,
        oversample=False,  # Never oversample validation
    )

    print(f"Dataset sizes -> Train (oversampled): {len(train_ds)}, Val: {len(val_ds)}")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # 2. Instantiate Model
    diff_method = "backprop" if args.fast else args.diff_method
    model = DRScreeningModel(
        backbone_name=args.backbone,
        active_head=args.head,
        num_classes=5,
        num_qubits=4,
        variational_layers=args.variational_layers,
        quantum_backend=args.quantum_backend,
        diff_method=diff_method,
        freeze_backbone=True,
    )
    model.to(device)

    # If quantum head, display parameter counts
    if args.head == "hybrid_quantum":
        counts = model.hybrid_quantum_head.count_parameters()
        print(f"Hybrid Quantum Head Parameter Breakdown:")
        print(f"  Classical W1 (1280->4): {counts['classical_w1']}")
        print(f"  Classical W2 (4->5):    {counts['classical_w2']}")
        print(f"  Total Classical:        {counts['classical_total']}")
        print(f"  Quantum Circuit (4L):   {counts['quantum_params']} (L={args.variational_layers})")
        print(f"  Total Head Parameters:  {counts['total_params']}")

    # 3. Optimizer and Loss Function
    # Only train the active head parameters (backbone is strictly frozen)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    criterion = get_loss_function(loss_type=args.loss, gamma=2.0)

    # 4. Train
    checkpoint_name = f"checkpoint_{args.head}_seed{args.seed}.pt"
    trainer = DRTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=str(device),
        num_epochs=num_epochs,
        early_stopping_patience=args.patience,
        save_dir=args.checkpoints_dir,
        checkpoint_name=checkpoint_name,
        verbose=True,
    )

    results = trainer.fit()
    print(f"\nTraining Complete for {args.head.upper()} (Seed {args.seed})")
    print(f"Best Validation QWK: {results['best_val_qwk']:.4f}")
    print(f"Saved Checkpoint: {results['checkpoint_path']}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DR Screening Head.")
    parser.add_argument("--head", default="hybrid_quantum", choices=["classical", "hybrid_quantum", "matched_dense"], help="Head architecture")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--manifest", default="data/manifests/demo_manifest.csv", help="Path to manifest")
    parser.add_argument("--cache_dir", default="data/cached_features", help="Path to cached feature directory")
    parser.add_argument("--checkpoints_dir", default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--backbone", default="efficientnet_b0", help="Backbone model")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0001, help="Weight decay")
    parser.add_argument("--loss", default="focal", choices=["focal", "weighted_ce"], help="Loss function")
    parser.add_argument("--diff_method", default="parameter-shift", choices=["parameter-shift", "backprop"], help="Quantum diff method")
    parser.add_argument("--quantum_backend", default="default.qubit", help="PennyLane backend")
    parser.add_argument("--variational_layers", type=int, default=3, help="L quantum variational layers")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--device", default="cpu", help="Compute device (cpu or cuda)")
    parser.add_argument("--fast", action="store_true", help="Fast smoke test flag")
    args = parser.parse_args()

    train_single_run(args)
