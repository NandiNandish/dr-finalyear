"""
Unit tests for model architectures, switchable heads, exact parameter counts,
and quantum circuit gradient flow.
"""

import pytest
import torch
from src.models.classical_head import ClassicalHead
from src.models.matched_head import MatchedDenseHead
from src.models.quantum_head import HybridQuantumHead
from src.models.full_model import DRScreeningModel


def test_classical_head_parameters_and_shape():
    head = ClassicalHead(in_features=1280, num_classes=5)
    # Expected: 1280 * 5 + 5 = 6405
    total_params = sum(p.numel() for p in head.parameters())
    assert total_params == 6405

    dummy_f = torch.randn(4, 1280)
    out = head(dummy_f)
    assert out.shape == (4, 5)


def test_matched_dense_head_parameters_and_shape():
    head = MatchedDenseHead(in_features=1280, bottleneck_dim=4, num_classes=5)
    # W1: 1280 * 4 + 4 = 5124
    # W2: 4 * 5 + 5 = 25
    # Total = 5149
    w1_params = sum(p.numel() for p in head.fc1.parameters())
    w2_params = sum(p.numel() for p in head.fc2.parameters())
    total_params = sum(p.numel() for p in head.parameters())

    assert w1_params == 5124
    assert w2_params == 25
    assert total_params == 5149

    dummy_f = torch.randn(3, 1280)
    out = head(dummy_f)
    assert out.shape == (3, 5)


def test_hybrid_quantum_head_exact_parameters():
    # Test for L in {2, 3, 4}
    for L in [2, 3, 4]:
        head = HybridQuantumHead(
            in_features=1280,
            num_qubits=4,
            num_layers=L,
            num_classes=5,
            diff_method="backprop",
        )
        counts = head.count_parameters()

        # Classical weights
        assert counts["classical_w1"] == 5124, f"W1 params {counts['classical_w1']} != 5124"
        assert counts["classical_w2"] == 25, f"W2 params {counts['classical_w2']} != 25"
        assert counts["classical_total"] == 5149

        # Quantum weights: 4 * L
        expected_quantum = 4 * L
        assert counts["quantum_params"] == expected_quantum, f"Quantum params {counts['quantum_params']} != {expected_quantum}"
        assert counts["total_params"] == 5149 + expected_quantum


def test_hybrid_quantum_head_gradient_flow():
    # Use backprop for rapid unit test
    head = HybridQuantumHead(
        in_features=1280,
        num_qubits=4,
        num_layers=2,
        num_classes=5,
        diff_method="backprop",
    )
    dummy_f = torch.randn(2, 1280, requires_grad=True)
    out = head(dummy_f)
    assert out.shape == (2, 5)

    loss = out.sum()
    loss.backward()

    # Verify gradients computed for quantum weights, W1, and W2
    assert head.q_weights.grad is not None
    assert torch.any(head.q_weights.grad != 0)
    assert head.fc1.weight.grad is not None
    assert head.fc2.weight.grad is not None


def test_full_model_switchable_heads():
    model = DRScreeningModel(
        backbone_name="efficientnet_b0",
        active_head="hybrid_quantum",
        variational_layers=2,
        diff_method="backprop",
        freeze_backbone=True,
    )

    # 1. Hybrid quantum head active
    assert model.active_head_type == "hybrid_quantum"
    dummy_features = torch.randn(2, 1280)
    out_q = model(dummy_features)
    assert out_q.shape == (2, 5)

    # 2. Switch to matched dense
    model.set_active_head("matched_dense")
    assert model.active_head_type == "matched_dense"
    out_m = model(dummy_features)
    assert out_m.shape == (2, 5)

    # 3. Switch to classical
    model.set_active_head("classical")
    assert model.active_head_type == "classical"
    out_c = model(dummy_features)
    assert out_c.shape == (2, 5)
