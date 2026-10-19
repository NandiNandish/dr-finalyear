"""
Hybrid Quantum-Classical Classification Head for Diabetic Retinopathy Screening.

Implementation Specifications:
1. Classical Projection W1 + b1:
   Linear(1280, 4) -> 5,124 parameters
   v = (pi / 2) * tanh(W1 f + b1)
2. Variational Quantum Circuit (VQC) via PennyLane:
   - 4 qubits: State |0>^(x4)
   - Hadamard H^(x4) + Angle Embedding RY(v_k)
   - L variational layers (L in {2, 3, 4}, default 3):
     * CNOT chain: (0,1), (1,2), (2,3), (3,0)
     * Trainable RY(w) per qubit -> 4L quantum parameters
   - Measure Pauli-Z expectation on each qubit -> z in [-1, 1]^4
   - Gradients: parameter-shift rule for circuit angles, backprop for classical layers
   - Configurable diff_method ('parameter-shift' or 'backprop')
   - Configurable noise model (PennyLane default.mixed with depolarizing noise)
3. Classical Readout W2 + b2:
   Linear(4, 5) -> 25 parameters
   y_hat = softmax(W2 z + b2)

Exact Parameter Counts:
Classical: 5,124 + 25 = 5,149
Quantum: 4L (12 for L=3)
Total: 5,149 + 4L (5,161 for L=3)
"""

import math
from typing import Optional, List, Dict, Any
import torch
import torch.nn as nn
import pennylane as qml


class HybridQuantumHead(nn.Module):
    """
    Hybrid Quantum Head with exact parameter accounting and configurable simulation options.
    """

    def __init__(
        self,
        in_features: int = 1280,
        num_qubits: int = 4,
        num_layers: int = 3,
        num_classes: int = 5,
        backend: str = "default.qubit",
        diff_method: str = "parameter-shift",
        noise_model: Optional[str] = None,
        depolarizing_prob: float = 0.01,
    ):
        super().__init__()
        self.in_features = in_features
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.backend_name = backend
        self.diff_method = diff_method
        self.noise_model = noise_model
        self.depolarizing_prob = depolarizing_prob

        # 1. Classical projection layer: 1280 -> 4 (5,124 parameters)
        self.fc1 = nn.Linear(in_features, num_qubits, bias=True)

        # 2. Quantum circuit variational parameters: shape (L, num_qubits) -> 4L params
        # Initialize uniformly around [0, 2*pi]
        self.q_weights = nn.Parameter(
            torch.empty(num_layers, num_qubits).uniform_(0.0, 2.0 * math.pi)
        )

        # 3. Classical readout layer: 4 -> 5 (25 parameters)
        self.fc2 = nn.Linear(num_qubits, num_classes, bias=True)

        # Build PennyLane device and QNode
        self._build_qnode()

    def _build_qnode(self):
        if self.noise_model == "depolarizing":
            dev = qml.device("default.mixed", wires=self.num_qubits)
            # Mixed state simulator uses parameter-shift or finite-diff
            circuit_diff = "parameter-shift" if self.diff_method == "parameter-shift" else "finite-diff"
        else:
            dev = qml.device(self.backend_name, wires=self.num_qubits)
            circuit_diff = self.diff_method

        def quantum_circuit(v_angles, weights):
            # Step A: Hadamard on each qubit
            for i in range(self.num_qubits):
                qml.Hadamard(wires=i)
                # Angle embedding: RY(v_k)
                qml.RY(v_angles[i], wires=i)

            # Step B: L Variational Layers
            for l in range(self.num_layers):
                # Circular entangling chain: (0,1), (1,2), (2,3), (3,0)
                for i in range(self.num_qubits):
                    qml.CNOT(wires=[i, (i + 1) % self.num_qubits])

                # Trainable rotations RY(w_{l, k})
                for i in range(self.num_qubits):
                    qml.RY(weights[l, i], wires=i)
                    if self.noise_model == "depolarizing":
                        qml.DepolarizingChannel(self.depolarizing_prob, wires=i)

            # Step C: Measure Pauli-Z expectation on each qubit
            return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]

        self.qnode = qml.QNode(quantum_circuit, dev, interface="torch", diff_method=circuit_diff)

    def count_parameters(self) -> Dict[str, int]:
        """
        Returns exact parameter breakdown.
        """
        w1_params = sum(p.numel() for p in self.fc1.parameters())
        w2_params = sum(p.numel() for p in self.fc2.parameters())
        q_params = self.q_weights.numel()
        return {
            "classical_w1": w1_params,
            "classical_w2": w2_params,
            "classical_total": w1_params + w2_params,
            "quantum_params": q_params,
            "total_params": w1_params + w2_params + q_params,
        }

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f: (B, 1280) pooled backbone features
        Returns:
            logits: (B, 5) classification logits
        """
        # Step 1: Classical reduction: v = (pi/2) * tanh(W1 f + b1)
        # Bounded in [-pi/2, pi/2]
        v = (math.pi / 2.0) * torch.tanh(self.fc1(f))

        batch_size = v.shape[0]
        # Step 2: Evaluate VQC per sample in batch
        z_list = []
        for i in range(batch_size):
            # v[i]: 1D tensor of shape (4,)
            expvals = self.qnode(v[i], self.q_weights)
            # Stack into tensor (4,)
            z_sample = torch.stack(expvals)
            z_list.append(z_sample)

        z = torch.stack(z_list)  # Shape: (B, 4), values in [-1, 1]

        # Ensure z is on the same device and dtype as fc2 weights
        z = z.to(dtype=f.dtype, device=f.device)

        # Step 3: Classical readout: W2 z + b2
        logits = self.fc2(z)  # Shape: (B, 5)
        return logits
