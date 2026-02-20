#!/usr/bin/env python3
"""
Export a trained PyTorch model to Rust source code for ESP32 deployment.

This script demonstrates the complete export pipeline:
  1. Define and train a model (or load a pre-trained one)
  2. Quantize weights to i8
  3. Calibrate requantization parameters
  4. Export to .rs file ready for `include!()` in firmware

Usage:
    python examples/export_for_esp32.py

Output:
    generated/sensor_model.rs  — Rust source with static weights + builder function

The generated .rs file can be copied to your ESP32 project's src/ directory.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Add scripts/ to path for nano_rust_utils
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from nano_rust_utils import (
    quantize_to_i8,
    quantize_weights,
    calibrate_model,
    export_to_rust,
)


def create_sensor_model() -> nn.Sequential:
    """Create a simple MLP for sensor classification.

    Input:  6 features (accel_x/y/z + gyro_x/y/z)
    Output: 4 classes (normal, vibration, temperature, emergency)

    Architecture chosen for MCU constraints:
      - Small hidden layers (32, 16) → fits in <2KB Flash
      - ReLU activations → fastest on MCU (no LUT needed)
      - No batch norm → simpler quantization
    """
    model = nn.Sequential(
        nn.Linear(6, 32),   # Layer 0: 6→32 (192 weights + 32 bias = 224 bytes)
        nn.ReLU(),           # Layer 1: zero cost
        nn.Linear(32, 16),  # Layer 2: 32→16 (512 weights + 16 bias = 528 bytes)
        nn.ReLU(),           # Layer 3: zero cost
        nn.Linear(16, 4),   # Layer 4: 16→4 (64 weights + 4 bias = 68 bytes)
    )
    return model


def train_model(model: nn.Sequential) -> torch.Tensor:
    """Train the model on synthetic sensor data.

    In a real project, replace this with your actual dataset:
      - Load CSV/binary sensor recordings
      - Split train/test
      - Train until convergence

    Returns:
        sample_input: A representative input tensor for calibration
    """
    print("📊 Training sensor classification model...")

    # Generate synthetic training data
    # Class 0: Normal (low values)
    # Class 1: Vibration (high accel variance)
    # Class 2: Temperature (high gyro values)
    # Class 3: Emergency (all high)
    rng = np.random.default_rng(42)
    n_per_class = 200

    X_train = np.vstack([
        rng.normal(0.0, 0.1, (n_per_class, 6)),           # Normal
        rng.normal(0.0, 1.0, (n_per_class, 6)),           # Vibration
        np.hstack([                                         # Temperature
            rng.normal(0.0, 0.1, (n_per_class, 3)),
            rng.normal(2.0, 0.5, (n_per_class, 3)),
        ]),
        rng.normal(2.0, 1.0, (n_per_class, 6)),           # Emergency
    ]).astype(np.float32)

    y_train = np.array([0]*n_per_class + [1]*n_per_class +
                        [2]*n_per_class + [3]*n_per_class)

    X_tensor = torch.from_numpy(X_train)
    y_tensor = torch.from_numpy(y_train).long()

    # Train
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(100):
        optimizer.zero_grad()
        out = model(X_tensor)
        loss = criterion(out, y_tensor)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 25 == 0:
            acc = (out.argmax(dim=1) == y_tensor).float().mean()
            print(f"  Epoch {epoch+1}/100 — Loss: {loss:.4f}, Acc: {acc:.1%}")

    # Return a representative input for calibration
    sample_input = X_tensor[0:1]  # shape [1, 6]
    return sample_input


def export_model(
    model: nn.Sequential,
    sample_input: torch.Tensor,
    output_path: Path,
) -> None:
    """Quantize, calibrate, and export model to Rust source code.

    This is the core export pipeline:
      1. quantize_weights() → i8 weight arrays
      2. calibrate_model() → requantization parameters
      3. export_to_rust() → complete .rs file
    """
    print("\n🔧 Quantizing weights...")
    q_weights = quantize_weights(model)
    for name, info in q_weights.items():
        w = info['weights']
        print(f"  Layer {name} ({info['type']}): {w.shape} → {w.size} bytes, "
              f"scale={info['weight_scale']:.6f}")

    print("\n📐 Calibrating requantization parameters...")
    q_input, input_scale = quantize_to_i8(sample_input.detach().numpy().flatten())
    print(f"  Input scale: {input_scale:.6f}")

    cal = calibrate_model(model, sample_input, q_weights, input_scale)
    for name, params in cal.items():
        if isinstance(params, tuple) and len(params) == 3:
            m, s, _ = params
            print(f"  Layer {name}: requant_m={m}, requant_shift={s}")

    print("\n📦 Exporting to Rust...")
    rust_code = export_to_rust(model, "sensor_model", input_shape=[6])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(rust_code)

    # Count total model size
    total_bytes = sum(
        info['weights'].size + info['bias'].size
        for info in q_weights.values()
    )
    print(f"\n✅ Exported to: {output_path}")
    print(f"   Total model size: {total_bytes} bytes (Flash)")
    print(f"   Estimated arena: ~{max(32, 16, 4) * 4} bytes (RAM)")
    print(f"\n💡 Copy this file to your ESP32 project's src/ directory:")
    print(f"   cp {output_path} my_esp32_project/src/model.rs")


def main():
    print("=" * 60)
    print("  NANO-RUST-AI: Export Pipeline for ESP32")
    print("=" * 60)

    # Step 1: Create model
    model = create_sensor_model()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n📐 Model: {total_params} parameters")
    print(f"   Float32 size: {total_params * 4} bytes")
    print(f"   Int8 size:    {total_params} bytes (4× smaller)")

    # Step 2: Train
    sample_input = train_model(model)

    # Step 3: Export
    output_path = PROJECT_ROOT / "generated" / "sensor_model.rs"
    export_model(model, sample_input, output_path)

    print("\n" + "=" * 60)
    print("  Done! Next steps:")
    print("  1. Copy generated/sensor_model.rs → ESP32 src/model.rs")
    print("  2. Add nano-rust-core to Cargo.toml dependencies")
    print("  3. Use include!(\"model.rs\") in main.rs")
    print("  4. See examples/esp32_deploy.rs for firmware template")
    print("=" * 60)


if __name__ == "__main__":
    main()
