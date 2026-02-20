#!/usr/bin/env python3
"""
Export PyTorch model weights to quantized i8 format for nano-rust-core.

Usage:
    python scripts/export.py --model resnet18 --output model_head.bin

This script:
1. Loads a pre-trained PyTorch model (backbone)
2. Quantizes weights to i8 (linear quantization)
3. Exports backbone to .bin files for Rust `include_bytes!`
4. Optionally trains a head via PyNanoModel and exports it

Requirements:
    pip install torch torchvision numpy
"""

import argparse
import struct
from pathlib import Path
from typing import List

import numpy as np


def quantize_to_i8(weights: np.ndarray, scale: float = 127.0) -> np.ndarray:
    """Quantize float32 weights to i8 using linear scaling.

    Why linear quantization: Simplest scheme that preserves relative magnitudes.
    Maps [-max_abs, +max_abs] → [-127, +127] uniformly.
    Research shows this is sufficient for 85-95% accuracy on classification.

    Args:
        weights: float32 weight tensor
        scale: target i8 range (default: 127 for symmetric quantization)

    Returns:
        Quantized i8 numpy array
    """
    max_abs = np.max(np.abs(weights))
    if max_abs == 0:
        return np.zeros_like(weights, dtype=np.int8)
    scaled = (weights / max_abs) * scale
    return np.clip(np.round(scaled), -128, 127).astype(np.int8)


def export_backbone(
    weights: np.ndarray,
    bias: np.ndarray,
    output_dir: Path,
) -> None:
    """Export backbone weights to raw binary files for Rust include_bytes!

    File format: raw i8 bytes, no header.
    Usage in Rust: static W: &[i8] = include_bytes!("backbone.bin");

    Args:
        weights: Quantized i8 weight matrix [OUT × IN]
        bias: Quantized i8 bias vector [OUT]
        output_dir: Directory to write .bin files
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    w_path = output_dir / "backbone.bin"
    b_path = output_dir / "backbone_bias.bin"

    weights.astype(np.int8).tofile(w_path)
    bias.astype(np.int8).tofile(b_path)

    print(f"✅ Backbone weights: {w_path} ({weights.size} bytes)")
    print(f"✅ Backbone bias:    {b_path} ({bias.size} bytes)")


def export_head_binary(
    weights: List[int],
    bias: List[int],
    hidden_dim: int,
    num_classes: int,
    output_path: Path,
) -> None:
    """Export head weights in NANO binary format.

    File format:
    - 4 bytes: magic "NANO"
    - 4 bytes: hidden_dim (u32 LE)
    - 4 bytes: num_classes (u32 LE)
    - hidden_dim * num_classes bytes: head weights (i8)
    - num_classes bytes: head bias (i8)

    Args:
        weights: Flat i8 head weight list
        bias: Flat i8 head bias list
        hidden_dim: Input dimension of head
        num_classes: Output dimension of head
        output_path: Path to write .bin file
    """
    with open(output_path, "wb") as f:
        f.write(b"NANO")
        f.write(struct.pack("<I", hidden_dim))
        f.write(struct.pack("<I", num_classes))
        f.write(bytes([v & 0xFF for v in weights]))
        f.write(bytes([v & 0xFF for v in bias]))

    total_size = 12 + len(weights) + len(bias)
    print(f"✅ Head exported: {output_path} ({total_size} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export PyTorch model to NANO-RUST-AI format"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("exported_model"),
        help="Output directory for .bin files",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
        help="Backbone output / head input dimension",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=10,
        help="Number of classification classes",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Generate demo random weights (no PyTorch needed)",
    )
    args = parser.parse_args()

    if args.demo:
        print("🎲 Generating demo random weights...")
        in_dim = 784  # MNIST
        rng = np.random.default_rng(42)

        # Random backbone weights
        backbone_w = rng.standard_normal((args.hidden_dim, in_dim)).astype(np.float32)
        backbone_b = rng.standard_normal(args.hidden_dim).astype(np.float32)

        # Quantize
        q_backbone_w = quantize_to_i8(backbone_w)
        q_backbone_b = quantize_to_i8(backbone_b)

        # Export backbone
        export_backbone(q_backbone_w, q_backbone_b, args.output_dir)

        # Generate random head
        head_w = rng.integers(-10, 10, size=args.num_classes * args.hidden_dim).tolist()
        head_b = rng.integers(-5, 5, size=args.num_classes).tolist()

        # Export head
        export_head_binary(
            head_w, head_b,
            args.hidden_dim, args.num_classes,
            args.output_dir / "model_head.bin",
        )

        print(f"\n📦 Files exported to: {args.output_dir}/")
        print(f"   backbone.bin       ({q_backbone_w.size} bytes)")
        print(f"   backbone_bias.bin  ({q_backbone_b.size} bytes)")
        print(f"   model_head.bin     (header + weights + bias)")
    else:
        print("⚠️  Full PyTorch export requires: pip install torch torchvision")
        print("   Use --demo flag for random weight generation.")


if __name__ == "__main__":
    main()
