"""
nano_rust_py — TinyML inference engine for embedded devices.

This package provides:
    - PySequentialModel: Rust-powered i8 inference engine (compiled extension)
    - nano_rust_py.utils: Python utilities for quantization, calibration, and export

Quick Start:
    >>> import nano_rust_py
    >>> model = nano_rust_py.PySequentialModel([784], arena_size=4096)
    >>> model.add_dense(weights, bias)
    >>> model.add_relu()
    >>> output = model.forward(input_i8)

    # Quantization utilities:
    >>> from nano_rust_py.utils import quantize_weights, calibrate_model, export_to_rust
"""

# Re-export everything from the compiled Rust extension module
from .nano_rust_py import *  # noqa: F401, F403

__version__ = "0.2.0"
