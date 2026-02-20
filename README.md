# 🧠 NANO-RUST-AI

**TinyML Inference Engine — Train in PyTorch, Run on Microcontrollers**

[![PyPI](https://img.shields.io/pypi/v/nano-rust-py)](https://pypi.org/project/nano-rust-py/)  ![Version](https://img.shields.io/badge/version-0.2.0-brightgreen)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

```
Train (PyTorch, GPU) → Quantize (float32 → int8) → Verify (Python) → Deploy (ESP32/STM32)
```

---

## 📦 Installation

```bash
pip install nano-rust-py
```

That's it. No Rust toolchain needed for using the library.
Includes both the Rust inference engine **and** Python quantization utilities.

```python
import nano_rust_py
print(nano_rust_py.__name__)  # → "nano_rust_py"
```

> **For development** (modifying Rust source): see [Development Setup](#-development-setup) below.

---

## 🚀 Quick Start — 3-Step Example

```python
import nano_rust_py

# Step 1: Create model (input: 4 features, arena: 4KB scratch memory)
model = nano_rust_py.PySequentialModel(input_shape=[4], arena_size=4096)

# Step 2: Add layers with i8 weights
#   Dense layer: 4 inputs → 3 outputs
#   weights = [4×3] matrix flattened, bias = [3] vector
model.add_dense(
    weights=[10, -5, 3, 7, -2, 8, -4, 6, 1, 5, -3, 9],  # 4×3 = 12 values
    bias=[1, -1, 2]                                        # 3 values
)
model.add_relu()

# Step 3: Run inference
input_data = [100, -50, 30, 70]  # i8 values: [-128, 127]
output = model.forward(input_data)
print(output)  # → [15, 0, 22]  (i8 values after ReLU)

# Get predicted class
prediction = model.predict(input_data)
print(prediction)  # → 2  (argmax index)
```

---

## 📖 Complete Python API Reference

### `PySequentialModel` — The Core Model Class

#### Constructor

```python
model = nano_rust_py.PySequentialModel(
    input_shape,   # List[int] — shape of input tensor
    arena_size     # int — scratch memory in bytes
)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `input_shape` | `List[int]` | `[N]` for 1D, `[C, H, W]` for 3D (e.g., `[1, 28, 28]` for MNIST) |
| `arena_size` | `int` | Bytes for intermediate computation. Rule: `2 × largest_layer_output × sizeof(i8)` |

```python
# 1D input (e.g., sensor features)
model = nano_rust_py.PySequentialModel([128], 4096)

# 3D input (e.g., MNIST image: 1 channel, 28×28)
model = nano_rust_py.PySequentialModel([1, 28, 28], 32768)
```

---

### Layer Methods

#### `add_dense(weights, bias)` — Fully-Connected Layer (Frozen)

Weights stored in Flash (0 bytes RAM). Uses simple requantization.

```python
# 4 inputs → 2 outputs
model.add_dense(
    weights=[10, -5, 3, 7, -2, 8],  # flat [out × in] = [2 × 4] = 8 values
    bias=[1, -1]                     # [out] = 2 values
)
```

| Parameter | Type | Shape | Description |
|-----------|------|-------|-------------|
| `weights` | `List[int]` | `[out_features × in_features]` | i8 weight matrix, **row-major** |
| `bias` | `List[int]` | `[out_features]` | i8 bias vector |

**Output**: `out[j] = clamp(Σ(w[j,i] × x[i]) + bias[j])` requantized to i8

---

#### `add_dense_with_requant(weights, bias, requant_m, requant_shift)` — Dense with Calibrated Requantization

For **high-accuracy** inference. Uses TFLite-style `(acc × M) >> shift`.

```python
model.add_dense_with_requant(
    weights=[10, -5, 3, 7, -2, 8],
    bias=[1, -1],
    requant_m=1234,       # int32 multiplier (from calibration)
    requant_shift=15      # uint32 bit-shift (from calibration)
)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `requant_m` | `int` | Fixed-point multiplier from `calibrate_model()` |
| `requant_shift` | `int` | Bit-shift from `calibrate_model()` |

> **When to use**: Always prefer this over `add_dense()` when you have calibration data. Accuracy improves from ~85% to ~97%.

---

#### `add_conv2d(kernel, bias, in_ch, out_ch, kh, kw, stride, padding)` — 2D Convolution (Frozen)

```python
# 1 input channel → 8 output channels, 3×3 kernel
model.add_conv2d(
    kernel=[...],   # [out_ch × in_ch × kh × kw] = 8×1×3×3 = 72 values
    bias=[...],     # [out_ch] = 8 values
    in_ch=1, out_ch=8, kh=3, kw=3,
    stride=1, padding=1
)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `kernel` | `List[int]` | i8 kernel, shape `[out_ch × in_ch × kh × kw]`, row-major |
| `bias` | `List[int]` | i8 bias, shape `[out_ch]` |
| `in_ch` | `int` | Input channels |
| `out_ch` | `int` | Output channels (number of filters) |
| `kh`, `kw` | `int` | Kernel height and width |
| `stride` | `int` | Stride (typically 1 or 2) |
| `padding` | `int` | Zero-padding (use `kh // 2` to preserve spatial size) |

**Output shape**: `[out_ch, (H + 2*pad - kh) / stride + 1, (W + 2*pad - kw) / stride + 1]`

---

#### `add_conv2d_with_requant(kernel, bias, in_ch, out_ch, kh, kw, stride, padding, requant_m, requant_shift)` — Conv2D with Calibrated Requantization

Same as `add_conv2d` but with calibrated requantization for accuracy.

```python
model.add_conv2d_with_requant(
    kernel=[...], bias=[...],
    in_ch=1, out_ch=8, kh=3, kw=3, stride=1, padding=1,
    requant_m=2048, requant_shift=14
)
```

---

#### `add_trainable_dense(in_features, out_features)` — Trainable Layer (RAM)

Weights live in RAM (for on-device fine-tuning). **Not for frozen inference**.

```python
model.add_trainable_dense(128, 10)  # 128 → 10, weights in RAM
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `in_features` | `int` | Input dimension |
| `out_features` | `int` | Output dimension |

> **RAM cost**: `in_features × out_features + out_features` bytes

---

#### Activation Layers

| Method | Formula | When to use |
|--------|---------|-------------|
| `add_relu()` | `max(0, x)` | Default choice, fastest |
| `add_sigmoid()` | `1 / (1 + e^(-x/16))` | Binary classification, fixed-scale |
| `add_sigmoid_scaled(mult, shift)` | Calibrated sigmoid LUT | After calibration |
| `add_tanh()` | `tanh(x/32)` | Centered output [-1, 1], fixed-scale |
| `add_tanh_scaled(mult, shift)` | Calibrated tanh LUT | After calibration |
| `add_softmax()` | Pseudo-softmax approximation | Multi-class output (last layer) |

```python
# Simple (no calibration needed)
model.add_relu()

# Calibrated (from calibrate_model output)
model.add_sigmoid_scaled(scale_mult=42, scale_shift=8)
model.add_tanh_scaled(scale_mult=84, scale_shift=8)
```

> **Important**: `add_sigmoid()` and `add_tanh()` use a fixed scale divisor (16 and 32 respectively). For best accuracy, use the `_scaled` variants with parameters from `calibrate_model()`.

---

#### Structural Layers

| Method | Parameters | Description |
|--------|------------|-------------|
| `add_flatten()` | — | Reshape 3D `[C,H,W]` → 1D `[C×H×W]`. Use between conv and dense. |
| `add_max_pool2d(kernel, stride, padding)` | `int, int, int` | Reduce spatial dims by taking max over kernel window |

```python
model.add_max_pool2d(kernel=2, stride=2, padding=0)
# Input [8, 28, 28] → Output [8, 14, 14]
```

---

### Inference Methods

#### `model.forward(input_data)` → `List[int]`

Run forward pass, get raw i8 output vector.

```python
output = model.forward([100, -50, 30, 70])
print(output)  # → [15, -8, 22]  (raw i8 activations)
```

#### `model.predict(input_data)` → `int`

Run forward pass, get argmax class index.

```python
class_id = model.predict([100, -50, 30, 70])
print(class_id)  # → 2
```

---

## 🔧 Python Utilities — `nano_rust_py.utils`

All utilities are **bundled in the PyPI package** — no need to clone the repo.

```python
from nano_rust_py.utils import (
    quantize_to_i8,
    quantize_weights,
    calibrate_model,
    compute_requant_params,
    compute_activation_scale_params,
    export_to_rust,
    export_weights_bin,
)
```

> **Note**: `numpy` is installed as a dependency. `torch` is only needed if you use
> `quantize_weights()` or `calibrate_model()` — install with `pip install nano-rust-py[train]`.

These utilities bridge PyTorch training and NANO-RUST inference.

### `quantize_to_i8(tensor, scale=127.0)` → `(np.ndarray, float)`

Quantize any float32 tensor to i8 using symmetric linear scaling.

```python
import numpy as np
from nano_rust_py.utils import quantize_to_i8


float_data = np.array([0.5, -0.3, 1.0, -1.0], dtype=np.float32)
q_data, scale = quantize_to_i8(float_data)
print(q_data)   # → [ 64, -38, 127, -127]
print(scale)    # → 0.00787  (max_abs / 127)

# To dequantize: float_value ≈ i8_value × scale
print(q_data[0] * scale)  # → 0.503 ≈ 0.5 ✓
```

---

### `quantize_weights(model)` → `Dict`

Walk a PyTorch model and quantize all weight tensors.

```python
import torch.nn as nn
from nano_rust_py.utils import quantize_weights


model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
)

q = quantize_weights(model)
# Returns: {
#   '0': {
#     'type': 'Linear',
#     'weights': np.ndarray (i8, shape [128, 784]),
#     'bias': np.ndarray (i8, shape [128]),
#     'weight_scale': 0.00312,
#     'bias_scale': 0.00156,
#     'params': {'in_features': 784, 'out_features': 128}
#   },
#   '2': {
#     'type': 'Linear',
#     'weights': np.ndarray (i8, shape [10, 128]),
#     ...
#   }
# }
# Note: ReLU (layer '1') has no weights, so it is skipped.
```

---

### `calibrate_model(model, input_tensor, q_weights, input_scale)` → `Dict`

Run float model and compute per-layer requantization parameters.

```python
from nano_rust_py.utils import calibrate_model, quantize_to_i8, quantize_weights


# 1. Quantize weights
q_weights = quantize_weights(model)

# 2. Prepare a representative input
sample_input = torch.randn(1, 784)
q_input, input_scale = quantize_to_i8(sample_input.numpy().flatten())

# 3. Calibrate
cal = calibrate_model(model, sample_input, q_weights, input_scale)
# Returns: {
#   '0': (requant_m=1234, requant_shift=15, bias_corrected=[...]),
#   '2': (requant_m=5678, requant_shift=14, bias_corrected=[...]),
# }
```

> **Why calibrate?** Without calibration, the library uses a generic `shift = ceil(log2(k)) + 7` which is approximate. Calibration computes the *exact* scale ratio between input, weights, and output — raising accuracy from ~85% to 95-99%.

---

### `compute_requant_params(input_scale, weight_scale, output_scale)` → `(int, int)`

Compute TFLite-style fixed-point multiplier and shift.

```python
from nano_rust_py.utils import compute_requant_params


M, shift = compute_requant_params(
    input_scale=0.00787,    # from quantize_to_i8(input)
    weight_scale=0.00312,   # from quantize_weights(model)
    output_scale=0.00450    # from quantize_to_i8(expected_output)
)
print(M, shift)  # → (1407, 15)
# Meaning: output_i8 ≈ (accumulator × 1407) >> 15
```

---

### `export_to_rust(model, model_name, input_shape)` → `str`

Generate complete Rust source code for the model weights and builder function.

```python
from nano_rust_py.utils import export_to_rust


rust_code = export_to_rust(model, "digit_classifier", input_shape=[1, 28, 28])
with open("generated/digit_classifier.rs", "w") as f:
    f.write(rust_code)
```

**Output file contains**:
```rust
// Auto-generated by nano_rust_utils
static LAYER_0_W: &[i8] = &[10, -5, 3, ...];
static LAYER_0_B: &[i8] = &[1, -1, ...];

pub fn build_digit_classifier() -> SequentialModel<'static> {
    let mut model = SequentialModel::new();
    model.add(Box::new(FrozenDense::new_with_requant(
        LAYER_0_W, LAYER_0_B, 784, 128, 1234, 15
    ).unwrap()));
    model.add(Box::new(ReLULayer));
    // ...
    model
}
```

---

### `export_weights_bin(q_weights, output_dir)` → `List[Path]`

Export quantized weights to binary files for `include_bytes!` in Rust.

```python
from nano_rust_py.utils import export_weights_bin


paths = export_weights_bin(q_weights, "output/")
# Creates:
#   output/0_w.bin  (128 × 784 = 100,352 bytes)
#   output/0_b.bin  (128 bytes)
#   output/2_w.bin  (10 × 128 = 1,280 bytes)
#   output/2_b.bin  (10 bytes)
```

---

## 📓 Notebooks — Learning Guide

### Prerequisites

```bash
pip install nano-rust-py numpy torch torchvision ipykernel
```

Open notebooks in Jupyter/VS Code and select your venv kernel.

### Validation Notebooks (`notebooks/`)

| # | Notebook | What You'll Learn |
|---|----------|-------------------|
| 01 | `01_pipeline_validation` | Full pipeline: Conv→ReLU→Flatten→Dense. Bit-exact comparison between float32 and i8. |
| 02 | `02_mlp_classification` | Dense→ReLU→Dense (MLP). Manual weight quantization and verification. |
| 03 | `03_deep_cnn` | Deep CNN with Conv→ReLU→MaxPool stacking. Memory estimation for MCU. |
| 04 | `04_activation_functions` | Side-by-side comparison: ReLU vs Sigmoid vs Tanh. Fixed vs scaled modes. |
| 05 | `05_transfer_learning` | Frozen backbone (Flash) + trainable head (RAM). Hybrid memory pattern. |

### Real-World Examples (`notebook-realworld-examples/`)

Full Jupyter notebooks — each follows the complete workflow:  
**Train (GPU) → Quantize → Calibrate → Build NANO Model → Verify Accuracy**

| # | Notebook | Task | Training Data | Accuracy |
|---|----------|------|---------------|----------|
| 06 | `06_mnist.ipynb` | Digit classification | MNIST (28×28) | ~97% |
| 07 | `07_fashion_mnist.ipynb` | Fashion item recognition | Fashion-MNIST (28×28) | ~87% |
| 08 | `08_sensor_anomaly.ipynb` | Industrial anomaly detection | Synthetic sensor data | ~98% |
| 09 | `09_keyword_spotting.ipynb` | Voice keyword detection | Speech Commands MFCC | ~79% |
| 10 | `10_text_classifier.ipynb` | Text topic classification | Bag-of-words features | 100% |

Open any notebook in Jupyter or VS Code and run all cells.

---

## 🏗️ The Complete Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│  STEP 1: Train in PyTorch (PC/GPU)                              │
│  ─────────────────────────────────────                          │
│  • Define nn.Sequential model                                   │
│  • Train on dataset (MNIST, sensor data, audio, etc.)           │
│  • Achieve desired float32 accuracy                             │
├─────────────────────────────────────────────────────────────────┤
│  STEP 2: Quantize & Calibrate (Python)                          │
│  ─────────────────────────────────────                          │
│  • quantize_weights(model) → i8 weights + scales               │
│  • calibrate_model() → requant_m, requant_shift per layer       │
│  • Memory shrinks 4× (float32 → int8)                          │
├─────────────────────────────────────────────────────────────────┤
│  STEP 3: Build NANO Model & Verify (Python)                     │
│  ─────────────────────────────────────                          │
│  • Create PySequentialModel with i8 weights                     │
│  • Run same test inputs → compare with PyTorch                  │
│  • Verify accuracy loss < 5% (typically < 2%)                   │
├─────────────────────────────────────────────────────────────────┤
│  STEP 4: Export to Rust (Python)                                │
│  ─────────────────────────────────────                          │
│  • export_to_rust(model, "my_model") → .rs file                │
│  • Contains: static weight arrays + builder function            │
│  • Or export_weights_bin() → .bin files for include_bytes!      │
├─────────────────────────────────────────────────────────────────┤
│  STEP 5: Deploy to MCU (Rust)                                   │
│  ─────────────────────────────────────                          │
│  • include!("my_model.rs") in firmware                          │
│  • Allocate arena buffer (stack/static)                         │
│  • Read sensor → quantize input → inference → action            │
│  • See examples/esp32_deploy.rs                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🏗️ Architecture

```
┌──────────────────────────────────────┐
│  Python (PyTorch + nano_rust_utils)  │  ← Train & Quantize
├──────────────────────────────────────┤
│  PyO3 Binding (nano_rust_py)         │  ← Bridge
├──────────────────────────────────────┤
│  Rust Core (nano-rust-core)          │  ← Inference Engine
│  ┌────────┐ ┌────────┐ ┌─────────┐  │
│  │ math.rs│ │layers/ │ │arena.rs │  │
│  │ matmul │ │dense   │ │bump ptr │  │
│  │ conv2d │ │conv    │ │ckpt/rst │  │
│  │ relu   │ │pool    │ └─────────┘  │
│  │sigmoid │ │flatten │              │
│  │ tanh   │ │activate│              │
│  └────────┘ └────────┘              │
└──────────────────────────────────────┘
```

### Memory Layout on MCU

```
FLASH (read-only)                RAM (read-write)
┌─────────────────────┐          ┌──────────────────┐
│ Frozen weights      │          │ Arena Buffer      │
│ - Conv2D kernels    │          │ ┌──────────────┐  │
│ - Dense weights     │          │ │ Intermediate │  │
│ - Bias arrays       │          │ │ activations  │  │
│ (.rs static arrays) │          │ ├──────────────┤  │
│                     │          │ │ Trainable    │  │
│ Cost: N bytes       │          │ │ head weights │  │
│ RAM cost: 0 bytes   │          │ └──────────────┘  │
└─────────────────────┘          └──────────────────┘
```

### Memory Budget Rules

| Component | Formula | Example (MNIST MLP) |
|-----------|---------|---------------------|
| Frozen weights | `Σ(in × out)` per dense + `Σ(in_ch × out_ch × kh × kw)` per conv | 100KB Flash |
| Arena buffer | `2 × max(layer_output_size)` | 2 × 784 = 1.6KB RAM |
| Bias arrays | `Σ(out_features)` per layer | 138 bytes Flash |
| Trainable head (if any) | `in × out + out` | 1.3KB RAM |

> **ESP32 budget**: 4MB Flash, 520KB RAM. A typical model uses <100KB Flash + <20KB RAM.

---

## 🚀 ESP32 Deployment

See the complete examples:

- [`examples/esp32_deploy.rs`](examples/esp32_deploy.rs) — Rust firmware template
- [`examples/export_for_esp32.py`](examples/export_for_esp32.py) — Python export pipeline

### Quick Summary

```python
# Python: export model
from nano_rust_utils import quantize_weights, calibrate_model, export_to_rust

rust_code = export_to_rust(trained_model, "my_model", input_shape=[416])
with open("src/model.rs", "w") as f:
    f.write(rust_code)
```

```rust
// Rust firmware: use exported model
#![no_std]
include!("model.rs");

let mut arena_buf = [0u8; 16384];
let mut arena = Arena::new(&mut arena_buf);
let model = build_my_model();
let (output, _) = model.forward(&input_i8, &[416], &mut arena).unwrap();
let class = nano_rust_core::math::argmax_i8(output);
```

---

## 🔧 Rust Core API (for Firmware Developers)

### Layers

```rust
use nano_rust_core::layers::*;

// Frozen layers (weights in Flash — 0 bytes RAM)
let dense = FrozenDense::new_with_requant(weights, bias, 784, 128, 1234, 15)?;
let conv = FrozenConv2D::new_with_requant(kernel, bias, 1, 8, 3, 3, 1, 1, 2048, 14)?;

// Trainable layer (weights in RAM — for fine-tuning)
let head = TrainableDense::new(128, 10);

// Activations
let _ = ReLULayer;
let _ = ScaledSigmoidLayer { scale_mult: 42, scale_shift: 8 };
let _ = ScaledTanhLayer { scale_mult: 84, scale_shift: 8 };
let _ = SoftmaxLayer;

// Structural
let _ = FlattenLayer;
let pool = MaxPool2DLayer::new(2, 2, 0)?;
```

### Arena Allocator

```rust
use nano_rust_core::Arena;

let mut buf = [0u8; 32768];
let mut arena = Arena::new(&mut buf);

// Checkpoint/restore for scratch memory reuse
let cp = arena.checkpoint();
let scratch = arena.alloc_i8_slice(1024)?;
arena.restore(cp);  // reclaim scratch memory
```

### Sequential Model

```rust
use nano_rust_core::model::SequentialModel;

let mut model = SequentialModel::new();
model.add(Box::new(dense));
model.add(Box::new(ReLULayer));
model.add(Box::new(dense2));

let (output, out_shape) = model.forward(input, &[784], &mut arena)?;
let class = nano_rust_core::math::argmax_i8(output);
```

---

## 🗂️ Project Structure

```
nano-rust/
├── core/                       # Rust no_std core library
│   └── src/
│       ├── lib.rs              # Crate root & re-exports
│       ├── arena.rs            # Bump pointer allocator
│       ├── math.rs             # Quantized matmul, conv2d, activations
│       ├── error.rs            # NanoError, NanoResult
│       ├── model.rs            # SequentialModel (layer pipeline)
│       └── layers/
│           ├── mod.rs          # Layer trait + Shape struct
│           ├── dense.rs        # FrozenDense + TrainableDense
│           ├── conv.rs         # FrozenConv2D (im2col+matmul)
│           ├── activations.rs  # ReLU, Sigmoid, Tanh, Softmax (LUT)
│           ├── flatten.rs      # Flatten 3D→1D
│           └── pooling.rs      # MaxPool2D
├── py_binding/                 # PyO3 Python bindings (compiled Rust)
│   └── src/lib.rs              # PySequentialModel wrapper
├── python/                     # Pure Python modules (bundled in PyPI)
│   └── nano_rust_py/
│       ├── __init__.py         # Package init — re-exports Rust types
│       └── utils.py            # Quantization, calibration, export tools
├── scripts/                    # Standalone scripts (not in PyPI)
│   ├── nano_rust_utils.py      # Legacy utils (now in nano_rust_py.utils)
│   └── export.py               # CLI weight exporter
├── notebooks/                  # Validation notebooks (01-05)
├── notebook-realworld-examples/ # Real-world Jupyter notebooks (06-10)
├── examples/                   # ESP32 deployment examples
├── generated/                  # Exported Rust weight files
├── pyproject.toml              # pip/maturin build config
├── Cargo.toml                  # Rust workspace config
├── LICENSE                     # MIT
└── README.md
```

---

## 🛠️ Development Setup

Only needed if you want to modify the Rust source code:

```bash
# 1. Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
# or: winget install Rustlang.Rust.MSVC

# 2. Clone and setup
git clone https://github.com/LeeNim/nano-rust.git
cd nano-rust
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
# .venv\Scripts\activate    # Windows

# 3. Install deps
pip install maturin numpy torch torchvision ipykernel

# 4. Build from source
# Windows: set CARGO_TARGET_DIR outside OneDrive!
$env:CARGO_TARGET_DIR = "$env:USERPROFILE\.nanorust_target"
maturin develop --release

# 5. Verify
python -c "import nano_rust_py; print('OK')"
```

---

## 📜 License

[MIT](LICENSE) © 2026 Niem Le

## 🔮 Roadmap

- [x] v0.1.0: Core inference engine with scale-aware requantization
- [x] v0.2.0: Bundled Python utilities (`nano_rust_py.utils`) in PyPI package
- [ ] v0.3.0: Const Generics refactor for compile-time optimization
- [ ] v0.4.0: On-device training (backprop for trainable head)
- [ ] v0.5.0: ARM SIMD intrinsics (SMLAD) for Cortex-M
