# 🧠 NANO-RUST-AI

**TinyML Framework for Embedded Devices — Rust `no_std` Core + Python Bindings**

Train in PyTorch → Quantize (i8) → Run on MCU (ESP32, STM32, Cortex-M)

---

## ✨ Features

- **🔒 No Heap**: Pure `no_std` — zero `malloc`, zero dynamic allocation
- **⚡ Int8 Quantization**: All compute in i8/i32 for 4× memory savings over f32
- **🧊 Hybrid Memory**: Frozen weights in Flash (0 bytes RAM), trainable head in RAM
- **🎯 Scale-Aware Requantization**: TFLite-style `(acc × M) >> shift` for accurate i8 output
- **🐍 Python Bindings**: PyO3 wrapper for seamless PyTorch → NANO-RUST pipeline
- **📦 Arena Allocator**: User provides `&mut [u8]` buffer — library self-manages within it

---

## 📋 Quick Start

### 1. Prerequisites

| Tool | Version |
|------|---------|
| Rust | 1.70+ (`rustup install stable`) |
| Python | 3.9+ |
| maturin | `pip install maturin` |

### 2. Create Virtual Environment

```bash
# Create and activate venv
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate

# Install dependencies
pip install maturin numpy torch torchvision jupyter ipykernel
```

### 3. Build & Install the Library

```bash
# IMPORTANT: Set CARGO_TARGET_DIR outside OneDrive to avoid file locking
# Windows PowerShell:
$env:CARGO_TARGET_DIR = "$env:USERPROFILE\.nanorust_target"

# Build and install into the active venv
maturin develop --release
```

### 4. Register Jupyter Kernel (for notebooks)

```bash
python -m ipykernel install --user --name nanorust --display-name "NanoRust (venv)"
```

Then select the **"NanoRust (venv)"** kernel in Jupyter when running notebooks.

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
FLASH (4MB)                    RAM (320KB)
┌─────────────────────┐        ┌──────────────────┐
│ Frozen Backbone     │        │ Arena Buffer      │
│ - Conv2D weights    │        │ ┌──────────────┐  │
│ - Dense weights     │        │ │ Intermediate │  │
│ - Bias arrays       │        │ │ activations  │  │
│ (read-only, static) │        │ ├──────────────┤  │
│                     │        │ │ Trainable    │  │
│                     │        │ │ Head weights │  │
│                     │        │ └──────────────┘  │
└─────────────────────┘        └──────────────────┘
```

---

## 🐍 Python API Reference

### `nano_rust_py.PySequentialModel`

```python
model = nano_rust_py.PySequentialModel(
    input_shape=[C, H, W],    # or [N] for 1D
    arena_size=32768           # bytes for scratch memory
)
```

### Layer Methods

| Method | Description |
|--------|-------------|
| `add_dense(weights, bias)` | Dense layer (i8 weights/bias as lists) |
| `add_dense_with_requant(weights, bias, M, shift)` | Dense with calibrated requant |
| `add_conv2d(kernel, bias, in_ch, out_ch, kh, kw, stride, padding)` | Conv2D layer |
| `add_conv2d_with_requant(kernel, bias, in_ch, out_ch, kh, kw, stride, padding, M, shift)` | Conv2D with calibrated requant |
| `add_trainable_dense(in_features, out_features)` | Trainable Dense (RAM weights) |
| `add_relu()` | ReLU activation |
| `add_sigmoid()` | Sigmoid (fixed scale, for general use) |
| `add_sigmoid_scaled(scale_mult, scale_shift)` | Sigmoid with scale-aware LUT |
| `add_tanh()` | Tanh (fixed scale, for general use) |
| `add_tanh_scaled(scale_mult, scale_shift)` | Tanh with scale-aware LUT |
| `add_softmax()` | Softmax (pseudo-probabilities) |
| `add_flatten()` | Flatten 3D→1D |
| `add_max_pool2d(kernel, stride, padding)` | MaxPool2D |

### Inference

```python
output = model.forward(input_i8_list)  # Returns list of i8 values
```

### Python Utilities (`scripts/nano_rust_utils.py`)

```python
from nano_rust_utils import quantize_to_i8, quantize_weights, calibrate_model

# Quantize input
q_input, input_scale = quantize_to_i8(float_array)

# Quantize model weights
q_weights = quantize_weights(pytorch_model)

# Calibrate requantization parameters
requant = calibrate_model(model, input_tensor, q_weights, input_scale)
# Returns dict: layer_name → (M, shift, bias_corrected) for parametric layers
#                             ('sigmoid', mult, shift) for Sigmoid
#                             ('tanh', mult, shift) for Tanh
```

---

## 📓 Notebooks

### Validation Notebooks (`notebooks/`)

Quick-run notebooks using `_setup.py` for auto-build:

| # | File | Description |
|---|------|-------------|
| 01 | `01_pipeline_validation.ipynb` | Conv→ReLU→Flatten→Dense end-to-end |
| 02 | `02_mlp_classification.ipynb` | MLP (Dense→ReLU→Dense) |
| 03 | `03_deep_cnn.ipynb` | Deep CNN with MaxPool |
| 04 | `04_activation_functions.ipynb` | ReLU vs Sigmoid vs Tanh comparison |
| 05 | `05_transfer_learning.ipynb` | Frozen backbone + trainable head |

### Real-World Test Scripts (`notebooks-for-test/`)

GPU-accelerated training → i8 quantization → NANO-RUST verification:

| # | File | Task | Accuracy |
|---|------|------|----------|
| 06 | `run_06_mnist.py` | MNIST digit classification (CNN) | ~97% |
| 07 | `run_07_fashion.py` | Fashion item classification (CNN) | ~87% |
| 08 | `run_08_sensor.py` | Industrial sensor fusion (MLP) | ~98% |
| 09 | `run_09_keyword_spotting.py` | Voice keyword spotting (MFCC+MLP) | ~79% |
| 10 | `run_10_text_classifier.py` | Text classification (BoW+MLP) | 100% |

Run all tests:
```bash
python notebooks-for-test/run_06_mnist.py
# ... etc
```

---

## 🚀 ESP32 Deployment Guide

### Step 1: Train & Export in Python

```python
import torch.nn as nn
from nano_rust_utils import quantize_weights, calibrate_model, export_to_rust

# 1. Train your PyTorch model
model = nn.Sequential(
    nn.Linear(416, 128), nn.ReLU(),
    nn.Linear(128, 64), nn.ReLU(),
    nn.Linear(64, 10),
)
# ... train on GPU ...

# 2. Quantize & calibrate
q_weights = quantize_weights(model)
requant = calibrate_model(model, sample_input, q_weights, input_scale)

# 3. Export to Rust source code
rust_code = export_to_rust(model, "keyword_model", input_shape=[416])
with open("model.rs", "w") as f:
    f.write(rust_code)
```

### Step 2: Use in ESP32 Rust Firmware

```rust
#![no_std]
use nano_rust_core::{Arena, model::SequentialModel};

// Generated model from export_to_rust()
include!("model.rs");

#[entry]
fn main() -> ! {
    // Arena in RAM — size from model.estimate_arena_size()
    let mut arena_buf = [0u8; 16384];

    loop {
        // Get sensor/audio data → quantize to i8
        let input: [i8; 416] = read_mfcc_features();

        // Run inference (< 1ms on ESP32 @ 240MHz)
        let mut arena = Arena::new(&mut arena_buf);
        let model = build_keyword_model();  // From generated code
        let (output, _) = model.forward(&input, &[416], &mut arena).unwrap();

        let predicted_class = output.iter()
            .enumerate()
            .max_by_key(|(_, v)| **v)
            .map(|(i, _)| i)
            .unwrap();
    }
}
```

### Memory Budget (ESP32)

| Component | Flash | RAM |
|-----------|-------|-----|
| Frozen weights | 60KB | 0B |
| Arena buffer | 0B | 16KB |
| Code + stack | ~20KB | ~4KB |
| **Total** | **~80KB** | **~20KB** |
| **Available** | **4MB** | **520KB** |

---

## 🔧 Rust Core API (`nano-rust-core`)

### Layers

```rust
use nano_rust_core::layers::*;

// Frozen (Flash) — 0 bytes RAM for weights
let dense = FrozenDense::new_with_requant(weights, bias, in_f, out_f, M, shift)?;
let conv = FrozenConv2D::new_with_requant(kernel, bias, in_ch, out_ch, kh, kw, s, p, M, shift)?;

// Trainable (RAM) — weights allocated in Arena
let head = TrainableDense::new(in_features, out_features);

// Activations
let _ = ReLULayer;
let _ = ScaledSigmoidLayer { scale_mult: 42, scale_shift: 8 };
let _ = ScaledTanhLayer { scale_mult: 84, scale_shift: 8 };
let _ = SoftmaxLayer;

// Structural
let _ = FlattenLayer;
let _ = MaxPool2DLayer::new(2, 2, 0)?;
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
let (output, shape) = model.forward(input, &input_shape, &mut arena)?;
```

---

## 📊 Accuracy Targets

| Model Type | Expected Max Diff (vs PyTorch) |
|------------|-------------------------------|
| Dense + ReLU | ≤ 3 |
| Conv + ReLU + Dense | ≤ 5 |
| Deep CNN + Pool | ≤ 10 |
| Sigmoid/Tanh (scaled) | ≤ 20 |

---

## 🗂️ Project Structure

```
nano-rust/
├── core/                    # Rust no_std core library
│   └── src/
│       ├── lib.rs           # Crate root
│       ├── arena.rs         # Bump pointer allocator
│       ├── math.rs          # Matmul, conv2d, activations
│       ├── error.rs         # Error types
│       ├── model.rs         # SequentialModel
│       └── layers/
│           ├── mod.rs       # Layer trait + Shape
│           ├── dense.rs     # FrozenDense + TrainableDense
│           ├── conv.rs      # FrozenConv2D
│           ├── activations.rs  # ReLU, Sigmoid, Tanh, Softmax
│           ├── flatten.rs   # Flatten layer
│           └── pooling.rs   # MaxPool2D
├── py_binding/              # PyO3 Python bindings
│   └── src/lib.rs
├── scripts/
│   ├── nano_rust_utils.py   # Quantization + calibration utilities
│   └── export.py            # CLI weight exporter
├── notebooks/               # Quick validation notebooks (01-05)
├── notebooks-for-test/      # Real-world test scripts (06-10)
├── pyproject.toml           # pip install configuration
├── Cargo.toml               # Workspace config
├── LICENSE                  # MIT License
└── README.md
```

---

## 📜 License

[MIT](LICENSE)

---

## 🔮 Roadmap

- [x] v0.1.0: Core inference engine with scale-aware requantization
- [ ] v0.2.0: Const Generics refactor for compile-time optimization
- [ ] v0.3.0: On-device training (backprop for trainable head)
- [ ] v0.4.0: ARM SIMD intrinsics (SMLAD) for Cortex-M
