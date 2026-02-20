//! # nano-rust-core: Universal TinyML Framework
//!
//! A `no_std` Rust library for quantized i8 neural network inference and
//! training on microcontrollers (ESP32, STM32, Cortex-M).
//!
//! ## Architecture
//!
//! - **Layer Zoo**: Modular layers (Dense, Conv2D, Flatten, Pooling, Activations)
//! - **Dual Memory**: Frozen (Flash `&'static`) + Trainable (RAM)
//! - **Arena Allocator**: Zero-heap scratch memory via `&mut [u8]`
//! - **i8 Quantization**: All math in i8 with i32 accumulators
//!
//! ## Usage
//!
//! ```ignore
//! use nano_rust_core::*;
//!
//! // Build a model: Conv2D → ReLU → Flatten → Dense
//! let conv = FrozenConv2D::new(CONV_W, CONV_B, 1, 8, 3, 3, 1, 1)?;
//! let relu = ReLULayer;
//! let flat = FlattenLayer;
//! let dense = FrozenDense::new(DENSE_W, DENSE_B, 8*28*28, 10)?;
//!
//! let layers: &[&dyn Layer] = &[&conv, &relu, &flat, &dense];
//! let model = SequentialModel::new(layers, Shape::d3(1, 28, 28))?;
//!
//! let mut scratch = [0u8; 8192];
//! let mut arena = Arena::new(&mut scratch);
//! let class = model.predict(&input, &mut arena)?;
//! ```

// Why #![no_std]: Compiles for bare-metal MCU targets.
#![no_std]

#[cfg(feature = "std")]
extern crate std;

pub mod arena;
pub mod error;
pub mod layers;
pub mod math;
pub mod model;
pub mod tensor;

// Re-export primary types
pub use arena::Arena;
pub use error::{NanoError, NanoResult};
pub use layers::{Layer, Shape};
pub use layers::{
    FrozenDense, TrainableDense,
    FrozenConv2D,
    ReLULayer, SigmoidLayer, TanhLayer, SoftmaxLayer,
    FlattenLayer,
    MaxPool2DLayer,
};
pub use math::{
    argmax_i8, quantized_add_bias, quantized_matmul_i8,
    relu_i8, sigmoid_i8, tanh_i8, softmax_i8,
    conv2d_i8, conv2d_output_size, conv2d_im2col_size,
    max_pool_2d, pool_output_size,
};
pub use model::SequentialModel;
pub use tensor::Tensor;
