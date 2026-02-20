//! Modular Layer Zoo for TinyML.
//!
//! Every neural network layer implements the `Layer` trait, enabling
//! layers to be composed into arbitrary `SequentialModel` pipelines.
//!
//! Dual Memory Strategy:
//! - **Frozen** layers: weights in Flash (`&'static [i8]`), 0 bytes RAM
//! - **Trainable** layers: weights in RAM, support `backward()`
//!
//! Shape propagation: Each layer declares its output `Shape` given an
//! input `Shape`, enabling compile-time/construction-time validation
//! of the entire model pipeline.

pub mod activations;
pub mod conv;
pub mod dense;
pub mod flatten;
pub mod pooling;

use crate::arena::Arena;
use crate::error::NanoResult;

/// Tensor shape descriptor supporting 1D–4D tensors.
///
/// Layout convention: [N, C, H, W] (batch, channels, height, width).
/// - 1D: [N] → ndim=1, dims=[N, 0, 0, 0]
/// - 2D: [N, C] → ndim=2, dims=[N, C, 0, 0] (used for Dense output)
/// - 3D: [C, H, W] → ndim=3, dims=[C, H, W, 0] (used for feature maps)
/// - 4D: [N, C, H, W] → ndim=4 (batch of feature maps)
///
/// Why fixed [usize; 4]: No heap allocation, fits in 4 registers on 32-bit MCU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Shape {
    pub dims: [usize; 4],
    pub ndim: usize,
}

impl Shape {
    /// Create a 1D shape [size].
    pub const fn d1(size: usize) -> Self {
        Self { dims: [size, 0, 0, 0], ndim: 1 }
    }

    /// Create a 2D shape [rows, cols] or [batch, features].
    pub const fn d2(d0: usize, d1: usize) -> Self {
        Self { dims: [d0, d1, 0, 0], ndim: 2 }
    }

    /// Create a 3D shape [C, H, W].
    pub const fn d3(c: usize, h: usize, w: usize) -> Self {
        Self { dims: [c, h, w, 0], ndim: 3 }
    }

    /// Create a 4D shape [N, C, H, W].
    pub const fn d4(n: usize, c: usize, h: usize, w: usize) -> Self {
        Self { dims: [n, c, h, w], ndim: 4 }
    }

    /// Total number of elements.
    pub fn total(&self) -> usize {
        let mut t = 1;
        for i in 0..self.ndim {
            t *= self.dims[i];
        }
        t
    }

    /// Get channels (dim[0] for 3D CHW layout).
    pub fn channels(&self) -> usize {
        if self.ndim >= 3 { self.dims[0] } else { 1 }
    }

    /// Get height (dim[1] for 3D CHW layout).
    pub fn height(&self) -> usize {
        if self.ndim >= 3 { self.dims[1] } else { 0 }
    }

    /// Get width (dim[2] for 3D CHW layout).
    pub fn width(&self) -> usize {
        if self.ndim >= 3 { self.dims[2] } else { 0 }
    }
}

/// The Layer trait: universal interface for all neural network layers.
///
/// Why `&dyn Layer`: Enables heterogeneous layer lists in `SequentialModel`.
/// The vtable overhead is 1 pointer per call — negligible vs. matmul cost.
///
/// All layers:
/// 1. Accept input data + input shape
/// 2. Allocate output from the arena (zero-heap)
/// 3. Return output data slice + output shape
pub trait Layer: Send {
    /// Human-readable layer name (for debugging/export).
    fn name(&self) -> &'static str;

    /// Compute the output shape given an input shape.
    /// Called once during model construction to validate the pipeline.
    fn output_shape(&self, input_shape: &Shape) -> NanoResult<Shape>;

    /// Forward pass: transform input data into output data.
    ///
    /// # Arguments
    /// - `input`: flat i8 slice with `input_shape.total()` elements
    /// - `input_shape`: shape descriptor for the input
    /// - `arena`: scratch memory for output allocation
    ///
    /// # Returns
    /// `(output_slice, output_shape)` — output is arena-allocated
    fn forward<'a>(
        &self,
        input: &[i8],
        input_shape: &Shape,
        arena: &mut Arena<'a>,
    ) -> NanoResult<(&'a mut [i8], Shape)>;
}

// Re-export all layer types for convenience
pub use activations::{ReLULayer, SigmoidLayer, TanhLayer, SoftmaxLayer, ScaledSigmoidLayer, ScaledTanhLayer};
pub use conv::FrozenConv2D;
pub use dense::{FrozenDense, TrainableDense};
pub use flatten::FlattenLayer;
pub use pooling::MaxPool2DLayer;
