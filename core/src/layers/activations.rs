//! Activation layers: ReLU, Sigmoid, Tanh, Softmax.
//!
//! Each activation is a zero-weight Layer that transforms data in-place
//! (or with minimal arena allocation). This modular approach lets
//! activations be placed anywhere in a Sequential pipeline.

use crate::arena::Arena;
use crate::error::NanoResult;
use crate::math;
use super::{Layer, Shape};

// =============================================================================
// ReLU
// =============================================================================

/// ReLU activation layer. max(0, x). Zero parameters.
pub struct ReLULayer;

impl Layer for ReLULayer {
    fn name(&self) -> &'static str { "ReLU" }

    fn output_shape(&self, input_shape: &Shape) -> NanoResult<Shape> {
        Ok(*input_shape) // Shape unchanged
    }

    fn forward<'a>(
        &self,
        input: &[i8],
        input_shape: &Shape,
        arena: &mut Arena<'a>,
    ) -> NanoResult<(&'a mut [i8], Shape)> {
        let total = input_shape.total();
        let output = arena.alloc_i8_slice(total)?;
        output.copy_from_slice(input);
        math::relu_i8(output);
        Ok((output, *input_shape))
    }
}

// =============================================================================
// Sigmoid
// =============================================================================

/// Sigmoid activation layer (LUT-based). σ(x) → [0, 127].
pub struct SigmoidLayer;

impl Layer for SigmoidLayer {
    fn name(&self) -> &'static str { "Sigmoid" }

    fn output_shape(&self, input_shape: &Shape) -> NanoResult<Shape> {
        Ok(*input_shape)
    }

    fn forward<'a>(
        &self,
        input: &[i8],
        input_shape: &Shape,
        arena: &mut Arena<'a>,
    ) -> NanoResult<(&'a mut [i8], Shape)> {
        let total = input_shape.total();
        let output = arena.alloc_i8_slice(total)?;
        output.copy_from_slice(input);
        math::sigmoid_i8(output);
        Ok((output, *input_shape))
    }
}

// =============================================================================
// Tanh
// =============================================================================

/// Tanh activation layer (LUT-based). tanh(x) → [-127, 127].
pub struct TanhLayer;

impl Layer for TanhLayer {
    fn name(&self) -> &'static str { "Tanh" }

    fn output_shape(&self, input_shape: &Shape) -> NanoResult<Shape> {
        Ok(*input_shape)
    }

    fn forward<'a>(
        &self,
        input: &[i8],
        input_shape: &Shape,
        arena: &mut Arena<'a>,
    ) -> NanoResult<(&'a mut [i8], Shape)> {
        let total = input_shape.total();
        let output = arena.alloc_i8_slice(total)?;
        output.copy_from_slice(input);
        math::tanh_i8(output);
        Ok((output, *input_shape))
    }
}

// =============================================================================
// Scaled Sigmoid (scale-aware — uses input scale for correct LUT lookup)
// =============================================================================

/// Scale-aware Sigmoid. Rescales input before LUT lookup.
///
/// Required when the preceding layer's output scale S_y ≠ 1/16.
/// `scale_mult / 2^scale_shift ≈ S_y * 16.0`
pub struct ScaledSigmoidLayer {
    pub scale_mult: i32,
    pub scale_shift: u32,
}

impl Layer for ScaledSigmoidLayer {
    fn name(&self) -> &'static str { "ScaledSigmoid" }

    fn output_shape(&self, input_shape: &Shape) -> NanoResult<Shape> {
        Ok(*input_shape)
    }

    fn forward<'a>(
        &self,
        input: &[i8],
        input_shape: &Shape,
        arena: &mut Arena<'a>,
    ) -> NanoResult<(&'a mut [i8], Shape)> {
        let total = input_shape.total();
        let output = arena.alloc_i8_slice(total)?;
        output.copy_from_slice(input);
        math::sigmoid_i8_scaled(output, self.scale_mult, self.scale_shift);
        Ok((output, *input_shape))
    }
}

// =============================================================================
// Scaled Tanh (scale-aware — uses input scale for correct LUT lookup)
// =============================================================================

/// Scale-aware Tanh. Rescales input before LUT lookup.
///
/// Required when the preceding layer's output scale S_y ≠ 1/32.
/// `scale_mult / 2^scale_shift ≈ S_y * 32.0`
pub struct ScaledTanhLayer {
    pub scale_mult: i32,
    pub scale_shift: u32,
}

impl Layer for ScaledTanhLayer {
    fn name(&self) -> &'static str { "ScaledTanh" }

    fn output_shape(&self, input_shape: &Shape) -> NanoResult<Shape> {
        Ok(*input_shape)
    }

    fn forward<'a>(
        &self,
        input: &[i8],
        input_shape: &Shape,
        arena: &mut Arena<'a>,
    ) -> NanoResult<(&'a mut [i8], Shape)> {
        let total = input_shape.total();
        let output = arena.alloc_i8_slice(total)?;
        output.copy_from_slice(input);
        math::tanh_i8_scaled(output, self.scale_mult, self.scale_shift);
        Ok((output, *input_shape))
    }
}

// =============================================================================
// Softmax
// =============================================================================

/// Softmax activation layer. Outputs pseudo-probabilities in [0, 127].
///
/// Typically the last layer in a classification pipeline.
pub struct SoftmaxLayer;

impl Layer for SoftmaxLayer {
    fn name(&self) -> &'static str { "Softmax" }

    fn output_shape(&self, input_shape: &Shape) -> NanoResult<Shape> {
        Ok(*input_shape)
    }

    fn forward<'a>(
        &self,
        input: &[i8],
        input_shape: &Shape,
        arena: &mut Arena<'a>,
    ) -> NanoResult<(&'a mut [i8], Shape)> {
        let total = input_shape.total();
        let output = arena.alloc_i8_slice(total)?;
        math::softmax_i8(input, output)?;
        Ok((output, *input_shape))
    }
}
