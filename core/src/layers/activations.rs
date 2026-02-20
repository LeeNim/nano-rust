//! Activation layers: ReLU, Sigmoid, Tanh, Softmax.

use crate::arena::Arena;
use crate::error::NanoResult;
use crate::math;
use super::{Layer, Shape};

pub struct ReLULayer;
impl Layer for ReLULayer {
    fn name(&self) -> &'static str { "ReLU" }
    fn output_shape(&self, input_shape: &Shape) -> NanoResult<Shape> { Ok(*input_shape) }
    fn forward<'a>(&self, input: &[i8], input_shape: &Shape, arena: &mut Arena<'a>) -> NanoResult<(&'a mut [i8], Shape)> {
        let total = input_shape.total();
        let output = arena.alloc_i8_slice(total)?;
        output.copy_from_slice(input);
        math::relu_i8(output);
        Ok((output, *input_shape))
    }
}

pub struct SigmoidLayer;
impl Layer for SigmoidLayer {
    fn name(&self) -> &'static str { "Sigmoid" }
    fn output_shape(&self, input_shape: &Shape) -> NanoResult<Shape> { Ok(*input_shape) }
    fn forward<'a>(&self, input: &[i8], input_shape: &Shape, arena: &mut Arena<'a>) -> NanoResult<(&'a mut [i8], Shape)> {
        let total = input_shape.total();
        let output = arena.alloc_i8_slice(total)?;
        output.copy_from_slice(input);
        math::sigmoid_i8(output);
        Ok((output, *input_shape))
    }
}

pub struct TanhLayer;
impl Layer for TanhLayer {
    fn name(&self) -> &'static str { "Tanh" }
    fn output_shape(&self, input_shape: &Shape) -> NanoResult<Shape> { Ok(*input_shape) }
    fn forward<'a>(&self, input: &[i8], input_shape: &Shape, arena: &mut Arena<'a>) -> NanoResult<(&'a mut [i8], Shape)> {
        let total = input_shape.total();
        let output = arena.alloc_i8_slice(total)?;
        output.copy_from_slice(input);
        math::tanh_i8(output);
        Ok((output, *input_shape))
    }
}

pub struct ScaledSigmoidLayer {
    pub scale_mult: i32,
    pub scale_shift: u32,
}
impl Layer for ScaledSigmoidLayer {
    fn name(&self) -> &'static str { "ScaledSigmoid" }
    fn output_shape(&self, input_shape: &Shape) -> NanoResult<Shape> { Ok(*input_shape) }
    fn forward<'a>(&self, input: &[i8], input_shape: &Shape, arena: &mut Arena<'a>) -> NanoResult<(&'a mut [i8], Shape)> {
        let total = input_shape.total();
        let output = arena.alloc_i8_slice(total)?;
        output.copy_from_slice(input);
        math::sigmoid_i8_scaled(output, self.scale_mult, self.scale_shift);
        Ok((output, *input_shape))
    }
}

pub struct ScaledTanhLayer {
    pub scale_mult: i32,
    pub scale_shift: u32,
}
impl Layer for ScaledTanhLayer {
    fn name(&self) -> &'static str { "ScaledTanh" }
    fn output_shape(&self, input_shape: &Shape) -> NanoResult<Shape> { Ok(*input_shape) }
    fn forward<'a>(&self, input: &[i8], input_shape: &Shape, arena: &mut Arena<'a>) -> NanoResult<(&'a mut [i8], Shape)> {
        let total = input_shape.total();
        let output = arena.alloc_i8_slice(total)?;
        output.copy_from_slice(input);
        math::tanh_i8_scaled(output, self.scale_mult, self.scale_shift);
        Ok((output, *input_shape))
    }
}

pub struct SoftmaxLayer;
impl Layer for SoftmaxLayer {
    fn name(&self) -> &'static str { "Softmax" }
    fn output_shape(&self, input_shape: &Shape) -> NanoResult<Shape> { Ok(*input_shape) }
    fn forward<'a>(&self, input: &[i8], input_shape: &Shape, arena: &mut Arena<'a>) -> NanoResult<(&'a mut [i8], Shape)> {
        let total = input_shape.total();
        let output = arena.alloc_i8_slice(total)?;
        math::softmax_i8(input, output)?;
        Ok((output, *input_shape))
    }
}
