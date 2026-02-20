//! Dense (fully-connected) layers: Frozen (Flash) and Trainable (RAM).

use crate::arena::Arena;
use crate::error::{NanoError, NanoResult};
use crate::math::compute_requant_shift;
use super::{Layer, Shape};

pub struct FrozenDense {
    weights: &'static [i8],
    bias: &'static [i8],
    in_features: usize,
    out_features: usize,
    requant_m: i32,
    requant_shift: u32,
}

impl FrozenDense {
    pub fn new(
        weights: &'static [i8], bias: &'static [i8],
        in_features: usize, out_features: usize,
    ) -> NanoResult<Self> {
        Self::new_with_requant(weights, bias, in_features, out_features, 1, compute_requant_shift(in_features))
    }

    pub fn new_with_requant(
        weights: &'static [i8], bias: &'static [i8],
        in_features: usize, out_features: usize,
        requant_m: i32, requant_shift: u32,
    ) -> NanoResult<Self> {
        if weights.len() != out_features * in_features {
            return Err(NanoError::DimensionMismatch { expected: out_features * in_features, actual: weights.len() });
        }
        if bias.len() != out_features {
            return Err(NanoError::DimensionMismatch { expected: out_features, actual: bias.len() });
        }
        Ok(Self { weights, bias, in_features, out_features, requant_m, requant_shift })
    }
}

impl Layer for FrozenDense {
    fn name(&self) -> &'static str { "FrozenDense" }
    fn output_shape(&self, _input_shape: &Shape) -> NanoResult<Shape> {
        Ok(Shape::d1(self.out_features))
    }
    fn forward<'a>(&self, input: &[i8], input_shape: &Shape, arena: &mut Arena<'a>) -> NanoResult<(&'a mut [i8], Shape)> {
        let in_total = input_shape.total();
        if in_total != self.in_features {
            return Err(NanoError::DimensionMismatch { expected: self.in_features, actual: in_total });
        }
        let output = arena.alloc_i8_slice(self.out_features)?;
        let m = self.requant_m as i64;
        let shift = self.requant_shift;
        for o in 0..self.out_features {
            let mut acc: i32 = 0;
            for i in 0..self.in_features {
                acc += (self.weights[o * self.in_features + i] as i32) * (input[i] as i32);
            }
            let scaled = ((acc as i64 * m) >> shift) as i32;
            let with_bias = scaled + (self.bias[o] as i32);
            output[o] = with_bias.clamp(-128, 127) as i8;
        }
        let out_shape = self.output_shape(input_shape)?;
        Ok((output, out_shape))
    }
}

pub struct TrainableDense {
    #[cfg(feature = "std")]
    weights: std::vec::Vec<i8>,
    #[cfg(not(feature = "std"))]
    weights: [i8; 8192],

    #[cfg(feature = "std")]
    bias: std::vec::Vec<i8>,
    #[cfg(not(feature = "std"))]
    bias: [i8; 256],

    in_features: usize,
    out_features: usize,
}

impl TrainableDense {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        Self {
            #[cfg(feature = "std")]
            weights: std::vec![0i8; out_features * in_features],
            #[cfg(not(feature = "std"))]
            weights: [0i8; 8192],
            #[cfg(feature = "std")]
            bias: std::vec![0i8; out_features],
            #[cfg(not(feature = "std"))]
            bias: [0i8; 256],
            in_features,
            out_features,
        }
    }

    pub fn from_weights(weights: &[i8], bias: &[i8], in_features: usize, out_features: usize) -> NanoResult<Self> {
        if weights.len() != out_features * in_features {
            return Err(NanoError::DimensionMismatch { expected: out_features * in_features, actual: weights.len() });
        }
        if bias.len() != out_features {
            return Err(NanoError::DimensionMismatch { expected: out_features, actual: bias.len() });
        }
        let mut layer = Self::new(in_features, out_features);
        layer.weight_slice_mut().copy_from_slice(weights);
        layer.bias_slice_mut().copy_from_slice(bias);
        Ok(layer)
    }

    fn weight_slice(&self) -> &[i8] {
        #[cfg(feature = "std")] { &self.weights }
        #[cfg(not(feature = "std"))] { &self.weights[..self.out_features * self.in_features] }
    }
    fn weight_slice_mut(&mut self) -> &mut [i8] {
        #[cfg(feature = "std")] { &mut self.weights }
        #[cfg(not(feature = "std"))] { let len = self.out_features * self.in_features; &mut self.weights[..len] }
    }
    fn bias_slice(&self) -> &[i8] {
        #[cfg(feature = "std")] { &self.bias }
        #[cfg(not(feature = "std"))] { &self.bias[..self.out_features] }
    }
    fn bias_slice_mut(&mut self) -> &mut [i8] {
        #[cfg(feature = "std")] { &mut self.bias }
        #[cfg(not(feature = "std"))] { let len = self.out_features; &mut self.bias[..len] }
    }

    pub fn backward(&mut self, input: &[i8], output_grad: &[i8], learning_rate: i8) -> NanoResult<()> {
        let in_f = self.in_features;
        let out_f = self.out_features;
        if input.len() != in_f { return Err(NanoError::DimensionMismatch { expected: in_f, actual: input.len() }); }
        if output_grad.len() != out_f { return Err(NanoError::DimensionMismatch { expected: out_f, actual: output_grad.len() }); }
        let weights = self.weight_slice_mut();
        for o in 0..out_f {
            for i in 0..in_f {
                let grad = (output_grad[o] as i32) * (input[i] as i32);
                let update = (grad * (learning_rate as i32)) >> 10;
                let new_w = (weights[o * in_f + i] as i32) - update;
                weights[o * in_f + i] = new_w.clamp(-128, 127) as i8;
            }
        }
        let bias = self.bias_slice_mut();
        for o in 0..out_f {
            let bias_update = ((output_grad[o] as i32) * (learning_rate as i32)) >> 8;
            let new_b = (bias[o] as i32) - bias_update;
            bias[o] = new_b.clamp(-128, 127) as i8;
        }
        Ok(())
    }

    pub fn weights(&self) -> &[i8] { self.weight_slice() }
    pub fn bias(&self) -> &[i8] { self.bias_slice() }

    pub fn load_weights(&mut self, weights: &[i8], bias: &[i8]) -> NanoResult<()> {
        if weights.len() != self.out_features * self.in_features {
            return Err(NanoError::DimensionMismatch { expected: self.out_features * self.in_features, actual: weights.len() });
        }
        if bias.len() != self.out_features {
            return Err(NanoError::DimensionMismatch { expected: self.out_features, actual: bias.len() });
        }
        self.weight_slice_mut().copy_from_slice(weights);
        self.bias_slice_mut().copy_from_slice(bias);
        Ok(())
    }
}

impl Layer for TrainableDense {
    fn name(&self) -> &'static str { "TrainableDense" }
    fn output_shape(&self, _input_shape: &Shape) -> NanoResult<Shape> {
        Ok(Shape::d1(self.out_features))
    }
    fn forward<'a>(&self, input: &[i8], input_shape: &Shape, arena: &mut Arena<'a>) -> NanoResult<(&'a mut [i8], Shape)> {
        let in_total = input_shape.total();
        if in_total != self.in_features {
            return Err(NanoError::DimensionMismatch { expected: self.in_features, actual: in_total });
        }
        let output = arena.alloc_i8_slice(self.out_features)?;
        let shift = compute_requant_shift(self.in_features);
        let w = self.weight_slice();
        let b = self.bias_slice();
        for o in 0..self.out_features {
            let mut acc: i32 = 0;
            for i in 0..self.in_features {
                acc += (w[o * self.in_features + i] as i32) * (input[i] as i32);
            }
            let scaled = acc >> shift;
            let with_bias = scaled + (b[o] as i32);
            output[o] = with_bias.clamp(-128, 127) as i8;
        }
        let out_shape = self.output_shape(input_shape)?;
        Ok((output, out_shape))
    }
}
