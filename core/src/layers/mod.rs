//! Modular Layer Zoo for TinyML.

pub mod activations;
pub mod conv;
pub mod dense;
pub mod flatten;
pub mod pooling;

use crate::arena::Arena;
use crate::error::NanoResult;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Shape {
    pub dims: [usize; 4],
    pub ndim: usize,
}

impl Shape {
    pub const fn d1(size: usize) -> Self {
        Self { dims: [size, 0, 0, 0], ndim: 1 }
    }
    pub const fn d2(d0: usize, d1: usize) -> Self {
        Self { dims: [d0, d1, 0, 0], ndim: 2 }
    }
    pub const fn d3(c: usize, h: usize, w: usize) -> Self {
        Self { dims: [c, h, w, 0], ndim: 3 }
    }
    pub const fn d4(n: usize, c: usize, h: usize, w: usize) -> Self {
        Self { dims: [n, c, h, w], ndim: 4 }
    }
    pub fn total(&self) -> usize {
        let mut t = 1;
        for i in 0..self.ndim { t *= self.dims[i]; }
        t
    }
    pub fn channels(&self) -> usize {
        if self.ndim >= 3 { self.dims[0] } else { 1 }
    }
    pub fn height(&self) -> usize {
        if self.ndim >= 3 { self.dims[1] } else { 0 }
    }
    pub fn width(&self) -> usize {
        if self.ndim >= 3 { self.dims[2] } else { 0 }
    }
}

pub trait Layer: Send {
    fn name(&self) -> &'static str;
    fn output_shape(&self, input_shape: &Shape) -> NanoResult<Shape>;
    fn forward<'a>(
        &self,
        input: &[i8],
        input_shape: &Shape,
        arena: &mut Arena<'a>,
    ) -> NanoResult<(&'a mut [i8], Shape)>;
}

pub use activations::{ReLULayer, SigmoidLayer, TanhLayer, SoftmaxLayer, ScaledSigmoidLayer, ScaledTanhLayer};
pub use conv::FrozenConv2D;
pub use dense::{FrozenDense, TrainableDense};
pub use flatten::FlattenLayer;
pub use pooling::MaxPool2DLayer;
