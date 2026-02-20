//! Pooling layers: MaxPool2D.
//!
//! Reduces spatial dimensions of feature maps.
//! No learnable parameters — pure spatial downsampling.

use crate::arena::Arena;
use crate::error::{NanoError, NanoResult};
use crate::math;
use super::{Layer, Shape};

/// Max pooling 2D layer.
///
/// Reduces [C, H, W] → [C, H', W'] where H' = (H - pool_h) / stride + 1.
/// Takes the maximum value in each pooling window.
///
/// Why max over avg: Max is a simple comparison (no division needed on MCU).
pub struct MaxPool2DLayer {
    pub pool_h: usize,
    pub pool_w: usize,
    pub stride: usize,
}

impl MaxPool2DLayer {
    /// Create a new max pooling layer.
    pub fn new(pool_h: usize, pool_w: usize, stride: usize) -> NanoResult<Self> {
        if stride == 0 {
            return Err(NanoError::InvalidQuantization);
        }
        Ok(Self { pool_h, pool_w, stride })
    }

    /// Convenience: square pooling (e.g., 2×2 with stride 2).
    pub fn square(size: usize, stride: usize) -> NanoResult<Self> {
        Self::new(size, size, stride)
    }
}

impl Layer for MaxPool2DLayer {
    fn name(&self) -> &'static str { "MaxPool2D" }

    fn output_shape(&self, input_shape: &Shape) -> NanoResult<Shape> {
        if input_shape.ndim != 3 {
            return Err(NanoError::DimensionMismatch {
                expected: 3, actual: input_shape.ndim,
            });
        }

        let channels = input_shape.dims[0];
        let in_h = input_shape.dims[1];
        let in_w = input_shape.dims[2];

        let (out_h, out_w) = math::pool_output_size(
            in_h, in_w, self.pool_h, self.pool_w, self.stride,
        );

        Ok(Shape::d3(channels, out_h, out_w))
    }

    fn forward<'a>(
        &self,
        input: &[i8],
        input_shape: &Shape,
        arena: &mut Arena<'a>,
    ) -> NanoResult<(&'a mut [i8], Shape)> {
        let out_shape = self.output_shape(input_shape)?;
        let channels = input_shape.dims[0];
        let in_h = input_shape.dims[1];
        let in_w = input_shape.dims[2];

        let out_total = out_shape.total();
        let output = arena.alloc_i8_slice(out_total)?;

        math::max_pool_2d(
            input, output,
            channels, in_h, in_w,
            self.pool_h, self.pool_w, self.stride,
        )?;

        Ok((output, out_shape))
    }
}
