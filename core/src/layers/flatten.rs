//! Flatten layer: reshapes multi-dimensional tensor to 1D.
//!
//! Zero-copy conceptually: the data doesn't move, only the Shape changes.
//! However, since the Layer trait returns arena-allocated output, we
//! do a copy to maintain the uniform interface. The compiler may optimize
//! this away in release mode.

use crate::arena::Arena;
use crate::error::NanoResult;
use super::{Layer, Shape};

/// Flatten layer: [C, H, W] → [C*H*W].
///
/// Used between Conv2D/Pooling and Dense layers.
/// No parameters, no computation — just shape transformation.
pub struct FlattenLayer;

impl Layer for FlattenLayer {
    fn name(&self) -> &'static str { "Flatten" }

    fn output_shape(&self, input_shape: &Shape) -> NanoResult<Shape> {
        Ok(Shape::d1(input_shape.total()))
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
        Ok((output, Shape::d1(total)))
    }
}
