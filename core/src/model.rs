//! SequentialModel: chain layers into a pipeline.

use crate::arena::Arena;
use crate::error::NanoResult;
use crate::layers::{Layer, Shape};
use crate::math;

pub struct SequentialModel<'m> {
    layers: &'m [&'m dyn Layer],
    input_shape: Shape,
}

impl<'m> SequentialModel<'m> {
    pub fn new(
        layers: &'m [&'m dyn Layer],
        input_shape: Shape,
    ) -> NanoResult<Self> {
        let mut shape = input_shape;
        for layer in layers.iter() {
            shape = layer.output_shape(&shape)?;
        }
        Ok(Self { layers, input_shape })
    }

    pub fn forward<'a>(
        &self,
        input: &[i8],
        arena: &mut Arena<'a>,
    ) -> NanoResult<(&'a mut [i8], Shape)> {
        if input.len() != self.input_shape.total() {
            return Err(crate::error::NanoError::DimensionMismatch {
                expected: self.input_shape.total(),
                actual: input.len(),
            });
        }

        if self.layers.is_empty() {
            let out = arena.alloc_i8_slice(input.len())?;
            out.copy_from_slice(input);
            return Ok((out, self.input_shape));
        }

        let (first_out, first_shape) = self.layers[0].forward(input, &self.input_shape, arena)?;

        if self.layers.len() == 1 {
            return Ok((first_out, first_shape));
        }

        let mut current_shape = first_shape;
        let mut current_ptr = first_out.as_ptr();
        let mut current_len = first_out.len();
        let mut final_result: (&'a mut [i8], Shape) = (first_out, first_shape);

        for layer in self.layers.iter().skip(1) {
            let prev_data = unsafe { core::slice::from_raw_parts(current_ptr, current_len) };
            let (out, out_shape) = layer.forward(prev_data, &current_shape, arena)?;
            current_ptr = out.as_ptr();
            current_len = out.len();
            current_shape = out_shape;
            final_result = (out, out_shape);
        }

        Ok(final_result)
    }

    pub fn predict(
        &self,
        input: &[i8],
        arena: &mut Arena<'_>,
    ) -> NanoResult<usize> {
        let (logits, _shape) = self.forward(input, arena)?;
        math::argmax_i8(logits)
    }

    pub fn output_shape(&self) -> NanoResult<Shape> {
        let mut shape = self.input_shape;
        for layer in self.layers.iter() {
            shape = layer.output_shape(&shape)?;
        }
        Ok(shape)
    }

    pub fn input_shape(&self) -> Shape { self.input_shape }
    pub fn num_layers(&self) -> usize { self.layers.len() }

    pub fn layer(&self, idx: usize) -> Option<&dyn Layer> {
        self.layers.get(idx).map(|l| *l)
    }

    pub fn estimate_arena_size(&self) -> NanoResult<usize> {
        let mut total: usize = 0;
        let mut shape = self.input_shape;
        for layer in self.layers.iter() {
            shape = layer.output_shape(&shape)?;
            total += shape.total();
        }
        Ok(total)
    }
}
