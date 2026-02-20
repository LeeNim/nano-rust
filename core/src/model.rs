//! SequentialModel: chain layers into a pipeline.
//!
//! Replaces the old hardcoded `NanoModel<IN, HIDDEN, CLASSES>` with a
//! dynamic, composable model that accepts any sequence of Layer objects.
//!
//! ```text
//! Input → Layer[0] → Layer[1] → ... → Layer[N-1] → Output
//! ```
//!
//! Each layer declares its output shape, enabling shape validation
//! at model construction time (before any data flows through).

use crate::arena::Arena;
use crate::error::NanoResult;
use crate::layers::{Layer, Shape};
use crate::math;

/// A sequential model: layers executed in order, output feeds into next.
///
/// # Lifetime `'m`
/// The model borrows layer references — layers must live at least as long
/// as the model. On MCU, layers are typically `static` or stack-allocated.
pub struct SequentialModel<'m> {
    /// Layer references in execution order.
    layers: &'m [&'m dyn Layer],

    /// Cached input shape for validation.
    input_shape: Shape,
}

impl<'m> SequentialModel<'m> {
    /// Create a new sequential model from a slice of layer references.
    ///
    /// Validates shape compatibility: output of each layer must be a valid
    /// input for the next layer. Fails fast at construction time.
    pub fn new(
        layers: &'m [&'m dyn Layer],
        input_shape: Shape,
    ) -> NanoResult<Self> {
        // Validate layer chain at construction time
        let mut shape = input_shape;
        for layer in layers.iter() {
            shape = layer.output_shape(&shape)?;
        }

        Ok(Self { layers, input_shape })
    }

    /// Run forward pass through all layers.
    ///
    /// # Returns
    /// `(output_slice, output_shape)` — output is arena-allocated.
    ///
    /// # Memory
    /// All intermediate activations live in the arena. The arena must have
    /// enough space for all layers' outputs simultaneously (bump allocator,
    /// no free until reset).
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
            // No layers — copy input to arena and return
            let out = arena.alloc_i8_slice(input.len())?;
            out.copy_from_slice(input);
            return Ok((out, self.input_shape));
        }

        // First layer takes original input
        let (first_out, first_shape) = self.layers[0].forward(input, &self.input_shape, arena)?;

        if self.layers.len() == 1 {
            return Ok((first_out, first_shape));
        }

        // Remaining layers chain: each layer's output feeds the next.
        // We cast &mut [i8] to &[i8] for the next layer's input.
        let mut current_shape = first_shape;
        let mut current_ptr = first_out.as_ptr();
        let mut current_len = first_out.len();

        let mut final_result: (&'a mut [i8], Shape) = (first_out, first_shape);

        for layer in self.layers.iter().skip(1) {
            // SAFETY: The previous output is arena-allocated and will not be
            // overwritten (bump allocator). We re-borrow as immutable for the
            // next layer's input while letting the arena allocate new output.
            let prev_data = unsafe { core::slice::from_raw_parts(current_ptr, current_len) };

            let (out, out_shape) = layer.forward(prev_data, &current_shape, arena)?;
            current_ptr = out.as_ptr();
            current_len = out.len();
            current_shape = out_shape;
            final_result = (out, out_shape);
        }

        Ok(final_result)
    }

    /// Predict class index: forward + argmax on final output.
    pub fn predict(
        &self,
        input: &[i8],
        arena: &mut Arena<'_>,
    ) -> NanoResult<usize> {
        let (logits, _shape) = self.forward(input, arena)?;
        math::argmax_i8(logits)
    }

    /// Get the expected output shape (computed from layer chain).
    pub fn output_shape(&self) -> NanoResult<Shape> {
        let mut shape = self.input_shape;
        for layer in self.layers.iter() {
            shape = layer.output_shape(&shape)?;
        }
        Ok(shape)
    }

    /// Get the input shape.
    pub fn input_shape(&self) -> Shape {
        self.input_shape
    }

    /// Number of layers.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Get layer by index.
    pub fn layer(&self, idx: usize) -> Option<&dyn Layer> {
        self.layers.get(idx).map(|l| *l)
    }

    /// Estimate minimum arena size needed for a forward pass.
    ///
    /// Sum of all layers' output sizes. Conservative estimate since
    /// bump allocator doesn't free between layers.
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
