//! Conv2D layer with frozen weights in Flash.

use crate::arena::Arena;
use crate::error::{NanoError, NanoResult};
use crate::math::{self, compute_requant_shift};
use super::{Layer, Shape};

pub struct FrozenConv2D {
    kernel: &'static [i8],
    bias: &'static [i8],
    pub in_ch: usize,
    pub out_ch: usize,
    pub kh: usize,
    pub kw: usize,
    pub stride: usize,
    pub padding: usize,
    requant_m: i32,
    requant_shift: u32,
}

impl FrozenConv2D {
    pub fn new(
        kernel: &'static [i8], bias: &'static [i8],
        in_ch: usize, out_ch: usize, kh: usize, kw: usize,
        stride: usize, padding: usize,
    ) -> NanoResult<Self> {
        let k_per_filter = in_ch * kh * kw;
        Self::new_with_requant(kernel, bias, in_ch, out_ch, kh, kw, stride, padding,
                               1, compute_requant_shift(k_per_filter))
    }

    pub fn new_with_requant(
        kernel: &'static [i8], bias: &'static [i8],
        in_ch: usize, out_ch: usize, kh: usize, kw: usize,
        stride: usize, padding: usize,
        requant_m: i32, requant_shift: u32,
    ) -> NanoResult<Self> {
        let expected_kernel = out_ch * in_ch * kh * kw;
        if kernel.len() != expected_kernel {
            return Err(NanoError::DimensionMismatch { expected: expected_kernel, actual: kernel.len() });
        }
        if bias.len() != out_ch {
            return Err(NanoError::DimensionMismatch { expected: out_ch, actual: bias.len() });
        }
        if stride == 0 { return Err(NanoError::InvalidQuantization); }
        Ok(Self { kernel, bias, in_ch, out_ch, kh, kw, stride, padding, requant_m, requant_shift })
    }
}

impl Layer for FrozenConv2D {
    fn name(&self) -> &'static str { "FrozenConv2D" }
    fn output_shape(&self, input_shape: &Shape) -> NanoResult<Shape> {
        if input_shape.ndim != 3 {
            return Err(NanoError::DimensionMismatch { expected: 3, actual: input_shape.ndim });
        }
        let in_h = input_shape.dims[1];
        let in_w = input_shape.dims[2];
        let (out_h, out_w) = math::conv2d_output_size(in_h, in_w, self.kh, self.kw, self.stride, self.padding);
        Ok(Shape::d3(self.out_ch, out_h, out_w))
    }
    fn forward<'a>(&self, input: &[i8], input_shape: &Shape, arena: &mut Arena<'a>) -> NanoResult<(&'a mut [i8], Shape)> {
        let out_shape = self.output_shape(input_shape)?;
        let in_h = input_shape.dims[1];
        let in_w = input_shape.dims[2];
        let out_h = out_shape.dims[1];
        let out_w = out_shape.dims[2];
        let out_total = out_shape.total();
        let output = arena.alloc_i8_slice(out_total)?;
        let im2col_size = math::conv2d_im2col_size(self.in_ch, self.kh, self.kw, out_h, out_w);
        let im2col_buf = arena.alloc_i8_slice(im2col_size)?;
        math::conv2d_i8_requant(
            input, self.kernel, self.bias, output, im2col_buf,
            self.in_ch, in_h, in_w, self.out_ch,
            self.kh, self.kw, self.stride, self.padding,
            self.requant_m, self.requant_shift,
        )?;
        Ok((output, out_shape))
    }
}
