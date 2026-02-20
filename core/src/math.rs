//! Quantized i8 math operations for TinyML inference and training.
//!
//! Why i8 quantization: Cortex-M MCUs have 8-bit SIMD instructions (e.g.,
//! ARM SXTB16, SMUAD) that process 4x i8 values in parallel per 32-bit register.
//! Using i8 instead of f32 gives ~4x throughput and ~4x memory savings.
//!
//! Accumulation strategy: All multiply-accumulate uses i32 accumulators to
//! prevent overflow. For two i8 values: i8 * i8 = max 127*127 = 16129,
//! and summing N products: max N * 16129. With i32 range ±2B, we can safely
//! accumulate up to ~133,000 products without overflow.

use crate::error::{NanoError, NanoResult};

/// Quantization scale factor used when converting i32 accumulator back to i8.
///
/// Why 128: Simple bit-shift (>> 7) on MCU. Avoids expensive division.
pub const QUANT_SCALE: i32 = 128;

/// Compute the optimal requantization shift for a given inner-product dimension.
///
/// Why adaptive: With i8×i8 accumulation over `k` terms, max accumulator value
/// is `k × 127 × 127 = k × 16129`. To fit back into i8 (±127), we need to
/// divide by approximately `k × 127`. Using `>> (log2(k) + 7)` achieves this
/// with a single ARM ASR instruction.
///
/// Examples:
/// - k=9   (Conv2D 3×3, 1ch): shift=10 → ÷1024 ≈ ÷(9×127)
/// - k=27  (Conv2D 3×3, 3ch): shift=12 → ÷4096 ≈ ÷(27×127)
/// - k=144 (Dense 144→N):     shift=14 → ÷16384 ≈ ÷(144×127)
/// - k=784 (Dense 784→N):     shift=17 → ÷131072 ≈ ÷(784×127)
#[inline]
pub fn compute_requant_shift(k: usize) -> u32 {
    if k == 0 {
        return 7; // fallback
    }
    // shift = ceil(log2(k)) + 7
    // ceil(log2(k)) = 32 - leading_zeros(k-1) for k>1, or 0 for k=1
    let log2_k = if k <= 1 {
        0u32
    } else {
        32u32 - ((k - 1) as u32).leading_zeros()
    };
    log2_k + 7
}

// =============================================================================
// Matrix Operations
// =============================================================================

/// Quantized matrix multiplication: C = A × B
///
/// Computes A[m×k] × B[k×n] → output[m×n].
/// Uses i32 accumulators internally, then re-quantizes to i8.
///
/// Why this layout:
/// - `a` row-major [m*k]: sequential access → cache hit
/// - `b` row-major [k*n]: column access, but k typically small on MCU
pub fn quantized_matmul_i8(
    a: &[i8],          // [m × k] row-major
    b: &[i8],          // [k × n] row-major
    output: &mut [i8], // [m × n] row-major
    m: usize,
    k: usize,
    n: usize,
) -> NanoResult<()> {
    // Default: M=1, adaptive shift
    quantized_matmul_i8_requant(a, b, output, m, k, n, 1, compute_requant_shift(k))
}

/// Quantized matmul with TFLite-style requantization: `(acc * M) >> shift`
///
/// `requant_m`: fixed-point multiplier computed from weight/input/output scales
/// `requant_shift`: bit-shift
///
/// Uses i64 accumulator for the multiply to prevent overflow.
pub fn quantized_matmul_i8_requant(
    a: &[i8],
    b: &[i8],
    output: &mut [i8],
    m: usize,
    k: usize,
    n: usize,
    requant_m: i32,
    requant_shift: u32,
) -> NanoResult<()> {
    if a.len() != m * k {
        return Err(NanoError::DimensionMismatch { expected: m * k, actual: a.len() });
    }
    if b.len() != k * n {
        return Err(NanoError::DimensionMismatch { expected: k * n, actual: b.len() });
    }
    if output.len() != m * n {
        return Err(NanoError::DimensionMismatch { expected: m * n, actual: output.len() });
    }

    let rm = requant_m as i64;
    for i in 0..m {
        for j in 0..n {
            let mut acc: i32 = 0;
            for p in 0..k {
                acc += (a[i * k + p] as i32) * (b[p * n + j] as i32);
            }
            // TFLite-style: (acc * M) >> shift
            let scaled = ((acc as i64 * rm) >> requant_shift) as i32;
            output[i * n + j] = scaled.clamp(-128, 127) as i8;
        }
    }
    Ok(())
}

/// Add bias vector to each row of a matrix (in-place).
///
/// For matrix [m × n], adds bias[n] to every row.
pub fn quantized_add_bias(
    output: &mut [i8],
    bias: &[i8],
    m: usize,
    n: usize,
) -> NanoResult<()> {
    if bias.len() != n {
        return Err(NanoError::DimensionMismatch { expected: n, actual: bias.len() });
    }
    if output.len() != m * n {
        return Err(NanoError::DimensionMismatch { expected: m * n, actual: output.len() });
    }

    for i in 0..m {
        for j in 0..n {
            let sum = (output[i * n + j] as i16) + (bias[j] as i16);
            output[i * n + j] = sum.clamp(-128, 127) as i8;
        }
    }
    Ok(())
}

// =============================================================================
// Convolution Operations
// =============================================================================

/// Quantized 2D convolution (im2col + matmul pattern).
///
/// Input:  [in_ch × in_h × in_w]  (CHW layout)
/// Kernel: [out_ch × in_ch × kh × kw]
/// Bias:   [out_ch]
/// Output: [out_ch × out_h × out_w]
///
/// Why im2col: Converts conv2d into a matmul, reusing the optimized
/// `quantized_matmul_i8`. This is the standard approach in TFLite Micro
/// and CMSIS-NN for Cortex-M targets. The im2col buffer lives in the arena.
///
/// # Layout
/// - `im2col_buf`: Arena-allocated scratch of size `(in_ch * kh * kw) * (out_h * out_w)`
/// - After im2col, we compute: output = kernel_matrix × im2col_matrix
pub fn conv2d_i8(
    input: &[i8],         // [in_ch × in_h × in_w] CHW
    kernel: &[i8],        // [out_ch × in_ch × kh × kw]
    bias: &[i8],          // [out_ch]
    output: &mut [i8],    // [out_ch × out_h × out_w]
    im2col_buf: &mut [i8], // scratch: [(in_ch*kh*kw) × (out_h*out_w)]
    in_ch: usize,
    in_h: usize,
    in_w: usize,
    out_ch: usize,
    kh: usize,
    kw: usize,
    stride: usize,
    padding: usize,
) -> NanoResult<()> {
    if stride == 0 {
        return Err(NanoError::InvalidQuantization);
    }

    let out_h = (in_h + 2 * padding - kh) / stride + 1;
    let out_w = (in_w + 2 * padding - kw) / stride + 1;

    // Validate dimensions
    let expected_input = in_ch * in_h * in_w;
    if input.len() != expected_input {
        return Err(NanoError::DimensionMismatch { expected: expected_input, actual: input.len() });
    }
    let k_per_filter = in_ch * kh * kw;
    let expected_kernel = out_ch * k_per_filter;
    if kernel.len() != expected_kernel {
        return Err(NanoError::DimensionMismatch { expected: expected_kernel, actual: kernel.len() });
    }
    if bias.len() != out_ch {
        return Err(NanoError::DimensionMismatch { expected: out_ch, actual: bias.len() });
    }
    let spatial_out = out_h * out_w;
    let expected_output = out_ch * spatial_out;
    if output.len() != expected_output {
        return Err(NanoError::DimensionMismatch { expected: expected_output, actual: output.len() });
    }
    let expected_im2col = k_per_filter * spatial_out;
    if im2col_buf.len() < expected_im2col {
        return Err(NanoError::BufferTooSmall { required: expected_im2col, available: im2col_buf.len() });
    }

    // Phase 1: im2col — unroll input patches into columns.
    // Each column is one receptive field: [in_ch × kh × kw] elements.
    // Why: Converts sliding-window conv into a single matmul call.
    for oy in 0..out_h {
        for ox in 0..out_w {
            let col_idx = oy * out_w + ox;
            for c in 0..in_ch {
                for ky in 0..kh {
                    for kx in 0..kw {
                        let iy = (oy * stride + ky) as isize - padding as isize;
                        let ix = (ox * stride + kx) as isize - padding as isize;

                        let row_idx = c * kh * kw + ky * kw + kx;
                        // Zero-padding: out-of-bounds reads as 0
                        im2col_buf[row_idx * spatial_out + col_idx] =
                            if iy >= 0 && iy < in_h as isize && ix >= 0 && ix < in_w as isize {
                                input[c * in_h * in_w + iy as usize * in_w + ix as usize]
                            } else {
                                0 // Zero padding
                            };
                    }
                }
            }
        }
    }

    // Phase 2: matmul with default requant
    quantized_matmul_i8(
        kernel,
        &im2col_buf[..expected_im2col],
        output,
        out_ch,
        k_per_filter,
        spatial_out,
    )?;

    // Phase 3: add bias per output channel
    for c in 0..out_ch {
        for s in 0..spatial_out {
            let idx = c * spatial_out + s;
            let val = (output[idx] as i16) + (bias[c] as i16);
            output[idx] = val.clamp(-128, 127) as i8;
        }
    }

    Ok(())
}

/// Conv2D with calibrated requantization parameters.
///
/// Same as `conv2d_i8` but uses `(acc * requant_m) >> requant_shift`
/// for more accurate scale-aware re-quantization.
pub fn conv2d_i8_requant(
    input: &[i8],
    kernel: &[i8],
    bias: &[i8],
    output: &mut [i8],
    im2col_buf: &mut [i8],
    in_ch: usize,
    in_h: usize,
    in_w: usize,
    out_ch: usize,
    kh: usize,
    kw: usize,
    stride: usize,
    padding: usize,
    requant_m: i32,
    requant_shift: u32,
) -> NanoResult<()> {
    if stride == 0 {
        return Err(NanoError::InvalidQuantization);
    }

    let out_h = (in_h + 2 * padding - kh) / stride + 1;
    let out_w = (in_w + 2 * padding - kw) / stride + 1;

    let expected_input = in_ch * in_h * in_w;
    if input.len() != expected_input {
        return Err(NanoError::DimensionMismatch { expected: expected_input, actual: input.len() });
    }
    let k_per_filter = in_ch * kh * kw;
    let expected_kernel = out_ch * k_per_filter;
    if kernel.len() != expected_kernel {
        return Err(NanoError::DimensionMismatch { expected: expected_kernel, actual: kernel.len() });
    }
    if bias.len() != out_ch {
        return Err(NanoError::DimensionMismatch { expected: out_ch, actual: bias.len() });
    }
    let spatial_out = out_h * out_w;
    let expected_output = out_ch * spatial_out;
    if output.len() != expected_output {
        return Err(NanoError::DimensionMismatch { expected: expected_output, actual: output.len() });
    }
    let expected_im2col = k_per_filter * spatial_out;
    if im2col_buf.len() < expected_im2col {
        return Err(NanoError::BufferTooSmall { required: expected_im2col, available: im2col_buf.len() });
    }

    // Phase 1: im2col
    for oy in 0..out_h {
        for ox in 0..out_w {
            let col_idx = oy * out_w + ox;
            for c in 0..in_ch {
                for ky in 0..kh {
                    for kx in 0..kw {
                        let iy = (oy * stride + ky) as isize - padding as isize;
                        let ix = (ox * stride + kx) as isize - padding as isize;
                        let row_idx = c * kh * kw + ky * kw + kx;
                        im2col_buf[row_idx * spatial_out + col_idx] =
                            if iy >= 0 && iy < in_h as isize && ix >= 0 && ix < in_w as isize {
                                input[c * in_h * in_w + iy as usize * in_w + ix as usize]
                            } else {
                                0
                            };
                    }
                }
            }
        }
    }

    // Phase 2: matmul with calibrated requant
    quantized_matmul_i8_requant(
        kernel,
        &im2col_buf[..expected_im2col],
        output,
        out_ch,
        k_per_filter,
        spatial_out,
        requant_m,
        requant_shift,
    )?;

    // Phase 3: add bias
    for c in 0..out_ch {
        for s in 0..spatial_out {
            let idx = c * spatial_out + s;
            let val = (output[idx] as i16) + (bias[c] as i16);
            output[idx] = val.clamp(-128, 127) as i8;
        }
    }

    Ok(())
}

/// Calculate output dimensions for conv2d.
///
/// Returns (out_h, out_w) given input spatial dims and conv parameters.
pub fn conv2d_output_size(
    in_h: usize, in_w: usize,
    kh: usize, kw: usize,
    stride: usize, padding: usize,
) -> (usize, usize) {
    let out_h = (in_h + 2 * padding - kh) / stride + 1;
    let out_w = (in_w + 2 * padding - kw) / stride + 1;
    (out_h, out_w)
}

/// Calculate im2col buffer size needed for conv2d.
pub fn conv2d_im2col_size(
    in_ch: usize, kh: usize, kw: usize,
    out_h: usize, out_w: usize,
) -> usize {
    in_ch * kh * kw * out_h * out_w
}

// =============================================================================
// Pooling Operations
// =============================================================================

/// Max pooling 2D for quantized i8 tensors.
///
/// Input:  [channels × in_h × in_w]
/// Output: [channels × out_h × out_w]
///
/// Why max pool over avg pool: Max is a single comparison (no division).
/// On MCU without FPU, avoiding division is critical for throughput.
pub fn max_pool_2d(
    input: &[i8],
    output: &mut [i8],
    channels: usize,
    in_h: usize,
    in_w: usize,
    pool_h: usize,
    pool_w: usize,
    stride: usize,
) -> NanoResult<()> {
    if stride == 0 {
        return Err(NanoError::InvalidQuantization);
    }

    let out_h = (in_h - pool_h) / stride + 1;
    let out_w = (in_w - pool_w) / stride + 1;

    if input.len() != channels * in_h * in_w {
        return Err(NanoError::DimensionMismatch {
            expected: channels * in_h * in_w, actual: input.len(),
        });
    }
    if output.len() != channels * out_h * out_w {
        return Err(NanoError::DimensionMismatch {
            expected: channels * out_h * out_w, actual: output.len(),
        });
    }

    for c in 0..channels {
        for oy in 0..out_h {
            for ox in 0..out_w {
                let mut max_val: i8 = i8::MIN;
                for py in 0..pool_h {
                    for px in 0..pool_w {
                        let iy = oy * stride + py;
                        let ix = ox * stride + px;
                        let val = input[c * in_h * in_w + iy * in_w + ix];
                        if val > max_val {
                            max_val = val;
                        }
                    }
                }
                output[c * out_h * out_w + oy * out_w + ox] = max_val;
            }
        }
    }
    Ok(())
}

/// Calculate output dimensions for max_pool_2d.
pub fn pool_output_size(
    in_h: usize, in_w: usize,
    pool_h: usize, pool_w: usize,
    stride: usize,
) -> (usize, usize) {
    let out_h = (in_h - pool_h) / stride + 1;
    let out_w = (in_w - pool_w) / stride + 1;
    (out_h, out_w)
}

// =============================================================================
// Activation Functions
// =============================================================================

/// ReLU activation for i8 (in-place).
///
/// ReLU(x) = max(0, x). Zero-cost on i8: just compare + zero.
pub fn relu_i8(data: &mut [i8]) {
    for val in data.iter_mut() {
        if *val < 0 {
            *val = 0;
        }
    }
}

/// Sigmoid activation for i8 using 256-entry lookup table.
///
/// LUT maps i8 input [-128..127] → i8 output [0..127].
/// σ(x) scaled to i8: output = round(sigmoid(x / scale) * 127)
///
/// Why LUT: sigmoid requires exp(), which costs ~100 cycles on Cortex-M0.
/// A 256-byte LUT is O(1) — single array index — and fits in one cache line
/// on most MCUs.
///
/// Scale factor: input is treated as x/16 to spread the sigmoid curve
/// across the i8 range. sigmoid(±8) ≈ 0/1, so ±128/16 = ±8 covers full range.
pub fn sigmoid_i8(data: &mut [i8]) {
    // Pre-computed LUT: sigmoid_lut[i + 128] = round(sigmoid(i / 16.0) * 127)
    // Generated offline; values verified against f64 reference.
    static SIGMOID_LUT: [i8; 256] = [
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,
        1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  2,
        2,  2,  3,  3,  3,  3,  3,  4,  4,  4,  4,  4,  5,  5,  5,  6,
        6,  6,  7,  7,  8,  8,  9,  9, 10, 10, 11, 11, 12, 13, 14, 14,
       15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 27, 28, 30, 31, 33,
       34, 36, 37, 39, 41, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62,
       64, 65, 67, 69, 71, 73, 75, 77, 79, 81, 83, 85, 86, 88, 90, 91,
       93, 94, 96, 97, 99,100,101,103,104,105,106,107,108,109,110,111,
      112,113,113,114,115,116,116,117,117,118,118,119,119,120,120,121,
      121,121,122,122,122,123,123,123,123,123,124,124,124,124,124,125,
      125,125,125,125,125,125,125,126,126,126,126,126,126,126,126,126,
      126,126,126,126,126,126,126,126,126,127,127,127,127,127,127,127,
      127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,
      127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,
    ];

    for val in data.iter_mut() {
        let idx = (*val as i16 + 128) as usize;
        *val = SIGMOID_LUT[idx];
    }
}

/// Tanh activation for i8 using 256-entry lookup table.
///
/// LUT maps i8 input [-128..127] → i8 output [-127..127].
/// tanh(x) scaled to i8: output = round(tanh(x / scale) * 127)
///
/// Why LUT: Same as sigmoid — O(1) lookup vs ~200 cycle exp() pair.
/// Scale factor: x/32, so tanh(±4) ≈ ±1, and ±128/32 = ±4.
pub fn tanh_i8(data: &mut [i8]) {
    // Pre-computed LUT: tanh_lut[i + 128] = round(tanh(i / 32.0) * 127)
    static TANH_LUT: [i8; 256] = [
      -127,-127,-127,-127,-127,-127,-127,-127,-127,-127,-127,-127,-127,-127,-127,-127,
      -127,-127,-127,-127,-127,-127,-127,-127,-127,-127,-127,-127,-127,-126,-126,-126,
      -126,-126,-126,-126,-126,-126,-126,-126,-126,-126,-126,-126,-126,-126,-125,-125,
      -125,-125,-125,-125,-125,-125,-125,-124,-124,-124,-124,-124,-123,-123,-123,-123,
      -122,-122,-122,-122,-121,-121,-120,-120,-120,-119,-119,-118,-118,-117,-116,-116,
      -115,-114,-113,-113,-112,-111,-110,-109,-108,-107,-105,-104,-103,-101,-100, -98,
       -97, -95, -93, -91, -89, -87, -85, -83, -81, -78, -76, -73, -70, -68, -65, -62,
       -59, -56, -52, -49, -46, -42, -38, -35, -31, -27, -24, -20, -16, -12,  -8,  -4,
         0,   4,   8,  12,  16,  20,  24,  27,  31,  35,  38,  42,  46,  49,  52,  56,
        59,  62,  65,  68,  70,  73,  76,  78,  81,  83,  85,  87,  89,  91,  93,  95,
        97,  98, 100, 101, 103, 104, 105, 107, 108, 109, 110, 111, 112, 113, 113, 114,
       115, 116, 116, 117, 118, 118, 119, 119, 120, 120, 120, 121, 121, 122, 122, 122,
       122, 123, 123, 123, 123, 124, 124, 124, 124, 124, 125, 125, 125, 125, 125, 125,
       125, 125, 125, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126,
       126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127,
       127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127,
    ];

    for val in data.iter_mut() {
        let idx = (*val as i16 + 128) as usize;
        *val = TANH_LUT[idx];
    }
}

/// Scale-aware sigmoid for i8 using the SAME 256-entry LUT.
///
/// The standard `sigmoid_i8` assumes input i8 represents `val / 16.0` in float.
/// But after requantization, i8 actually represents `val * S_y` (output scale).
///
/// Fix: rescale BEFORE LUT lookup so the LUT sees the correct float-equivalent:
///   index = clamp( (val * scale_mult) >> scale_shift, -128, 127 )
///   output = SIGMOID_LUT[index + 128]
///
/// Where `scale_mult / 2^scale_shift ≈ S_y * 16.0` (maps from S_y units to /16 units).
///
/// Why this works: LUT[i+128] = σ(i/16) * 127. After rescaling, i = val*S_y*16,
/// so LUT lookups become σ(val*S_y) * 127 = σ(float_val) * 127. Exact.
pub fn sigmoid_i8_scaled(data: &mut [i8], scale_mult: i32, scale_shift: u32) {
    static SIGMOID_LUT: [i8; 256] = [
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,
        1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  2,
        2,  2,  3,  3,  3,  3,  3,  4,  4,  4,  4,  4,  5,  5,  5,  6,
        6,  6,  7,  7,  8,  8,  9,  9, 10, 10, 11, 11, 12, 13, 14, 14,
       15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 27, 28, 30, 31, 33,
       34, 36, 37, 39, 41, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62,
       64, 65, 67, 69, 71, 73, 75, 77, 79, 81, 83, 85, 86, 88, 90, 91,
       93, 94, 96, 97, 99,100,101,103,104,105,106,107,108,109,110,111,
      112,113,113,114,115,116,116,117,117,118,118,119,119,120,120,121,
      121,121,122,122,122,123,123,123,123,123,124,124,124,124,124,125,
      125,125,125,125,125,125,125,126,126,126,126,126,126,126,126,126,
      126,126,126,126,126,126,126,126,126,127,127,127,127,127,127,127,
      127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,
      127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,
    ];

    for val in data.iter_mut() {
        // Rescale: index = (val * scale_mult) >> scale_shift
        // This converts from S_y units to the LUT's expected /16 units
        let rescaled = ((*val as i32 * scale_mult) >> scale_shift).clamp(-128, 127);
        let idx = (rescaled + 128) as usize;
        *val = SIGMOID_LUT[idx];
    }
}

/// Scale-aware tanh for i8 using the SAME 256-entry LUT.
///
/// Same principle as `sigmoid_i8_scaled`: rescale before LUT lookup.
/// `scale_mult / 2^scale_shift ≈ S_y * 32.0` (maps from S_y units to /32 units).
pub fn tanh_i8_scaled(data: &mut [i8], scale_mult: i32, scale_shift: u32) {
    static TANH_LUT: [i8; 256] = [
      -127,-127,-127,-127,-127,-127,-127,-127,-127,-127,-127,-127,-127,-127,-127,-127,
      -127,-127,-127,-127,-127,-127,-127,-127,-127,-127,-127,-127,-127,-126,-126,-126,
      -126,-126,-126,-126,-126,-126,-126,-126,-126,-126,-126,-126,-126,-126,-125,-125,
      -125,-125,-125,-125,-125,-125,-125,-124,-124,-124,-124,-124,-123,-123,-123,-123,
      -122,-122,-122,-122,-121,-121,-120,-120,-120,-119,-119,-118,-118,-117,-116,-116,
      -115,-114,-113,-113,-112,-111,-110,-109,-108,-107,-105,-104,-103,-101,-100, -98,
       -97, -95, -93, -91, -89, -87, -85, -83, -81, -78, -76, -73, -70, -68, -65, -62,
       -59, -56, -52, -49, -46, -42, -38, -35, -31, -27, -24, -20, -16, -12,  -8,  -4,
         0,   4,   8,  12,  16,  20,  24,  27,  31,  35,  38,  42,  46,  49,  52,  56,
        59,  62,  65,  68,  70,  73,  76,  78,  81,  83,  85,  87,  89,  91,  93,  95,
        97,  98, 100, 101, 103, 104, 105, 107, 108, 109, 110, 111, 112, 113, 113, 114,
       115, 116, 116, 117, 118, 118, 119, 119, 120, 120, 120, 121, 121, 122, 122, 122,
       122, 123, 123, 123, 123, 124, 124, 124, 124, 124, 125, 125, 125, 125, 125, 125,
       125, 125, 125, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126,
       126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127,
       127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127,
    ];

    for val in data.iter_mut() {
        let rescaled = ((*val as i32 * scale_mult) >> scale_shift).clamp(-128, 127);
        let idx = (rescaled + 128) as usize;
        *val = TANH_LUT[idx];
    }
}

/// Approximate softmax for i8 values.
///
/// Converts logits into pseudo-probabilities in i8 range [0, 127].
pub fn softmax_i8(logits: &[i8], output: &mut [i8]) -> NanoResult<()> {
    if logits.len() != output.len() {
        return Err(NanoError::DimensionMismatch {
            expected: logits.len(), actual: output.len(),
        });
    }

    let n = logits.len();
    if n == 0 { return Ok(()); }

    // Find max for numerical stability
    let mut max_val = logits[0];
    for &v in logits.iter().skip(1) {
        if v > max_val { max_val = v; }
    }

    // Piecewise-linear exp approximation + normalize
    let mut exp_vals = [0i32; 256];
    let exp_slice = &mut exp_vals[..n];
    let mut sum: i32 = 0;

    for i in 0..n {
        let shifted = (logits[i] as i32) - (max_val as i32);
        let exp_approx = (128i32 + 2 * shifted).max(0);
        exp_slice[i] = exp_approx;
        sum += exp_approx;
    }

    if sum == 0 {
        let uniform = (127i32 / n as i32) as i8;
        for val in output.iter_mut() { *val = uniform; }
    } else {
        for i in 0..n {
            output[i] = ((exp_slice[i] * 127) / sum).clamp(0, 127) as i8;
        }
    }
    Ok(())
}

/// Argmax of an i8 slice. Returns index of the maximum value.
pub fn argmax_i8(data: &[i8]) -> NanoResult<usize> {
    if data.is_empty() {
        return Err(NanoError::InvalidInputLength { expected: 1, actual: 0 });
    }

    let mut max_idx = 0;
    let mut max_val = data[0];
    for (i, &val) in data.iter().enumerate().skip(1) {
        if val > max_val {
            max_val = val;
            max_idx = i;
        }
    }
    Ok(max_idx)
}
