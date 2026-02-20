//! Fixed-size i8 Tensor with compile-time size checking via const generics.
//!
//! Why const generics: The compiler knows exact tensor size at compile time,
//! enabling it to unroll loops, inline operations, and catch size mismatches as
//! compilation errors rather than runtime panics. On MCU, this means zero overhead
//! from dynamic size checks in the hot path.
//!
//! Design note: We use a single const generic `N` for total element count
//! (instead of `ROWS * COLS`) because `generic_const_exprs` is unstable in Rust.
//! The caller tracks row/col semantics externally.

use crate::error::{NanoError, NanoResult};

/// A fixed-size tensor of `i8` values with compile-time element count.
///
/// Memory layout: Contiguous `[i8; N]` on the stack.
/// This guarantees optimal cache line utilization — sequential access patterns
/// hit the hardware prefetcher perfectly on Cortex-M with D-cache.
///
/// # Type Parameters
/// - `N`: Total number of elements (compile-time constant)
///
/// For a matrix of shape [rows × cols], set N = rows * cols.
/// The caller is responsible for maintaining row/col semantics.
#[derive(Debug, Clone)]
pub struct Tensor<const N: usize> {
    /// Why [i8; N]: Flat layout ensures contiguous memory for SIMD-friendly
    /// access patterns. No extra metadata overhead (zero-cost abstraction).
    data: [i8; N],
}

impl<const N: usize> Tensor<N> {
    /// Create a zero-initialized tensor.
    ///
    /// Why: Default state for intermediate activation buffers.
    /// Cost: Just zeroing stack memory — compiler optimizes to memset.
    pub const fn zeros() -> Self {
        Self { data: [0i8; N] }
    }

    /// Create a tensor from a flat slice.
    ///
    /// Returns error if slice length != N.
    /// Why Result instead of panic: MCU cannot recover from panic.
    pub fn from_slice(slice: &[i8]) -> NanoResult<Self> {
        if slice.len() != N {
            return Err(NanoError::InvalidInputLength {
                expected: N,
                actual: slice.len(),
            });
        }
        let mut data = [0i8; N];
        data.copy_from_slice(slice);
        Ok(Self { data })
    }

    /// Create a tensor from a static reference (Flash memory on MCU).
    ///
    /// Why 'static: Backbone weights live in Flash for the entire program lifetime.
    /// The compiler places `static` data in the `.rodata` section → Flash.
    pub fn from_static(static_data: &'static [i8]) -> NanoResult<Self> {
        Self::from_slice(static_data)
    }

    /// Get value at index. Returns None if out of bounds.
    #[inline(always)]
    pub fn get(&self, index: usize) -> Option<i8> {
        if index < N {
            Some(self.data[index])
        } else {
            None
        }
    }

    /// Set value at index. Returns error if out of bounds.
    #[inline(always)]
    pub fn set(&mut self, index: usize, value: i8) -> NanoResult<()> {
        if index >= N {
            return Err(NanoError::DimensionMismatch {
                expected: N,
                actual: index,
            });
        }
        self.data[index] = value;
        Ok(())
    }

    /// Get immutable reference to the underlying flat data.
    ///
    /// Why expose raw slice: Allows math functions to operate on contiguous memory
    /// without copying — zero overhead for matmul/activation functions.
    #[inline(always)]
    pub fn as_slice(&self) -> &[i8] {
        &self.data
    }

    /// Get mutable reference to the underlying flat data.
    #[inline(always)]
    pub fn as_mut_slice(&mut self) -> &mut [i8] {
        &mut self.data
    }

    /// Total number of elements.
    #[inline(always)]
    pub const fn len(&self) -> usize {
        N
    }

    /// Whether the tensor has zero elements.
    #[inline(always)]
    pub const fn is_empty(&self) -> bool {
        N == 0
    }
}
