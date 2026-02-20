//! Fixed-size i8 Tensor with compile-time size checking via const generics.

use crate::error::{NanoError, NanoResult};

#[derive(Debug, Clone)]
pub struct Tensor<const N: usize> {
    data: [i8; N],
}

impl<const N: usize> Tensor<N> {
    pub const fn zeros() -> Self {
        Self { data: [0i8; N] }
    }

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

    pub fn from_static(static_data: &'static [i8]) -> NanoResult<Self> {
        Self::from_slice(static_data)
    }

    #[inline(always)]
    pub fn get(&self, index: usize) -> Option<i8> {
        if index < N { Some(self.data[index]) } else { None }
    }

    #[inline(always)]
    pub fn set(&mut self, index: usize, value: i8) -> NanoResult<()> {
        if index >= N {
            return Err(NanoError::DimensionMismatch { expected: N, actual: index });
        }
        self.data[index] = value;
        Ok(())
    }

    #[inline(always)]
    pub fn as_slice(&self) -> &[i8] { &self.data }

    #[inline(always)]
    pub fn as_mut_slice(&mut self) -> &mut [i8] { &mut self.data }

    #[inline(always)]
    pub const fn len(&self) -> usize { N }

    #[inline(always)]
    pub const fn is_empty(&self) -> bool { N == 0 }
}
