//! Bump Arena Allocator with checkpoint support for microcontrollers.
//!
//! Why Arena: MCUs have no heap allocator. The caller owns a stack-allocated
//! `[u8; N]` buffer and passes `&mut [u8]` into the model. The Arena hands out
//! sub-slices — zero fragmentation, O(1) alloc/reset.
//!
//! Checkpoint support: Conv2D needs temporary im2col buffers. Checkpoints
//! let you alloc a temp buffer, use it, then "release" it while keeping
//! the actual output alive. This maximizes arena reuse.

use crate::error::{NanoError, NanoResult};

/// Bump allocator over a borrowed byte buffer.
///
/// Lifetime `'a` ties all allocations to the buffer — Rust's borrow checker
/// enforces no dangling references at zero runtime cost.
pub struct Arena<'a> {
    buf: &'a mut [u8],
    offset: usize,
}

impl<'a> Arena<'a> {
    /// Create a new arena from a mutable byte buffer.
    pub fn new(buf: &'a mut [u8]) -> Self {
        Self { buf, offset: 0 }
    }

    /// Allocate a mutable slice of `i8` with the given length.
    ///
    /// SAFETY: i8 and u8 have identical size (1) and alignment (1).
    /// Reinterpreting is safe per Rust's type layout guarantees.
    pub fn alloc_i8_slice(&mut self, len: usize) -> NanoResult<&'a mut [i8]> {
        let bytes_needed = len;

        if self.offset + bytes_needed > self.buf.len() {
            return Err(NanoError::ArenaExhausted {
                requested: bytes_needed,
                remaining: self.buf.len().saturating_sub(self.offset),
            });
        }

        let start = self.offset;
        self.offset += bytes_needed;

        let slice = &mut self.buf[start..start + bytes_needed];
        let ptr = slice.as_mut_ptr() as *mut i8;
        let result = unsafe { core::slice::from_raw_parts_mut(ptr, len) };

        // Zero-initialize to prevent stale data leaks on MCU
        for val in result.iter_mut() {
            *val = 0;
        }

        Ok(result)
    }

    /// Allocate a mutable slice of raw `u8` bytes.
    pub fn alloc_u8_slice(&mut self, len: usize) -> NanoResult<&'a mut [u8]> {
        if self.offset + len > self.buf.len() {
            return Err(NanoError::ArenaExhausted {
                requested: len,
                remaining: self.buf.len().saturating_sub(self.offset),
            });
        }

        let start = self.offset;
        self.offset += len;

        let slice = &mut self.buf[start..start + len];
        let ptr = slice.as_mut_ptr();
        let result = unsafe { core::slice::from_raw_parts_mut(ptr, len) };

        for val in result.iter_mut() {
            *val = 0;
        }

        Ok(result)
    }

    /// Save a checkpoint of the current allocation offset.
    #[inline(always)]
    pub fn save_checkpoint(&self) -> usize {
        self.offset
    }

    /// Restore arena to a previous checkpoint.
    #[inline(always)]
    pub fn restore_checkpoint(&mut self, checkpoint: usize) {
        if checkpoint <= self.offset {
            self.offset = checkpoint;
        }
    }

    /// Reset the arena fully.
    pub fn reset(&mut self) {
        self.offset = 0;
    }

    /// Bytes remaining in the arena.
    #[inline(always)]
    pub fn remaining(&self) -> usize {
        self.buf.len().saturating_sub(self.offset)
    }

    /// Total capacity in bytes.
    #[inline(always)]
    pub fn capacity(&self) -> usize {
        self.buf.len()
    }

    /// Bytes currently allocated.
    #[inline(always)]
    pub fn used(&self) -> usize {
        self.offset
    }
}
