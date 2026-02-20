//! Error types for the nano-rust-core library.
//!
//! Why a dedicated error module: Every function in this `no_std` library returns
//! `NanoResult<T>` instead of panicking. On a microcontroller, a panic halts the
//! entire device — unacceptable for production IoT/Edge deployments.

/// All possible error conditions in the nano-rust-core library.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NanoError {
    BufferTooSmall {
        required: usize,
        available: usize,
    },
    DimensionMismatch {
        expected: usize,
        actual: usize,
    },
    ArenaExhausted {
        requested: usize,
        remaining: usize,
    },
    InvalidQuantization,
    InvalidInputLength {
        expected: usize,
        actual: usize,
    },
}

pub type NanoResult<T> = Result<T, NanoError>;
