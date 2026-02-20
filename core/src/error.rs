//! Error types for the nano-rust-core library.
//!
//! Why a dedicated error module: Every function in this `no_std` library returns
//! `NanoResult<T>` instead of panicking. On a microcontroller, a panic halts the
//! entire device — unacceptable for production IoT/Edge deployments.

/// All possible error conditions in the nano-rust-core library.
///
/// Each variant carries enough context for the caller to understand
/// *what* went wrong without needing a heap-allocated error message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NanoError {
    /// The provided buffer is too small for the requested operation.
    /// Fields: (required_bytes, available_bytes)
    BufferTooSmall {
        required: usize,
        available: usize,
    },

    /// Matrix/tensor dimensions do not match for the operation.
    /// Fields: (expected_dim, actual_dim)
    DimensionMismatch {
        expected: usize,
        actual: usize,
    },

    /// The arena allocator has exhausted its available memory.
    /// Fields: (requested_bytes, remaining_bytes)
    ArenaExhausted {
        requested: usize,
        remaining: usize,
    },

    /// A quantization parameter is out of valid range.
    InvalidQuantization,

    /// Input slice length does not match the expected tensor size.
    InvalidInputLength {
        expected: usize,
        actual: usize,
    },
}

/// Convenience type alias — every fallible function in this crate returns this.
pub type NanoResult<T> = Result<T, NanoError>;

// Why no Display/fmt impl: we're no_std without alloc. The caller (Python binding
// or test harness) can pattern-match on the error variant and format as needed.
// On MCU, the debug probe reads the variant discriminant directly.
