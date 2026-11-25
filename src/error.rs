use std::fmt;

/// Every way a call can fail. The mapping to the Go errors is in `docs/parity.md`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// Not a valid WebP file (Go `ErrFormat`).
    Format,
    /// The file ends before its sizes say (Go `ErrTruncated`).
    Truncated,
    /// `decode_planar` on an animated file (Go `ErrAnimated`).
    Animated,
    /// `decode_planar` on a lossless image (Go `ErrNotLossy`).
    NotLossy,
    /// The cancel flag was set.
    Cancelled,
    /// A call needs more than `max_alloc` bytes at once (`got` is what it asked for, `max` the cap;
    /// Go `ErrMemoryLimit` and the memory `*LimitError`). Permanent for the same input and limits.
    MemoryBudget { got: u64, max: u64 },
    /// The image has more than `max_pixels` pixels (Go `*LimitError`).
    MaxPixels { got: u64, max: u64 },
    /// The image is wider than 16383 (Go `*LimitError`; `decode` only, `encode` gives
    /// [`Error::InvalidArgument`]).
    Width { got: u64, max: u64 },
    /// The image is taller than 16383 (Go `*LimitError`; `decode` only, `encode` gives
    /// [`Error::InvalidArgument`]).
    Height { got: u64, max: u64 },
    /// `metadata` would return more than `max_metadata_bytes` (Go `*LimitError`).
    MetadataBytes { got: u64, max: u64 },
    /// The allocator refused a request that passed the limits.
    OutOfMemory,
    /// A VP8 frame header is invalid (Go `vp8dec.errHeader`).
    Vp8Header,
    /// A VP8 partition ends early (Go `vp8dec.errTruncated`).
    Vp8Truncated,
    /// A VP8L header is invalid (Go `vp8ldec.errSig`).
    Vp8lHeader,
    /// A VP8L stream ends early (Go `vp8ldec.errTruncated`).
    Vp8lTruncated,
    /// A VP8L stream breaks the format (Go `vp8ldec.errBitstream`).
    Vp8lBitstream,
    /// A VP8L prefix code is not a valid code (Go `vp8ldec.errCode`).
    PrefixCode,
    /// A caller argument is out of range.
    InvalidArgument(&'static str),
    /// The image or option cannot be encoded.
    Unsupported(&'static str),
    /// A panic on a pool thread, caught and returned: a bug, not a bad input.
    Internal(String),
}

impl Error {
    /// A short stable token naming the error's class, the same for the Go oracle's output.
    pub fn class(&self) -> &'static str {
        match self {
            Error::Format => "format",
            Error::Truncated => "truncated",
            Error::Animated => "animated",
            Error::NotLossy => "not-lossy",
            Error::Cancelled => "cancelled",
            Error::MemoryBudget { .. }
            | Error::MaxPixels { .. }
            | Error::Width { .. }
            | Error::Height { .. }
            | Error::MetadataBytes { .. }
            | Error::OutOfMemory => "limit",
            Error::Vp8Header => "vp8-header",
            Error::Vp8Truncated => "vp8-truncated",
            Error::Vp8lHeader => "vp8l-header",
            Error::Vp8lTruncated => "vp8l-truncated",
            Error::Vp8lBitstream => "vp8l-bitstream",
            Error::PrefixCode => "prefix-code",
            Error::InvalidArgument(_) => "argument",
            Error::Unsupported(_) => "unsupported",
            Error::Internal(_) => "internal",
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Format => f.write_str("webp: invalid format"),
            Error::Truncated => f.write_str("webp: truncated file"),
            Error::Animated => f.write_str("webp: animated images are not supported"),
            Error::NotLossy => f.write_str("webp: image is lossless"),
            Error::Cancelled => f.write_str("webp: cancelled"),
            Error::MemoryBudget { got, max } => {
                write!(f, "webp: MemoryBudget {got} over limit {max}")
            }
            Error::MaxPixels { got, max } => write!(f, "webp: MaxPixels {got} over limit {max}"),
            Error::Width { got, max } => write!(f, "webp: Width {got} over limit {max}"),
            Error::Height { got, max } => write!(f, "webp: Height {got} over limit {max}"),
            Error::MetadataBytes { got, max } => {
                write!(f, "webp: MetadataBytes {got} over limit {max}")
            }
            Error::OutOfMemory => f.write_str("webp: out of memory"),
            Error::Vp8Header => f.write_str("webp: invalid VP8 frame header"),
            Error::Vp8Truncated => f.write_str("webp: truncated VP8 data"),
            Error::Vp8lHeader => f.write_str("webp: invalid VP8L header"),
            Error::Vp8lTruncated => f.write_str("webp: truncated VP8L stream"),
            Error::Vp8lBitstream => f.write_str("webp: invalid VP8L bitstream"),
            Error::PrefixCode => f.write_str("webp: invalid prefix code"),
            Error::InvalidArgument(s) => write!(f, "webp: {s}"),
            Error::Unsupported(s) => write!(f, "webp: {s}"),
            Error::Internal(m) => write!(f, "webp: internal error: {m}"),
        }
    }
}

impl std::error::Error for Error {}

#[cfg(test)]
mod tests {
    use super::Error;

    /// Every limit has its own variant, which a caller can match, and all of them keep the parity
    /// class `limit` of the Go errors.
    #[test]
    fn limit_variants_are_distinct_and_classed() {
        let (got, max) = (2, 1);
        let limits = [
            Error::MemoryBudget { got, max },
            Error::MaxPixels { got, max },
            Error::Width { got, max },
            Error::Height { got, max },
            Error::MetadataBytes { got, max },
        ];
        for (i, a) in limits.iter().enumerate() {
            assert_eq!(a.class(), "limit", "{a}");
            for (j, b) in limits.iter().enumerate() {
                assert_eq!(a == b, i == j, "{a} and {b}");
            }
        }
        assert_eq!(Error::OutOfMemory.class(), "limit");
        assert_eq!(
            Error::MaxPixels { got: 5, max: 4 }.to_string(),
            "webp: MaxPixels 5 over limit 4"
        );
    }
}
