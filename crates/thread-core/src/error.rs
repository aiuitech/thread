// Dense-side file written before rustfmt was adopted; kept as is (`cargo fmt` skips it).
#![cfg_attr(rustfmt, rustfmt::skip)]
use core::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FormatError {
    TooShort { need: usize, have: usize },
    BadMagic,
    BadVersion(u16),
    Inconsistent(&'static str),
}

impl fmt::Display for FormatError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FormatError::TooShort { need, have } => {
                write!(f, "buffer too short: need {need} bytes, have {have}")
            }
            FormatError::BadMagic => write!(f, "bad magic"),
            FormatError::BadVersion(v) => write!(f, "unsupported format version {v}"),
            FormatError::Inconsistent(what) => write!(f, "inconsistent header: {what}"),
        }
    }
}

impl std::error::Error for FormatError {}

/// Little-endian read helpers with bounds checking.
pub(crate) fn need(buf: &[u8], len: usize) -> Result<(), FormatError> {
    if buf.len() < len {
        Err(FormatError::TooShort { need: len, have: buf.len() })
    } else {
        Ok(())
    }
}

#[inline]
pub(crate) fn rd_u32(buf: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]])
}

#[inline]
pub(crate) fn rd_u16(buf: &[u8], off: usize) -> u16 {
    u16::from_le_bytes([buf[off], buf[off + 1]])
}

#[inline]
pub(crate) fn rd_f32(buf: &[u8], off: usize) -> f32 {
    f32::from_bits(rd_u32(buf, off))
}

/// Reinterprets a little-endian `f32` byte buffer as a `Vec<f32>`.
pub fn f32s_from_le(buf: &[u8]) -> Vec<f32> {
    buf.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

pub fn f32s_to_le(v: &[f32], out: &mut Vec<u8>) {
    out.reserve(v.len() * 4);
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
}

/// Errors of the lexical/docs side (ported with Wiper's thread modules): a read failure, where
/// `Source(SourceError::Missing(..))` is the demand-paging signal, or bytes that are not a valid
/// section/container (bad magic, version, header, config).
#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    Source(crate::source::SourceError),
    Format(String),
}

impl Error {
    /// Collapses into a `SourceError` (format errors become `Io` messages), for callers whose
    /// signature is `Result<_, SourceError>`.
    pub fn into_source(self) -> crate::source::SourceError {
        match self {
            Error::Source(e) => e,
            Error::Format(m) => crate::source::SourceError::Io(m),
        }
    }
    /// The ranges to fetch when this is a `Missing` error.
    pub fn missing(&self) -> Option<&[crate::source::Range]> {
        match self {
            Error::Source(crate::source::SourceError::Missing(r)) => Some(r),
            _ => None,
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Source(e) => write!(f, "{e}"),
            Error::Format(m) => write!(f, "{m}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<crate::source::SourceError> for Error {
    fn from(e: crate::source::SourceError) -> Self {
        Error::Source(e)
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Source(crate::source::SourceError::Io(e.to_string()))
    }
}

impl From<FormatError> for Error {
    fn from(e: FormatError) -> Self {
        Error::Format(e.to_string())
    }
}
