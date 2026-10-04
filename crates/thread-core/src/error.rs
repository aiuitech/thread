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
