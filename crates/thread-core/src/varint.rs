//! LEB128 unsigned varints. Values below 128 take one byte, which is the common case for both
//! the document-id gaps inside a posting list and the term frequencies.

#[inline(always)]
pub fn len(mut v: u64) -> usize {
    let mut n = 1;
    while v >= 0x80 {
        v >>= 7;
        n += 1;
    }
    n
}

/// Appends `v` to `out`.
#[inline(always)]
pub fn put(out: &mut Vec<u8>, mut v: u64) {
    while v >= 0x80 {
        out.push((v as u8) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

/// Writes `v` at `buf[*pos..]` and advances `pos`. The caller guarantees capacity (see `len`).
#[inline(always)]
pub fn write_at(buf: &mut [u8], pos: &mut usize, mut v: u64) {
    while v >= 0x80 {
        buf[*pos] = (v as u8) | 0x80;
        *pos += 1;
        v >>= 7;
    }
    buf[*pos] = v as u8;
    *pos += 1;
}

/// Reads one varint from `buf[*pos..]` and advances `pos`. Out-of-range input returns 0 and
/// leaves `pos` at the end, so a truncated file degrades to zeros instead of panicking.
#[inline(always)]
pub fn read(buf: &[u8], pos: &mut usize) -> u64 {
    let mut p = *pos;
    // Fast path: single byte.
    if let Some(&b0) = buf.get(p) {
        if b0 < 0x80 {
            *pos = p + 1;
            return b0 as u64;
        }
    } else {
        return 0;
    }
    let mut result: u64 = 0;
    let mut shift = 0u32;
    while let Some(&b) = buf.get(p) {
        p += 1;
        result |= ((b & 0x7f) as u64) << shift;
        if b < 0x80 {
            *pos = p;
            return result;
        }
        shift += 7;
        if shift >= 64 {
            break;
        }
    }
    *pos = buf.len();
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        let values = [
            0u64,
            1,
            127,
            128,
            255,
            300,
            16383,
            16384,
            u32::MAX as u64,
            u64::MAX,
        ];
        let mut buf = Vec::new();
        for &v in &values {
            let before = buf.len();
            put(&mut buf, v);
            assert_eq!(buf.len() - before, len(v));
        }
        let mut pos = 0;
        for &v in &values {
            assert_eq!(read(&buf, &mut pos), v);
        }
        assert_eq!(pos, buf.len());
    }
}
