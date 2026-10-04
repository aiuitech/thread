//! One blob holding named sections (`meta`, `terms`, `postings`, `docs`): the container of a
//! thread lexical/docs segment (ported from Wiper's thread engine).
//!
//! Layout (little-endian):
//!
//! ```text
//! 0   magic "THREADIX"            8 bytes
//! 8   container version u32       = 1
//! 12  section count u32
//! 16  reserved, zero              up to byte 64
//! 64  section table               count × 32-byte entries:
//!       name  [u8; 16]            ASCII, zero-padded
//!       offset u64                absolute, 8-byte aligned
//!       len    u64
//! ..  sections in table order     each starting at an 8-byte aligned offset, zero-padded between
//! ```
//!
//! [`Container::open`] reads only the header and the table. A section's bytes are addressed
//! through [`Container::sub`], a [`SubSource`] window, so an index opened on it uses offsets
//! relative to the section while plans and `Missing` errors stay blob-absolute.

use std::io::{self, Write};
use std::path::Path;

use crate::source::{ByteSource, Plan, Range, SubSource};
use crate::Error;

pub const MAGIC: &[u8; 8] = b"THREADIX";
pub const VERSION: u32 = 1;
pub const HEADER_LEN: u64 = 64;
pub const ENTRY_LEN: u64 = 32;
pub const NAME_LEN: usize = 16;
/// Sections the table can hold within [`Container::plan_open`]'s single fetch.
pub const PLAN_OPEN_SECTIONS: u64 = 16;

/// Reserved section names of a thread segment (`segment.rs`).
/// UTF-8 JSON, free-form (`format`, `version`, `tokenizer`, per-section summaries); the core
/// parses only `tokenizer` out of it.
pub const SEC_META: &str = "meta";
/// Bucketed sorted term table with document frequencies and posting ranges (`lex.rs`).
pub const SEC_TERMS: &str = "terms";
/// Posting lists (`lex.rs`).
pub const SEC_POSTINGS: &str = "postings";
/// Document ids -> metadata (+ optional text), bucketed by id (`docs.rs`).
pub const SEC_DOCS: &str = "docs";

#[inline]
fn align8(x: u64) -> u64 {
    (x + 7) & !7
}

fn check_name(name: &str) -> Result<(), Error> {
    if name.is_empty() || name.len() > NAME_LEN || !name.bytes().all(|b| b.is_ascii_graphic()) {
        return Err(Error::Format(format!(
            "section name {name:?} must be 1..={NAME_LEN} printable ASCII bytes"
        )));
    }
    Ok(())
}

/// The section table of an opened container.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Container {
    sections: Vec<(String, Range)>,
    len: u64,
}

impl Container {
    /// Reads and validates the header and the section table (nothing else).
    pub fn open<S: ByteSource + ?Sized>(source: &S) -> Result<Container, Error> {
        let len = source.len();
        let head = source.read(Range::new(0, HEADER_LEN))?;
        if &head[0..8] != MAGIC {
            return Err(Error::Format("not a thread container (bad magic)".into()));
        }
        let version = u32::from_le_bytes(head[8..12].try_into().unwrap());
        if version != VERSION {
            return Err(Error::Format(format!(
                "unsupported container version {version} (expected {VERSION})"
            )));
        }
        let count = u32::from_le_bytes(head[12..16].try_into().unwrap()) as u64;
        if HEADER_LEN + count * ENTRY_LEN > len {
            return Err(Error::Format(format!(
                "section table of {count} entries does not fit in {len} bytes"
            )));
        }
        drop(head);
        let table = source.read(Range::new(HEADER_LEN, count * ENTRY_LEN))?;
        let mut sections = Vec::with_capacity(count as usize);
        let mut cursor = align8(HEADER_LEN + count * ENTRY_LEN);
        for i in 0..count as usize {
            let e = &table[i * ENTRY_LEN as usize..(i + 1) * ENTRY_LEN as usize];
            let name_end = e[..NAME_LEN]
                .iter()
                .position(|&b| b == 0)
                .unwrap_or(NAME_LEN);
            let name = std::str::from_utf8(&e[..name_end])
                .map_err(|_| Error::Format(format!("section {i}: name is not ASCII")))?
                .to_string();
            check_name(&name)?;
            let offset = u64::from_le_bytes(e[16..24].try_into().unwrap());
            let slen = u64::from_le_bytes(e[24..32].try_into().unwrap());
            if offset % 8 != 0
                || offset < cursor
                || offset.checked_add(slen).is_none_or(|end| end > len)
            {
                return Err(Error::Format(format!(
                    "section {name:?}: bad range [{offset}, +{slen}) in {len} bytes"
                )));
            }
            if sections.iter().any(|(n, _)| n == &name) {
                return Err(Error::Format(format!("duplicate section {name:?}")));
            }
            cursor = offset + slen;
            sections.push((name, Range::new(offset, slen)));
        }
        Ok(Container { sections, len })
    }

    /// What a host fetches before `open`: the header plus a table of up to
    /// [`PLAN_OPEN_SECTIONS`] entries, in one range (clip it to the blob's length). If the table
    /// is longer, `open` returns `Missing` with the rest.
    pub fn plan_open() -> Plan {
        Plan::from(vec![Range::new(
            0,
            HEADER_LEN + PLAN_OPEN_SECTIONS * ENTRY_LEN,
        )])
    }

    /// Size of the blob the table was read from.
    pub fn len(&self) -> u64 {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.sections.is_empty()
    }
    /// Sections in table (= file) order.
    pub fn sections(&self) -> &[(String, Range)] {
        &self.sections
    }
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.sections.iter().map(|(n, _)| n.as_str())
    }
    /// Absolute range of a section.
    pub fn section(&self, name: &str) -> Option<Range> {
        self.sections
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, r)| *r)
    }
    /// A window over `source` for one section (`source` must be the one `open` read).
    pub fn sub<'a, S: ByteSource + ?Sized>(
        &self,
        source: &'a S,
        name: &str,
    ) -> Option<SubSource<'a, S>> {
        let r = self.section(name)?;
        SubSource::new(source, r).ok()
    }
}

/// Appends sections and writes the blob.
#[derive(Default)]
pub struct ContainerWriter {
    sections: Vec<(String, Vec<u8>)>,
}

impl ContainerWriter {
    pub fn new() -> ContainerWriter {
        ContainerWriter::default()
    }

    /// Appends a section. Names are 1..=16 printable ASCII bytes and unique.
    pub fn add(
        &mut self,
        name: &str,
        bytes: impl Into<Vec<u8>>,
    ) -> Result<&mut ContainerWriter, Error> {
        check_name(name)?;
        if self.sections.iter().any(|(n, _)| n == name) {
            return Err(Error::Format(format!("duplicate section {name:?}")));
        }
        self.sections.push((name.to_string(), bytes.into()));
        Ok(self)
    }

    /// Appends a section produced by `f` into a fresh buffer.
    pub fn add_from_writer<F>(&mut self, name: &str, f: F) -> Result<&mut ContainerWriter, Error>
    where
        F: FnOnce(&mut Vec<u8>) -> Result<(), Error>,
    {
        let mut buf = Vec::new();
        f(&mut buf)?;
        self.add(name, buf)
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.sections.iter().map(|(n, _)| n.as_str())
    }

    /// Total size of the blob `finish` produces.
    pub fn total_len(&self) -> u64 {
        let mut cursor = align8(HEADER_LEN + self.sections.len() as u64 * ENTRY_LEN);
        for (_, b) in &self.sections {
            cursor = align8(cursor + b.len() as u64);
        }
        cursor
    }

    /// Writes the blob.
    pub fn write_to<W: Write>(&self, mut w: W) -> io::Result<()> {
        let count = self.sections.len() as u64;
        let mut head = vec![0u8; HEADER_LEN as usize];
        head[0..8].copy_from_slice(MAGIC);
        head[8..12].copy_from_slice(&VERSION.to_le_bytes());
        head[12..16].copy_from_slice(&(count as u32).to_le_bytes());
        w.write_all(&head)?;
        let mut cursor = align8(HEADER_LEN + count * ENTRY_LEN);
        let mut table = Vec::with_capacity((count * ENTRY_LEN) as usize);
        for (name, bytes) in &self.sections {
            let mut e = [0u8; ENTRY_LEN as usize];
            e[..name.len()].copy_from_slice(name.as_bytes());
            e[16..24].copy_from_slice(&cursor.to_le_bytes());
            e[24..32].copy_from_slice(&(bytes.len() as u64).to_le_bytes());
            table.extend_from_slice(&e);
            cursor = align8(cursor + bytes.len() as u64);
        }
        w.write_all(&table)?;
        let mut written = HEADER_LEN + count * ENTRY_LEN;
        for (_, bytes) in &self.sections {
            let pad = align8(written) - written;
            w.write_all(&[0u8; 8][..pad as usize])?;
            w.write_all(bytes)?;
            written = align8(written) + bytes.len() as u64;
        }
        let pad = align8(written) - written;
        w.write_all(&[0u8; 8][..pad as usize])?;
        w.flush()
    }

    pub fn finish(&self) -> Vec<u8> {
        let mut v = Vec::with_capacity(self.total_len() as usize);
        self.write_to(&mut v).expect("writing to a Vec cannot fail");
        v
    }

    pub fn write_path(&self, path: &Path) -> io::Result<()> {
        let f = std::fs::File::create(path)?;
        self.write_to(io::BufWriter::with_capacity(1 << 20, f))
    }
}

/// Whether `head` (at least 8 bytes) starts with the container magic.
pub fn is_container(head: &[u8]) -> bool {
    head.len() >= 8 && &head[..8] == MAGIC
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::{Prefetched, SourceError};

    #[test]
    fn roundtrip() {
        let mut w = ContainerWriter::new();
        w.add(SEC_META, br#"{"format":"thread"}"#.as_slice())
            .unwrap();
        w.add(SEC_TERMS, vec![7u8; 1001]).unwrap();
        w.add_from_writer(SEC_POSTINGS, |buf| {
            buf.extend_from_slice(b"abc");
            Ok(())
        })
        .unwrap();
        w.add("empty", Vec::new()).unwrap();
        assert!(w.add("terms", vec![]).is_err());
        assert!(w.add("", vec![]).is_err());
        assert!(w.add("a name with spaces", vec![]).is_err());
        let blob = w.finish();
        assert_eq!(blob.len() as u64, w.total_len());
        assert_eq!(blob.len() % 8, 0);
        let c = Container::open(&blob).unwrap();
        assert_eq!(c.len(), blob.len() as u64);
        assert_eq!(
            c.names().collect::<Vec<_>>(),
            vec!["meta", "terms", "postings", "empty"]
        );
        let meta = c.section(SEC_META).unwrap();
        assert_eq!(meta, Range::new(HEADER_LEN + 4 * ENTRY_LEN, 19));
        assert_eq!(
            &blob[meta.offset as usize..meta.end() as usize],
            br#"{"format":"thread"}"#
        );
        let bm25 = c.section(SEC_TERMS).unwrap();
        assert_eq!(bm25.offset % 8, 0);
        assert_eq!(bm25.len, 1001);
        assert!(blob[bm25.offset as usize..bm25.end() as usize]
            .iter()
            .all(|&b| b == 7));
        let sub = c.sub(&blob, SEC_POSTINGS).unwrap();
        assert_eq!(&*sub.read(Range::new(0, 3)).unwrap(), b"abc");
        assert_eq!(
            sub.to_absolute(Range::new(1, 1)),
            Range::new(c.section(SEC_POSTINGS).unwrap().offset + 1, 1)
        );
        assert_eq!(c.section("empty").unwrap().len, 0);
        assert!(c.section("nope").is_none());
        let path = std::env::temp_dir().join(format!("thread-container-{}", std::process::id()));
        w.write_path(&path).unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), blob);
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn open_over_prefetched_and_rejects_corruption() {
        let mut w = ContainerWriter::new();
        w.add(SEC_TERMS, vec![1u8; 100]).unwrap();
        let blob = w.finish();
        let mut p = Prefetched::new(blob.len() as u64);
        assert!(matches!(
            Container::open(&p),
            Err(Error::Source(SourceError::Missing(_)))
        ));
        let plan = Container::plan_open();
        let r = plan.ranges()[0];
        let end = r.end().min(blob.len() as u64) as usize;
        p.insert(0, blob[..end].to_vec()).unwrap();
        assert_eq!(
            Container::open(&p).unwrap(),
            Container::open(&blob).unwrap()
        );

        let mut bad = blob.clone();
        bad[0] = b'X';
        assert!(matches!(Container::open(&bad), Err(Error::Format(_))));
        let mut bad = blob.clone();
        bad[12] = 200; // 200 sections do not fit
        assert!(matches!(Container::open(&bad), Err(Error::Format(_))));
        let mut bad = blob.clone();
        bad[64 + 24] = 255; // section longer than the blob
        assert!(matches!(Container::open(&bad), Err(Error::Format(_))));
        let mut bad = blob.clone();
        bad[64 + 16] = 4; // unaligned offset
        assert!(matches!(Container::open(&bad), Err(Error::Format(_))));
        assert!(matches!(
            Container::open(&blob[..40]),
            Err(Error::Source(SourceError::OutOfBounds(_)))
        ));
        assert!(is_container(&blob) && !is_container(b"THRDTERM"));
    }
}
