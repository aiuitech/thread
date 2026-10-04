// Dense-side file written before rustfmt was adopted; kept as is (`cargo fmt` skips it).
#![cfg_attr(rustfmt, rustfmt::skip)]
//! Tombstones: the set of deleted vector ids.
//!
//! Serialized as a plain array of `u32` little-endian ids, sorted ascending
//! and unique (`4 * n` bytes, no header), so a host can store it as one small
//! object and the empty set is the empty buffer. Queries skip tombstoned ids
//! inside the scan, before top-k insertion (the `skip` argument of
//! `search::scan_slab`, `scan_slab_pq`, `scan_delta`), so deleted entries do not
//! take heap slots; filtering a finished top-k ([`Tombstones::filter_hits`])
//! can return fewer than k hits. Compaction skips them when re-emitting slab
//! and delta records.
//!
//! Deletes are by id: a tombstone hides every copy of the id (in slabs and in
//! delta segments). To update a vector, give the new version a new id, or
//! drop the id from the tombstones once the compaction that removed the old
//! copies is published.

use crate::error::{rd_u32, FormatError};
use crate::topk::Hit;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Tombstones {
    ids: Vec<u32>,
}

impl Tombstones {
    pub fn new() -> Self {
        Tombstones { ids: Vec::new() }
    }

    /// From any id list (sorted and deduplicated here).
    pub fn from_ids(ids: &[u32]) -> Self {
        let mut ids = ids.to_vec();
        ids.sort_unstable();
        ids.dedup();
        Tombstones { ids }
    }

    /// Parses `4 * n` bytes of little-endian ids. Unsorted or duplicate input
    /// is accepted and normalized. Errors if the length is not a multiple of 4.
    pub fn from_bytes(buf: &[u8]) -> Result<Self, FormatError> {
        if buf.len() % 4 != 0 {
            return Err(FormatError::Inconsistent("tombstone buffer length is not a multiple of 4"));
        }
        let ids: Vec<u32> = (0..buf.len() / 4).map(|i| rd_u32(buf, i * 4)).collect();
        if ids.windows(2).all(|w| w[0] < w[1]) {
            Ok(Tombstones { ids })
        } else {
            Ok(Tombstones::from_ids(&ids))
        }
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.ids.len() * 4);
        for id in &self.ids {
            out.extend_from_slice(&id.to_le_bytes());
        }
        out
    }

    /// The union with `add` (any order, duplicates allowed).
    pub fn merge(&self, add: &[u32]) -> Self {
        let add = Tombstones::from_ids(add);
        let (a, b) = (&self.ids, &add.ids);
        let mut ids = Vec::with_capacity(a.len() + b.len());
        let (mut i, mut j) = (0, 0);
        while i < a.len() && j < b.len() {
            match a[i].cmp(&b[j]) {
                core::cmp::Ordering::Less => {
                    ids.push(a[i]);
                    i += 1;
                }
                core::cmp::Ordering::Greater => {
                    ids.push(b[j]);
                    j += 1;
                }
                core::cmp::Ordering::Equal => {
                    ids.push(a[i]);
                    i += 1;
                    j += 1;
                }
            }
        }
        ids.extend_from_slice(&a[i..]);
        ids.extend_from_slice(&b[j..]);
        Tombstones { ids }
    }

    #[inline]
    pub fn contains(&self, id: u32) -> bool {
        self.ids.binary_search(&id).is_ok()
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    pub fn ids(&self) -> &[u32] {
        &self.ids
    }

    /// Keeps the hits whose id is not tombstoned (order preserved).
    pub fn filter_hits(&self, hits: Vec<Hit>) -> Vec<Hit> {
        if self.ids.is_empty() {
            return hits;
        }
        hits.into_iter().filter(|h| !self.contains(h.id)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topk::{pack_hits, unpack_hits};

    #[test]
    fn merge_contains_roundtrip() {
        let t = Tombstones::from_ids(&[9, 3, 3, 7]);
        assert_eq!(t.ids(), &[3, 7, 9]);
        let m = t.merge(&[8, 1, 9, 100, 1]);
        assert_eq!(m.ids(), &[1, 3, 7, 8, 9, 100]);
        for id in [1, 3, 7, 8, 9, 100] {
            assert!(m.contains(id));
        }
        for id in [0, 2, 4, 99, 101, u32::MAX] {
            assert!(!m.contains(id));
        }
        let bytes = m.to_bytes();
        assert_eq!(bytes.len(), 24);
        assert_eq!(Tombstones::from_bytes(&bytes).unwrap(), m);
        // unsorted input bytes are normalized
        let raw: Vec<u8> = [5u32, 2, 5].iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(Tombstones::from_bytes(&raw).unwrap().ids(), &[2, 5]);
        assert!(Tombstones::from_bytes(&[1, 2, 3]).is_err());
        let e = Tombstones::from_bytes(&[]).unwrap();
        assert!(e.is_empty() && !e.contains(0));
        assert_eq!(e.merge(&[]).to_bytes(), Vec::<u8>::new());
        assert_eq!(Tombstones::new().merge(&[4, 4]).ids(), &[4]);
    }

    #[test]
    fn filter_packed_hits() {
        let hits: Vec<Hit> = (0..10).map(|i| Hit { id: i, score: 1.0 - i as f32 * 0.1, idx: i * 2 }).collect();
        // two concatenated packed buffers, as the host merges them
        let mut packed = pack_hits(&hits[..4]);
        packed.extend(pack_hits(&hits[4..]));
        let t = Tombstones::from_ids(&[0, 5, 9, 42]);
        let got = t.filter_hits(unpack_hits(&packed));
        assert_eq!(got.iter().map(|h| h.id).collect::<Vec<_>>(), vec![1, 2, 3, 4, 6, 7, 8]);
        assert_eq!(got[3], hits[4]);
        assert_eq!(Tombstones::new().filter_hits(hits.clone()), hits);
    }
}
