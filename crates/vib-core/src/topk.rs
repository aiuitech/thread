//! Bounded top-k with a min-heap, plus a merge/dedupe for multi-slab results.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hit {
    pub id: u32,
    pub score: f32,
    /// Entry index inside the slab the hit was scanned from (locates its
    /// rerank record / payload). Only meaningful together with that slab.
    pub idx: u32,
}

/// Wrapper whose ordering is reversed on score so `BinaryHeap` (a max-heap)
/// pops the *lowest* score first.
#[derive(Clone, Copy, PartialEq)]
struct MinHit(Hit);

impl Eq for MinHit {}

impl PartialOrd for MinHit {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MinHit {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .0
            .score
            .total_cmp(&self.0.score)
            .then_with(|| other.0.id.cmp(&self.0.id))
    }
}

pub struct TopK {
    k: usize,
    heap: BinaryHeap<MinHit>,
}

impl TopK {
    pub fn new(k: usize) -> Self {
        TopK { k, heap: BinaryHeap::with_capacity(k + 1) }
    }

    /// Lowest score currently retained, or -inf while the heap is not full.
    /// Candidates must beat this to enter.
    #[inline]
    pub fn threshold(&self) -> f32 {
        if self.heap.len() < self.k {
            f32::NEG_INFINITY
        } else {
            self.heap.peek().map(|h| h.0.score).unwrap_or(f32::NEG_INFINITY)
        }
    }

    #[inline]
    pub fn push(&mut self, id: u32, score: f32, idx: u32) {
        if self.k == 0 {
            return;
        }
        if self.heap.len() < self.k {
            self.heap.push(MinHit(Hit { id, score, idx }));
        } else if score > self.threshold() {
            self.heap.pop();
            self.heap.push(MinHit(Hit { id, score, idx }));
        }
    }

    pub fn len(&self) -> usize {
        self.heap.len()
    }

    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    /// Drains into a vector sorted by descending score.
    pub fn into_sorted(self) -> Vec<Hit> {
        let mut v: Vec<Hit> = self.heap.into_iter().map(|h| h.0).collect();
        v.sort_by(|a, b| b.score.total_cmp(&a.score).then_with(|| a.id.cmp(&b.id)));
        v
    }
}

/// Merges hits from several slabs. The same id can appear in more than one
/// slab because of multi-assignment; keep its best score once (with the
/// `idx` of that best-scoring copy).
pub fn merge_dedup(mut hits: Vec<Hit>, k: usize) -> Vec<Hit> {
    hits.sort_by(|a, b| a.id.cmp(&b.id).then_with(|| b.score.total_cmp(&a.score)));
    hits.dedup_by_key(|h| h.id);
    hits.sort_by(|a, b| b.score.total_cmp(&a.score).then_with(|| a.id.cmp(&b.id)));
    hits.truncate(k);
    hits
}

/// Packed wire form used across the wasm boundary:
/// `[count: u32][id: u32, score: f32, idx: u32] * count`, little-endian.
pub fn pack_hits(hits: &[Hit]) -> Vec<u8> {
    let mut out = Vec::with_capacity(4 + hits.len() * HIT_BYTES);
    out.extend_from_slice(&(hits.len() as u32).to_le_bytes());
    for h in hits {
        out.extend_from_slice(&h.id.to_le_bytes());
        out.extend_from_slice(&h.score.to_le_bytes());
        out.extend_from_slice(&h.idx.to_le_bytes());
    }
    out
}

/// Bytes per packed hit.
pub const HIT_BYTES: usize = 12;

/// Parses one or more concatenated `pack_hits` buffers.
pub fn unpack_hits(mut buf: &[u8]) -> Vec<Hit> {
    let mut out = Vec::new();
    while buf.len() >= 4 {
        let n = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]) as usize;
        let body = &buf[4..];
        let take = n.saturating_mul(HIT_BYTES).min(body.len());
        for rec in body[..take].chunks_exact(HIT_BYTES) {
            out.push(Hit {
                id: u32::from_le_bytes([rec[0], rec[1], rec[2], rec[3]]),
                score: f32::from_le_bytes([rec[4], rec[5], rec[6], rec[7]]),
                idx: u32::from_le_bytes([rec[8], rec[9], rec[10], rec[11]]),
            });
        }
        buf = &body[take..];
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_best_k() {
        let mut t = TopK::new(3);
        for (i, s) in [0.1, 0.9, 0.3, 0.8, 0.2, 0.7].iter().enumerate() {
            t.push(i as u32, *s, 100 + i as u32);
        }
        let v = t.into_sorted();
        assert_eq!(v.iter().map(|h| h.id).collect::<Vec<_>>(), vec![1, 3, 5]);
        assert_eq!(v.iter().map(|h| h.idx).collect::<Vec<_>>(), vec![101, 103, 105]);
    }

    #[test]
    fn merge_removes_duplicates() {
        let hits = vec![
            Hit { id: 7, score: 0.5, idx: 1 },
            Hit { id: 7, score: 0.51, idx: 2 },
            Hit { id: 2, score: 0.9, idx: 3 },
            Hit { id: 9, score: 0.1, idx: 4 },
        ];
        let m = merge_dedup(hits, 2);
        assert_eq!(m.len(), 2);
        assert_eq!(m[0].id, 2);
        assert_eq!(m[1].id, 7);
        assert!((m[1].score - 0.51).abs() < 1e-6);
        assert_eq!(m[1].idx, 2);
    }

    #[test]
    fn pack_roundtrip() {
        let hits = vec![Hit { id: 1, score: 0.5, idx: 9 }, Hit { id: 2, score: -1.0, idx: 0 }];
        let mut buf = pack_hits(&hits);
        assert_eq!(buf.len(), 4 + 2 * HIT_BYTES);
        buf.extend(pack_hits(&hits[..1]));
        let back = unpack_hits(&buf);
        assert_eq!(back.len(), 3);
        assert_eq!(back[2], hits[0]);
        assert_eq!(&back[..2], &hits[..]);
    }
}
