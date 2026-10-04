//! Attribute filters: a predicate over the 32-bit `attrs` word every dense entry (slab and delta)
//! and every `docs` entry carries, applied inside a scan before top-k insertion (like tombstones),
//! so a filtered-out entry never takes a candidate slot.
//!
//! A filter is a conjunction of clauses. Each clause masks the word (`x = attrs & mask`) and tests
//! it: equal to / not equal to a value, in / not in a sorted value set, in a range `[a, b]`, or a
//! bit test on the masked bits (any set, all set, none set). The host compiles a field-level
//! expression (`{tenant: 5, lang: {$in: ["en", "fr"]}}`, see the SDKs' `attrs` modules) into this
//! form against the index's attribute schema; the kernel only knows masks and values.
//!
//! Wire form (`u32` words, the argument the scan kernels take):
//! ```text
//! [clause count n]
//! n x [mask, op, a, b, value count v, v values (sorted, unique)]
//! ```
//! An empty word list means "no filter"; a clause list that can never match (e.g. a range with
//! `a > b`) is a valid filter that drops every entry.

use crate::error::FormatError;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum Op {
    /// `x == a`
    Eq = 0,
    /// `x != a`
    Ne = 1,
    /// `x` in `values`
    In = 2,
    /// `x` not in `values`
    NotIn = 3,
    /// `a <= x <= b` (never matches when `a > b`)
    Range = 4,
    /// `x != 0` (some masked bit set)
    BitsAny = 5,
    /// `x == mask` (every masked bit set)
    BitsAll = 6,
    /// `x == 0` (no masked bit set)
    BitsNone = 7,
}

impl Op {
    pub fn from_u32(v: u32) -> Option<Op> {
        Some(match v {
            0 => Op::Eq,
            1 => Op::Ne,
            2 => Op::In,
            3 => Op::NotIn,
            4 => Op::Range,
            5 => Op::BitsAny,
            6 => Op::BitsAll,
            7 => Op::BitsNone,
            _ => return None,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Clause {
    pub mask: u32,
    pub op: Op,
    pub a: u32,
    pub b: u32,
    /// Sorted, unique (for `In` / `NotIn`).
    pub values: Vec<u32>,
}

impl Clause {
    #[inline]
    fn matches(&self, attrs: u32) -> bool {
        let x = attrs & self.mask;
        match self.op {
            Op::Eq => x == self.a,
            Op::Ne => x != self.a,
            Op::In => self.contains(x),
            Op::NotIn => !self.contains(x),
            Op::Range => self.a <= x && x <= self.b,
            Op::BitsAny => x != 0,
            Op::BitsAll => x == self.mask,
            Op::BitsNone => x == 0,
        }
    }

    #[inline]
    fn contains(&self, x: u32) -> bool {
        if self.values.len() <= 8 {
            self.values.contains(&x)
        } else {
            self.values.binary_search(&x).is_ok()
        }
    }
}

/// A conjunction of [`Clause`]s.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AttrFilter {
    clauses: Vec<Clause>,
}

impl AttrFilter {
    pub fn new(clauses: Vec<Clause>) -> AttrFilter {
        let clauses = clauses
            .into_iter()
            .map(|mut c| {
                c.values.sort_unstable();
                c.values.dedup();
                c
            })
            .collect();
        AttrFilter { clauses }
    }

    /// A filter no entry matches.
    pub fn nothing() -> AttrFilter {
        AttrFilter {
            clauses: vec![Clause {
                mask: 0,
                op: Op::Range,
                a: 1,
                b: 0,
                values: Vec::new(),
            }],
        }
    }

    pub fn clauses(&self) -> &[Clause] {
        &self.clauses
    }

    /// No clauses: matches everything (hosts pass no filter at all instead).
    pub fn is_empty(&self) -> bool {
        self.clauses.is_empty()
    }

    /// Parses the wire form. An empty slice (or zero clauses) is `None`: no filter.
    pub fn from_words(words: &[u32]) -> Result<Option<AttrFilter>, FormatError> {
        if words.is_empty() {
            return Ok(None);
        }
        let n = words[0] as usize;
        let mut pos = 1;
        let mut clauses = Vec::with_capacity(n);
        for _ in 0..n {
            if pos + 5 > words.len() {
                return Err(FormatError::Inconsistent("filter: truncated clause"));
            }
            let (mask, op, a, b, nv) = (
                words[pos],
                words[pos + 1],
                words[pos + 2],
                words[pos + 3],
                words[pos + 4] as usize,
            );
            let op = Op::from_u32(op).ok_or(FormatError::Inconsistent("filter: unknown op"))?;
            pos += 5;
            if pos + nv > words.len() {
                return Err(FormatError::Inconsistent("filter: truncated value list"));
            }
            let values = words[pos..pos + nv].to_vec();
            pos += nv;
            clauses.push(Clause {
                mask,
                op,
                a,
                b,
                values,
            });
        }
        if pos != words.len() {
            return Err(FormatError::Inconsistent("filter: trailing words"));
        }
        if clauses.is_empty() {
            return Ok(None);
        }
        Ok(Some(AttrFilter::new(clauses)))
    }

    pub fn to_words(&self) -> Vec<u32> {
        let mut out = vec![self.clauses.len() as u32];
        for c in &self.clauses {
            out.extend_from_slice(&[c.mask, c.op as u32, c.a, c.b, c.values.len() as u32]);
            out.extend_from_slice(&c.values);
        }
        out
    }

    /// Whether an entry with this `attrs` word passes every clause.
    #[inline]
    pub fn matches(&self, attrs: u32) -> bool {
        self.clauses.iter().all(|c| c.matches(attrs))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clause(mask: u32, op: Op, a: u32, b: u32, values: &[u32]) -> Clause {
        Clause {
            mask,
            op,
            a,
            b,
            values: values.to_vec(),
        }
    }

    #[test]
    fn ops() {
        let f = |c: Clause| AttrFilter::new(vec![c]);
        // a 4-bit field at bits 4..8
        let m = 0xF0;
        assert!(f(clause(m, Op::Eq, 0x30, 0, &[])).matches(0x1234 & 0xFF | 0x30));
        assert!(!f(clause(m, Op::Eq, 0x30, 0, &[])).matches(0x40));
        assert!(f(clause(m, Op::Ne, 0x30, 0, &[])).matches(0x40));
        assert!(f(clause(m, Op::In, 0, 0, &[0x50, 0x10, 0x30])).matches(0x1f));
        assert!(!f(clause(m, Op::In, 0, 0, &[0x50, 0x10, 0x30])).matches(0x2f));
        assert!(f(clause(m, Op::NotIn, 0, 0, &[0x50, 0x10, 0x30])).matches(0x2f));
        assert!(f(clause(m, Op::Range, 0x20, 0x40, &[])).matches(0x3a));
        assert!(!f(clause(m, Op::Range, 0x20, 0x40, &[])).matches(0x5a));
        assert!(!f(clause(m, Op::Range, 0x40, 0x20, &[])).matches(0x30)); // a > b: nothing
        assert!(f(clause(0b101, Op::BitsAny, 0, 0, &[])).matches(0b100));
        assert!(!f(clause(0b101, Op::BitsAny, 0, 0, &[])).matches(0b010));
        assert!(f(clause(0b101, Op::BitsAll, 0, 0, &[])).matches(0b111));
        assert!(!f(clause(0b101, Op::BitsAll, 0, 0, &[])).matches(0b100));
        assert!(f(clause(0b101, Op::BitsNone, 0, 0, &[])).matches(0b010));
        assert!(!AttrFilter::nothing().matches(0));
        assert!(AttrFilter::default().matches(123));
        // a large value set uses the binary search
        let many: Vec<u32> = (0..100).map(|i| i * 3).collect();
        let big = f(clause(u32::MAX, Op::In, 0, 0, &many));
        assert!(big.matches(99) && !big.matches(100));
        // conjunction
        let both = AttrFilter::new(vec![
            clause(m, Op::Eq, 0x30, 0, &[]),
            clause(0b1, Op::BitsAll, 0, 0, &[]),
        ]);
        assert!(both.matches(0x31) && !both.matches(0x30) && !both.matches(0x41));
    }

    #[test]
    fn wire_roundtrip_and_errors() {
        let f = AttrFilter::new(vec![
            clause(0xF0, Op::In, 0, 0, &[0x50, 0x10, 0x30, 0x10]),
            clause(0b1, Op::BitsNone, 0, 0, &[]),
        ]);
        let w = f.to_words();
        assert_eq!(
            w,
            vec![2, 0xF0, 2, 0, 0, 3, 0x10, 0x30, 0x50, 1, 7, 0, 0, 0]
        );
        assert_eq!(AttrFilter::from_words(&w).unwrap().unwrap(), f);
        assert_eq!(AttrFilter::from_words(&[]).unwrap(), None);
        assert_eq!(AttrFilter::from_words(&[0]).unwrap(), None);
        assert!(AttrFilter::from_words(&[1, 0xF0, 2, 0, 0]).is_err());
        assert!(AttrFilter::from_words(&[1, 0xF0, 9, 0, 0, 0]).is_err());
        assert!(AttrFilter::from_words(&[1, 0xF0, 0, 0, 0, 0, 5]).is_err());
        assert!(AttrFilter::from_words(&[1, 0xF0, 2, 0, 0, 2, 1]).is_err());
    }
}
