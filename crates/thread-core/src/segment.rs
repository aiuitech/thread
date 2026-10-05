//! A segment: one immutable object holding, for a batch of documents, its `meta`, a lexical
//! index over them (`terms` + `postings`, see [`crate::lex`]), their `docs` entries (see
//! [`crate::docs`]) and/or their `keys` (unique string keys -> ids, see [`crate::keys`]), as a
//! [`Container`]. An upsert writes one segment next to its dense delta
//! segment; a compaction merges every segment of a generation into one base segment. The dense
//! slabs, the segments and the tombstone sets share the index's u32 ids, so a hit from either
//! side resolves through the same `docs` and a replaced or deleted id is hidden everywhere by
//! the one tombstone rule (a segment added at manifest rev `a` is filtered by the sets with
//! `rev > a`).
//!
//! `meta` is JSON: `{"format":"thread-seg","version":1,"tokenizer":{..},"lex":{..},"docs":{..},"keys":{..}}`;
//! the core reads only `tokenizer` back (to tokenize queries exactly like the documents).

use crate::container::{self, Container, ContainerWriter};
use crate::docs::{self, Docs, DocsBuilder};
use crate::error::Error;
use crate::keys::{self, Keys, KeysBuilder};
use crate::lex::{self, LexBuilder, LexParams, Postings, Terms};
use crate::source::{ByteSource, Range, SubSource};
use crate::tokenize::{Tokenizer, TokenizerConfig};
use crate::tomb::Tombstones;

pub const FORMAT: &str = "thread-seg";
pub const VERSION: u32 = 1;

/// What a segment stores.
#[derive(Clone, Debug, PartialEq)]
pub struct SegmentOptions {
    /// Tokenizer of the lexical index; `None` builds no `terms`/`postings`.
    pub tokenizer: Option<TokenizerConfig>,
    pub lex: LexParams,
    /// Write a `docs` section.
    pub docs: bool,
    /// Store the texts in `docs` (needs `docs`).
    pub text: bool,
    pub bucket_docs: u32,
    /// Write a `keys` section (the documents' unique string keys -> ids). A merge writes one
    /// whenever an input has one.
    pub keys: bool,
    pub bucket_keys: u32,
}

impl Default for SegmentOptions {
    fn default() -> Self {
        SegmentOptions {
            tokenizer: Some(TokenizerConfig::bm25s_default()),
            lex: LexParams::default(),
            docs: true,
            text: false,
            bucket_docs: docs::DEFAULT_BUCKET_DOCS,
            keys: false,
            bucket_keys: keys::DEFAULT_BUCKET_KEYS,
        }
    }
}

/// Builds one segment: tokenizes texts, collects docs entries, writes the container.
pub struct SegmentBuilder {
    cfg: Option<TokenizerConfig>,
    tokenizer: Option<Tokenizer>,
    lex: Option<LexBuilder>,
    docs: Option<DocsBuilder>,
    keys: Option<KeysBuilder>,
    tokens: u64,
}

impl SegmentBuilder {
    pub fn new(opts: &SegmentOptions) -> Result<SegmentBuilder, Error> {
        if opts.text && !opts.docs {
            return Err(Error::Format(
                "segment: storing texts needs the docs section".into(),
            ));
        }
        if opts.tokenizer.is_none() && !opts.docs && !opts.keys {
            return Err(Error::Format(
                "segment: nothing to build (no tokenizer, no docs, no keys)".into(),
            ));
        }
        let tokenizer = match &opts.tokenizer {
            Some(c) => Some(Tokenizer::new(c)?),
            None => None,
        };
        Ok(SegmentBuilder {
            cfg: opts.tokenizer.clone(),
            tokenizer,
            lex: opts.tokenizer.as_ref().map(|_| LexBuilder::new(opts.lex)),
            docs: opts.docs.then(|| {
                DocsBuilder::new()
                    .bucket_docs(opts.bucket_docs)
                    .with_text(opts.text)
            }),
            keys: opts
                .keys
                .then(|| KeysBuilder::new().bucket_keys(opts.bucket_keys)),
            tokens: 0,
        })
    }

    pub fn has_lex(&self) -> bool {
        self.lex.is_some()
    }
    pub fn has_docs(&self) -> bool {
        self.docs.is_some()
    }
    pub fn has_keys(&self) -> bool {
        self.keys.is_some()
    }
    pub fn len(&self) -> u32 {
        self.lex
            .as_ref()
            .map(|l| l.num_docs())
            .or_else(|| self.docs.as_ref().map(|d| d.len()))
            .or_else(|| self.keys.as_ref().map(|k| k.len()))
            .unwrap_or(0)
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Tokens indexed so far.
    pub fn tokens(&self) -> u64 {
        self.tokens
    }
    /// The tokenizer the segment indexes with (for tokenizing queries the same way).
    pub fn tokenizer_config(&self) -> Option<&TokenizerConfig> {
        self.cfg.as_ref()
    }

    /// Adds a document: `text` is tokenized into the lexical index (when the segment has one;
    /// `None` indexes an empty document) and stored when the segment keeps texts; `meta` (default
    /// `""`) and `attrs` (the document's attribute word, the same its dense entry carries) go into
    /// `docs`. Returns the number of tokens indexed. Fails on a duplicate id.
    pub fn add(
        &mut self,
        id: u32,
        text: Option<&str>,
        meta: Option<&str>,
        attrs: u32,
    ) -> Result<usize, Error> {
        self.add_keyed(id, None, text, meta, attrs)
    }

    /// [`add`](Self::add) with the document's unique string `key`, stored in `keys` when the
    /// segment has that section (a segment without one ignores it; a document may have no key).
    /// Fails on a duplicate id or key (nothing is added then).
    pub fn add_keyed(
        &mut self,
        id: u32,
        key: Option<&str>,
        text: Option<&str>,
        meta: Option<&str>,
        attrs: u32,
    ) -> Result<usize, Error> {
        if let (Some(k), Some(key)) = (&self.keys, key) {
            if k.contains(key) {
                return Err(Error::Format(format!("duplicate key {key:?}")));
            }
        }
        let mut n = 0;
        if let (Some(tok), Some(lex)) = (&mut self.tokenizer, &mut self.lex) {
            let tokens = tok.tokenize(text.unwrap_or(""));
            n = tokens.len();
            lex.add(id, &tokens)?;
        }
        if let Some(d) = &mut self.docs {
            // a duplicate id is caught above when there is a lexical index, here otherwise
            d.add_attrs(id, meta.unwrap_or(""), text, attrs)?;
        }
        if let (Some(k), Some(key)) = (&mut self.keys, key) {
            k.add(key, id)?;
        }
        self.tokens += n as u64;
        Ok(n)
    }

    /// Adds a pre-tokenized document (tokens must come from the same tokenizer config).
    pub fn add_tokens(
        &mut self,
        id: u32,
        tokens: &[String],
        meta: Option<&str>,
        text: Option<&str>,
        attrs: u32,
    ) -> Result<(), Error> {
        if let Some(lex) = &mut self.lex {
            lex.add(id, tokens)?;
        }
        if let Some(d) = &mut self.docs {
            d.add_attrs(id, meta.unwrap_or(""), text, attrs)?;
        }
        self.tokens += tokens.len() as u64;
        Ok(())
    }

    /// The container bytes.
    pub fn finish(&self) -> Result<Vec<u8>, Error> {
        let lex = match &self.lex {
            Some(l) => Some(l.finish()?),
            None => None,
        };
        let docs = match &self.docs {
            Some(d) => Some(d.finish()?),
            None => None,
        };
        let keys = match &self.keys {
            Some(k) => Some(k.finish()?),
            None => None,
        };
        write_container(self.cfg.as_ref(), lex, docs, keys)
    }
}

/// Assembles the container from finished sections (`meta` is derived from them).
pub fn write_container(
    tokenizer: Option<&TokenizerConfig>,
    lex: Option<lex::LexBytes>,
    docs: Option<Vec<u8>>,
    keys: Option<Vec<u8>>,
) -> Result<Vec<u8>, Error> {
    let mut meta = format!(r#"{{"format":"{FORMAT}","version":{VERSION}"#);
    if let Some(cfg) = tokenizer {
        meta.push_str(",\"tokenizer\":");
        meta.push_str(&cfg.to_json());
    }
    if let Some(l) = &lex {
        let t = Terms::open(l.terms.as_slice())?;
        meta.push_str(",\"lex\":");
        meta.push_str(&t.meta_json(l.postings.len() as u64));
    }
    if let Some(d) = &docs {
        let r = Docs::open(d.as_slice())?;
        meta.push_str(",\"docs\":");
        meta.push_str(&r.meta_json());
    }
    if let Some(k) = &keys {
        let r = Keys::open(k.as_slice())?;
        meta.push_str(",\"keys\":");
        meta.push_str(&r.meta_json());
    }
    meta.push('}');
    let mut w = ContainerWriter::new();
    w.add(container::SEC_META, meta)?;
    if let Some(l) = lex {
        w.add(container::SEC_TERMS, l.terms)?;
        w.add(container::SEC_POSTINGS, l.postings)?;
    }
    if let Some(d) = docs {
        w.add(container::SEC_DOCS, d)?;
    }
    if let Some(k) = keys {
        w.add(container::SEC_KEYS, k)?;
    }
    Ok(w.finish())
}

/// An opened segment: the container table and the `meta` section, from which the section
/// readers are opened on demand (each over a [`SubSource`] of the same source, so plans and
/// `Missing` ranges stay blob-absolute). On a demand-paged source, `open` needs the container
/// header + table ([`Container::plan_open`]) and the `meta` section.
pub struct Segment {
    container: Container,
    meta: String,
    tokenizer: Option<TokenizerConfig>,
}

impl Segment {
    pub fn open<S: ByteSource + ?Sized>(source: &S) -> Result<Segment, Error> {
        let container = Container::open(source)?;
        let meta = match container.section(container::SEC_META) {
            Some(r) => String::from_utf8_lossy(&source.read(r)?).into_owned(),
            None => String::from("{}"),
        };
        let tokenizer = TokenizerConfig::from_meta_json(&meta)?;
        Ok(Segment {
            container,
            meta,
            tokenizer,
        })
    }

    pub fn container(&self) -> &Container {
        &self.container
    }
    /// The `meta` JSON.
    pub fn meta(&self) -> &str {
        &self.meta
    }
    /// The tokenizer config recorded in `meta` (`None` for a docs-only segment).
    pub fn tokenizer(&self) -> Option<&TokenizerConfig> {
        self.tokenizer.as_ref()
    }
    pub fn has_lex(&self) -> bool {
        self.container.section(container::SEC_TERMS).is_some()
            && self.container.section(container::SEC_POSTINGS).is_some()
    }
    pub fn has_docs(&self) -> bool {
        self.container.section(container::SEC_DOCS).is_some()
    }
    pub fn has_keys(&self) -> bool {
        self.container.section(container::SEC_KEYS).is_some()
    }
    pub fn section(&self, name: &str) -> Option<Range> {
        self.container.section(name)
    }

    fn sub<'a, S: ByteSource + ?Sized>(
        &self,
        source: &'a S,
        name: &str,
    ) -> Option<Result<SubSource<'a, S>, Error>> {
        self.container
            .section(name)
            .map(|r| SubSource::new(source, r).map_err(Error::Source))
    }

    /// The `terms` reader (`None` without a lexical index).
    pub fn terms<'a, S: ByteSource + ?Sized>(
        &self,
        source: &'a S,
    ) -> Option<Result<Terms<SubSource<'a, S>>, Error>> {
        self.sub(source, container::SEC_TERMS)
            .map(|s| Terms::open(s?))
    }
    pub fn postings<'a, S: ByteSource + ?Sized>(
        &self,
        source: &'a S,
    ) -> Option<Result<Postings<SubSource<'a, S>>, Error>> {
        self.sub(source, container::SEC_POSTINGS)
            .map(|s| Postings::open(s?))
    }
    pub fn docs<'a, S: ByteSource + ?Sized>(
        &self,
        source: &'a S,
    ) -> Option<Result<Docs<SubSource<'a, S>>, Error>> {
        self.sub(source, container::SEC_DOCS)
            .map(|s| Docs::open(s?))
    }
    /// The `keys` reader (`None` without a keys section).
    pub fn keys<'a, S: ByteSource + ?Sized>(
        &self,
        source: &'a S,
    ) -> Option<Result<Keys<SubSource<'a, S>>, Error>> {
        self.sub(source, container::SEC_KEYS)
            .map(|s| Keys::open(s?))
    }
}

type DocsInput<'a> = (&'a Docs<SubSource<'a, [u8]>>, Option<&'a Tombstones>);
type KeysInput<'a> = (&'a Keys<SubSource<'a, [u8]>>, Option<&'a Tombstones>);

/// Merges segments (given **oldest first**, each with the tombstones that apply to it) into one:
/// the lexical indexes through [`lex::merge`] with `lex` as the new parameters, the docs through
/// [`docs::merge`] (newest copy of an id wins). The tokenizer config is taken from the first
/// segment that has one (every lexical segment of an index shares it). `text` stores texts when
/// the inputs have them; `docs` false drops the docs section. A `keys` section (newest copy of a
/// key wins, [`keys::merge`]) is written when `keys` is set or any input has one.
pub fn merge(
    inputs: &[(&[u8], Option<&Tombstones>)],
    opts: &SegmentOptions,
) -> Result<Vec<u8>, Error> {
    let mut segs = Vec::with_capacity(inputs.len());
    for (bytes, skip) in inputs {
        segs.push((Segment::open(*bytes)?, *bytes, *skip));
    }
    let tokenizer = opts
        .tokenizer
        .clone()
        .or_else(|| segs.iter().find_map(|(s, _, _)| s.tokenizer().cloned()));
    let mut lex_out = None;
    if tokenizer.is_some() {
        let mut readers = Vec::new();
        for (s, bytes, skip) in &segs {
            if let (Some(t), Some(p)) = (s.terms(*bytes), s.postings(*bytes)) {
                readers.push((t?, p?, *skip));
            }
        }
        let ins: Vec<lex::MergeInput<'_, SubSource<'_, [u8]>>> = readers
            .iter()
            .map(|(t, p, skip)| lex::MergeInput {
                terms: t,
                postings: p,
                skip: *skip,
            })
            .collect();
        lex_out = Some(lex::merge(&ins, opts.lex)?.finish()?);
    }
    let mut docs_out = None;
    if opts.docs {
        let mut readers = Vec::new();
        for (s, bytes, skip) in &segs {
            if let Some(d) = s.docs(*bytes) {
                readers.push((d?, *skip));
            }
        }
        let ins: Vec<DocsInput<'_>> = readers.iter().map(|(d, s)| (d, *s)).collect();
        docs_out = Some(docs::merge(&ins, opts.bucket_docs, opts.text)?.finish()?);
    }
    let mut keys_out = None;
    if opts.keys || segs.iter().any(|(s, _, _)| s.has_keys()) {
        let mut readers = Vec::new();
        for (s, bytes, skip) in &segs {
            if let Some(k) = s.keys(*bytes) {
                readers.push((k?, *skip));
            }
        }
        let ins: Vec<KeysInput<'_>> = readers.iter().map(|(k, s)| (k, *s)).collect();
        keys_out = Some(keys::merge(&ins, opts.bucket_keys)?.finish()?);
    }
    write_container(tokenizer.as_ref(), lex_out, docs_out, keys_out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lex::{QueryTerm, Stats};
    use crate::source::Prefetched;

    const DOCS: [(u32, &str, &str); 6] = [
        (10, "The deployments failed on Tuesday", "t10"),
        (3, "Retry the deployment, then retry again", "t3"),
        (1000, "Budget review for the galaxy budget", "t1000"),
        (42, "galaxy galaxy budget", "t42"),
        (7, "", "t7"),
        (99, "Nothing to see here", "t99"),
    ];

    fn build(opts: &SegmentOptions, rows: &[(u32, &str, &str)]) -> Vec<u8> {
        let mut b = SegmentBuilder::new(opts).unwrap();
        for (id, text, meta) in rows {
            let key = format!("doc/{id}");
            b.add_keyed(*id, Some(&key), Some(text), Some(meta), *id % 3)
                .unwrap();
        }
        assert_eq!(b.len() as usize, rows.len());
        b.finish().unwrap()
    }

    #[test]
    fn build_open_search_and_docs() {
        let opts = SegmentOptions {
            text: true,
            ..SegmentOptions::default()
        };
        let bytes = build(&opts, &DOCS);
        let seg = Segment::open(&bytes).unwrap();
        assert!(seg.has_lex() && seg.has_docs() && !seg.has_keys());
        assert!(seg.keys(&bytes).is_none());
        assert_eq!(seg.tokenizer(), Some(&TokenizerConfig::bm25s_default()));
        assert!(
            seg.meta().contains(r#""format":"thread-seg""#)
                && seg.meta().contains(r#""lex":{"#)
                && seg.meta().contains(r#""docs":{"#)
        );
        let terms = seg.terms(&bytes).unwrap().unwrap();
        let postings = seg.postings(&bytes).unwrap().unwrap();
        let docs = seg.docs(&bytes).unwrap().unwrap();
        assert_eq!(terms.num_docs(), 6);
        assert_eq!(docs.n(), 6);
        // a query tokenized with the segment's config: "deployment" and "deployments" both stem to "deploy"
        let mut tok = Tokenizer::new(seg.tokenizer().unwrap()).unwrap();
        let q = tok.tokenize("deployment retry");
        let qs: Vec<&str> = q.iter().map(String::as_str).collect();
        let stats: Stats = terms.stats();
        let infos = terms.lookup(&qs).unwrap();
        let qt: Vec<QueryTerm> = infos
            .iter()
            .flatten()
            .map(|i| QueryTerm {
                info: *i,
                idf: stats.idf(i.df as u64),
                weight: 1.0,
            })
            .collect();
        let hits = postings
            .search(&qt, &stats.scoring(terms.k1(), terms.b()), 5, None)
            .unwrap();
        assert_eq!(hits.iter().map(|h| h.id).collect::<Vec<_>>(), vec![3, 10]);
        let got = docs.get(&[3, 10, 7, 5]).unwrap();
        assert_eq!(
            docs.lookup(&[3, 10])
                .unwrap()
                .iter()
                .map(|e| e.as_ref().unwrap().attrs)
                .collect::<Vec<_>>(),
            vec![0, 1]
        );
        assert_eq!(got[0], Some(("t3".into(), DOCS[1].1.into())));
        assert_eq!(got[1], Some(("t10".into(), DOCS[0].1.into())));
        assert_eq!(got[2], Some(("t7".into(), String::new())));
        assert_eq!(got[3], None);
        // docs-only and lex-only segments
        let docs_only = build(
            &SegmentOptions {
                tokenizer: None,
                ..SegmentOptions::default()
            },
            &DOCS,
        );
        let s = Segment::open(&docs_only).unwrap();
        assert!(!s.has_lex() && s.has_docs() && s.tokenizer().is_none());
        let lex_only = build(
            &SegmentOptions {
                docs: false,
                ..SegmentOptions::default()
            },
            &DOCS,
        );
        let s = Segment::open(&lex_only).unwrap();
        assert!(s.has_lex() && !s.has_docs());
        assert!(SegmentBuilder::new(&SegmentOptions {
            tokenizer: None,
            docs: false,
            ..SegmentOptions::default()
        })
        .is_err());
        let mut b = SegmentBuilder::new(&SegmentOptions::default()).unwrap();
        b.add(1, Some("x y"), None, 0).unwrap();
        assert!(b.add(1, Some("z"), None, 0).is_err());
    }

    #[test]
    fn keys_section_and_keys_only_segments() {
        let opts = SegmentOptions {
            keys: true,
            bucket_keys: 2,
            ..SegmentOptions::default()
        };
        let bytes = build(&opts, &DOCS);
        let seg = Segment::open(&bytes).unwrap();
        assert!(seg.has_lex() && seg.has_docs() && seg.has_keys());
        assert!(seg
            .meta()
            .contains(r#""keys":{"version":1,"n":6,"bucket_keys":2"#));
        let k = seg.keys(&bytes).unwrap().unwrap();
        assert_eq!(k.n(), 6);
        assert_eq!(
            k.lookup(&["doc/3", "doc/1000", "doc/4"]).unwrap(),
            vec![Some(3), Some(1000), None]
        );
        let page = k.list("doc/1", None, 10).unwrap();
        assert_eq!(
            page.entries,
            vec![("doc/10".into(), 10), ("doc/1000".into(), 1000)]
        );
        // a duplicate key is refused before anything is added
        let mut b = SegmentBuilder::new(&opts).unwrap();
        b.add_keyed(1, Some("k"), Some("x"), None, 0).unwrap();
        assert!(b.add_keyed(2, Some("k"), Some("y"), None, 0).is_err());
        assert_eq!(b.len(), 1);
        // a document without a key, and a segment without a keys section ignoring keys
        b.add_keyed(2, None, Some("y"), None, 0).unwrap();
        let bytes = b.finish().unwrap();
        let s = Segment::open(&bytes).unwrap();
        let k = s.keys(&bytes).unwrap().unwrap();
        assert_eq!(k.entries().unwrap(), vec![("k".into(), 1)]);
        let mut b = SegmentBuilder::new(&SegmentOptions::default()).unwrap();
        b.add_keyed(1, Some("k"), Some("x"), None, 0).unwrap();
        assert!(!Segment::open(&b.finish().unwrap()).unwrap().has_keys());
        // keys only: no tokenizer, no docs
        let only = SegmentOptions {
            tokenizer: None,
            docs: false,
            keys: true,
            ..SegmentOptions::default()
        };
        let mut b = SegmentBuilder::new(&only).unwrap();
        b.add_keyed(7, Some("seven"), None, None, 0).unwrap();
        assert_eq!(b.len(), 1);
        let bytes = b.finish().unwrap();
        let s = Segment::open(&bytes).unwrap();
        assert!(!s.has_lex() && !s.has_docs() && s.has_keys());
        assert_eq!(
            s.keys(&bytes).unwrap().unwrap().lookup(&["seven"]).unwrap(),
            vec![Some(7)]
        );
        // merge: the newest copy of a key wins, tombstoned ids drop their keys, inputs without
        // keys are fine, and the section appears whenever one input has it
        let a = build(&opts, &DOCS[..3]);
        let plain = build(&SegmentOptions::default(), &DOCS[3..5]);
        let mut cb = SegmentBuilder::new(&opts).unwrap();
        cb.add_keyed(99, Some("doc/99-renamed"), Some("Nothing"), Some("t99"), 0)
            .unwrap();
        cb.add_keyed(3, Some("doc/3"), Some("Retry"), Some("t3-v2"), 0)
            .unwrap();
        let c = cb.finish().unwrap();
        let dead_a = Tombstones::from_ids(&[3, 10]);
        let merged = merge(
            &[(&a, Some(&dead_a)), (&plain, None), (&c, None)],
            &SegmentOptions::default(),
        )
        .unwrap();
        let s = Segment::open(&merged).unwrap();
        assert!(s.has_keys());
        let k = s.keys(&merged).unwrap().unwrap();
        assert_eq!(
            k.entries().unwrap(),
            vec![
                ("doc/1000".into(), 1000),
                ("doc/3".into(), 3),
                ("doc/99-renamed".into(), 99)
            ]
        );
        let d = s.docs(&merged).unwrap().unwrap();
        assert_eq!(d.n(), 5);
    }

    #[test]
    fn merge_segments_equals_one_build() {
        let opts = SegmentOptions {
            text: true,
            ..SegmentOptions::default()
        };
        let a = build(&opts, &DOCS[..3]);
        let b = build(&opts, &DOCS[3..]);
        // a replaces id 3 in a later segment, and 99 is deleted after segment b
        let c = build(
            &opts,
            &[(3, "Retry the deployment, then retry again", "t3-v2")],
        );
        let dead_a = Tombstones::from_ids(&[3, 99]);
        let dead_b = Tombstones::from_ids(&[99]);
        let merged = merge(
            &[(&a, Some(&dead_a)), (&b, Some(&dead_b)), (&c, None)],
            &opts,
        )
        .unwrap();
        let mut rows: Vec<(u32, &str, &str)> = DOCS
            .iter()
            .filter(|r| r.0 != 99 && r.0 != 7)
            .copied()
            .collect();
        rows.iter_mut().find(|r| r.0 == 3).unwrap().2 = "t3-v2";
        // the lexical part equals a fresh build over the live non-empty documents; docs keep the
        // empty document 7 too (it has no tokens but is a document)
        let seg = Segment::open(&merged).unwrap();
        let want = build(&opts, &rows);
        let ws = Segment::open(&want).unwrap();
        let (mt, wt) = (
            seg.section(container::SEC_TERMS).unwrap(),
            ws.section(container::SEC_TERMS).unwrap(),
        );
        assert_eq!(
            &merged[mt.offset as usize..mt.end() as usize],
            &want[wt.offset as usize..wt.end() as usize]
        );
        let (mp, wp) = (
            seg.section(container::SEC_POSTINGS).unwrap(),
            ws.section(container::SEC_POSTINGS).unwrap(),
        );
        assert_eq!(
            &merged[mp.offset as usize..mp.end() as usize],
            &want[wp.offset as usize..wp.end() as usize]
        );
        let d = seg.docs(&merged).unwrap().unwrap();
        assert_eq!(d.n(), 5);
        assert_eq!(d.lookup(&[10]).unwrap()[0].as_ref().unwrap().attrs, 1);
        let got = d.get(&[3, 7, 99]).unwrap();
        assert_eq!(got[0], Some(("t3-v2".into(), DOCS[1].1.into())));
        assert_eq!(got[1], Some(("t7".into(), String::new())));
        assert_eq!(got[2], None);
        assert_eq!(seg.tokenizer(), Some(&TokenizerConfig::bm25s_default()));
    }

    #[test]
    fn opening_a_paged_segment_needs_the_table_and_meta() {
        let bytes = build(&SegmentOptions::default(), &DOCS);
        let mut p = Prefetched::new(bytes.len() as u64);
        assert!(Segment::open(&p).is_err());
        let c = Container::open(&bytes).unwrap();
        let table = container::HEADER_LEN + c.sections().len() as u64 * container::ENTRY_LEN;
        p.insert(0, bytes[..table as usize].to_vec()).unwrap();
        let e = Segment::open(&p).err().unwrap();
        let meta = c.section(container::SEC_META).unwrap();
        assert_eq!(e.missing(), Some(&[meta][..]));
        // `plan_open` (header + a 16-entry table) covers the small meta of this segment too
        let mut p2 = Prefetched::new(bytes.len() as u64);
        for r in Container::plan_open().iter() {
            let end = r.end().min(bytes.len() as u64) as usize;
            p2.insert(r.offset, bytes[r.offset as usize..end].to_vec())
                .unwrap();
        }
        assert!(Segment::open(&p2).is_ok());
        p.insert(
            meta.offset,
            bytes[meta.offset as usize..meta.end() as usize].to_vec(),
        )
        .unwrap();
        let s = Segment::open(&p).unwrap();
        assert!(s.has_lex());
        // the terms reader then wants its header
        let t = s.terms(&p).unwrap();
        assert!(t.err().unwrap().missing().is_some());
    }
}
