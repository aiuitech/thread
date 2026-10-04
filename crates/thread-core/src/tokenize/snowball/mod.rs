//! Snowball 3.1.0's English stemmer, vendored so that `tokenize` stems exactly like the
//! PyStemmer Wiper's `bm25/v1` and `bm25/v2` use (PyStemmer 3.1.0 bundles libstemmer 3.1.0).
//!
//! `rust-stemmers` 1.2.0 was generated from Snowball 2.x; the English algorithm changed in 3.0
//! (new exception lists such as `inter`/`organ`/`univers`/`past`, `-ogist`, `-edly` handling),
//! and 30 of 338k English words from a sweep stemmed differently (`added`, `internal`,
//! `organization`, `university`, `interval`, `biologist`, `paste`, ...). Snowball ships a Rust
//! backend, so the files here are:
//!
//! * `snowball_env.rs`, `among.rs`: the runtime from `rust/src/snowball/` of the Snowball
//!   repository at tag `v3.1.0` (commit `77e07c9`), imports adjusted to this module;
//! * `english.rs`: `snowball algorithms/english.sbl -rust` from the same tag, unchanged but for
//!   the imports.
//!
//! Snowball is BSD-3-Clause: Copyright (c) 2001, Dr Martin Porter; (c) 2004,2005, Richard
//! Boulton; (c) 2013, Yoshiki Shibukawa; (c) 2006-2025, Olly Betts. Redistribution with the
//! copyright notice, conditions and disclaimer of its `COPYING` file. To refresh: build the
//! compiler (`zig cc -o snowball compiler/*.c`), regenerate `english.rs`, re-run the Python
//! sweep in `tests/test_thread_tokenize.py`.
#![allow(clippy::all)]
#![allow(dead_code)] // the runtime offers more than the English algorithm uses

#[rustfmt::skip]
mod among;
#[rustfmt::skip]
mod english;
#[rustfmt::skip]
mod snowball_env;

pub use among::Among;
pub use snowball_env::SnowballEnv;

/// Snowball English (3.1.0) stem of one lowercase word.
pub fn stem_english(word: &str) -> String {
    let mut env = SnowballEnv::create(word);
    english::stem(&mut env);
    env.get_current().into_owned()
}
