//! Ports of webp-go's metadata tests (internal/webp/metadata_test.go).
#![allow(clippy::type_complexity)]

use webp::{Error, Limits};

const CHUNK_COST: u64 = 32;

/// A RIFF/WEBP file from raw chunks: tag, declared size and body, which need not agree. `riff` of
/// `None` writes the true RIFF size.
fn meta_file(riff: Option<u32>, chunks: &[(&str, u32, &str)]) -> Vec<u8> {
    let mut b = b"RIFF\x00\x00\x00\x00WEBP".to_vec();
    for &(tag, size, body) in chunks {
        b.extend_from_slice(tag.as_bytes());
        b.extend_from_slice(&size.to_le_bytes());
        b.extend_from_slice(body.as_bytes());
        if body.len() & 1 == 1 && size as usize == body.len() {
            b.push(0);
        }
    }
    let riff = riff.unwrap_or((b.len() - 8) as u32);
    b[4..8].copy_from_slice(&riff.to_le_bytes());
    b
}

fn ok<'a>(tag: &'a str, body: &'a str) -> (&'a str, u32, &'a str) {
    (tag, body.len() as u32, body)
}

const IMG: (&str, u32, &str) = ("VP8L", 5, "\x2f\x00\x00\x00\x00");

#[test]
fn metadata() {
    type W = (&'static str, bool);
    let cases: Vec<(&str, Vec<u8>, Vec<W>, Vec<W>)> = vec![
        ("none", meta_file(None, &[IMG]), vec![], vec![]),
        (
            "order",
            meta_file(None, &[ok("EXIF", "e1"), IMG, ok("XMP ", "x1")]),
            vec![("e1", false)],
            vec![("x1", false)],
        ),
        (
            "duplicate EXIF",
            meta_file(None, &[ok("EXIF", "a"), ok("EXIF", "bb"), IMG]),
            vec![("a", false), ("bb", false)],
            vec![],
        ),
        (
            "duplicate XMP",
            meta_file(None, &[IMG, ok("XMP ", "a"), ok("XMP ", "bb")]),
            vec![],
            vec![("a", false), ("bb", false)],
        ),
        (
            "truncated EXIF",
            meta_file(None, &[IMG, ("EXIF", 10, "abc")]),
            vec![("abc", true)],
            vec![],
        ),
        (
            "truncated XMP",
            meta_file(None, &[IMG, ("XMP ", 10, "abc")]),
            vec![],
            vec![("abc", true)],
        ),
        (
            "size near 2^32",
            meta_file(None, &[IMG, ("EXIF", 0xFFFFFFFF, "abc")]),
            vec![("abc", true)],
            vec![],
        ),
        (
            "size 2^32-2",
            meta_file(None, &[IMG, ("XMP ", 0xFFFFFFFE, "abc")]),
            vec![],
            vec![("abc", true)],
        ),
        (
            "RIFF size short cuts chunk",
            meta_file(Some(27), &[IMG, ok("EXIF", "abcd")]),
            vec![("a", true)],
            vec![],
        ),
        (
            "RIFF size short drops later chunk",
            meta_file(Some(18), &[IMG, ok("EXIF", "abcd")]),
            vec![],
            vec![],
        ),
        (
            "RIFF size long",
            meta_file(Some(0xFFFFFFF0), &[IMG, ok("EXIF", "ab")]),
            vec![("ab", false)],
            vec![],
        ),
        (
            "RIFF size long, truncated",
            meta_file(Some(0xFFFFFFF0), &[IMG, ("EXIF", 9, "ab")]),
            vec![("ab", true)],
            vec![],
        ),
        (
            "partial chunk header",
            {
                let mut f = meta_file(None, &[IMG]);
                f.extend_from_slice(b"EXI");
                f
            },
            vec![],
            vec![],
        ),
    ];
    for (name, file, exif, xmp) in cases {
        let m = webp::metadata(&file, &Limits::default()).unwrap_or_else(|e| panic!("{name}: {e}"));
        for (kind, got, want) in [("EXIF", &m.exif, &exif), ("XMP", &m.xmp, &xmp)] {
            let got: Vec<(&[u8], bool)> = got.iter().map(|c| (c.data, c.truncated)).collect();
            let want: Vec<(&[u8], bool)> = want.iter().map(|&(d, t)| (d.as_bytes(), t)).collect();
            assert_eq!(got, want, "{name}: {kind}");
        }
    }
}

#[test]
fn metadata_errors() {
    let l = Limits::default();
    for b in [
        &b""[..],
        b"RIFF",
        b"RIFF\x04\x00\x00\x00WAVE",
        b"RIFF\x03\x00\x00\x00WEBP",
    ] {
        assert_eq!(webp::metadata(b, &l).err(), Some(Error::Format), "{b:?}");
    }
}

#[test]
fn metadata_limit() {
    let f = meta_file(None, &[ok("EXIF", "abcd"), ok("XMP ", "efgh")]);
    let lim = |n: u64| Limits {
        max_metadata_bytes: n,
        ..Limits::default()
    };
    assert!(
        webp::metadata(&f, &lim(8 + 2 * CHUNK_COST)).is_ok(),
        "at limit"
    );
    let r = webp::metadata(&f, &lim(7 + 2 * CHUNK_COST));
    assert_eq!(r.err().map(|e| e.class()), Some("limit"), "over limit");
    // A huge declared size counts only the bytes the file holds.
    let f = meta_file(None, &[("EXIF", 0xFFFFFFFF, "abcd")]);
    let r = webp::metadata(&f, &lim(3 + CHUNK_COST));
    assert_eq!(
        r.err().map(|e| e.class()),
        Some("limit"),
        "truncated over limit"
    );
}

/// Empty chunks hold no bytes but each costs a chunk; the limit must stop the walk before they are
/// all collected.
#[test]
fn metadata_limit_empty_chunks() {
    const N: usize = 1 << 21;
    let mut f = meta_file(None, &[IMG]);
    for _ in 0..N {
        f.extend_from_slice(b"EXIF\x00\x00\x00\x00");
    }
    let n = (f.len() - 8) as u32;
    f[4..8].copy_from_slice(&n.to_le_bytes());
    let r = webp::metadata(&f, &Limits::default());
    assert_eq!(r.err().map(|e| e.class()), Some("limit"), "default limit");
    let l = Limits {
        max_metadata_bytes: 1,
        ..Limits::default()
    };
    let r = webp::metadata(&f, &l);
    assert_eq!(r.err().map(|e| e.class()), Some("limit"), "limit 1");
}

/// The chunk entries `metadata` collects are charged against `max_alloc`: 524288 empty EXIF
/// chunks (a 4 MiB file, 16 MiB at 32 each, so inside the default `max_metadata_bytes`) need about
/// 12 MiB of entries and are refused with `Error::MemoryBudget` under a 4 MiB `max_alloc`, while
/// a few chunks pass under a tiny one.
#[test]
fn metadata_charges_max_alloc() {
    const N: usize = 1 << 19;
    let mut f = meta_file(None, &[IMG]);
    for _ in 0..N {
        f.extend_from_slice(b"EXIF\x00\x00\x00\x00");
    }
    let n = (f.len() - 8) as u32;
    f[4..8].copy_from_slice(&n.to_le_bytes());
    let l = |max_alloc| Limits {
        max_alloc,
        ..Limits::default()
    };
    assert_eq!(webp::metadata(&f, &l(1 << 30)).unwrap().exif.len(), N);
    let r = webp::metadata(&f, &l(4 << 20));
    assert!(
        matches!(r, Err(Error::MemoryBudget { max, .. }) if max == 4 << 20),
        "{:?}",
        r.err()
    );
    let few = meta_file(None, &[IMG, ok("EXIF", "abcd"), ok("XMP ", "efgh")]);
    assert!(webp::metadata(&few, &l(1 << 12)).is_ok());
}

/// A file that ends inside an EXIF or XMP chunk header, within the RIFF size, has a cut chunk:
/// truncated, no data.
#[test]
fn metadata_partial_header_in_riff() {
    let cases: [(&[u8], usize, usize); 13] = [
        (b"E", 1, 0),
        (b"EXI", 1, 0),
        (b"EXIF", 1, 0),
        (b"EXIF\x1a\x00", 1, 0),
        (b"EXIF\x1a\x00\x00", 1, 0),
        (b"X", 0, 1),
        (b"XMP", 0, 1),
        (b"XMP ", 0, 1),
        (b"XMP \x01\x00\x00", 0, 1),
        (b"JUNK", 0, 0),
        (b"EXIX", 0, 0),
        (b"\x00", 0, 0),
        (b"ALPHA\x00\x00", 0, 0),
    ];
    for (tail, exif, xmp) in cases {
        let mut b = meta_file(None, &[IMG]);
        b.extend_from_slice(tail);
        let n = (b.len() - 8) as u32;
        b[4..8].copy_from_slice(&n.to_le_bytes());
        let m = webp::metadata(&b, &Limits::default()).unwrap();
        assert_eq!((m.exif.len(), m.xmp.len()), (exif, xmp), "tail {tail:?}");
        for c in m.exif.iter().chain(&m.xmp) {
            assert!(c.truncated && c.data.is_empty(), "tail {tail:?}");
        }
    }
}
