#![no_main]
use libfuzzer_sys::fuzz_target;
use webp::{Error, Limits};

/// Whether `s` lies inside `data`.
fn inside(data: &[u8], s: &[u8]) -> bool {
    let (d, p) = (data.as_ptr() as usize, s.as_ptr() as usize);
    s.is_empty() || (p >= d && p + s.len() <= d + data.len())
}

// data = [max: u16 LE][a WebP file]. The chunks alias the file, cost 32 bytes each besides their
// bytes, and together stay within the limit; one that would not is a limit error, which a large
// enough limit does not give.
fuzz_target!(|data: &[u8]| {
    let Some((&[a, b], file)) = data.split_first_chunk::<2>() else {
        return;
    };
    let max = u64::from(u16::from_le_bytes([a, b]));
    let small = Limits {
        max_metadata_bytes: max,
        ..Limits::default()
    };
    match webp::metadata(file, &small) {
        Ok(m) => {
            let mut cost = 0u64;
            for c in m.exif.iter().chain(&m.xmp) {
                assert!(inside(file, c.data));
                cost += 32 + c.data.len() as u64;
            }
            assert!(cost <= max, "cost {cost} over the limit {max}");
        }
        Err(Error::MetadataBytes { .. }) => {
            let big = Limits {
                max_metadata_bytes: 1 << 40,
                ..Limits::default()
            };
            if let Ok(m) = webp::metadata(file, &big) {
                let cost: u64 = m
                    .exif
                    .iter()
                    .chain(&m.xmp)
                    .map(|c| 32 + c.data.len() as u64)
                    .sum();
                assert!(cost > max, "limit error for cost {cost} within {max}");
            }
        }
        Err(_) => {}
    }
});
