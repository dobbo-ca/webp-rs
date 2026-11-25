//! Port of libwebp 1.2.2 utils/huffman_encode_utils.c. Copyright 2012 Google Inc. BSD-3, see
//! LICENSE.libwebp.

use std::cmp::Ordering;

/// MAX_ALLOWED_CODE_LENGTH, format_constants.h.
const MAX_ALLOWED_CODE_LENGTH: usize = 15;

/// HuffmanTreeToken.
#[derive(Clone, Copy, Default)]
pub(super) struct Token {
    /// 0..15 or escape 16, 17, 18.
    pub code: u8,
    pub extra_bits: u8,
}

/// HuffmanTree, a node of the tree-building pool.
#[derive(Clone, Copy, Default)]
pub(super) struct Tree {
    total_count: u32,
    value: i32,
    left: i32,
    right: i32,
}

/// HuffmanTreeCode over its slice of the code storage.
pub(super) struct Code<'a> {
    pub num_symbols: usize,
    pub code_lengths: &'a mut [u8],
    pub codes: &'a mut [u16],
}

fn values_should_be_collapsed_to_stride_average(a: i32, b: i32) -> bool {
    (a - b).abs() < 4
}

/// OptimizeHuffmanForRle, huffman_encode_utils.c:31.
fn optimize_huffman_for_rle(good_for_rle: &mut [u8], counts: &mut [u32]) {
    let mut length = counts.len();
    loop {
        if length == 0 {
            return; // all zeros
        }
        if counts[length - 1] != 0 {
            break;
        }
        length -= 1;
    }
    // Mark runs that already code well with RLE.
    {
        let mut symbol = counts[0];
        let mut stride = 0usize;
        for i in 0..length + 1 {
            if i == length || counts[i] != symbol {
                if (symbol == 0 && stride >= 5) || (symbol != 0 && stride >= 7) {
                    for k in 0..stride {
                        good_for_rle[i - k - 1] = 1;
                    }
                }
                stride = 1;
                if i != length {
                    symbol = counts[i];
                }
            } else {
                stride += 1;
            }
        }
    }
    // Replace counts that lead to more RLE codes.
    {
        let mut stride = 0u32;
        let mut limit = counts[0];
        let mut sum = 0u32;
        for i in 0..length + 1 {
            if i == length
                || good_for_rle[i] != 0
                || (i != 0 && good_for_rle[i - 1] != 0)
                || !values_should_be_collapsed_to_stride_average(counts[i] as i32, limit as i32)
            {
                if stride >= 4 || (stride >= 3 && sum == 0) {
                    let mut count = (sum + stride / 2) / stride;
                    if count < 1 {
                        count = 1;
                    }
                    if sum == 0 {
                        count = 0;
                    }
                    for k in 0..stride as usize {
                        counts[i - k - 1] = count;
                    }
                }
                stride = 0;
                sum = 0;
                if i + 3 < length {
                    limit = (counts[i] + counts[i + 1] + counts[i + 2] + counts[i + 3] + 2) / 4;
                } else if i < length {
                    limit = counts[i];
                } else {
                    limit = 0;
                }
            }
            stride += 1;
            if i != length {
                sum += counts[i];
                if stride >= 4 {
                    limit = (sum + stride / 2) / stride;
                }
            }
        }
    }
}

/// CompareHuffmanTrees: count descending, value ascending.
fn compare_huffman_trees(a: &Tree, b: &Tree) -> Ordering {
    match b.total_count.cmp(&a.total_count) {
        Ordering::Equal => a.value.cmp(&b.value),
        o => o,
    }
}

/// SetBitDepths, huffman_encode_utils.c:135.
fn set_bit_depths(t: &Tree, pool: &[Tree], depths: &mut [u8], level: u8) {
    if t.left >= 0 {
        set_bit_depths(&pool[t.left as usize], pool, depths, level + 1);
        set_bit_depths(&pool[t.right as usize], pool, depths, level + 1);
    } else {
        depths[t.value as usize] = level;
    }
}

/// GenerateOptimalTree: `tree` has room for 3 * histogram.len() nodes.
fn generate_optimal_tree(histogram: &[u32], tree: &mut [Tree], limit: usize, depths: &mut [u8]) {
    let size_orig = histogram.iter().filter(|&&v| v != 0).count();
    if size_orig == 0 {
        return;
    }
    let (tree, pool) = tree.split_at_mut(size_orig);
    let mut count_min = 1u32;
    loop {
        let mut size = size_orig;
        let mut idx = 0;
        for (j, &v) in histogram.iter().enumerate() {
            if v != 0 {
                tree[idx] = Tree {
                    total_count: v.max(count_min),
                    value: j as i32,
                    left: -1,
                    right: -1,
                };
                idx += 1;
            }
        }
        // Values are distinct, so the order is total and qsort-equivalent.
        tree[..size].sort_by(compare_huffman_trees);

        if size > 1 {
            // The tree and the pool are one array in C; pool nodes start after `size_orig` slots, so
            // the first `size_orig` nodes of `tree` and the pool never overlap.
            let mut pool_size = 0;
            while size > 1 {
                pool[pool_size] = tree[size - 1];
                pool_size += 1;
                pool[pool_size] = tree[size - 2];
                pool_size += 1;
                let count = pool[pool_size - 1].total_count + pool[pool_size - 2].total_count;
                size -= 2;
                let mut k = 0;
                while k < size {
                    if tree[k].total_count <= count {
                        break;
                    }
                    k += 1;
                }
                tree.copy_within(k..size, k + 1);
                tree[k] = Tree {
                    total_count: count,
                    value: -1,
                    left: pool_size as i32 - 1,
                    right: pool_size as i32 - 2,
                };
                size += 1;
            }
            set_bit_depths(&tree[0], pool, depths, 0);
        } else if size == 1 {
            depths[tree[0].value as usize] = 1;
        }

        let max_depth = depths.iter().copied().max().unwrap_or(0) as usize;
        if max_depth <= limit {
            break;
        }
        count_min *= 2;
    }
}

/// CodeRepeatedValues, huffman_encode_utils.c:261.
fn code_repeated_values(tokens: &mut Vec<Token>, mut reps: usize, value: usize, prev: usize) {
    if value != prev {
        tokens.push(Token {
            code: value as u8,
            extra_bits: 0,
        });
        reps -= 1;
    }
    while reps >= 1 {
        if reps < 3 {
            for _ in 0..reps {
                tokens.push(Token {
                    code: value as u8,
                    extra_bits: 0,
                });
            }
            break;
        } else if reps < 7 {
            tokens.push(Token {
                code: 16,
                extra_bits: (reps - 3) as u8,
            });
            break;
        } else {
            tokens.push(Token {
                code: 16,
                extra_bits: 3,
            });
            reps -= 6;
        }
    }
}

/// CodeRepeatedZeros, huffman_encode_utils.c:295.
fn code_repeated_zeros(tokens: &mut Vec<Token>, mut reps: usize) {
    while reps >= 1 {
        if reps < 3 {
            for _ in 0..reps {
                tokens.push(Token::default());
            }
            break;
        } else if reps < 11 {
            tokens.push(Token {
                code: 17,
                extra_bits: (reps - 3) as u8,
            });
            break;
        } else if reps < 139 {
            tokens.push(Token {
                code: 18,
                extra_bits: (reps - 11) as u8,
            });
            break;
        } else {
            tokens.push(Token {
                code: 18,
                extra_bits: 0x7f,
            });
            reps -= 138;
        }
    }
}

/// VP8LCreateCompressedHuffmanTree, huffman_encode_utils.c:326: the tokens of the code lengths.
pub(super) fn create_compressed_huffman_tree(code_lengths: &[u8], tokens: &mut Vec<Token>) {
    tokens.clear();
    let n = code_lengths.len();
    let mut prev = 8; // initial RLE value
    let mut i = 0;
    while i < n {
        let value = usize::from(code_lengths[i]);
        let mut k = i + 1;
        while k < n && usize::from(code_lengths[k]) == value {
            k += 1;
        }
        let runs = k - i;
        if value == 0 {
            code_repeated_zeros(tokens, runs);
        } else {
            code_repeated_values(tokens, runs, value, prev);
            prev = value;
        }
        i += runs;
    }
}

/// kReversedBits, huffman_encode_utils.c:356.
const REVERSED_BITS: [u8; 16] = [
    0x0, 0x8, 0x4, 0xc, 0x2, 0xa, 0x6, 0xe, 0x1, 0x9, 0x5, 0xd, 0x3, 0xb, 0x7, 0xf,
];

/// ReverseBits, huffman_encode_utils.c:361.
fn reverse_bits(n: usize, mut bits: u32) -> u32 {
    let mut r = 0u32;
    let mut i = 0;
    while i < n {
        i += 4;
        r |= u32::from(REVERSED_BITS[(bits & 0xf) as usize]) << (MAX_ALLOWED_CODE_LENGTH + 1 - i);
        bits >>= 4;
    }
    r >> (MAX_ALLOWED_CODE_LENGTH + 1 - n)
}

/// ConvertBitDepthsToSymbols, huffman_encode_utils.c:374.
fn convert_bit_depths_to_symbols(c: &mut Code) {
    let mut next_code = [0u32; MAX_ALLOWED_CODE_LENGTH + 1];
    let mut depth_count = [0u32; MAX_ALLOWED_CODE_LENGTH + 1];
    let n = c.num_symbols;
    for i in 0..n {
        depth_count[c.code_lengths[i] as usize] += 1;
    }
    depth_count[0] = 0; // unused symbols
    let mut code = 0u32;
    for i in 1..=MAX_ALLOWED_CODE_LENGTH {
        code = (code + depth_count[i - 1]) << 1;
        next_code[i] = code;
    }
    for i in 0..n {
        let l = c.code_lengths[i] as usize;
        c.codes[i] = reverse_bits(l, next_code[l]) as u16;
        next_code[l] += 1;
    }
}

/// VP8LCreateHuffmanTree: mutates `histogram` (RLE optimisation) and fills `c`. `buf_rle` and
/// `tree` are scratch of histogram.len() and 3 * histogram.len(). `c.code_lengths` must be zeroed,
/// as C's calloc'd buffer is.
pub(super) fn create_huffman_tree(
    histogram: &mut [u32],
    limit: usize,
    buf_rle: &mut [u8],
    tree: &mut [Tree],
    c: &mut Code,
) {
    let n = c.num_symbols;
    buf_rle[..n].fill(0);
    optimize_huffman_for_rle(buf_rle, &mut histogram[..n]);
    generate_optimal_tree(&histogram[..n], tree, limit, c.code_lengths);
    convert_bit_depths_to_symbols(c);
}
