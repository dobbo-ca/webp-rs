//! Go's `slices.SortFunc` (pattern-defeating quicksort, Go 1.27 zsortanyfunc.go), so an unstable
//! sort breaks ties as Go's does. Copyright 2009 The Go Authors; the licence is in NOTICE.

use std::cmp::Ordering;

#[derive(PartialEq, Clone, Copy)]
enum Hint {
    Unknown,
    Increasing,
    Decreasing,
}

/// `slices.SortFunc(data, cmp)`; `lt(a, b)` is `cmp(a, b) < 0`.
pub(crate) fn sort_func<T: Copy>(data: &mut [T], cmp: impl Fn(&T, &T) -> Ordering) {
    let lt = |a: &T, b: &T| cmp(a, b) == Ordering::Less;
    let n = data.len();
    pdqsort(data, 0, n, (usize::BITS - n.leading_zeros()) as usize, &lt);
}

fn insertion_sort<T: Copy>(data: &mut [T], a: usize, b: usize, lt: &impl Fn(&T, &T) -> bool) {
    for i in a + 1..b {
        let mut j = i;
        while j > a && lt(&data[j], &data[j - 1]) {
            data.swap(j, j - 1);
            j -= 1;
        }
    }
}

fn sift_down<T: Copy>(
    data: &mut [T],
    lo: usize,
    hi: usize,
    first: usize,
    lt: &impl Fn(&T, &T) -> bool,
) {
    let mut root = lo;
    loop {
        let mut child = 2 * root + 1;
        if child >= hi {
            break;
        }
        if child + 1 < hi && lt(&data[first + child], &data[first + child + 1]) {
            child += 1;
        }
        if !lt(&data[first + root], &data[first + child]) {
            return;
        }
        data.swap(first + root, first + child);
        root = child;
    }
}

fn heap_sort<T: Copy>(data: &mut [T], a: usize, b: usize, lt: &impl Fn(&T, &T) -> bool) {
    let first = a;
    let hi = b - a;
    for i in (0..=(hi - 1) / 2).rev() {
        sift_down(data, i, hi, first, lt);
    }
    for i in (0..hi).rev() {
        data.swap(first, first + i);
        sift_down(data, 0, i, first, lt);
    }
}

fn pdqsort<T: Copy>(
    data: &mut [T],
    mut a: usize,
    mut b: usize,
    mut limit: usize,
    lt: &impl Fn(&T, &T) -> bool,
) {
    const MAX_INSERTION: usize = 12;
    let (mut was_balanced, mut was_partitioned) = (true, true);
    loop {
        let length = b - a;
        if length <= MAX_INSERTION {
            insertion_sort(data, a, b, lt);
            return;
        }
        if limit == 0 {
            heap_sort(data, a, b, lt);
            return;
        }
        if !was_balanced {
            break_patterns(data, a, b);
            limit -= 1;
        }
        let (mut pivot, mut hint) = choose_pivot(data, a, b, lt);
        if hint == Hint::Decreasing {
            reverse_range(data, a, b);
            pivot = (b - 1) - (pivot - a);
            hint = Hint::Increasing;
        }
        if was_balanced
            && was_partitioned
            && hint == Hint::Increasing
            && partial_insertion_sort(data, a, b, lt)
        {
            return;
        }
        if a > 0 && !lt(&data[a - 1], &data[pivot]) {
            a = partition_equal(data, a, b, pivot, lt);
            continue;
        }
        let (mid, already) = partition(data, a, b, pivot, lt);
        was_partitioned = already;
        let (left_len, right_len) = (mid - a, b - mid);
        let balance_threshold = length / 8;
        if left_len < right_len {
            was_balanced = left_len >= balance_threshold;
            pdqsort(data, a, mid, limit, lt);
            a = mid + 1;
        } else {
            was_balanced = right_len >= balance_threshold;
            pdqsort(data, mid + 1, b, limit, lt);
            b = mid;
        }
    }
}

fn partition<T: Copy>(
    data: &mut [T],
    a: usize,
    b: usize,
    pivot: usize,
    lt: &impl Fn(&T, &T) -> bool,
) -> (usize, bool) {
    data.swap(a, pivot);
    // i and j are inclusive of the elements remaining to be partitioned; j can pass below i, so
    // they are signed.
    let (mut i, mut j) = (a as isize + 1, b as isize - 1);
    let at = |k: isize| k as usize;
    while i <= j && lt(&data[at(i)], &data[a]) {
        i += 1;
    }
    while i <= j && !lt(&data[at(j)], &data[a]) {
        j -= 1;
    }
    if i > j {
        data.swap(at(j), a);
        return (at(j), true);
    }
    data.swap(at(i), at(j));
    i += 1;
    j -= 1;
    loop {
        while i <= j && lt(&data[at(i)], &data[a]) {
            i += 1;
        }
        while i <= j && !lt(&data[at(j)], &data[a]) {
            j -= 1;
        }
        if i > j {
            break;
        }
        data.swap(at(i), at(j));
        i += 1;
        j -= 1;
    }
    data.swap(at(j), a);
    (at(j), false)
}

fn partition_equal<T: Copy>(
    data: &mut [T],
    a: usize,
    b: usize,
    pivot: usize,
    lt: &impl Fn(&T, &T) -> bool,
) -> usize {
    data.swap(a, pivot);
    let (mut i, mut j) = (a as isize + 1, b as isize - 1);
    let at = |k: isize| k as usize;
    loop {
        while i <= j && !lt(&data[a], &data[at(i)]) {
            i += 1;
        }
        while i <= j && lt(&data[a], &data[at(j)]) {
            j -= 1;
        }
        if i > j {
            break;
        }
        data.swap(at(i), at(j));
        i += 1;
        j -= 1;
    }
    at(i)
}

fn partial_insertion_sort<T: Copy>(
    data: &mut [T],
    a: usize,
    b: usize,
    lt: &impl Fn(&T, &T) -> bool,
) -> bool {
    const MAX_STEPS: usize = 5;
    const SHORTEST_SHIFTING: usize = 50;
    let mut i = a + 1;
    for _ in 0..MAX_STEPS {
        while i < b && !lt(&data[i], &data[i - 1]) {
            i += 1;
        }
        if i == b {
            return true;
        }
        if b - a < SHORTEST_SHIFTING {
            return false;
        }
        data.swap(i, i - 1);
        if i - a >= 2 {
            for j in (1..i).rev() {
                if !lt(&data[j], &data[j - 1]) {
                    break;
                }
                data.swap(j, j - 1);
            }
        }
        if b - i >= 2 {
            for j in i + 1..b {
                if !lt(&data[j], &data[j - 1]) {
                    break;
                }
                data.swap(j, j - 1);
            }
        }
    }
    false
}

fn break_patterns<T: Copy>(data: &mut [T], a: usize, b: usize) {
    let length = b - a;
    if length >= 8 {
        let mut random = length as u64;
        let modulus = 1usize << (usize::BITS - length.leading_zeros());
        for idx in a + (length / 4) * 2 - 1..=a + (length / 4) * 2 + 1 {
            random ^= random << 13;
            random ^= random >> 7;
            random ^= random << 17;
            let mut other = (random as usize) & (modulus - 1);
            if other >= length {
                other -= length;
            }
            data.swap(idx, a + other);
        }
    }
}

fn choose_pivot<T: Copy>(
    data: &[T],
    a: usize,
    b: usize,
    lt: &impl Fn(&T, &T) -> bool,
) -> (usize, Hint) {
    const SHORTEST_NINTHER: usize = 50;
    const MAX_SWAPS: usize = 4 * 3;
    let l = b - a;
    let mut swaps = 0;
    let (mut i, mut j, mut k) = (a + l / 4, a + l / 4 * 2, a + l / 4 * 3);
    if l >= 8 {
        if l >= SHORTEST_NINTHER {
            i = median(data, i - 1, i, i + 1, &mut swaps, lt);
            j = median(data, j - 1, j, j + 1, &mut swaps, lt);
            k = median(data, k - 1, k, k + 1, &mut swaps, lt);
        }
        j = median(data, i, j, k, &mut swaps, lt);
    }
    match swaps {
        0 => (j, Hint::Increasing),
        MAX_SWAPS => (j, Hint::Decreasing),
        _ => (j, Hint::Unknown),
    }
}

fn order2<T: Copy>(
    data: &[T],
    a: usize,
    b: usize,
    swaps: &mut usize,
    lt: &impl Fn(&T, &T) -> bool,
) -> (usize, usize) {
    if lt(&data[b], &data[a]) {
        *swaps += 1;
        return (b, a);
    }
    (a, b)
}

fn median<T: Copy>(
    data: &[T],
    a: usize,
    b: usize,
    c: usize,
    swaps: &mut usize,
    lt: &impl Fn(&T, &T) -> bool,
) -> usize {
    let (a, b) = order2(data, a, b, swaps, lt);
    let (b, _) = order2(data, b, c, swaps, lt);
    let (_, b) = order2(data, a, b, swaps, lt);
    b
}

fn reverse_range<T: Copy>(data: &mut [T], a: usize, b: usize) {
    data[a..b].reverse();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sorts() {
        for n in [0usize, 1, 5, 12, 13, 40, 49, 50, 300, 1000] {
            let mut v: Vec<(u32, u32)> = (0..n as u32).map(|i| (i * 7919 % 13, i)).collect();
            sort_func(&mut v, |a, b| a.0.cmp(&b.0));
            assert!(v.windows(2).all(|w| w[0].0 <= w[1].0), "n {n}");
            let mut ids: Vec<u32> = v.iter().map(|x| x.1).collect();
            ids.sort_unstable();
            assert!(ids.iter().enumerate().all(|(i, &x)| i as u32 == x), "n {n}");
        }
    }
}
