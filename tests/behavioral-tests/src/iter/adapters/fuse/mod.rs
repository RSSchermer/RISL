use risl::gpu;

mod count;
mod find;
mod fold;
mod is_empty;
mod last;
mod len;
mod next;
mod next_back;
mod nth;
mod nth_back;
mod rfind;
mod rfold;
mod size_hint;
mod try_fold;
mod try_rfold;

const NONE: u32 = u32::MAX;

// Each slot is one call result, so `NONE` consumes a slot without ending iteration permanently.
// Both directions share the remaining slots; only an empty slice is permanently exhausted.
#[gpu]
struct NonFusedIter<'a> {
    values: &'a [u32],
}

#[gpu]
impl<'a> NonFusedIter<'a> {
    fn new(values: &'a [u32]) -> Self {
        Self { values }
    }
}

#[gpu]
impl Iterator for NonFusedIter<'_> {
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        if let [first, rest @ ..] = self.values {
            self.values = rest;

            if *first == NONE { None } else { Some(*first) }
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        // Slots may contain `NONE`, so their count is only an upper bound on the item count.
        (0, Some(self.values.len()))
    }
}

#[gpu]
impl DoubleEndedIterator for NonFusedIter<'_> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if let [rest @ .., last] = self.values {
            self.values = rest;

            if *last == NONE { None } else { Some(*last) }
        } else {
            None
        }
    }
}
