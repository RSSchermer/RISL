use core::ops::RangeInclusive;

use super::{AbstractI32, MAX_INTEGER_INTERVALS};

const TOP_INTERVALS: &[RangeInclusive<u32>] = &[0..=u32::MAX];

/// A (possibly constrained) unsigned 32-bit integer value.
///
/// See also [AbstractValue](super::AbstractValue).
///
/// The value is represented by a set of non-overlapping, non-adjacent inclusive intervals in
/// increasing order. An empty set is the "bottom" value and an interval spanning the full `u32`
/// domain is the "top" value. At most [`MAX_INTEGER_INTERVALS`] intervals are retained.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct AbstractU32(Vec<RangeInclusive<u32>>);

impl AbstractU32 {
    /// Returns a new abstract value constrained to values in the given `intervals`.
    ///
    /// Intervals are sorted and overlapping or adjacent intervals are merged. If more than
    /// [`MAX_INTEGER_INTERVALS`] disjoint intervals remain, the result is widened to "top".
    ///
    /// # Panics
    ///
    /// Panics if an interval is reversed.
    pub fn from_intervals(intervals: impl IntoIterator<Item = RangeInclusive<u32>>) -> Self {
        let mut intervals: Vec<_> = intervals.into_iter().collect();

        assert!(
            intervals.iter().all(|interval| !interval.is_empty()),
            "integer intervals must not be reversed"
        );

        intervals.sort_unstable_by_key(|interval| (*interval.start(), *interval.end()));
        let mut normalized_len = 0;

        for index in 0..intervals.len() {
            let start = *intervals[index].start();
            let end = *intervals[index].end();

            if normalized_len > 0 {
                let previous = &mut intervals[normalized_len - 1];

                if start <= previous.end().saturating_add(1) {
                    let previous_start = *previous.start();
                    *previous = previous_start..=(*previous.end()).max(end);
                    continue;
                }
            }

            intervals.swap(normalized_len, index);
            normalized_len += 1;
        }

        intervals.truncate(normalized_len);

        if intervals.len() > MAX_INTEGER_INTERVALS {
            Self::top()
        } else {
            intervals.shrink_to_fit();
            Self(intervals)
        }
    }

    /// Returns a new abstract value constrained to exactly the given `value`.
    pub fn from_constant(value: u32) -> Self {
        Self(vec![value..=value])
    }

    /// Returns a new unconstrained "top" value.
    pub fn top() -> Self {
        Self(TOP_INTERVALS.to_vec())
    }

    /// Returns a new impossible "bottom" value.
    pub fn bottom() -> Self {
        Self(Vec::new())
    }

    /// Returns the canonical intervals represented by this value.
    ///
    /// "Bottom" returns an empty slice and "top" returns one interval spanning the entire `u32`
    /// domain.
    pub fn intervals(&self) -> &[RangeInclusive<u32>] {
        &self.0
    }

    /// Returns the constrained value if this abstract value represents exactly one value, `None`
    /// otherwise.
    pub fn to_singleton(&self) -> Option<u32> {
        match self.0.as_slice() {
            [interval] if interval.start() == interval.end() => Some(*interval.start()),
            _ => None,
        }
    }

    /// Returns a signed abstract value representing the same 32-bit patterns.
    ///
    /// Intervals spanning [`i32::MAX`] are split to preserve their bit patterns across the change
    /// in signedness.
    pub fn to_abstract_i32(&self) -> AbstractI32 {
        AbstractI32::from_intervals(self.0.iter().flat_map(|interval| {
            let start = *interval.start();
            let end = *interval.end();

            if end <= i32::MAX as u32 || start > i32::MAX as u32 {
                vec![start as i32..=end as i32]
            } else {
                vec![i32::MIN..=end as i32, start as i32..=i32::MAX]
            }
        }))
    }

    /// Returns whether the value is an unconstrained "top" value.
    ///
    /// See also [AbstractValue::is_top](super::AbstractValue::is_top).
    pub fn is_top(&self) -> bool {
        self.0.as_slice() == TOP_INTERVALS
    }

    /// Returns whether the value is an impossible "bottom" value.
    ///
    /// See also [AbstractValue::is_bottom](super::AbstractValue::is_bottom).
    pub fn is_bottom(&self) -> bool {
        self.0.is_empty()
    }

    /// Returns `true` if this abstract value's constraints are compatible with `value`, `false`
    /// otherwise.
    ///
    /// See also [AbstractValue::contains](super::AbstractValue::contains).
    pub fn contains(&self, value: u32) -> bool {
        self.0.iter().any(|interval| interval.contains(&value))
    }

    /// Returns a new abstract value representing at least the union of `self` and `other`.
    ///
    /// A union requiring more than [`MAX_INTEGER_INTERVALS`] intervals widens to "top".
    ///
    /// See also [AbstractValue::join](super::AbstractValue::join).
    pub fn join(&self, other: &Self) -> Self {
        Self::from_intervals(self.0.iter().cloned().chain(other.0.iter().cloned()))
    }

    /// Returns a new abstract value approximating the intersection of `self` and `other`.
    ///
    /// If the exact intersection would require more than [`MAX_INTEGER_INTERVALS`] intervals,
    /// `self` is retained as a sound approximation. This operation is therefore not generally
    /// commutative.
    ///
    /// See also [AbstractValue::refine](super::AbstractValue::refine).
    pub fn refine(&self, other: &Self) -> Self {
        if self.is_bottom() || other.is_top() {
            return self.clone();
        }
        if other.is_bottom() || self.is_top() {
            return other.clone();
        }

        let intervals: Vec<_> = self
            .0
            .iter()
            .flat_map(|left| {
                other.0.iter().filter_map(move |right| {
                    let start = (*left.start()).max(*right.start());
                    let end = (*left.end()).min(*right.end());
                    (start <= end).then_some(start..=end)
                })
            })
            .collect();

        if intervals.len() > MAX_INTEGER_INTERVALS {
            self.clone()
        } else {
            Self::from_intervals(intervals)
        }
    }

    /// Returns `true` if there is no overlap between value sets representable by both operands,
    /// `false` otherwise.
    ///
    /// See also [AbstractValue::is_disjoint](super::AbstractValue::is_disjoint).
    pub fn is_disjoint(&self, other: &Self) -> bool {
        !self.0.iter().any(|left| {
            other
                .0
                .iter()
                .any(|right| left.start() <= right.end() && right.start() <= left.end())
        })
    }

    /// Returns `true` if the value set representable by `self` is a subset of the value set
    /// representable by `other`, `false` otherwise.
    ///
    /// See also [AbstractValue::is_subset](super::AbstractValue::is_subset).
    pub fn is_subset(&self, other: &Self) -> bool {
        self.0.iter().all(|left| {
            other
                .0
                .iter()
                .any(|right| right.start() <= left.start() && left.end() <= right.end())
        })
    }

    /// Returns a new abstract value excluding the values represented by the 32-bit encodings in
    /// `cases`.
    ///
    /// Encodings outside the 32-bit domain are ignored. If exact exclusion would require more than
    /// [`MAX_INTEGER_INTERVALS`] intervals, `self` is retained.
    pub fn exclude_cases(&self, cases: &[u128]) -> Self {
        let mut cases: Vec<_> = cases
            .iter()
            .filter_map(|&value| u32::try_from(value).ok())
            .collect();
        cases.sort_unstable();
        cases.dedup();

        let mut remaining = Vec::new();

        for interval in &self.0 {
            let start = *interval.start();
            let end = *interval.end();

            let mut cursor = u64::from(start);

            for &case in &cases {
                if case < start || case > end {
                    continue;
                }

                if cursor < u64::from(case) {
                    remaining.push(cursor as u32..=case - 1);

                    if remaining.len() > MAX_INTEGER_INTERVALS {
                        return self.clone();
                    }
                }

                cursor = u64::from(case) + 1;
            }

            if cursor <= u64::from(end) {
                remaining.push(cursor as u32..=end);

                if remaining.len() > MAX_INTEGER_INTERVALS {
                    return self.clone();
                }
            }
        }

        Self::from_intervals(remaining)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_intervals() {
        assert_eq!(
            AbstractU32::from_intervals([8..=9, 2..=4, 3..=7, 0..=1]),
            AbstractU32(vec![0..=9])
        );
        assert_eq!(AbstractU32::from_intervals([]), AbstractU32::bottom());
        assert_eq!(
            AbstractU32::from_intervals([0..=u32::MAX]),
            AbstractU32::top()
        );
        assert_eq!(
            AbstractU32::from_intervals([0..=0, 2..=2, 4..=4, 6..=6, 8..=8]),
            AbstractU32::top()
        );
    }

    #[test]
    fn intervals() {
        let value = AbstractU32::from_intervals([4..=5, 0..=2, 3..=3]);

        assert_eq!(value.intervals(), [0..=5]);
    }

    #[test]
    fn to_singleton() {
        assert_eq!(AbstractU32::from_constant(4).to_singleton(), Some(4));
        assert_eq!(AbstractU32::from_intervals([4..=5]).to_singleton(), None);
        assert_eq!(AbstractU32::bottom().to_singleton(), None);
    }

    #[test]
    fn to_i32() {
        assert_eq!(
            AbstractU32::from_intervals([0..=1, u32::MAX..=u32::MAX]).to_abstract_i32(),
            AbstractI32::from_intervals([-1..=1])
        );
        assert_eq!(AbstractU32::top().to_abstract_i32(), AbstractI32::top());
        assert_eq!(
            AbstractU32::bottom().to_abstract_i32(),
            AbstractI32::bottom()
        );
    }

    #[test]
    fn is_top() {
        assert!(AbstractU32::top().is_top());
        assert!(!AbstractU32::from_constant(u32::MAX).is_top());
        assert!(!AbstractU32::bottom().is_top());
    }

    #[test]
    fn is_bottom() {
        assert!(AbstractU32::bottom().is_bottom());
        assert!(!AbstractU32::from_constant(0).is_bottom());
        assert!(!AbstractU32::top().is_bottom());
    }

    #[test]
    fn contains() {
        let value = AbstractU32::from_intervals([1..=3, 8..=10]);

        assert!(value.contains(2));
        assert!(value.contains(10));
        assert!(!value.contains(4));
        assert!(AbstractU32::top().contains(0));
        assert!(!AbstractU32::bottom().contains(0));
    }

    #[test]
    fn join() {
        let left = AbstractU32::from_intervals([0..=2, 8..=9]);
        let right = AbstractU32::from_intervals([3..=7]);

        assert_eq!(left.join(&right), AbstractU32::from_intervals([0..=9]));
        assert_eq!(left.join(&AbstractU32::bottom()), left);
        assert_eq!(AbstractU32::bottom().join(&left), left);
        assert_eq!(left.join(&AbstractU32::top()), AbstractU32::top());
        assert_eq!(AbstractU32::top().join(&left), AbstractU32::top());
    }

    #[test]
    fn refine() {
        let left = AbstractU32::from_intervals([0..=5, 10..=15]);
        let right = AbstractU32::from_intervals([3..=12]);

        assert_eq!(
            left.refine(&right),
            AbstractU32::from_intervals([3..=5, 10..=12])
        );
        assert_eq!(left.refine(&AbstractU32::top()), left);
        assert_eq!(AbstractU32::top().refine(&left), left);
        assert_eq!(left.refine(&AbstractU32::bottom()), AbstractU32::bottom());
        assert_eq!(AbstractU32::bottom().refine(&left), AbstractU32::bottom());
    }

    #[test]
    fn is_disjoint() {
        let value = AbstractU32::from_intervals([1..=3]);

        assert!(value.is_disjoint(&AbstractU32::from_constant(4)));
        assert!(!value.is_disjoint(&AbstractU32::from_constant(3)));
        assert!(!AbstractU32::top().is_disjoint(&value));
        assert!(AbstractU32::bottom().is_disjoint(&AbstractU32::top()));
    }

    #[test]
    fn is_subset() {
        let value = AbstractU32::from_intervals([1..=3, 8..=9]);

        assert!(AbstractU32::bottom().is_subset(&AbstractU32::bottom()));
        assert!(AbstractU32::bottom().is_subset(&value));
        assert!(value.is_subset(&value));
        assert!(value.is_subset(&AbstractU32::from_intervals([0..=10])));
        assert!(!AbstractU32::from_intervals([0..=10]).is_subset(&value));
        assert!(value.is_subset(&AbstractU32::top()));
        assert!(!AbstractU32::top().is_subset(&value));
    }

    #[test]
    fn exclude_cases() {
        let value = AbstractU32::from_intervals([0..=10]);

        assert_eq!(
            value.exclude_cases(&[0, 2, 10]),
            AbstractU32::from_intervals([1..=1, 3..=9])
        );
        assert_eq!(
            AbstractU32::from_constant(4).exclude_cases(&[4]),
            AbstractU32::bottom()
        );

        let wide_value = AbstractU32::from_intervals([0..=10]);

        assert_eq!(wide_value.exclude_cases(&[1, 3, 5, 7]), wide_value);
    }
}
