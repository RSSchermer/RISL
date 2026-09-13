use core::ops::RangeInclusive;

use super::{AbstractU32, MAX_INTEGER_INTERVALS};

const TOP_INTERVALS: &[RangeInclusive<i32>] = &[i32::MIN..=i32::MAX];

/// A (possibly constrained) signed 32-bit integer value.
///
/// See also [AbstractValue](super::AbstractValue).
///
/// The value is represented by a set of non-overlapping, non-adjacent inclusive intervals in
/// increasing order. An empty set is the "bottom" value and an interval spanning the full `i32`
/// domain is the "top" value. At most [`MAX_INTEGER_INTERVALS`] intervals are retained.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct AbstractI32(Vec<RangeInclusive<i32>>);

impl AbstractI32 {
    /// Returns a new abstract value constrained to values in the given `intervals`.
    ///
    /// Intervals are sorted and overlapping or adjacent intervals are merged. If more than
    /// [`MAX_INTEGER_INTERVALS`] disjoint intervals remain, the result is conservatively widened to
    /// "top".
    ///
    /// # Panics
    ///
    /// Panics if an interval is reversed (`start > end`). If the intent is to represent a wrapping
    /// interval, use two separate intervals instead (e.g., `0..=N` and `M..=i32::MAX` instead of
    /// `M..=N` where `M > N`).
    pub fn from_intervals(intervals: impl IntoIterator<Item = RangeInclusive<i32>>) -> Self {
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
    pub fn from_constant(value: i32) -> Self {
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

    /// Returns the intervals that constrain this value.
    ///
    /// Intervals are non-empty, non-overlapping, non-adjacent, and sorted in ascending order.
    /// Returns at most [MAX_INTEGER_INTERVALS] intervals. "Bottom" returns an empty slice and "top"
    /// returns one interval spanning the entire `i32` domain.
    pub fn intervals(&self) -> &[RangeInclusive<i32>] {
        &self.0
    }

    /// Returns the constrained value if this abstract value represents exactly one value, `None`
    /// otherwise.
    pub fn to_singleton(&self) -> Option<i32> {
        match self.0.as_slice() {
            [interval] if interval.start() == interval.end() => Some(*interval.start()),
            _ => None,
        }
    }

    /// Returns an unsigned abstract value representing the same 32-bit patterns.
    ///
    /// Intervals spanning zero are split to preserve their bit patterns across the change in
    /// signedness.
    pub fn to_abstract_u32(&self) -> AbstractU32 {
        AbstractU32::from_intervals(self.0.iter().flat_map(|interval| {
            let start = *interval.start();
            let end = *interval.end();

            if start < 0 && end >= 0 {
                vec![0..=end as u32, start as u32..=u32::MAX]
            } else {
                vec![start as u32..=end as u32]
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
    pub fn contains(&self, value: i32) -> bool {
        self.0.iter().any(|interval| interval.contains(&value))
    }

    /// Returns a new abstract value representing at least the union of `self` and `other`.
    ///
    /// A union requiring more than [`MAX_INTEGER_INTERVALS`] intervals conservatively widens to
    /// "top".
    ///
    /// See also [AbstractValue::join](super::AbstractValue::join).
    pub fn join(&self, other: &Self) -> Self {
        Self::from_intervals(self.0.iter().cloned().chain(other.0.iter().cloned()))
    }

    /// Returns a new abstract value approximating the intersection of `self` and `other`.
    ///
    /// If the exact intersection would require more than [`MAX_INTEGER_INTERVALS`] intervals,
    /// `self` is retained as a conservative approximation. This operation is therefore not
    /// generally commutative.
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

    /// Returns a new abstract value excluding the signed values represented by the 32-bit encodings
    /// in `cases`.
    ///
    /// Encodings outside the 32-bit domain are ignored. If exact exclusion would require more than
    /// [`MAX_INTEGER_INTERVALS`] intervals, `self` is retained.
    pub fn exclude_cases(&self, cases: &[u128]) -> Self {
        let mut cases: Vec<_> = cases
            .iter()
            .filter_map(|&value| u32::try_from(value).ok())
            .map(|value| value as i32)
            .collect();
        cases.sort_unstable();
        cases.dedup();

        let mut remaining = Vec::new();

        for interval in &self.0 {
            let start = *interval.start();
            let end = *interval.end();

            let mut cursor = i64::from(start);

            for &case in &cases {
                if case < start || case > end {
                    continue;
                }

                if cursor < i64::from(case) {
                    remaining.push(cursor as i32..=case - 1);
                    if remaining.len() > MAX_INTEGER_INTERVALS {
                        return self.clone();
                    }
                }

                cursor = i64::from(case) + 1;
            }

            if cursor <= i64::from(end) {
                remaining.push(cursor as i32..=end);

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
            AbstractI32::from_intervals([8..=9, -4..=-2, -1..=7]),
            AbstractI32(vec![-4..=9])
        );
        assert_eq!(AbstractI32::from_intervals([]), AbstractI32::bottom());
        assert_eq!(
            AbstractI32::from_intervals([i32::MIN..=i32::MAX]),
            AbstractI32::top()
        );
        assert_eq!(
            AbstractI32::from_intervals([-8..=-8, -6..=-6, -4..=-4, -2..=-2, 0..=0]),
            AbstractI32::top()
        );
    }

    #[test]
    fn intervals() {
        let value = AbstractI32::from_intervals([2..=3, -2..=0, 1..=1]);

        assert_eq!(value.intervals(), [-2..=3]);
    }

    #[test]
    fn to_singleton() {
        assert_eq!(AbstractI32::from_constant(-4).to_singleton(), Some(-4));
        assert_eq!(AbstractI32::from_intervals([-4..=-3]).to_singleton(), None);
        assert_eq!(AbstractI32::bottom().to_singleton(), None);
    }

    #[test]
    fn to_u32() {
        assert_eq!(
            AbstractI32::from_intervals([-1..=1]).to_abstract_u32(),
            AbstractU32::from_intervals([0..=1, u32::MAX..=u32::MAX])
        );
        assert_eq!(AbstractI32::top().to_abstract_u32(), AbstractU32::top());
        assert_eq!(
            AbstractI32::bottom().to_abstract_u32(),
            AbstractU32::bottom()
        );
    }

    #[test]
    fn is_top() {
        assert!(AbstractI32::top().is_top());
        assert!(!AbstractI32::from_constant(i32::MIN).is_top());
        assert!(!AbstractI32::bottom().is_top());
    }

    #[test]
    fn is_bottom() {
        assert!(AbstractI32::bottom().is_bottom());
        assert!(!AbstractI32::from_constant(0).is_bottom());
        assert!(!AbstractI32::top().is_bottom());
    }

    #[test]
    fn contains() {
        let value = AbstractI32::from_intervals([-10..=-8, 1..=3]);

        assert!(value.contains(-9));
        assert!(value.contains(3));
        assert!(!value.contains(0));
        assert!(AbstractI32::top().contains(0));
        assert!(!AbstractI32::bottom().contains(0));
    }

    #[test]
    fn join() {
        let left = AbstractI32::from_intervals([-5..=-2, 3..=4]);
        let right = AbstractI32::from_intervals([-1..=2]);

        assert_eq!(left.join(&right), AbstractI32::from_intervals([-5..=4]));
        assert_eq!(left.join(&AbstractI32::bottom()), left);
        assert_eq!(AbstractI32::bottom().join(&left), left);
        assert_eq!(left.join(&AbstractI32::top()), AbstractI32::top());
        assert_eq!(AbstractI32::top().join(&left), AbstractI32::top());
    }

    #[test]
    fn refine() {
        let left = AbstractI32::from_intervals([-10..=-5, 0..=10]);
        let right = AbstractI32::from_intervals([-7..=3]);

        assert_eq!(
            left.refine(&right),
            AbstractI32::from_intervals([-7..=-5, 0..=3])
        );
        assert_eq!(left.refine(&AbstractI32::top()), left);
        assert_eq!(AbstractI32::top().refine(&left), left);
        assert_eq!(left.refine(&AbstractI32::bottom()), AbstractI32::bottom());
        assert_eq!(AbstractI32::bottom().refine(&left), AbstractI32::bottom());
    }

    #[test]
    fn is_disjoint() {
        let value = AbstractI32::from_intervals([-3..=-1]);

        assert!(value.is_disjoint(&AbstractI32::from_constant(0)));
        assert!(!value.is_disjoint(&AbstractI32::from_constant(-1)));
        assert!(!AbstractI32::top().is_disjoint(&value));
        assert!(AbstractI32::bottom().is_disjoint(&AbstractI32::top()));
    }

    #[test]
    fn is_subset() {
        let value = AbstractI32::from_intervals([-3..=-1, 2..=4]);

        assert!(AbstractI32::bottom().is_subset(&AbstractI32::bottom()));
        assert!(AbstractI32::bottom().is_subset(&value));
        assert!(value.is_subset(&value));
        assert!(value.is_subset(&AbstractI32::from_intervals([-5..=5])));
        assert!(!AbstractI32::from_intervals([-5..=5]).is_subset(&value));
        assert!(value.is_subset(&AbstractI32::top()));
        assert!(!AbstractI32::top().is_subset(&value));
    }

    #[test]
    fn exclude_cases() {
        let value = AbstractI32::from_intervals([-3..=3]);

        assert_eq!(
            value.exclude_cases(&[u128::from(u32::MAX), 0]),
            AbstractI32::from_intervals([-3..=-2, 1..=3])
        );
        assert_eq!(
            AbstractI32::from_constant(-1).exclude_cases(&[u128::from(u32::MAX)]),
            AbstractI32::bottom()
        );

        let wide_value = AbstractI32::from_intervals([-10..=10]);
        let cases = [-7i32, -3, 1, 5].map(|value| u128::from(value as u32));

        assert_eq!(wide_value.exclude_cases(&cases), wide_value);
    }
}
