use core::ops::RangeInclusive;

use smallvec::SmallVec;

use super::{AbstractBool, AbstractF32, AbstractU32, MAX_INTEGER_INTERVALS};

const TOP_INTERVALS: &[RangeInclusive<i32>] = &[i32::MIN..=i32::MAX];
const I32_MODULUS: i64 = 1i64 << 32;

/// Wraps an `i64` interval into the `i32` domain.
fn wrapping_intervals(start: i64, end: i64) -> SmallVec<[RangeInclusive<i32>; 2]> {
    let span = end - start;

    if span >= I32_MODULUS - 1 {
        return smallvec::smallvec![i32::MIN..=i32::MAX];
    }

    let start =
        ((start - i64::from(i32::MIN)).rem_euclid(I32_MODULUS) + i64::from(i32::MIN)) as i32;
    let end = i64::from(start) + span;

    if end <= i64::from(i32::MAX) {
        smallvec::smallvec![start..=end as i32]
    } else {
        smallvec::smallvec![start..=i32::MAX, i32::MIN..=(end - I32_MODULUS) as i32]
    }
}

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

    /// Returns the abstract floating-point value of this signed integer.
    pub fn to_abstract_f32(&self) -> AbstractF32 {
        if self.is_bottom() {
            AbstractF32::Bottom
        } else {
            self.to_singleton()
                .map(|value| AbstractF32::from_constant(value as f32))
                .unwrap_or(AbstractF32::Top)
        }
    }

    /// Returns the abstract boolean value of this integer.
    pub fn to_abstract_bool(&self) -> AbstractBool {
        if self.is_bottom() {
            AbstractBool::Bottom
        } else if self.to_singleton() == Some(0) {
            AbstractBool::Const(false)
        } else if self.contains(0) {
            AbstractBool::Top
        } else {
            AbstractBool::Const(true)
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

    /// Returns the abstract result of arithmetically negating this value.
    pub fn abstract_neg(&self) -> Self {
        Self::from_intervals(self.0.iter().flat_map(|interval| {
            wrapping_intervals(-i64::from(*interval.end()), -i64::from(*interval.start()))
        }))
    }

    /// Returns the operand constraints implied by `res` being the result of arithmetically
    /// negating this value.
    ///
    /// The constraints returned include the prior constraints on the operand, not just the
    /// additional constraints implied by the result.
    pub fn abstract_neg_inv(&self, res: &Self) -> Self {
        self.refine(&res.abstract_neg())
    }

    /// Returns the abstract result of adding `other` to this value.
    pub fn abstract_add(&self, other: &Self) -> Self {
        Self::from_intervals(self.0.iter().flat_map(|left| {
            other.0.iter().flat_map(|right| {
                wrapping_intervals(
                    i64::from(*left.start()) + i64::from(*right.start()),
                    i64::from(*left.end()) + i64::from(*right.end()),
                )
            })
        }))
    }

    /// Returns the operand constraints implied by `res` being the result of adding `other` to this
    /// value.
    ///
    /// The constraints returned include the prior constraints on the operands, not just the
    /// additional constraints implied by the result.
    ///
    /// Returns `("bottom", "bottom")` if the abstract inverse evaluation produced a contradiction.
    pub fn abstract_add_inv(&self, other: &Self, res: &Self) -> (Self, Self) {
        let refinements = (
            self.refine(&res.abstract_sub(other)),
            other.refine(&res.abstract_sub(self)),
        );

        if refinements.0.is_bottom() || refinements.1.is_bottom() {
            (Self::bottom(), Self::bottom())
        } else {
            refinements
        }
    }

    /// Returns the abstract result of subtracting `other` from this value.
    pub fn abstract_sub(&self, other: &Self) -> Self {
        Self::from_intervals(self.0.iter().flat_map(|left| {
            other.0.iter().flat_map(|right| {
                wrapping_intervals(
                    i64::from(*left.start()) - i64::from(*right.end()),
                    i64::from(*left.end()) - i64::from(*right.start()),
                )
            })
        }))
    }

    /// Returns the operand constraints implied by `res` being the result of subtracting `other`
    /// from this value.
    ///
    /// The constraints returned include the prior constraints on the operands, not just the
    /// additional constraints implied by the result.
    ///
    /// Returns `("bottom", "bottom")` if the abstract inverse evaluation produced a contradiction.
    pub fn abstract_sub_inv(&self, other: &Self, res: &Self) -> (Self, Self) {
        let refinements = (
            self.refine(&res.abstract_add(other)),
            other.refine(&self.abstract_sub(res)),
        );

        if refinements.0.is_bottom() || refinements.1.is_bottom() {
            (Self::bottom(), Self::bottom())
        } else {
            refinements
        }
    }

    /// Returns the abstract result of multiplying this value by `other`.
    ///
    /// For each pair of intervals, this multiplies all four combinations of their endpoints and
    /// uses the minimum and maximum products as an "interval hull". This does not compute the
    /// tightest possible constraints we could infer, but rather an overapproximation that can be
    /// represented in a small number of intervals. For example, multiplying `2..=3` by `4..=5`
    /// produces the hull interval `8..=15`. The tightest possible set of values we could prove the
    /// abstract multiplication can produce is `{8, 10, 12, 15}`. However, this would require four
    /// intervals (`8..=8`, `10..=10`, `12..=12`, and `15..=15`). The hull approximation requires
    /// only one interval, at a tradeoff for accuracy: it also includes `9`, `11`, `13`, and `14`.
    ///
    /// In this example, the "tight" evaluation produces only four intervals, but the number of
    /// intervals rapidly increases with the size of the input intervals; it could easily produce
    /// very large numbers of intervals. This would force us to fall back to "top". Therefore, while
    /// the hull approximation is less precise in some cases, it is more accurate than the tight
    /// evaluation with a fallback to "top" when the number of intervals produced exceeds
    /// [MAX_INTEGER_INTERVALS].
    ///
    /// Note that the hull approximation is equal to the tight evaluation when:
    ///
    /// - Both the left-hand-side and right-hand-side are singletons.
    /// - Either side is `0`.
    /// - Either side is `1`.
    /// - Either side is `-1`.
    ///
    /// If any hull interval spans `2^32` or more integers, the result will be a "top" value.
    pub fn abstract_mul(&self, other: &Self) -> Self {
        Self::from_intervals(self.0.iter().flat_map(|left| {
            other.0.iter().flat_map(|right| {
                let products = [
                    i64::from(*left.start()) * i64::from(*right.start()),
                    i64::from(*left.start()) * i64::from(*right.end()),
                    i64::from(*left.end()) * i64::from(*right.start()),
                    i64::from(*left.end()) * i64::from(*right.end()),
                ];

                wrapping_intervals(
                    *products.iter().min().unwrap(),
                    *products.iter().max().unwrap(),
                )
            })
        }))
    }

    /// Returns the operand constraints implied by `res` being the result of multiplying this value
    /// by `other`.
    ///
    /// Currently, this is a "placeholder" implementation that does not infer any additional
    /// constraints on the operands; the implementation may be "upgraded" later.
    pub fn abstract_mul_inv(&self, other: &Self, _res: &Self) -> (Self, Self) {
        (self.clone(), other.clone())
    }

    /// Returns the abstract result of dividing this value by `other`.
    pub fn abstract_div(&self, other: &Self) -> Self {
        if self.is_bottom() || other.is_bottom() {
            Self::bottom()
        } else if other.to_singleton() == Some(0) {
            Self::top()
        } else if self.to_singleton() == Some(0) {
            Self::from_constant(0)
        } else if other.to_singleton() == Some(1) {
            self.clone()
        } else if let (Some(left), Some(right)) = (self.to_singleton(), other.to_singleton())
            && right != 0
        {
            Self::from_constant(left.wrapping_div(right))
        } else {
            Self::top()
        }
    }

    /// Returns the operand constraints implied by `res` being the result of dividing this value by
    /// `other`.
    ///
    /// Currently, this is a "placeholder" implementation that does not infer any additional
    /// constraints on the operands; the implementation may be "upgraded" later.
    pub fn abstract_div_inv(&self, other: &Self, _res: &Self) -> (Self, Self) {
        (self.clone(), other.clone())
    }

    /// Returns the abstract result of taking the remainder of this value divided by `other`.
    pub fn abstract_mod(&self, other: &Self) -> Self {
        if self.is_bottom() || other.is_bottom() {
            Self::bottom()
        } else if other.to_singleton() == Some(0) {
            Self::top()
        } else if self.to_singleton() == Some(0) || matches!(other.to_singleton(), Some(1 | -1)) {
            Self::from_constant(0)
        } else if let (Some(left), Some(right)) = (self.to_singleton(), other.to_singleton())
            && right != 0
        {
            Self::from_constant(left.wrapping_rem(right))
        } else {
            Self::top()
        }
    }

    /// Returns the operand constraints implied by `res` being the remainder of dividing this value
    /// by `other`.
    ///
    /// Currently, this is a "placeholder" implementation that does not infer any additional
    /// constraints on the operands; the implementation may be "upgraded" later.
    pub fn abstract_mod_inv(&self, other: &Self, _res: &Self) -> (Self, Self) {
        (self.clone(), other.clone())
    }

    /// Returns the abstract result of taking the bitwise AND of this value and `other`.
    pub fn abstract_bit_and(&self, other: &Self) -> Self {
        if self.is_bottom() || other.is_bottom() {
            Self::bottom()
        } else if self.to_singleton() == Some(0) || other.to_singleton() == Some(0) {
            Self::from_constant(0)
        } else if self.to_singleton() == Some(-1) {
            other.clone()
        } else if other.to_singleton() == Some(-1) {
            self.clone()
        } else if let (Some(left), Some(right)) = (self.to_singleton(), other.to_singleton()) {
            Self::from_constant(left & right)
        } else {
            Self::top()
        }
    }

    /// Returns the operand constraints implied by `res` being the result of taking the bitwise AND
    /// of this value and `other`.
    ///
    /// Currently, this is a "placeholder" implementation that does not infer any additional
    /// constraints on the operands; the implementation may be "upgraded" later.
    pub fn abstract_bit_and_inv(&self, other: &Self, _res: &Self) -> (Self, Self) {
        (self.clone(), other.clone())
    }

    /// Returns the abstract result of taking the bitwise OR of this value and `other`.
    pub fn abstract_bit_or(&self, other: &Self) -> Self {
        if self.is_bottom() || other.is_bottom() {
            Self::bottom()
        } else if self.to_singleton() == Some(-1) || other.to_singleton() == Some(-1) {
            Self::from_constant(-1)
        } else if self.to_singleton() == Some(0) {
            other.clone()
        } else if other.to_singleton() == Some(0) {
            self.clone()
        } else if let (Some(left), Some(right)) = (self.to_singleton(), other.to_singleton()) {
            Self::from_constant(left | right)
        } else {
            Self::top()
        }
    }

    /// Returns the operand constraints implied by `res` being the result of taking the bitwise OR
    /// of this value and `other`.
    ///
    /// Currently, this is a "placeholder" implementation that does not infer any additional
    /// constraints on the operands; the implementation may be "upgraded" later.
    pub fn abstract_bit_or_inv(&self, other: &Self, _res: &Self) -> (Self, Self) {
        (self.clone(), other.clone())
    }

    /// Returns the abstract result of taking the bitwise XOR of this value and `other`.
    pub fn abstract_bit_xor(&self, other: &Self) -> Self {
        if self.is_bottom() || other.is_bottom() {
            Self::bottom()
        } else if self.to_singleton() == Some(0) {
            other.clone()
        } else if other.to_singleton() == Some(0) {
            self.clone()
        } else if let (Some(left), Some(right)) = (self.to_singleton(), other.to_singleton()) {
            Self::from_constant(left ^ right)
        } else {
            Self::top()
        }
    }

    /// Returns the operand constraints implied by `res` being the result of taking the bitwise XOR
    /// of this value and `other`.
    ///
    /// Currently, this is a "placeholder" implementation that does not infer any additional
    /// constraints on the operands; the implementation may be "upgraded" later.
    pub fn abstract_bit_xor_inv(&self, other: &Self, _res: &Self) -> (Self, Self) {
        (self.clone(), other.clone())
    }

    /// Returns the abstract result of shifting this value left by the unsigned `other` modulo the
    /// bit-width.
    pub fn abstract_shl(&self, other: &AbstractU32) -> Self {
        if self.is_bottom() || other.is_bottom() {
            Self::bottom()
        } else if self.to_singleton() == Some(0) {
            Self::from_constant(0)
        } else if let Some(other) = other.to_singleton() {
            // SLIR's shift intrinsics follow the WGSL specification for shift operations. The spec
            // prescribes that the RHS shift amount is taken module the bit-width of the LHS.
            let shift = other % i32::BITS;

            if shift == 0 {
                self.clone()
            } else if let Some(value) = self.to_singleton() {
                // Rust also provides a wrapping_shl operation that matches the WGSL behavior.
                // However, since we want to special-case a `0` RHS above, we have to compute our
                // own masked RHS anyway, so we'll use that with a regular unmasked shift.
                Self::from_constant(value << shift)
            } else {
                Self::top()
            }
        } else {
            Self::top()
        }
    }

    /// Returns the operand constraints implied by `res` being the result of shifting this value
    /// left by the unsigned `other` modulo the bit-width.
    ///
    /// Currently, this is a "placeholder" implementation that does not infer any additional
    /// constraints on the operands; the implementation may be "upgraded" later.
    pub fn abstract_shl_inv(&self, other: &AbstractU32, _res: &Self) -> (Self, AbstractU32) {
        (self.clone(), other.clone())
    }

    /// Returns the abstract result of shifting this value right by the unsigned `other` modulo the
    /// bit-width.
    pub fn abstract_shr(&self, other: &AbstractU32) -> Self {
        if self.is_bottom() || other.is_bottom() {
            Self::bottom()
        } else if self.to_singleton() == Some(0) {
            Self::from_constant(0)
        } else if let Some(other) = other.to_singleton() {
            // SLIR's shift intrinsics follow the WGSL specification for shift operations. The spec
            // prescribes that the RHS shift amount is taken module the bit-width of the LHS.
            let shift = other % i32::BITS;

            if shift == 0 {
                self.clone()
            } else if let Some(value) = self.to_singleton() {
                // Rust also provides a wrapping_shr operation that matches the WGSL behavior.
                // However, since we want to special-case a `0` RHS above, we have to compute our
                // own masked RHS anyway, so we'll use that with a regular unmasked shift.
                Self::from_constant(value >> shift)
            } else {
                Self::top()
            }
        } else {
            Self::top()
        }
    }

    /// Returns the operand constraints implied by `res` being the result of shifting this value
    /// right by the unsigned `other` modulo the bit-width.
    ///
    /// Currently, this is a "placeholder" implementation that does not infer any additional
    /// constraints on the operands; the implementation may be "upgraded" later.
    pub fn abstract_shr_inv(&self, other: &AbstractU32, _res: &Self) -> (Self, AbstractU32) {
        (self.clone(), other.clone())
    }

    /// Returns the abstract result of comparing this value equal to `other`.
    pub fn abstract_eq(&self, other: &Self) -> AbstractBool {
        if self.is_bottom() || other.is_bottom() {
            AbstractBool::Bottom
        } else if self.is_disjoint(other) {
            AbstractBool::Const(false)
        } else if let (Some(left), Some(right)) = (self.to_singleton(), other.to_singleton()) {
            AbstractBool::Const(left == right)
        } else {
            AbstractBool::Top
        }
    }

    /// Returns the operand constraints implied by `res` being the result of comparing this value
    /// equal to `other`.
    ///
    /// The constraints returned include the prior constraints on the operands, not just the
    /// additional constraints implied by the result.
    ///
    /// Returns `("bottom", "bottom")` if the abstract inverse evaluation produced a contradiction.
    pub fn abstract_eq_inv(&self, other: &Self, res: &AbstractBool) -> (Self, Self) {
        let refinements = match res {
            AbstractBool::Const(true) => (self.refine(other), other.refine(self)),
            AbstractBool::Const(false) => {
                let exclude_singleton = |value: &Self, other: &Self| {
                    other.to_singleton().map_or_else(
                        || value.clone(),
                        |other| value.exclude_cases(&[other as u32 as u128]),
                    )
                };

                (
                    exclude_singleton(self, other),
                    exclude_singleton(other, self),
                )
            }
            AbstractBool::Top => (self.clone(), other.clone()),
            AbstractBool::Bottom => (Self::bottom(), Self::bottom()),
        };

        if refinements.0.is_bottom() || refinements.1.is_bottom() {
            (Self::bottom(), Self::bottom())
        } else {
            refinements
        }
    }

    /// Returns the abstract result of comparing this value not equal to `other`.
    pub fn abstract_not_eq(&self, other: &Self) -> AbstractBool {
        self.abstract_eq(other).abstract_not()
    }

    /// Returns the operand constraints implied by `res` being the result of comparing this value
    /// not equal to `other`.
    ///
    /// The constraints returned include the prior constraints on the operands, not just the
    /// additional constraints implied by the result.
    ///
    /// Returns `("bottom", "bottom")` if the abstract inverse evaluation produced a contradiction.
    pub fn abstract_not_eq_inv(&self, other: &Self, res: &AbstractBool) -> (Self, Self) {
        self.abstract_eq_inv(other, &res.abstract_not())
    }

    /// Returns the abstract result of comparing this value less than `other`.
    pub fn abstract_lt(&self, other: &Self) -> AbstractBool {
        if self.is_bottom() || other.is_bottom() {
            return AbstractBool::Bottom;
        }

        // We now neither interval set is empty because of the is_bottom check above, so we can
        // unwrap here
        let self_min = self.0.first().unwrap().start();
        let self_max = self.0.last().unwrap().end();
        let other_min = other.0.first().unwrap().start();
        let other_max = other.0.last().unwrap().end();

        if self_max < other_min {
            AbstractBool::Const(true)
        } else if self_min >= other_max {
            AbstractBool::Const(false)
        } else {
            AbstractBool::Top
        }
    }

    /// Returns the operand constraints implied by `res` being the result of comparing this value
    /// less than `other`.
    ///
    /// The constraints returned include the prior constraints on the operands, not just the
    /// additional constraints implied by the result.
    ///
    /// Returns `("bottom", "bottom")` if the abstract inverse evaluation produced a contradiction.
    pub fn abstract_lt_inv(&self, other: &Self, res: &AbstractBool) -> (Self, Self) {
        self.abstract_order_inv(other, res, true)
    }

    /// Returns the abstract result of comparing this value less than or equal to `other`.
    pub fn abstract_lt_eq(&self, other: &Self) -> AbstractBool {
        if self.is_bottom() || other.is_bottom() {
            return AbstractBool::Bottom;
        }

        // We now neither interval set is empty because of the is_bottom check above, so we can
        // unwrap here
        let self_min = self.0.first().unwrap().start();
        let self_max = self.0.last().unwrap().end();
        let other_min = other.0.first().unwrap().start();
        let other_max = other.0.last().unwrap().end();

        if self_max <= other_min {
            AbstractBool::Const(true)
        } else if self_min > other_max {
            AbstractBool::Const(false)
        } else {
            AbstractBool::Top
        }
    }

    /// Returns the operand constraints implied by `res` being the result of comparing this value
    /// less than or equal to `other`.
    ///
    /// The constraints returned include the prior constraints on the operands, not just the
    /// additional constraints implied by the result.
    ///
    /// Returns `("bottom", "bottom")` if the abstract inverse evaluation produced a contradiction.
    pub fn abstract_lt_eq_inv(&self, other: &Self, res: &AbstractBool) -> (Self, Self) {
        self.abstract_order_inv(other, res, false)
    }

    /// Returns the abstract result of comparing this value greater than `other`.
    pub fn abstract_gt(&self, other: &Self) -> AbstractBool {
        other.abstract_lt(self)
    }

    /// Returns the operand constraints implied by `res` being the result of comparing this value
    /// greater than `other`.
    ///
    /// The constraints returned include the prior constraints on the operands, not just the
    /// additional constraints implied by the result.
    ///
    /// Returns `("bottom", "bottom")` if the abstract inverse evaluation produced a contradiction.
    pub fn abstract_gt_inv(&self, other: &Self, res: &AbstractBool) -> (Self, Self) {
        let (right, left) = other.abstract_lt_inv(self, res);

        (left, right)
    }

    /// Returns the abstract result of comparing this value greater than or equal to `other`.
    pub fn abstract_gt_eq(&self, other: &Self) -> AbstractBool {
        other.abstract_lt_eq(self)
    }

    /// Returns the operand constraints implied by `res` being the result of comparing this value
    /// greater than or equal to `other`.
    ///
    /// The constraints returned include the prior constraints on the operands, not just the
    /// additional constraints implied by the result.
    ///
    /// Returns `("bottom", "bottom")` if the abstract inverse evaluation produced a contradiction.
    pub fn abstract_gt_eq_inv(&self, other: &Self, res: &AbstractBool) -> (Self, Self) {
        let (right, left) = other.abstract_lt_eq_inv(self, res);

        (left, right)
    }

    fn abstract_order_inv(&self, other: &Self, res: &AbstractBool, strict: bool) -> (Self, Self) {
        let refinements = match res {
            AbstractBool::Const(result) if !self.is_bottom() && !other.is_bottom() => {
                let self_min = *self.0.first().unwrap().start() as i64;
                let self_max = *self.0.last().unwrap().end() as i64;
                let other_min = *other.0.first().unwrap().start() as i64;
                let other_max = *other.0.last().unwrap().end() as i64;

                let ordered_interval = |start: i64, end: i64| {
                    let start = start.max(i32::MIN as i64);
                    let end = end.min(i32::MAX as i64);

                    if start > end {
                        Self::bottom()
                    } else {
                        Self::from_intervals([start as i32..=end as i32])
                    }
                };

                if *result {
                    let delta = strict as i64;

                    (
                        self.refine(&ordered_interval(i32::MIN as i64, other_max - delta)),
                        other.refine(&ordered_interval(self_min + delta, i32::MAX as i64)),
                    )
                } else {
                    let delta = (!strict) as i64;

                    (
                        self.refine(&ordered_interval(other_min + delta, i32::MAX as i64)),
                        other.refine(&ordered_interval(i32::MIN as i64, self_max - delta)),
                    )
                }
            }
            AbstractBool::Const(_) | AbstractBool::Bottom => (Self::bottom(), Self::bottom()),
            AbstractBool::Top => (self.clone(), other.clone()),
        };

        if refinements.0.is_bottom() || refinements.1.is_bottom() {
            (Self::bottom(), Self::bottom())
        } else {
            refinements
        }
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
    fn to_abstract_f32() {
        assert_eq!(
            AbstractI32::from_constant(-4).to_abstract_f32(),
            AbstractF32::from_constant(-4.0)
        );
        assert_eq!(
            AbstractI32::from_constant(i32::MAX).to_abstract_f32(),
            AbstractF32::from_constant(i32::MAX as f32)
        );
        assert_eq!(
            AbstractI32::from_intervals([-4..=-3]).to_abstract_f32(),
            AbstractF32::Top
        );
        assert_eq!(AbstractI32::top().to_abstract_f32(), AbstractF32::Top);
        assert_eq!(AbstractI32::bottom().to_abstract_f32(), AbstractF32::Bottom);
    }

    #[test]
    fn to_abstract_bool() {
        assert_eq!(
            AbstractI32::from_constant(0).to_abstract_bool(),
            AbstractBool::Const(false)
        );
        assert_eq!(
            AbstractI32::from_intervals([-4..=-1, 1..=4]).to_abstract_bool(),
            AbstractBool::Const(true)
        );
        assert_eq!(
            AbstractI32::from_intervals([-1..=1]).to_abstract_bool(),
            AbstractBool::Top
        );
        assert_eq!(AbstractI32::top().to_abstract_bool(), AbstractBool::Top);
        assert_eq!(
            AbstractI32::bottom().to_abstract_bool(),
            AbstractBool::Bottom
        );
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

    #[test]
    fn abstract_neg() {
        assert_eq!(
            AbstractI32::from_intervals([-5..=-3, 2..=4]).abstract_neg(),
            AbstractI32::from_intervals([-4..=-2, 3..=5])
        );
        assert_eq!(
            AbstractI32::from_intervals([i32::MIN..=i32::MIN + 2]).abstract_neg(),
            AbstractI32::from_intervals([i32::MIN..=i32::MIN, i32::MAX - 1..=i32::MAX])
        );
        assert_eq!(AbstractI32::top().abstract_neg(), AbstractI32::top());
        assert_eq!(AbstractI32::bottom().abstract_neg(), AbstractI32::bottom());
    }

    #[test]
    fn abstract_neg_inv() {
        assert_eq!(
            AbstractI32::top().abstract_neg_inv(&AbstractI32::from_intervals([3..=5])),
            AbstractI32::from_intervals([-5..=-3])
        );
        assert_eq!(
            AbstractI32::from_intervals([-5..=2])
                .abstract_neg_inv(&AbstractI32::from_intervals([3..=5])),
            AbstractI32::from_intervals([-5..=-3])
        );
        assert_eq!(
            AbstractI32::from_constant(i32::MIN)
                .abstract_neg_inv(&AbstractI32::from_constant(i32::MIN)),
            AbstractI32::from_constant(i32::MIN)
        );
        assert_eq!(
            AbstractI32::from_intervals([1..=3])
                .abstract_neg_inv(&AbstractI32::from_intervals([1..=3])),
            AbstractI32::bottom()
        );
        assert_eq!(
            AbstractI32::from_constant(1).abstract_neg_inv(&AbstractI32::top()),
            AbstractI32::from_constant(1)
        );
        assert_eq!(
            AbstractI32::top().abstract_neg_inv(&AbstractI32::bottom()),
            AbstractI32::bottom()
        );
        assert_eq!(
            AbstractI32::bottom().abstract_neg_inv(&AbstractI32::top()),
            AbstractI32::bottom()
        );
    }

    #[test]
    fn abstract_add() {
        assert_eq!(
            AbstractI32::from_intervals([1..=3])
                .abstract_add(&AbstractI32::from_intervals([4..=6])),
            AbstractI32::from_intervals([5..=9])
        );
        assert_eq!(
            AbstractI32::from_intervals([i32::MAX - 1..=i32::MAX])
                .abstract_add(&AbstractI32::from_intervals([1..=2])),
            AbstractI32::from_intervals([i32::MIN..=i32::MIN + 1, i32::MAX..=i32::MAX])
        );
        assert_eq!(
            AbstractI32::top().abstract_add(&AbstractI32::from_constant(1)),
            AbstractI32::top()
        );
        assert_eq!(
            AbstractI32::bottom().abstract_add(&AbstractI32::top()),
            AbstractI32::bottom()
        );
        assert_eq!(
            AbstractI32::top().abstract_add(&AbstractI32::bottom()),
            AbstractI32::bottom()
        );
    }

    #[test]
    fn abstract_add_inv() {
        assert_eq!(
            AbstractI32::from_intervals([1..=5]).abstract_add_inv(
                &AbstractI32::from_intervals([4..=8]),
                &AbstractI32::from_constant(6),
            ),
            (
                AbstractI32::from_intervals([1..=2]),
                AbstractI32::from_intervals([4..=5]),
            )
        );
        assert_eq!(
            AbstractI32::from_intervals([1..=10]).abstract_add_inv(
                &AbstractI32::from_intervals([4..=12]),
                &AbstractI32::from_intervals([8..=10]),
            ),
            (
                AbstractI32::from_intervals([1..=6]),
                AbstractI32::from_intervals([4..=9]),
            )
        );
        assert_eq!(
            AbstractI32::from_intervals([1..=5]).abstract_add_inv(
                &AbstractI32::from_intervals([4..=8]),
                &AbstractI32::from_intervals([8..=10]),
            ),
            (
                AbstractI32::from_intervals([1..=5]),
                AbstractI32::from_intervals([4..=8]),
            )
        );
        assert_eq!(
            AbstractI32::from_intervals([i32::MAX - 1..=i32::MAX]).abstract_add_inv(
                &AbstractI32::from_intervals([1..=2]),
                &AbstractI32::from_intervals([i32::MIN..=i32::MIN + 1]),
            ),
            (
                AbstractI32::from_intervals([i32::MAX - 1..=i32::MAX]),
                AbstractI32::from_intervals([1..=2]),
            )
        );
        assert_eq!(
            AbstractI32::from_intervals([0..=3, 10..=15]).abstract_add_inv(
                &AbstractI32::from_intervals([2..=6, 20..=25]),
                &AbstractI32::from_intervals([13..=18]),
            ),
            (
                AbstractI32::from_intervals([10..=15]),
                AbstractI32::from_intervals([2..=6]),
            )
        );
        assert_eq!(
            AbstractI32::from_intervals([-3..=-1, 1..=3]).abstract_add_inv(
                &AbstractI32::from_constant(2),
                &AbstractI32::from_intervals([-1..=1, 3..=5]),
            ),
            (
                AbstractI32::from_intervals([-3..=-1, 1..=3]),
                AbstractI32::from_constant(2),
            )
        );
        assert_eq!(
            AbstractI32::from_intervals([1..=3]).abstract_add_inv(
                &AbstractI32::from_intervals([4..=6]),
                &AbstractI32::from_intervals([10..=12]),
            ),
            (AbstractI32::bottom(), AbstractI32::bottom())
        );
        assert_eq!(
            AbstractI32::from_intervals([1..=3])
                .abstract_add_inv(&AbstractI32::from_intervals([4..=6]), &AbstractI32::top()),
            (
                AbstractI32::from_intervals([1..=3]),
                AbstractI32::from_intervals([4..=6]),
            )
        );
        assert_eq!(
            AbstractI32::top().abstract_add_inv(&AbstractI32::top(), &AbstractI32::bottom()),
            (AbstractI32::bottom(), AbstractI32::bottom())
        );
    }

    #[test]
    fn abstract_sub() {
        assert_eq!(
            AbstractI32::from_intervals([4..=6])
                .abstract_sub(&AbstractI32::from_intervals([1..=2])),
            AbstractI32::from_intervals([2..=5])
        );
        assert_eq!(
            AbstractI32::from_intervals([i32::MIN..=i32::MIN + 1])
                .abstract_sub(&AbstractI32::from_intervals([1..=2])),
            AbstractI32::from_intervals([i32::MIN..=i32::MIN, i32::MAX - 1..=i32::MAX])
        );
        assert_eq!(
            AbstractI32::top().abstract_sub(&AbstractI32::from_constant(1)),
            AbstractI32::top()
        );
        assert_eq!(
            AbstractI32::bottom().abstract_sub(&AbstractI32::top()),
            AbstractI32::bottom()
        );
        assert_eq!(
            AbstractI32::top().abstract_sub(&AbstractI32::bottom()),
            AbstractI32::bottom()
        );
    }

    #[test]
    fn abstract_sub_inv() {
        assert_eq!(
            AbstractI32::from_intervals([1..=10]).abstract_sub_inv(
                &AbstractI32::from_intervals([4..=12]),
                &AbstractI32::from_intervals([2..=4]),
            ),
            (
                AbstractI32::from_intervals([6..=10]),
                AbstractI32::from_intervals([4..=8]),
            )
        );
        assert_eq!(
            AbstractI32::from_intervals([i32::MIN..=i32::MIN + 1]).abstract_sub_inv(
                &AbstractI32::from_intervals([1..=2]),
                &AbstractI32::from_intervals([i32::MAX - 1..=i32::MAX]),
            ),
            (
                AbstractI32::from_intervals([i32::MIN..=i32::MIN + 1]),
                AbstractI32::from_intervals([1..=2]),
            )
        );
        assert_eq!(
            AbstractI32::from_intervals([0..=3, 10..=15]).abstract_sub_inv(
                &AbstractI32::from_intervals([2..=6, 20..=25]),
                &AbstractI32::from_intervals([-15..=-7]),
            ),
            (
                AbstractI32::from_intervals([10..=15]),
                AbstractI32::from_intervals([20..=25]),
            )
        );
        assert_eq!(
            AbstractI32::from_intervals([1..=3]).abstract_sub_inv(
                &AbstractI32::from_intervals([4..=6]),
                &AbstractI32::from_intervals([10..=12]),
            ),
            (AbstractI32::bottom(), AbstractI32::bottom())
        );
        assert_eq!(
            AbstractI32::from_intervals([1..=3])
                .abstract_sub_inv(&AbstractI32::from_intervals([4..=6]), &AbstractI32::top()),
            (
                AbstractI32::from_intervals([1..=3]),
                AbstractI32::from_intervals([4..=6]),
            )
        );
        assert_eq!(
            AbstractI32::top().abstract_sub_inv(&AbstractI32::top(), &AbstractI32::bottom()),
            (AbstractI32::bottom(), AbstractI32::bottom())
        );
    }

    #[test]
    fn abstract_mul() {
        assert_eq!(
            AbstractI32::from_constant(-6).abstract_mul(&AbstractI32::from_constant(7)),
            AbstractI32::from_constant(-42)
        );
        assert_eq!(
            AbstractI32::from_constant(i32::MAX).abstract_mul(&AbstractI32::from_constant(2)),
            AbstractI32::from_constant(-2)
        );
        assert_eq!(
            AbstractI32::from_constant(i32::MAX)
                .abstract_mul(&AbstractI32::from_constant(i32::MAX)),
            AbstractI32::from_constant(1)
        );
        assert_eq!(
            AbstractI32::from_constant(i32::MIN)
                .abstract_mul(&AbstractI32::from_constant(i32::MAX)),
            AbstractI32::from_constant(i32::MIN)
        );

        let value = AbstractI32::from_intervals([2..=3]);

        assert_eq!(
            AbstractI32::from_constant(0).abstract_mul(&value),
            AbstractI32::from_constant(0)
        );
        assert_eq!(
            value.abstract_mul(&AbstractI32::from_constant(0)),
            AbstractI32::from_constant(0)
        );
        assert_eq!(AbstractI32::from_constant(1).abstract_mul(&value), value);
        assert_eq!(value.abstract_mul(&AbstractI32::from_constant(1)), value);
        assert_eq!(
            AbstractI32::from_intervals([2..=3])
                .abstract_mul(&AbstractI32::from_intervals([4..=5])),
            AbstractI32::from_intervals([8..=15])
        );
        assert_eq!(
            AbstractI32::from_intervals([2..=3, 5..=6])
                .abstract_mul(&AbstractI32::from_intervals([4..=5])),
            AbstractI32::from_intervals([8..=15, 20..=30])
        );
        assert_eq!(
            AbstractI32::from_intervals([-2..=3])
                .abstract_mul(&AbstractI32::from_intervals([-4..=5])),
            AbstractI32::from_intervals([-12..=15])
        );
        assert_eq!(
            AbstractI32::from_intervals([i32::MAX - 1..=i32::MAX])
                .abstract_mul(&AbstractI32::from_constant(2)),
            AbstractI32::from_intervals([-4..=-2])
        );
        assert_eq!(
            AbstractI32::from_intervals([i32::MAX - 1..=i32::MAX])
                .abstract_mul(&AbstractI32::from_intervals([1..=2])),
            AbstractI32::from_intervals([i32::MIN..=-2, i32::MAX - 1..=i32::MAX])
        );
        assert_eq!(
            AbstractI32::from_constant(0).abstract_mul(&AbstractI32::top()),
            AbstractI32::from_constant(0)
        );
        assert_eq!(
            AbstractI32::top().abstract_mul(&AbstractI32::from_constant(0)),
            AbstractI32::from_constant(0)
        );
        assert_eq!(
            AbstractI32::from_constant(1).abstract_mul(&AbstractI32::top()),
            AbstractI32::top()
        );
        assert_eq!(
            AbstractI32::top().abstract_mul(&AbstractI32::from_constant(1)),
            AbstractI32::top()
        );
        assert_eq!(
            AbstractI32::bottom().abstract_mul(&AbstractI32::top()),
            AbstractI32::bottom()
        );
        assert_eq!(
            AbstractI32::top().abstract_mul(&AbstractI32::bottom()),
            AbstractI32::bottom()
        );
    }

    #[test]
    fn abstract_div() {
        assert_eq!(
            AbstractI32::from_constant(-43).abstract_div(&AbstractI32::from_constant(7)),
            AbstractI32::from_constant(-6)
        );
        assert_eq!(
            AbstractI32::from_constant(i32::MIN).abstract_div(&AbstractI32::from_constant(-1)),
            AbstractI32::from_constant(i32::MIN)
        );
        assert_eq!(
            AbstractI32::from_constant(1).abstract_div(&AbstractI32::from_constant(0)),
            AbstractI32::top()
        );

        let value = AbstractI32::from_intervals([4..=6]);

        assert_eq!(
            AbstractI32::from_constant(0).abstract_div(&value),
            AbstractI32::from_constant(0)
        );
        assert_eq!(value.abstract_div(&AbstractI32::from_constant(1)), value);
        assert_eq!(
            AbstractI32::from_constant(0).abstract_div(&AbstractI32::top()),
            AbstractI32::from_constant(0)
        );
        assert_eq!(
            AbstractI32::top().abstract_div(&AbstractI32::from_constant(1)),
            AbstractI32::top()
        );
        assert_eq!(
            AbstractI32::from_constant(0).abstract_div(&AbstractI32::from_constant(0)),
            AbstractI32::top()
        );
        assert_eq!(
            value.abstract_div(&AbstractI32::from_constant(2)),
            AbstractI32::top()
        );
        assert_eq!(
            AbstractI32::bottom().abstract_div(&AbstractI32::top()),
            AbstractI32::bottom()
        );
        assert_eq!(
            AbstractI32::top().abstract_div(&AbstractI32::bottom()),
            AbstractI32::bottom()
        );
    }

    #[test]
    fn abstract_mod() {
        assert_eq!(
            AbstractI32::from_constant(-43).abstract_mod(&AbstractI32::from_constant(7)),
            AbstractI32::from_constant(-1)
        );
        assert_eq!(
            AbstractI32::from_constant(i32::MIN).abstract_mod(&AbstractI32::from_constant(-1)),
            AbstractI32::from_constant(0)
        );
        assert_eq!(
            AbstractI32::from_constant(1).abstract_mod(&AbstractI32::from_constant(0)),
            AbstractI32::top()
        );

        let value = AbstractI32::from_intervals([4..=6]);

        assert_eq!(
            AbstractI32::from_constant(0).abstract_mod(&value),
            AbstractI32::from_constant(0)
        );
        assert_eq!(
            value.abstract_mod(&AbstractI32::from_constant(1)),
            AbstractI32::from_constant(0)
        );
        assert_eq!(
            value.abstract_mod(&AbstractI32::from_constant(-1)),
            AbstractI32::from_constant(0)
        );
        assert_eq!(
            AbstractI32::from_constant(0).abstract_mod(&AbstractI32::top()),
            AbstractI32::from_constant(0)
        );
        assert_eq!(
            AbstractI32::from_constant(0).abstract_mod(&AbstractI32::from_constant(0)),
            AbstractI32::top()
        );
        assert_eq!(
            value.abstract_mod(&AbstractI32::from_constant(2)),
            AbstractI32::top()
        );
        assert_eq!(
            AbstractI32::bottom().abstract_mod(&AbstractI32::top()),
            AbstractI32::bottom()
        );
        assert_eq!(
            AbstractI32::top().abstract_mod(&AbstractI32::bottom()),
            AbstractI32::bottom()
        );
    }

    #[test]
    fn abstract_bit_and() {
        assert_eq!(
            AbstractI32::from_constant(-1).abstract_bit_and(&AbstractI32::from_constant(42)),
            AbstractI32::from_constant(42)
        );

        let value = AbstractI32::from_intervals([2..=3]);

        assert_eq!(
            AbstractI32::from_constant(0).abstract_bit_and(&value),
            AbstractI32::from_constant(0)
        );
        assert_eq!(
            value.abstract_bit_and(&AbstractI32::from_constant(0)),
            AbstractI32::from_constant(0)
        );
        assert_eq!(
            AbstractI32::from_constant(-1).abstract_bit_and(&value),
            value
        );
        assert_eq!(
            value.abstract_bit_and(&AbstractI32::from_constant(-1)),
            value
        );
        assert_eq!(
            AbstractI32::from_constant(0).abstract_bit_and(&AbstractI32::top()),
            AbstractI32::from_constant(0)
        );
        assert_eq!(
            value.abstract_bit_and(&AbstractI32::from_constant(1)),
            AbstractI32::top()
        );
        assert_eq!(
            AbstractI32::bottom().abstract_bit_and(&AbstractI32::top()),
            AbstractI32::bottom()
        );
        assert_eq!(
            AbstractI32::top().abstract_bit_and(&AbstractI32::bottom()),
            AbstractI32::bottom()
        );
    }

    #[test]
    fn abstract_bit_or() {
        assert_eq!(
            AbstractI32::from_constant(i32::MIN)
                .abstract_bit_or(&AbstractI32::from_constant(i32::MAX)),
            AbstractI32::from_constant(-1)
        );

        let value = AbstractI32::from_intervals([2..=3]);

        assert_eq!(AbstractI32::from_constant(0).abstract_bit_or(&value), value);
        assert_eq!(value.abstract_bit_or(&AbstractI32::from_constant(0)), value);
        assert_eq!(
            AbstractI32::from_constant(-1).abstract_bit_or(&value),
            AbstractI32::from_constant(-1)
        );
        assert_eq!(
            value.abstract_bit_or(&AbstractI32::from_constant(-1)),
            AbstractI32::from_constant(-1)
        );
        assert_eq!(
            AbstractI32::from_constant(-1).abstract_bit_or(&AbstractI32::top()),
            AbstractI32::from_constant(-1)
        );
        assert_eq!(
            value.abstract_bit_or(&AbstractI32::from_constant(1)),
            AbstractI32::top()
        );
        assert_eq!(
            AbstractI32::bottom().abstract_bit_or(&AbstractI32::top()),
            AbstractI32::bottom()
        );
        assert_eq!(
            AbstractI32::top().abstract_bit_or(&AbstractI32::bottom()),
            AbstractI32::bottom()
        );
    }

    #[test]
    fn abstract_bit_xor() {
        assert_eq!(
            AbstractI32::from_constant(-1).abstract_bit_xor(&AbstractI32::from_constant(i32::MAX)),
            AbstractI32::from_constant(i32::MIN)
        );

        let value = AbstractI32::from_intervals([2..=3]);

        assert_eq!(
            AbstractI32::from_constant(0).abstract_bit_xor(&value),
            value
        );
        assert_eq!(
            value.abstract_bit_xor(&AbstractI32::from_constant(0)),
            value
        );
        assert_eq!(
            AbstractI32::from_constant(0).abstract_bit_xor(&AbstractI32::top()),
            AbstractI32::top()
        );
        assert_eq!(
            value.abstract_bit_xor(&AbstractI32::from_constant(1)),
            AbstractI32::top()
        );
        assert_eq!(
            AbstractI32::bottom().abstract_bit_xor(&AbstractI32::top()),
            AbstractI32::bottom()
        );
        assert_eq!(
            AbstractI32::top().abstract_bit_xor(&AbstractI32::bottom()),
            AbstractI32::bottom()
        );
    }

    #[test]
    fn abstract_shl() {
        assert_eq!(
            AbstractI32::from_constant(3).abstract_shl(&AbstractU32::from_constant(2)),
            AbstractI32::from_constant(12)
        );
        assert_eq!(
            AbstractI32::from_constant(1).abstract_shl(&AbstractU32::from_constant(u32::MAX)),
            AbstractI32::from_constant(i32::MIN)
        );

        let value = AbstractI32::from_intervals([4..=6]);

        assert_eq!(value.abstract_shl(&AbstractU32::from_constant(32)), value);
        assert_eq!(value.abstract_shl(&AbstractU32::from_constant(64)), value);
        assert_eq!(
            AbstractI32::from_constant(0).abstract_shl(&AbstractU32::top()),
            AbstractI32::from_constant(0)
        );
        assert_eq!(
            value.abstract_shl(&AbstractU32::from_constant(1)),
            AbstractI32::top()
        );
        assert_eq!(
            AbstractI32::bottom().abstract_shl(&AbstractU32::top()),
            AbstractI32::bottom()
        );
        assert_eq!(
            AbstractI32::top().abstract_shl(&AbstractU32::bottom()),
            AbstractI32::bottom()
        );
    }

    #[test]
    fn abstract_shr() {
        assert_eq!(
            AbstractI32::from_constant(-16).abstract_shr(&AbstractU32::from_constant(2)),
            AbstractI32::from_constant(-4)
        );
        assert_eq!(
            AbstractI32::from_constant(i32::MIN).abstract_shr(&AbstractU32::from_constant(63)),
            AbstractI32::from_constant(-1)
        );

        let value = AbstractI32::from_intervals([-6..=-4]);

        assert_eq!(value.abstract_shr(&AbstractU32::from_constant(32)), value);
        assert_eq!(value.abstract_shr(&AbstractU32::from_constant(64)), value);
        assert_eq!(
            AbstractI32::from_constant(0).abstract_shr(&AbstractU32::top()),
            AbstractI32::from_constant(0)
        );
        assert_eq!(
            value.abstract_shr(&AbstractU32::from_constant(1)),
            AbstractI32::top()
        );
        assert_eq!(
            AbstractI32::bottom().abstract_shr(&AbstractU32::top()),
            AbstractI32::bottom()
        );
        assert_eq!(
            AbstractI32::top().abstract_shr(&AbstractU32::bottom()),
            AbstractI32::bottom()
        );
    }

    #[test]
    fn abstract_eq() {
        assert_eq!(
            AbstractI32::from_constant(-1).abstract_eq(&AbstractI32::from_constant(-1)),
            AbstractBool::Const(true)
        );
        assert_eq!(
            AbstractI32::from_intervals([-3..=-1])
                .abstract_eq(&AbstractI32::from_intervals([1..=3])),
            AbstractBool::Const(false)
        );
        assert_eq!(
            AbstractI32::from_intervals([-1..=1]).abstract_eq(&AbstractI32::from_constant(0)),
            AbstractBool::Top
        );
        assert_eq!(
            AbstractI32::top().abstract_eq(&AbstractI32::top()),
            AbstractBool::Top
        );
        assert_eq!(
            AbstractI32::bottom().abstract_eq(&AbstractI32::top()),
            AbstractBool::Bottom
        );
        assert_eq!(
            AbstractI32::top().abstract_eq(&AbstractI32::bottom()),
            AbstractBool::Bottom
        );
    }

    #[test]
    fn abstract_eq_inv() {
        assert_eq!(
            AbstractI32::from_intervals([-3..=3]).abstract_eq_inv(
                &AbstractI32::from_intervals([1..=5]),
                &AbstractBool::Const(true),
            ),
            (
                AbstractI32::from_intervals([1..=3]),
                AbstractI32::from_intervals([1..=3]),
            )
        );
        assert_eq!(
            AbstractI32::from_intervals([-1..=1])
                .abstract_eq_inv(&AbstractI32::from_constant(0), &AbstractBool::Const(false),),
            (
                AbstractI32::from_intervals([-1..=-1, 1..=1]),
                AbstractI32::from_constant(0),
            )
        );
        assert_eq!(
            AbstractI32::from_intervals([-1..=1]).abstract_eq_inv(
                &AbstractI32::from_intervals([0..=2]),
                &AbstractBool::Const(false),
            ),
            (
                AbstractI32::from_intervals([-1..=1]),
                AbstractI32::from_intervals([0..=2]),
            )
        );
        assert_eq!(
            AbstractI32::from_constant(0)
                .abstract_eq_inv(&AbstractI32::from_constant(1), &AbstractBool::Const(true),),
            (AbstractI32::bottom(), AbstractI32::bottom())
        );
        assert_eq!(
            AbstractI32::top().abstract_eq_inv(&AbstractI32::top(), &AbstractBool::Top),
            (AbstractI32::top(), AbstractI32::top())
        );
        assert_eq!(
            AbstractI32::top().abstract_eq_inv(&AbstractI32::top(), &AbstractBool::Bottom),
            (AbstractI32::bottom(), AbstractI32::bottom())
        );
        assert_eq!(
            AbstractI32::bottom().abstract_eq_inv(&AbstractI32::top(), &AbstractBool::Top),
            (AbstractI32::bottom(), AbstractI32::bottom())
        );
    }

    #[test]
    fn abstract_not_eq() {
        assert_eq!(
            AbstractI32::from_constant(-1).abstract_not_eq(&AbstractI32::from_constant(-1)),
            AbstractBool::Const(false)
        );
        assert_eq!(
            AbstractI32::from_intervals([-3..=-1])
                .abstract_not_eq(&AbstractI32::from_intervals([1..=3])),
            AbstractBool::Const(true)
        );
        assert_eq!(
            AbstractI32::from_intervals([-1..=1]).abstract_not_eq(&AbstractI32::from_constant(0)),
            AbstractBool::Top
        );
        assert_eq!(
            AbstractI32::bottom().abstract_not_eq(&AbstractI32::top()),
            AbstractBool::Bottom
        );
    }

    #[test]
    fn abstract_not_eq_inv() {
        assert_eq!(
            AbstractI32::from_intervals([-1..=1])
                .abstract_not_eq_inv(&AbstractI32::from_constant(0), &AbstractBool::Const(true),),
            (
                AbstractI32::from_intervals([-1..=-1, 1..=1]),
                AbstractI32::from_constant(0),
            )
        );
        assert_eq!(
            AbstractI32::from_intervals([-3..=3]).abstract_not_eq_inv(
                &AbstractI32::from_intervals([1..=5]),
                &AbstractBool::Const(false),
            ),
            (
                AbstractI32::from_intervals([1..=3]),
                AbstractI32::from_intervals([1..=3]),
            )
        );
        assert_eq!(
            AbstractI32::top().abstract_not_eq_inv(&AbstractI32::top(), &AbstractBool::Bottom),
            (AbstractI32::bottom(), AbstractI32::bottom())
        );
    }

    #[test]
    fn abstract_lt() {
        assert_eq!(
            AbstractI32::from_intervals([-4..=-2])
                .abstract_lt(&AbstractI32::from_intervals([1..=3])),
            AbstractBool::Const(true)
        );
        assert_eq!(
            AbstractI32::from_intervals([1..=3])
                .abstract_lt(&AbstractI32::from_intervals([-4..=-2])),
            AbstractBool::Const(false)
        );
        assert_eq!(
            AbstractI32::from_constant(1).abstract_lt(&AbstractI32::from_constant(1)),
            AbstractBool::Const(false)
        );
        assert_eq!(
            AbstractI32::from_intervals([-2..=2])
                .abstract_lt(&AbstractI32::from_intervals([1..=3])),
            AbstractBool::Top
        );
        assert_eq!(
            AbstractI32::bottom().abstract_lt(&AbstractI32::top()),
            AbstractBool::Bottom
        );
        assert_eq!(
            AbstractI32::top().abstract_lt(&AbstractI32::bottom()),
            AbstractBool::Bottom
        );
    }

    #[test]
    fn abstract_lt_inv() {
        assert_eq!(
            AbstractI32::from_intervals([2..=7]).abstract_lt_inv(
                &AbstractI32::from_intervals([1..=5]),
                &AbstractBool::Const(true),
            ),
            (
                AbstractI32::from_intervals([2..=4]),
                AbstractI32::from_intervals([3..=5]),
            )
        );
        assert_eq!(
            AbstractI32::from_intervals([1..=5]).abstract_lt_inv(
                &AbstractI32::from_intervals([2..=7]),
                &AbstractBool::Const(false),
            ),
            (
                AbstractI32::from_intervals([2..=5]),
                AbstractI32::from_intervals([2..=5]),
            )
        );
        assert_eq!(
            AbstractI32::from_constant(i32::MIN).abstract_lt_inv(
                &AbstractI32::from_constant(i32::MIN),
                &AbstractBool::Const(true),
            ),
            (AbstractI32::bottom(), AbstractI32::bottom())
        );
        assert_eq!(
            AbstractI32::from_intervals([2..=7])
                .abstract_lt_inv(&AbstractI32::from_intervals([1..=5]), &AbstractBool::Top,),
            (
                AbstractI32::from_intervals([2..=7]),
                AbstractI32::from_intervals([1..=5]),
            )
        );
        assert_eq!(
            AbstractI32::from_intervals([2..=7])
                .abstract_lt_inv(&AbstractI32::from_intervals([1..=5]), &AbstractBool::Bottom,),
            (AbstractI32::bottom(), AbstractI32::bottom())
        );
        assert_eq!(
            AbstractI32::bottom()
                .abstract_lt_inv(&AbstractI32::from_intervals([1..=5]), &AbstractBool::Top,),
            (AbstractI32::bottom(), AbstractI32::bottom())
        );
        assert_eq!(
            AbstractI32::from_intervals([-10..=-8, 2..=7]).abstract_lt_inv(
                &AbstractI32::from_intervals([1..=5]),
                &AbstractBool::Const(true),
            ),
            (
                AbstractI32::from_intervals([-10..=-8, 2..=4]),
                AbstractI32::from_intervals([1..=5]),
            )
        );
        assert_eq!(
            AbstractI32::from_intervals([2..=7]).abstract_lt_inv(
                &AbstractI32::from_intervals([1..=3, 8..=10]),
                &AbstractBool::Const(true),
            ),
            (
                AbstractI32::from_intervals([2..=7]),
                AbstractI32::from_intervals([3..=3, 8..=10]),
            )
        );
        assert_eq!(
            AbstractI32::from_intervals([-5..=-2, 4..=8]).abstract_lt_inv(
                &AbstractI32::from_intervals([-3..=1, 6..=10]),
                &AbstractBool::Const(false),
            ),
            (
                AbstractI32::from_intervals([-3..=-2, 4..=8]),
                AbstractI32::from_intervals([-3..=1, 6..=8]),
            )
        );
        assert_eq!(
            AbstractI32::from_intervals([-5..=-2, 4..=8]).abstract_lt_inv(
                &AbstractI32::from_intervals([-3..=1, 6..=10]),
                &AbstractBool::Const(true),
            ),
            (
                AbstractI32::from_intervals([-5..=-2, 4..=8]),
                AbstractI32::from_intervals([-3..=1, 6..=10]),
            )
        );
    }

    #[test]
    fn abstract_lt_eq() {
        assert_eq!(
            AbstractI32::from_intervals([-4..=-2])
                .abstract_lt_eq(&AbstractI32::from_intervals([-2..=1])),
            AbstractBool::Const(true)
        );
        assert_eq!(
            AbstractI32::from_intervals([1..=3])
                .abstract_lt_eq(&AbstractI32::from_intervals([-4..=0])),
            AbstractBool::Const(false)
        );
        assert_eq!(
            AbstractI32::from_intervals([-2..=2])
                .abstract_lt_eq(&AbstractI32::from_intervals([1..=3])),
            AbstractBool::Top
        );
        assert_eq!(
            AbstractI32::bottom().abstract_lt_eq(&AbstractI32::top()),
            AbstractBool::Bottom
        );
    }

    #[test]
    fn abstract_lt_eq_inv() {
        assert_eq!(
            AbstractI32::from_intervals([2..=7]).abstract_lt_eq_inv(
                &AbstractI32::from_intervals([1..=5]),
                &AbstractBool::Const(true),
            ),
            (
                AbstractI32::from_intervals([2..=5]),
                AbstractI32::from_intervals([2..=5]),
            )
        );
        assert_eq!(
            AbstractI32::from_intervals([0..=5]).abstract_lt_eq_inv(
                &AbstractI32::from_intervals([2..=7]),
                &AbstractBool::Const(false),
            ),
            (
                AbstractI32::from_intervals([3..=5]),
                AbstractI32::from_intervals([2..=4]),
            )
        );
    }

    #[test]
    fn abstract_gt() {
        assert_eq!(
            AbstractI32::from_intervals([1..=3])
                .abstract_gt(&AbstractI32::from_intervals([-4..=-2])),
            AbstractBool::Const(true)
        );
        assert_eq!(
            AbstractI32::from_intervals([-4..=-2])
                .abstract_gt(&AbstractI32::from_intervals([1..=3])),
            AbstractBool::Const(false)
        );
        assert_eq!(
            AbstractI32::from_intervals([-2..=2])
                .abstract_gt(&AbstractI32::from_intervals([1..=3])),
            AbstractBool::Top
        );
        assert_eq!(
            AbstractI32::bottom().abstract_gt(&AbstractI32::top()),
            AbstractBool::Bottom
        );
    }

    #[test]
    fn abstract_gt_inv() {
        assert_eq!(
            AbstractI32::from_intervals([1..=5]).abstract_gt_inv(
                &AbstractI32::from_intervals([2..=7]),
                &AbstractBool::Const(false),
            ),
            (
                AbstractI32::from_intervals([1..=5]),
                AbstractI32::from_intervals([2..=7]),
            )
        );
        assert_eq!(
            AbstractI32::from_intervals([2..=7]).abstract_gt_inv(
                &AbstractI32::from_intervals([1..=5]),
                &AbstractBool::Const(true),
            ),
            (
                AbstractI32::from_intervals([2..=7]),
                AbstractI32::from_intervals([1..=5]),
            )
        );
    }

    #[test]
    fn abstract_gt_eq() {
        assert_eq!(
            AbstractI32::from_intervals([-2..=1])
                .abstract_gt_eq(&AbstractI32::from_intervals([-4..=-2])),
            AbstractBool::Const(true)
        );
        assert_eq!(
            AbstractI32::from_intervals([-4..=0])
                .abstract_gt_eq(&AbstractI32::from_intervals([1..=3])),
            AbstractBool::Const(false)
        );
        assert_eq!(
            AbstractI32::from_intervals([-2..=2])
                .abstract_gt_eq(&AbstractI32::from_intervals([1..=3])),
            AbstractBool::Top
        );
        assert_eq!(
            AbstractI32::top().abstract_gt_eq(&AbstractI32::bottom()),
            AbstractBool::Bottom
        );
    }

    #[test]
    fn abstract_gt_eq_inv() {
        assert_eq!(
            AbstractI32::from_intervals([2..=7]).abstract_gt_eq_inv(
                &AbstractI32::from_intervals([1..=5]),
                &AbstractBool::Const(false),
            ),
            (
                AbstractI32::from_intervals([2..=4]),
                AbstractI32::from_intervals([3..=5]),
            )
        );
        assert_eq!(
            AbstractI32::from_intervals([0..=5]).abstract_gt_eq_inv(
                &AbstractI32::from_intervals([2..=7]),
                &AbstractBool::Const(true),
            ),
            (
                AbstractI32::from_intervals([2..=5]),
                AbstractI32::from_intervals([2..=5]),
            )
        );
    }
}
