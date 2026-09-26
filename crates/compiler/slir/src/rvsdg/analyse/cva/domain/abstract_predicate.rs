use super::{AbstractBool, AbstractI32, AbstractU32};
use crate::BranchCase;

/// A (possibly constrained) branch-selector predicate value.
///
/// See also [AbstractValue](super::AbstractValue).
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum AbstractPredicate {
    /// The "top" value in a lattice-theory sense.
    ///
    /// The value is unconstrained; any branch index may occur.
    Top,

    /// The value is constrained to one of the contained branch indices.
    ///
    /// The set is deduplicated and in increasing order. An empty set is the "bottom" value.
    Set(Vec<u32>),
}

impl AbstractPredicate {
    /// Returns a new abstract value constrained to the given `values`.
    ///
    /// The values are sorted and deduplicated. An empty iterator produces "bottom".
    pub fn from_values(values: impl IntoIterator<Item = u32>) -> Self {
        let mut values: Vec<_> = values.into_iter().collect();
        values.sort_unstable();
        values.dedup();
        values.shrink_to_fit();
        Self::Set(values)
    }

    /// Returns a new abstract value constrained to exactly the given `value`.
    pub fn from_constant(value: u32) -> Self {
        Self::Set(vec![value])
    }

    /// Returns the abstract result of deriving a branch-selector value from a boolean value.
    pub fn abstract_from_bool(value: &AbstractBool) -> Self {
        match value {
            AbstractBool::Top => Self::from_values([0, 1]),
            AbstractBool::Const(true) => Self::from_constant(0),
            AbstractBool::Const(false) => Self::from_constant(1),
            AbstractBool::Bottom => Self::bottom(),
        }
    }

    /// Returns the possible branch indices when comparing a signed integer with `cases`.
    /// The default branch has index `cases.len()`; duplicate cases select their first occurrence.
    pub fn abstract_from_case_i32(value: &AbstractI32, cases: &[BranchCase]) -> Self {
        let mut branches = Vec::new();

        for (index, &case) in cases.iter().enumerate() {
            if let Ok(constant) = i32::try_from(case) {
                if !cases[..index].contains(&case) && value.contains(constant) {
                    branches.push(index as u32);
                }
            }
        }

        if !value.exclude_cases(cases).is_bottom() {
            branches.push(cases.len() as u32);
        }

        Self::from_values(branches)
    }

    /// Returns the possible branch indices when comparing an unsigned integer with `cases`.
    /// The default branch has index `cases.len()`; duplicate cases select their first occurrence.
    pub fn abstract_from_case_u32(value: &AbstractU32, cases: &[BranchCase]) -> Self {
        let mut branches = Vec::new();

        for (index, &case) in cases.iter().enumerate() {
            if let Ok(constant) = u32::try_from(case) {
                if !cases[..index].contains(&case) && value.contains(constant) {
                    branches.push(index as u32);
                }
            }
        }

        if !value.exclude_cases(cases).is_bottom() {
            branches.push(cases.len() as u32);
        }

        Self::from_values(branches)
    }

    /// Returns a new impossible "bottom" value.
    pub fn bottom() -> Self {
        Self::Set(Vec::new())
    }

    /// Returns the represented branch index if this abstract value contains exactly one value,
    /// `None` otherwise.
    pub fn to_singleton(&self) -> Option<u32> {
        match self {
            Self::Set(values) if values.len() == 1 => Some(values[0]),
            _ => None,
        }
    }

    /// Returns the constrained values if this abstract value represents a constrained set, `None`
    /// if it is "top".
    ///
    /// "Bottom" returns an empty slice.
    pub fn values(&self) -> Option<&[u32]> {
        match self {
            Self::Top => None,
            Self::Set(values) => Some(values),
        }
    }

    /// Returns whether the value is an unconstrained "top" value.
    ///
    /// See also [AbstractValue::is_top](super::AbstractValue::is_top).
    pub fn is_top(&self) -> bool {
        matches!(self, Self::Top)
    }

    /// Returns whether the value is an impossible "bottom" value.
    ///
    /// See also [AbstractValue::is_bottom](super::AbstractValue::is_bottom).
    pub fn is_bottom(&self) -> bool {
        matches!(self, Self::Set(values) if values.is_empty())
    }

    /// Returns `true` if this abstract value's constraints are compatible with `value`, `false`
    /// otherwise.
    ///
    /// See also [AbstractValue::contains](super::AbstractValue::contains).
    pub fn contains(&self, value: u32) -> bool {
        match self {
            Self::Top => true,
            Self::Set(values) => values.binary_search(&value).is_ok(),
        }
    }

    /// Returns a new abstract value representing the union of `self` and `other`.
    ///
    /// See also [AbstractValue::join](super::AbstractValue::join).
    pub fn join(&self, other: &Self) -> Self {
        match (self, other) {
            (Self::Top, _) | (_, Self::Top) => Self::Top,
            (Self::Set(left), Self::Set(right)) => {
                Self::from_values(left.iter().copied().chain(right.iter().copied()))
            }
        }
    }

    /// Returns a new abstract value representing the intersection of `self` and `other`.
    ///
    /// See also [AbstractValue::refine](super::AbstractValue::refine).
    pub fn refine(&self, other: &Self) -> Self {
        match (self, other) {
            (Self::Top, value) | (value, Self::Top) => value.clone(),
            (Self::Set(left), Self::Set(right)) => Self::from_values(
                left.iter()
                    .copied()
                    .filter(|value| right.binary_search(value).is_ok()),
            ),
        }
    }

    /// Returns `true` if there is no overlap between value sets representable by both operands,
    /// `false` otherwise.
    ///
    /// See also [AbstractValue::is_disjoint](super::AbstractValue::is_disjoint).
    pub fn is_disjoint(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Top, Self::Top) => false,
            (Self::Top, Self::Set(values)) | (Self::Set(values), Self::Top) => values.is_empty(),
            (Self::Set(left), Self::Set(right)) => {
                !left.iter().any(|value| right.binary_search(value).is_ok())
            }
        }
    }

    /// Returns `true` if the value set representable by `self` is a subset of the value set
    /// representable by `other`, `false` otherwise.
    ///
    /// See also [AbstractValue::is_subset](super::AbstractValue::is_subset).
    pub fn is_subset(&self, other: &Self) -> bool {
        match (self, other) {
            (_, Self::Top) => true,
            (Self::Top, Self::Set(_)) => false,
            (Self::Set(left), Self::Set(right)) => {
                left.iter().all(|value| right.binary_search(value).is_ok())
            }
        }
    }

    /// Returns the boolean values that can select the constrained branches.
    pub fn abstract_to_bool(&self) -> AbstractBool {
        let Self::Set(branches) = self else {
            return AbstractBool::Top;
        };

        let mut result = AbstractBool::Bottom;

        for &branch in branches {
            let value = match branch {
                0 => AbstractBool::Const(true),
                1 => AbstractBool::Const(false),
                _ => panic!("invalid Boolean selector index"),
            };
            result = result.join(&value);
        }

        result
    }

    /// Returns the `i32` values compatible with the abstract conversion of this value to an
    /// `i32` value using the given `cases` list, and compatible with the initial constraints on the
    /// target value as represented by `value`.
    ///
    /// We pass in the original constraints on the target as an argument here, rather than first
    /// deriving the constraints implied by the conversion of the predicate value, then intersecting
    /// the values afterward (with [AbstractI32::refine]) for increased precision: occasionally,
    /// producing the constraints implied by the conversion in isolation may require more integer
    /// intervals than refining the pre-existing constraints directly. If this causes us to exceed
    /// the maximum interval budget, we would need to fall back to a conservative overapproximation.
    /// Involving the prior constraints on the target value allows us to avoid this fallback in some
    /// instances.
    pub fn abstract_to_i32(&self, value: &AbstractI32, cases: &[BranchCase]) -> AbstractI32 {
        let Self::Set(branches) = self else {
            return value.clone();
        };

        let mut result = AbstractI32::bottom();

        for &branch in branches {
            assert!(
                branch as usize <= cases.len(),
                "invalid case selector index"
            );

            let branch_value = if let Some(&case) = cases.get(branch as usize) {
                if cases[..branch as usize].contains(&case) {
                    AbstractI32::bottom()
                } else if let Ok(constant) = i32::try_from(case) {
                    value.refine(&AbstractI32::from_constant(constant))
                } else {
                    AbstractI32::bottom()
                }
            } else {
                value.exclude_cases(cases)
            };

            result = result.join(&branch_value);
        }

        value.refine(&result)
    }

    /// Returns the `u32` values compatible with the abstract conversion of this value to an
    /// `u32` value using the given `cases` list, and compatible with the initial constraints on the
    /// target value as represented by `value`.
    ///
    /// We pass in the original constraints on the target as an argument here, rather than first
    /// deriving the constraints implied by the conversion of the predicate value, then intersecting
    /// the values afterward (with [AbstractU32::refine]) for increased precision: occasionally,
    /// producing the constraints implied by the conversion in isolation may require more integer
    /// intervals than refining the pre-existing constraints directly. If this causes us to exceed
    /// the maximum interval budget, we would need to fall back to a conservative overapproximation.
    /// Involving the prior constraints on the target value allows us to avoid this fallback in some
    /// instances.
    pub fn abstract_to_u32(&self, value: &AbstractU32, cases: &[BranchCase]) -> AbstractU32 {
        let Self::Set(branches) = self else {
            return value.clone();
        };

        let mut result = AbstractU32::bottom();

        for &branch in branches {
            assert!(
                branch as usize <= cases.len(),
                "invalid case selector index"
            );

            let branch_value = if let Some(&case) = cases.get(branch as usize) {
                if cases[..branch as usize].contains(&case) {
                    AbstractU32::bottom()
                } else if let Ok(constant) = u32::try_from(case) {
                    value.refine(&AbstractU32::from_constant(constant))
                } else {
                    AbstractU32::bottom()
                }
            } else {
                value.exclude_cases(cases)
            };

            result = result.join(&branch_value);
        }

        value.refine(&result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abstract_from_bool() {
        assert_eq!(
            AbstractPredicate::abstract_from_bool(&AbstractBool::Top),
            AbstractPredicate::from_values([0, 1])
        );
        assert_eq!(
            AbstractPredicate::abstract_from_bool(&AbstractBool::Const(true)),
            AbstractPredicate::from_constant(0)
        );
        assert_eq!(
            AbstractPredicate::abstract_from_bool(&AbstractBool::Const(false)),
            AbstractPredicate::from_constant(1)
        );
        assert_eq!(
            AbstractPredicate::abstract_from_bool(&AbstractBool::Bottom),
            AbstractPredicate::bottom()
        );
    }

    #[test]
    fn abstract_from_case_i32() {
        let cases = [
            BranchCase::from(-1i32),
            BranchCase::from(1i32),
            BranchCase::from(-1i32),
            BranchCase::from(u32::MAX as u128 + 1),
        ];

        assert_eq!(
            AbstractPredicate::abstract_from_case_i32(&AbstractI32::from_constant(-1), &cases),
            AbstractPredicate::from_constant(0)
        );
        assert_eq!(
            AbstractPredicate::abstract_from_case_i32(
                &AbstractI32::from_intervals([-1..=2]),
                &cases
            ),
            AbstractPredicate::from_values([0, 1, 4])
        );
        assert_eq!(
            AbstractPredicate::abstract_from_case_i32(&AbstractI32::bottom(), &cases),
            AbstractPredicate::bottom()
        );
        assert_eq!(
            AbstractPredicate::abstract_from_case_i32(
                &AbstractI32::top(),
                &[BranchCase::from(i32::MIN)],
            ),
            AbstractPredicate::from_values([0, 1])
        );
        assert_eq!(
            AbstractPredicate::abstract_from_case_i32(
                &AbstractI32::from_intervals([0..=4]),
                &[0i32, 1, 2, 3, 4].map(BranchCase::from),
            ),
            AbstractPredicate::from_values(0..=4)
        );
    }

    #[test]
    fn abstract_from_case_u32() {
        let cases = [
            BranchCase::from(1u32),
            BranchCase::from(3u32),
            BranchCase::from(1u32),
            BranchCase::from(u32::MAX as u128 + 1),
        ];

        assert_eq!(
            AbstractPredicate::abstract_from_case_u32(&AbstractU32::from_constant(1), &cases),
            AbstractPredicate::from_constant(0)
        );
        assert_eq!(
            AbstractPredicate::abstract_from_case_u32(
                &AbstractU32::from_intervals([0..=3]),
                &cases
            ),
            AbstractPredicate::from_values([0, 1, 4])
        );
        assert_eq!(
            AbstractPredicate::abstract_from_case_u32(&AbstractU32::bottom(), &cases),
            AbstractPredicate::bottom()
        );
        assert_eq!(
            AbstractPredicate::abstract_from_case_u32(&AbstractU32::top(), &[]),
            AbstractPredicate::from_constant(0)
        );
        assert_eq!(
            AbstractPredicate::abstract_from_case_u32(
                &AbstractU32::from_intervals([0..=4]),
                &[0u32, 1, 2, 3, 4].map(BranchCase::from),
            ),
            AbstractPredicate::from_values(0..=4)
        );
        assert_eq!(
            AbstractPredicate::abstract_from_case_u32(
                &AbstractU32::from_intervals([0..=10]),
                &[1u32, 3, 5, 7].map(BranchCase::from),
            ),
            AbstractPredicate::from_values(0..=4)
        );
    }

    #[test]
    fn to_singleton() {
        assert_eq!(AbstractPredicate::from_constant(3).to_singleton(), Some(3));
        assert_eq!(AbstractPredicate::from_values([1, 3]).to_singleton(), None);
        assert_eq!(AbstractPredicate::bottom().to_singleton(), None);
        assert_eq!(AbstractPredicate::Top.to_singleton(), None);
    }

    #[test]
    fn values() {
        let value = AbstractPredicate::from_values([4, 1, 4, 3, 1]);

        assert_eq!(value.values(), Some([1, 3, 4].as_slice()));
        assert_eq!(AbstractPredicate::Top.values(), None);
    }

    #[test]
    fn is_top() {
        assert!(AbstractPredicate::Top.is_top());
        assert!(!AbstractPredicate::bottom().is_top());
        assert!(!AbstractPredicate::from_constant(0).is_top());
    }

    #[test]
    fn is_bottom() {
        assert!(AbstractPredicate::bottom().is_bottom());
        assert!(!AbstractPredicate::from_constant(0).is_bottom());
        assert!(!AbstractPredicate::Top.is_bottom());
    }

    #[test]
    fn contains() {
        let value = AbstractPredicate::from_values([1, 3, 5]);

        assert!(value.contains(1));
        assert!(value.contains(5));
        assert!(!value.contains(4));
        assert!(AbstractPredicate::Top.contains(123));
        assert!(!AbstractPredicate::bottom().contains(123));
    }

    #[test]
    fn join() {
        let left = AbstractPredicate::from_values([1, 3, 5]);
        let right = AbstractPredicate::from_values([2, 3, 4]);

        assert_eq!(
            left.join(&right),
            AbstractPredicate::Set(vec![1, 2, 3, 4, 5])
        );
        assert_eq!(left.join(&AbstractPredicate::bottom()), left);
        assert_eq!(AbstractPredicate::bottom().join(&left), left);
        assert_eq!(left.join(&AbstractPredicate::Top), AbstractPredicate::Top);
        assert_eq!(AbstractPredicate::Top.join(&left), AbstractPredicate::Top);
    }

    #[test]
    fn refine() {
        let left = AbstractPredicate::from_values([1, 2, 3, 5]);
        let right = AbstractPredicate::from_values([2, 3, 4]);

        assert_eq!(left.refine(&right), AbstractPredicate::Set(vec![2, 3]));
        assert_eq!(left.refine(&AbstractPredicate::Top), left);
        assert_eq!(AbstractPredicate::Top.refine(&left), left);
        assert_eq!(
            left.refine(&AbstractPredicate::bottom()),
            AbstractPredicate::bottom()
        );
        assert_eq!(
            AbstractPredicate::bottom().refine(&left),
            AbstractPredicate::bottom()
        );
    }

    #[test]
    fn is_disjoint() {
        let value = AbstractPredicate::from_values([1, 3, 5]);

        assert!(!value.is_disjoint(&AbstractPredicate::from_values([3, 4])));
        assert!(value.is_disjoint(&AbstractPredicate::from_values([2, 4])));
        assert!(!AbstractPredicate::Top.is_disjoint(&value));
        assert!(AbstractPredicate::bottom().is_disjoint(&AbstractPredicate::Top));
    }

    #[test]
    fn is_subset() {
        let bottom = AbstractPredicate::bottom();
        let value = AbstractPredicate::from_values([1, 3]);

        assert!(bottom.is_subset(&bottom));
        assert!(bottom.is_subset(&value));
        assert!(value.is_subset(&value));
        assert!(value.is_subset(&AbstractPredicate::from_values([1, 2, 3])));
        assert!(!AbstractPredicate::from_values([1, 2, 3]).is_subset(&value));
        assert!(value.is_subset(&AbstractPredicate::Top));
        assert!(!AbstractPredicate::Top.is_subset(&value));
    }

    #[test]
    fn abstract_to_bool() {
        assert_eq!(AbstractPredicate::Top.abstract_to_bool(), AbstractBool::Top);
        assert_eq!(
            AbstractPredicate::bottom().abstract_to_bool(),
            AbstractBool::Bottom
        );
        assert_eq!(
            AbstractPredicate::from_constant(0).abstract_to_bool(),
            AbstractBool::Const(true)
        );
        assert_eq!(
            AbstractPredicate::from_constant(1).abstract_to_bool(),
            AbstractBool::Const(false)
        );
        assert_eq!(
            AbstractPredicate::from_values([0, 1]).abstract_to_bool(),
            AbstractBool::Top
        );
    }

    #[test]
    fn abstract_to_i32() {
        let cases = [
            BranchCase::from(-1i32),
            BranchCase::from(2i32),
            BranchCase::from(-1i32),
            BranchCase::from(u32::MAX as u128 + 1),
        ];
        let input = AbstractI32::from_intervals([-2..=3]);

        assert_eq!(
            AbstractPredicate::from_constant(0).abstract_to_i32(&input, &cases),
            AbstractI32::from_constant(-1)
        );
        // Note that branch 2 implies that the input was `-1`, which is inside the prior input
        // constraints, but `-1` will always take branch `0`, so this is still contradictory and we
        // expect "bottom".
        assert_eq!(
            AbstractPredicate::from_constant(2).abstract_to_i32(&input, &cases),
            AbstractI32::bottom()
        );
        // Branch 3 implies a value that cannot be represented by the input type, so branch 3
        // cannot ever be taken, so this is contradictory and we expect "bottom".
        assert_eq!(
            AbstractPredicate::from_constant(3).abstract_to_i32(&input, &cases),
            AbstractI32::bottom()
        );
        assert_eq!(
            AbstractPredicate::from_constant(4).abstract_to_i32(&input, &cases),
            AbstractI32::from_intervals([-2..=-2, 0..=1, 3..=3])
        );
        assert_eq!(
            AbstractPredicate::from_values([0, 4]).abstract_to_i32(&input, &cases),
            AbstractI32::from_intervals([-2..=1, 3..=3])
        );
        assert_eq!(
            AbstractPredicate::Top.abstract_to_i32(&input, &cases),
            input
        );
        assert_eq!(
            AbstractPredicate::bottom().abstract_to_i32(&input, &cases),
            AbstractI32::bottom()
        );
    }

    #[test]
    fn abstract_to_u32() {
        let cases = [
            BranchCase::from(1u32),
            BranchCase::from(3u32),
            BranchCase::from(1u32),
            BranchCase::from(u32::MAX as u128 + 1),
        ];
        let input = AbstractU32::from_intervals([0..=4]);

        assert_eq!(
            AbstractPredicate::from_constant(0).abstract_to_u32(&input, &cases),
            AbstractU32::from_constant(1)
        );
        // Note that branch 2 implies that the input was `1`, which is inside the prior input
        // constraints, but `1` will always take branch `0`, so this is still contradictory.
        assert_eq!(
            AbstractPredicate::from_constant(2).abstract_to_u32(&input, &cases),
            AbstractU32::bottom()
        );
        // Branch 3 implies a value that cannot be represented by the input type, so branch 3
        // cannot ever be taken, so this is contradictory and we expect "bottom".
        assert_eq!(
            AbstractPredicate::from_constant(3).abstract_to_u32(&input, &cases),
            AbstractU32::bottom()
        );
        assert_eq!(
            AbstractPredicate::from_constant(4).abstract_to_u32(&input, &cases),
            AbstractU32::from_intervals([0..=0, 2..=2, 4..=4])
        );
        assert_eq!(
            AbstractPredicate::from_values([0, 4]).abstract_to_u32(&input, &cases),
            AbstractU32::from_intervals([0..=2, 4..=4])
        );
        assert_eq!(
            AbstractPredicate::Top.abstract_to_u32(&input, &cases),
            input
        );
        assert_eq!(
            AbstractPredicate::bottom().abstract_to_u32(&input, &cases),
            AbstractU32::bottom()
        );
        assert_eq!(
            AbstractPredicate::from_constant(0)
                .abstract_to_u32(&AbstractU32::from_constant(2), &cases),
            AbstractU32::bottom()
        );
    }
}
