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
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
