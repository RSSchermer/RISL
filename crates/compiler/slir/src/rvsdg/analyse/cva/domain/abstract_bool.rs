/// A (possibly constrained) boolean value.
///
/// See also [AbstractValue](super::AbstractValue).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum AbstractBool {
    /// The "top" value in a lattice-theory sense.
    ///
    /// The value is unconstrained, it may be either `true` or `false`.
    Top,

    /// The value is constrained to exactly this boolean value.
    Const(bool),

    /// The "bottom" value in a lattice-theory sense.
    ///
    /// Represents the constraints leading to a contradiction; the value can neither be `true` nor
    /// `false`.
    Bottom,
}

impl AbstractBool {
    /// Returns a new abstract value constrained to exactly the given `value`.
    pub fn from_constant(value: bool) -> Self {
        Self::Const(value)
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
        matches!(self, Self::Bottom)
    }

    /// Returns `true` if this abstract value's constraints are compatible with `constant`, `false`
    /// otherwise.
    ///
    /// See also [AbstractValue::contains](super::AbstractValue::contains).
    pub fn contains(&self, value: bool) -> bool {
        match self {
            Self::Top => true,
            Self::Const(constant) => *constant == value,
            Self::Bottom => false,
        }
    }

    /// Returns a new abstract value representing at least the union of `self` and `other`.
    ///
    /// See also [AbstractValue::join](super::AbstractValue::join).
    pub fn join(&self, other: &Self) -> Self {
        match (self, other) {
            (Self::Top, _) | (_, Self::Top) => Self::Top,
            (Self::Bottom, value) | (value, Self::Bottom) => *value,
            (Self::Const(left), Self::Const(right)) if left == right => *self,
            (Self::Const(_), Self::Const(_)) => Self::Top,
        }
    }

    /// Returns a new abstract value approximating the intersection of `self` and `other`.
    ///
    /// See also [AbstractValue::refine](super::AbstractValue::refine).
    pub fn refine(&self, other: &Self) -> Self {
        match (self, other) {
            (Self::Bottom, _) | (_, Self::Bottom) => Self::Bottom,
            (Self::Top, value) | (value, Self::Top) => *value,
            (Self::Const(left), Self::Const(right)) if left == right => *self,
            (Self::Const(_), Self::Const(_)) => Self::Bottom,
        }
    }

    /// Returns `true` if there is no overlap between value sets representable by both operands,
    /// `false` otherwise.
    ///
    /// See also [AbstractValue::is_disjoint](super::AbstractValue::is_disjoint).
    pub fn is_disjoint(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Bottom, _) | (_, Self::Bottom) => true,
            (Self::Top, _) | (_, Self::Top) => false,
            (Self::Const(left), Self::Const(right)) => left != right,
        }
    }

    /// Returns `true` if the value set representable by `self` is a subset of the value set
    /// representable by `other`, `false` otherwise.
    ///
    /// See also [AbstractValue::is_subset](super::AbstractValue::is_subset).
    pub fn is_subset(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Bottom, _) | (_, Self::Top) => true,
            (Self::Top, _) | (_, Self::Bottom) => false,
            (Self::Const(left), Self::Const(right)) => left == right,
        }
    }

    /// Returns the abstract result of logically negating this value.
    pub fn abstract_not(&self) -> Self {
        match self {
            Self::Top => Self::Top,
            Self::Const(value) => Self::Const(!value),
            Self::Bottom => Self::Bottom,
        }
    }

    /// Returns the abstract result of logically AND-ing this value with `other`.
    pub fn abstract_and(&self, other: &Self) -> Self {
        match (self, other) {
            (Self::Bottom, _) | (_, Self::Bottom) => Self::Bottom,
            (Self::Const(false), _) | (_, Self::Const(false)) => Self::Const(false),
            (Self::Const(true), value) | (value, Self::Const(true)) => *value,
            (Self::Top, Self::Top) => Self::Top,
        }
    }

    /// Returns the abstract result of logically OR-ing this value with `other`.
    pub fn abstract_or(&self, other: &Self) -> Self {
        match (self, other) {
            (Self::Bottom, _) | (_, Self::Bottom) => Self::Bottom,
            (Self::Const(true), _) | (_, Self::Const(true)) => Self::Const(true),
            (Self::Const(false), value) | (value, Self::Const(false)) => *value,
            (Self::Top, Self::Top) => Self::Top,
        }
    }

    /// Returns the abstract result of comparing this value equal to `other`.
    pub fn abstract_eq(&self, other: &Self) -> Self {
        match (self, other) {
            (Self::Bottom, _) | (_, Self::Bottom) => Self::Bottom,
            (Self::Const(left), Self::Const(right)) => Self::Const(left == right),
            _ => Self::Top,
        }
    }

    /// Returns the abstract result of comparing this value not equal to `other`.
    pub fn abstract_not_eq(&self, other: &Self) -> Self {
        self.abstract_eq(other).abstract_not()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_top() {
        assert!(AbstractBool::Top.is_top());
        assert!(!AbstractBool::Const(false).is_top());
        assert!(!AbstractBool::Bottom.is_top());
    }

    #[test]
    fn is_bottom() {
        assert!(AbstractBool::Bottom.is_bottom());
        assert!(!AbstractBool::Const(false).is_bottom());
        assert!(!AbstractBool::Top.is_bottom());
    }

    #[test]
    fn contains() {
        assert!(AbstractBool::Top.contains(false));
        assert!(AbstractBool::Top.contains(true));
        assert!(!AbstractBool::Bottom.contains(false));
        assert!(!AbstractBool::Bottom.contains(true));
        assert!(AbstractBool::Const(true).contains(true));
        assert!(!AbstractBool::Const(true).contains(false));
    }

    #[test]
    fn join() {
        assert_eq!(
            AbstractBool::Const(true).join(&AbstractBool::Const(true)),
            AbstractBool::Const(true)
        );
        assert_eq!(
            AbstractBool::Const(true).join(&AbstractBool::Const(false)),
            AbstractBool::Top
        );

        let value = AbstractBool::Const(true);

        assert_eq!(value.join(&AbstractBool::Bottom), value);
        assert_eq!(AbstractBool::Bottom.join(&value), value);
        assert_eq!(value.join(&AbstractBool::Top), AbstractBool::Top);
        assert_eq!(AbstractBool::Top.join(&value), AbstractBool::Top);
    }

    #[test]
    fn refine() {
        assert_eq!(
            AbstractBool::Const(false).refine(&AbstractBool::Const(false)),
            AbstractBool::Const(false)
        );
        assert_eq!(
            AbstractBool::Const(false).refine(&AbstractBool::Const(true)),
            AbstractBool::Bottom
        );

        let value = AbstractBool::Const(false);

        assert_eq!(value.refine(&AbstractBool::Top), value);
        assert_eq!(AbstractBool::Top.refine(&value), value);
        assert_eq!(value.refine(&AbstractBool::Bottom), AbstractBool::Bottom);
        assert_eq!(AbstractBool::Bottom.refine(&value), AbstractBool::Bottom);
    }

    #[test]
    fn is_disjoint() {
        assert!(!AbstractBool::Const(true).is_disjoint(&AbstractBool::Const(true)));
        assert!(AbstractBool::Const(true).is_disjoint(&AbstractBool::Const(false)));
        assert!(!AbstractBool::Top.is_disjoint(&AbstractBool::Const(false)));
        assert!(AbstractBool::Bottom.is_disjoint(&AbstractBool::Top));
    }

    #[test]
    fn is_subset() {
        assert!(AbstractBool::Bottom.is_subset(&AbstractBool::Bottom));
        assert!(AbstractBool::Bottom.is_subset(&AbstractBool::Const(true)));
        assert!(AbstractBool::Const(true).is_subset(&AbstractBool::Const(true)));
        assert!(!AbstractBool::Const(true).is_subset(&AbstractBool::Const(false)));
        assert!(AbstractBool::Const(true).is_subset(&AbstractBool::Top));
        assert!(!AbstractBool::Top.is_subset(&AbstractBool::Const(true)));
    }

    #[test]
    fn abstract_not() {
        assert_eq!(
            AbstractBool::Const(false).abstract_not(),
            AbstractBool::Const(true)
        );
        assert_eq!(
            AbstractBool::Const(true).abstract_not(),
            AbstractBool::Const(false)
        );
        assert_eq!(AbstractBool::Top.abstract_not(), AbstractBool::Top);
        assert_eq!(AbstractBool::Bottom.abstract_not(), AbstractBool::Bottom);
    }

    #[test]
    fn abstract_and() {
        assert_eq!(
            AbstractBool::Const(true).abstract_and(&AbstractBool::Const(true)),
            AbstractBool::Const(true)
        );
        assert_eq!(
            AbstractBool::Const(true).abstract_and(&AbstractBool::Const(false)),
            AbstractBool::Const(false)
        );
        assert_eq!(
            AbstractBool::Const(false).abstract_and(&AbstractBool::Top),
            AbstractBool::Const(false)
        );
        assert_eq!(
            AbstractBool::Const(true).abstract_and(&AbstractBool::Top),
            AbstractBool::Top
        );
        assert_eq!(
            AbstractBool::Top.abstract_and(&AbstractBool::Const(false)),
            AbstractBool::Const(false)
        );
        assert_eq!(
            AbstractBool::Top.abstract_and(&AbstractBool::Const(true)),
            AbstractBool::Top
        );
        assert_eq!(
            AbstractBool::Top.abstract_and(&AbstractBool::Top),
            AbstractBool::Top
        );
        assert_eq!(
            AbstractBool::Const(false).abstract_and(&AbstractBool::Bottom),
            AbstractBool::Bottom
        );
        assert_eq!(
            AbstractBool::Bottom.abstract_and(&AbstractBool::Const(false)),
            AbstractBool::Bottom
        );
    }

    #[test]
    fn abstract_or() {
        assert_eq!(
            AbstractBool::Const(false).abstract_or(&AbstractBool::Const(false)),
            AbstractBool::Const(false)
        );
        assert_eq!(
            AbstractBool::Const(false).abstract_or(&AbstractBool::Const(true)),
            AbstractBool::Const(true)
        );
        assert_eq!(
            AbstractBool::Const(true).abstract_or(&AbstractBool::Top),
            AbstractBool::Const(true)
        );
        assert_eq!(
            AbstractBool::Const(false).abstract_or(&AbstractBool::Top),
            AbstractBool::Top
        );
        assert_eq!(
            AbstractBool::Top.abstract_or(&AbstractBool::Const(true)),
            AbstractBool::Const(true)
        );
        assert_eq!(
            AbstractBool::Top.abstract_or(&AbstractBool::Const(false)),
            AbstractBool::Top
        );
        assert_eq!(
            AbstractBool::Top.abstract_or(&AbstractBool::Top),
            AbstractBool::Top
        );
        assert_eq!(
            AbstractBool::Const(true).abstract_or(&AbstractBool::Bottom),
            AbstractBool::Bottom
        );
        assert_eq!(
            AbstractBool::Bottom.abstract_or(&AbstractBool::Const(true)),
            AbstractBool::Bottom
        );
    }

    #[test]
    fn abstract_eq() {
        assert_eq!(
            AbstractBool::Const(true).abstract_eq(&AbstractBool::Const(true)),
            AbstractBool::Const(true)
        );
        assert_eq!(
            AbstractBool::Const(true).abstract_eq(&AbstractBool::Const(false)),
            AbstractBool::Const(false)
        );
        assert_eq!(
            AbstractBool::Const(true).abstract_eq(&AbstractBool::Top),
            AbstractBool::Top
        );
        assert_eq!(
            AbstractBool::Top.abstract_eq(&AbstractBool::Top),
            AbstractBool::Top
        );
        assert_eq!(
            AbstractBool::Const(true).abstract_eq(&AbstractBool::Bottom),
            AbstractBool::Bottom
        );
        assert_eq!(
            AbstractBool::Bottom.abstract_eq(&AbstractBool::Const(true)),
            AbstractBool::Bottom
        );
    }

    #[test]
    fn abstract_not_eq() {
        assert_eq!(
            AbstractBool::Const(true).abstract_not_eq(&AbstractBool::Const(true)),
            AbstractBool::Const(false)
        );
        assert_eq!(
            AbstractBool::Const(true).abstract_not_eq(&AbstractBool::Const(false)),
            AbstractBool::Const(true)
        );
        assert_eq!(
            AbstractBool::Const(true).abstract_not_eq(&AbstractBool::Top),
            AbstractBool::Top
        );
        assert_eq!(
            AbstractBool::Top.abstract_not_eq(&AbstractBool::Top),
            AbstractBool::Top
        );
        assert_eq!(
            AbstractBool::Const(true).abstract_not_eq(&AbstractBool::Bottom),
            AbstractBool::Bottom
        );
        assert_eq!(
            AbstractBool::Bottom.abstract_not_eq(&AbstractBool::Const(true)),
            AbstractBool::Bottom
        );
    }
}
