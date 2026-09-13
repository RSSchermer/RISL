use core::hash::{Hash, Hasher};

/// A (possibly constrained) `f32` value.
///
/// See also [AbstractValue](super::AbstractValue).
#[derive(Clone, Copy, Debug)]
pub enum AbstractF32 {
    /// The "top" value in a lattice-theory sense.
    ///
    /// The value is unconstrained, it may be any `f32` bit pattern.
    Top,

    /// The value is constrained to exactly this `f32` bit pattern.
    Const(f32),

    /// The "bottom" value in a lattice-theory sense.
    ///
    /// Represents the constraints leading to a contradiction; the value cannot be any `f32` bit
    /// pattern.
    Bottom,
}

impl AbstractF32 {
    /// Returns a new abstract value constrained to exactly the given `value`'s bit pattern.
    pub fn from_constant(value: f32) -> Self {
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

    /// Returns `true` if this abstract value's constraints are compatible with `value`, `false`
    /// otherwise.
    ///
    /// See also [AbstractValue::contains](super::AbstractValue::contains).
    pub fn contains(&self, value: f32) -> bool {
        match self {
            Self::Top => true,
            Self::Const(constant) => constant.to_bits() == value.to_bits(),
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
            (Self::Const(left), Self::Const(right)) if left.to_bits() == right.to_bits() => *self,
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
            (Self::Const(left), Self::Const(right)) if left.to_bits() == right.to_bits() => *self,
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
            (Self::Const(left), Self::Const(right)) => left.to_bits() != right.to_bits(),
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
            (Self::Const(left), Self::Const(right)) => left.to_bits() == right.to_bits(),
        }
    }
}

impl PartialEq for AbstractF32 {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Top, Self::Top) | (Self::Bottom, Self::Bottom) => true,
            (Self::Const(left), Self::Const(right)) => left.to_bits() == right.to_bits(),
            _ => false,
        }
    }
}

impl Eq for AbstractF32 {}

impl Hash for AbstractF32 {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            Self::Top => 0u8.hash(state),
            Self::Const(value) => {
                1u8.hash(state);
                value.to_bits().hash(state);
            }
            Self::Bottom => 2u8.hash(state),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    const NAN_A: f32 = f32::from_bits(0x7fc0_0001);
    const NAN_B: f32 = f32::from_bits(0x7fc0_0002);

    #[test]
    fn equality_compares_float_bit_patterns() {
        assert_eq!(AbstractF32::Const(NAN_A), AbstractF32::Const(NAN_A));
        assert_ne!(AbstractF32::Const(NAN_A), AbstractF32::Const(NAN_B));
        assert_ne!(AbstractF32::Const(0.0), AbstractF32::Const(-0.0));
    }

    #[test]
    fn hash_uses_float_bit_patterns() {
        let values = HashSet::from([
            AbstractF32::Const(NAN_A),
            AbstractF32::Const(NAN_A),
            AbstractF32::Const(NAN_B),
            AbstractF32::Const(0.0),
            AbstractF32::Const(-0.0),
        ]);

        assert_eq!(values.len(), 4);
    }

    #[test]
    fn is_top() {
        assert!(AbstractF32::Top.is_top());
        assert!(!AbstractF32::Const(0.0).is_top());
        assert!(!AbstractF32::Bottom.is_top());
    }

    #[test]
    fn is_bottom() {
        assert!(AbstractF32::Bottom.is_bottom());
        assert!(!AbstractF32::Const(0.0).is_bottom());
        assert!(!AbstractF32::Top.is_bottom());
    }

    #[test]
    fn contains() {
        assert!(AbstractF32::Top.contains(NAN_A));
        assert!(!AbstractF32::Bottom.contains(0.0));
        assert!(AbstractF32::Const(NAN_A).contains(NAN_A));
        assert!(!AbstractF32::Const(NAN_A).contains(NAN_B));
        assert!(!AbstractF32::Const(0.0).contains(-0.0));
    }

    #[test]
    fn join() {
        assert_eq!(
            AbstractF32::Const(NAN_A).join(&AbstractF32::Const(NAN_A)),
            AbstractF32::Const(NAN_A)
        );
        assert_eq!(
            AbstractF32::Const(NAN_A).join(&AbstractF32::Const(NAN_B)),
            AbstractF32::Top
        );
        assert_eq!(
            AbstractF32::Const(0.0).join(&AbstractF32::Const(-0.0)),
            AbstractF32::Top
        );

        let value = AbstractF32::Const(1.0);

        assert_eq!(value.join(&AbstractF32::Bottom), value);
        assert_eq!(AbstractF32::Bottom.join(&value), value);
        assert_eq!(value.join(&AbstractF32::Top), AbstractF32::Top);
        assert_eq!(AbstractF32::Top.join(&value), AbstractF32::Top);
    }

    #[test]
    fn refine() {
        assert_eq!(
            AbstractF32::Const(NAN_A).refine(&AbstractF32::Const(NAN_A)),
            AbstractF32::Const(NAN_A)
        );
        assert_eq!(
            AbstractF32::Const(NAN_A).refine(&AbstractF32::Const(NAN_B)),
            AbstractF32::Bottom
        );
        assert_eq!(
            AbstractF32::Const(0.0).refine(&AbstractF32::Const(-0.0)),
            AbstractF32::Bottom
        );

        let value = AbstractF32::Const(1.0);

        assert_eq!(value.refine(&AbstractF32::Top), value);
        assert_eq!(AbstractF32::Top.refine(&value), value);
        assert_eq!(value.refine(&AbstractF32::Bottom), AbstractF32::Bottom);
        assert_eq!(AbstractF32::Bottom.refine(&value), AbstractF32::Bottom);
    }

    #[test]
    fn is_disjoint() {
        assert!(!AbstractF32::Const(NAN_A).is_disjoint(&AbstractF32::Const(NAN_A)));
        assert!(AbstractF32::Const(NAN_A).is_disjoint(&AbstractF32::Const(NAN_B)));
        assert!(AbstractF32::Const(0.0).is_disjoint(&AbstractF32::Const(-0.0)));
        assert!(!AbstractF32::Top.is_disjoint(&AbstractF32::Const(NAN_A)));
        assert!(AbstractF32::Bottom.is_disjoint(&AbstractF32::Top));
    }

    #[test]
    fn is_subset() {
        assert!(AbstractF32::Bottom.is_subset(&AbstractF32::Bottom));
        assert!(AbstractF32::Bottom.is_subset(&AbstractF32::Const(NAN_A)));
        assert!(AbstractF32::Const(NAN_A).is_subset(&AbstractF32::Const(NAN_A)));
        assert!(!AbstractF32::Const(NAN_A).is_subset(&AbstractF32::Const(NAN_B)));
        assert!(!AbstractF32::Const(0.0).is_subset(&AbstractF32::Const(-0.0)));
        assert!(AbstractF32::Const(NAN_A).is_subset(&AbstractF32::Top));
        assert!(!AbstractF32::Top.is_subset(&AbstractF32::Const(NAN_A)));
    }
}
