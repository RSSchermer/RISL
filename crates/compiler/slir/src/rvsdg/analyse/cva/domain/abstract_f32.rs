use core::hash::{Hash, Hasher};

use super::{AbstractBool, AbstractI32, AbstractU32};

// There are some discrepancies between floating point arithmetic in Rust and floating point
// arithmetic in WGSL. To avoid such discrepancies, we use this helper function to conservatively
// only apply concrete operations involving `f32` values when it is "safe" to do so, falling back
// to "top" otherwise.
fn is_safe_f32(value: f32) -> bool {
    value.is_normal() || value == 0.0
}

fn concrete_values(value: &AbstractF32) -> Option<&[f32]> {
    const ZERO_VALUES: [f32; 2] = [0.0, -0.0];

    match value {
        AbstractF32::Zero => Some(&ZERO_VALUES),
        AbstractF32::Const(value) if is_safe_f32(*value) => Some(core::slice::from_ref(value)),
        _ => None,
    }
}

fn abstract_binary_operation(
    left: &AbstractF32,
    right: &AbstractF32,
    operation: impl Fn(f32, f32) -> Option<f32>,
) -> AbstractF32 {
    if left.is_bottom() || right.is_bottom() {
        return AbstractF32::Bottom;
    }

    let (Some(left_values), Some(right_values)) = (concrete_values(left), concrete_values(right))
    else {
        return AbstractF32::Top;
    };

    let mut result = AbstractF32::Bottom;

    for left in left_values {
        for right in right_values {
            let Some(concrete_result) =
                operation(*left, *right).filter(|value| is_safe_f32(*value))
            else {
                return AbstractF32::Top;
            };

            result = result.join(&AbstractF32::Const(concrete_result));
        }
    }

    result
}

fn abstract_comparison(
    left: &AbstractF32,
    right: &AbstractF32,
    operation: impl Fn(f32, f32) -> bool,
) -> AbstractBool {
    abstract_binary_operation(left, right, |left, right| {
        Some(if operation(left, right) { 1.0 } else { 0.0 })
    })
    .to_abstract_bool()
}

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

    /// The value is constrained to either positive or negative zero.
    Zero,

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

    /// Returns the abstract boolean value of this floating-point value.
    pub fn to_abstract_bool(&self) -> AbstractBool {
        match self {
            Self::Zero => AbstractBool::Const(false),
            Self::Const(value) if is_safe_f32(*value) => AbstractBool::Const(*value != 0.0),
            Self::Bottom => AbstractBool::Bottom,
            _ => AbstractBool::Top,
        }
    }

    /// Returns the abstract signed integer value of this floating-point value.
    pub fn to_abstract_i32(&self) -> AbstractI32 {
        match self {
            Self::Zero => AbstractI32::from_constant(0),
            Self::Const(value) if is_safe_f32(*value) => AbstractI32::from_constant(*value as i32),
            Self::Bottom => AbstractI32::bottom(),
            _ => AbstractI32::top(),
        }
    }

    /// Returns the abstract unsigned integer value of this floating-point value.
    pub fn to_abstract_u32(&self) -> AbstractU32 {
        match self {
            Self::Zero => AbstractU32::from_constant(0),
            Self::Const(value) if is_safe_f32(*value) => AbstractU32::from_constant(*value as u32),
            Self::Bottom => AbstractU32::bottom(),
            _ => AbstractU32::top(),
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
        matches!(self, Self::Bottom)
    }

    /// Returns `true` if this abstract value's constraints are compatible with `value`, `false`
    /// otherwise.
    ///
    /// See also [AbstractValue::contains](super::AbstractValue::contains).
    pub fn contains(&self, value: f32) -> bool {
        match self {
            Self::Top => true,
            Self::Zero => value == 0.0,
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
            (Self::Zero, Self::Zero) => Self::Zero,
            (Self::Zero, Self::Const(value)) | (Self::Const(value), Self::Zero)
                if *value == 0.0 =>
            {
                Self::Zero
            }
            (Self::Const(left), Self::Const(right)) if left.to_bits() == right.to_bits() => *self,
            // Note that the pattern above will produce a constant when both values are `0.0` or
            // when both values are `-0.0`, so this pattern will only catch the "mixed" zero cases
            // where one value is `0.0` and the other value is `-0.0`.
            (Self::Const(left), Self::Const(right)) if *left == 0.0 && *right == 0.0 => Self::Zero,
            (Self::Const(_), Self::Const(_)) => Self::Top,
            (Self::Zero, Self::Const(_)) | (Self::Const(_), Self::Zero) => Self::Top,
        }
    }

    /// Returns a new abstract value approximating the intersection of `self` and `other`.
    ///
    /// See also [AbstractValue::refine](super::AbstractValue::refine).
    pub fn refine(&self, other: &Self) -> Self {
        match (self, other) {
            (Self::Bottom, _) | (_, Self::Bottom) => Self::Bottom,
            (Self::Top, value) | (value, Self::Top) => *value,
            (Self::Zero, Self::Zero) => Self::Zero,
            (Self::Zero, Self::Const(value)) | (Self::Const(value), Self::Zero)
                if *value == 0.0 =>
            {
                Self::Const(*value)
            }
            (Self::Const(left), Self::Const(right)) if left.to_bits() == right.to_bits() => *self,
            (Self::Const(_), Self::Const(_)) => Self::Bottom,
            (Self::Zero, Self::Const(_)) | (Self::Const(_), Self::Zero) => Self::Bottom,
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
            (Self::Zero, Self::Zero) => false,
            (Self::Zero, Self::Const(value)) | (Self::Const(value), Self::Zero) => *value != 0.0,
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
            (Self::Zero, Self::Zero) => true,
            (Self::Const(value), Self::Zero) => *value == 0.0,
            (Self::Zero, Self::Const(_)) => false,
            (Self::Const(left), Self::Const(right)) => left.to_bits() == right.to_bits(),
        }
    }

    /// Returns the abstract result of negating this value.
    pub fn abstract_neg(&self) -> Self {
        match self {
            Self::Bottom => Self::Bottom,
            Self::Zero => Self::Zero,
            Self::Const(value) if is_safe_f32(*value) => {
                let result = -*value;

                if is_safe_f32(result) {
                    Self::Const(result)
                } else {
                    Self::Top
                }
            }
            _ => Self::Top,
        }
    }

    /// Returns the abstract result of adding this value and `other`.
    pub fn abstract_add(&self, other: &Self) -> Self {
        abstract_binary_operation(self, other, |left, right| Some(left + right))
    }

    /// Returns the operand constraints implied by `res` being the result of adding this value and
    /// `other`.
    ///
    /// Currently, this is a "placeholder" implementation that does not infer any additional
    /// constraints on the operands; the implementation may be "upgraded" later.
    pub fn abstract_add_inv(&self, other: &Self, _res: &Self) -> (Self, Self) {
        (*self, *other)
    }

    /// Returns the abstract result of subtracting `other` from this value.
    pub fn abstract_sub(&self, other: &Self) -> Self {
        abstract_binary_operation(self, other, |left, right| Some(left - right))
    }

    /// Returns the operand constraints implied by `res` being the result of subtracting `other`
    /// from this value.
    ///
    /// Currently, this is a "placeholder" implementation that does not infer any additional
    /// constraints on the operands; the implementation may be "upgraded" later.
    pub fn abstract_sub_inv(&self, other: &Self, _res: &Self) -> (Self, Self) {
        (*self, *other)
    }

    /// Returns the abstract result of multiplying this value by `other`.
    pub fn abstract_mul(&self, other: &Self) -> Self {
        abstract_binary_operation(self, other, |left, right| Some(left * right))
    }

    /// Returns the operand constraints implied by `res` being the result of multiplying this value
    /// by `other`.
    ///
    /// Currently, this is a "placeholder" implementation that does not infer any additional
    /// constraints on the operands; the implementation may be "upgraded" later.
    pub fn abstract_mul_inv(&self, other: &Self, _res: &Self) -> (Self, Self) {
        (*self, *other)
    }

    /// Returns the abstract result of dividing this value by `other`.
    pub fn abstract_div(&self, other: &Self) -> Self {
        abstract_binary_operation(self, other, |left, right| {
            (right != 0.0).then(|| left / right)
        })
    }

    /// Returns the operand constraints implied by `res` being the result of dividing this value by
    /// `other`.
    ///
    /// Currently, this is a "placeholder" implementation that does not infer any additional
    /// constraints on the operands; the implementation may be "upgraded" later.
    pub fn abstract_div_inv(&self, other: &Self, _res: &Self) -> (Self, Self) {
        (*self, *other)
    }

    /// Returns the abstract result of taking the remainder of this value divided by `other`.
    pub fn abstract_mod(&self, other: &Self) -> Self {
        abstract_binary_operation(self, other, |left, right| {
            (right != 0.0).then(|| left % right)
        })
    }

    /// Returns the operand constraints implied by `res` being the remainder of dividing this value
    /// by `other`.
    ///
    /// Currently, this is a "placeholder" implementation that does not infer any additional
    /// constraints on the operands; the implementation may be "upgraded" later.
    pub fn abstract_mod_inv(&self, other: &Self, _res: &Self) -> (Self, Self) {
        (*self, *other)
    }

    /// Returns the abstract result of comparing this value equal to `other`.
    pub fn abstract_eq(&self, other: &Self) -> AbstractBool {
        abstract_comparison(self, other, |left, right| left == right)
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
            AbstractBool::Const(required) => match self.abstract_eq(other) {
                AbstractBool::Const(actual) if actual != *required => (Self::Bottom, Self::Bottom),
                AbstractBool::Bottom => (Self::Bottom, Self::Bottom),
                _ if *required => {
                    let equality_constraint = |other: &Self| match other {
                        Self::Zero => Self::Zero,
                        Self::Const(value) if *value == 0.0 => Self::Zero,
                        Self::Const(value) if is_safe_f32(*value) => Self::Const(*value),
                        Self::Bottom => Self::Bottom,
                        _ => Self::Top,
                    };

                    (
                        self.refine(&equality_constraint(other)),
                        other.refine(&equality_constraint(self)),
                    )
                }
                _ => (*self, *other),
            },
            AbstractBool::Top => (*self, *other),
            AbstractBool::Bottom => (Self::Bottom, Self::Bottom),
        };

        if refinements.0.is_bottom() || refinements.1.is_bottom() {
            (Self::Bottom, Self::Bottom)
        } else {
            refinements
        }
    }

    /// Returns the abstract result of comparing this value not equal to `other`.
    pub fn abstract_not_eq(&self, other: &Self) -> AbstractBool {
        abstract_comparison(self, other, |left, right| left != right)
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
        abstract_comparison(self, other, |left, right| left < right)
    }

    /// Returns the operand constraints implied by `res` being the result of comparing this value
    /// less than `other`.
    ///
    /// Currently, this is a "placeholder" implementation that does not infer any additional
    /// constraints on the operands; the implementation may be "upgraded" later.
    pub fn abstract_lt_inv(&self, other: &Self, _res: &AbstractBool) -> (Self, Self) {
        (*self, *other)
    }

    /// Returns the abstract result of comparing this value less than or equal to `other`.
    pub fn abstract_lt_eq(&self, other: &Self) -> AbstractBool {
        abstract_comparison(self, other, |left, right| left <= right)
    }

    /// Returns the operand constraints implied by `res` being the result of comparing this value
    /// less than or equal to `other`.
    ///
    /// Currently, this is a "placeholder" implementation that does not infer any additional
    /// constraints on the operands; the implementation may be "upgraded" later.
    pub fn abstract_lt_eq_inv(&self, other: &Self, _res: &AbstractBool) -> (Self, Self) {
        (*self, *other)
    }

    /// Returns the abstract result of comparing this value greater than `other`.
    pub fn abstract_gt(&self, other: &Self) -> AbstractBool {
        abstract_comparison(self, other, |left, right| left > right)
    }

    /// Returns the operand constraints implied by `res` being the result of comparing this value
    /// greater than `other`.
    ///
    /// Currently, this is a "placeholder" implementation that does not infer any additional
    /// constraints on the operands; the implementation may be "upgraded" later.
    pub fn abstract_gt_inv(&self, other: &Self, _res: &AbstractBool) -> (Self, Self) {
        (*self, *other)
    }

    /// Returns the abstract result of comparing this value greater than or equal to `other`.
    pub fn abstract_gt_eq(&self, other: &Self) -> AbstractBool {
        abstract_comparison(self, other, |left, right| left >= right)
    }

    /// Returns the operand constraints implied by `res` being the result of comparing this value
    /// greater than or equal to `other`.
    ///
    /// Currently, this is a "placeholder" implementation that does not infer any additional
    /// constraints on the operands; the implementation may be "upgraded" later.
    pub fn abstract_gt_eq_inv(&self, other: &Self, _res: &AbstractBool) -> (Self, Self) {
        (*self, *other)
    }
}

impl PartialEq for AbstractF32 {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Top, Self::Top) | (Self::Zero, Self::Zero) | (Self::Bottom, Self::Bottom) => {
                true
            }
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
            Self::Zero => 2u8.hash(state),
            Self::Bottom => 3u8.hash(state),
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
        assert_ne!(AbstractF32::Zero, AbstractF32::Const(0.0));
    }

    #[test]
    fn hash_uses_float_bit_patterns() {
        let values = HashSet::from([
            AbstractF32::Const(NAN_A),
            AbstractF32::Const(NAN_A),
            AbstractF32::Const(NAN_B),
            AbstractF32::Const(0.0),
            AbstractF32::Const(-0.0),
            AbstractF32::Zero,
        ]);

        assert_eq!(values.len(), 5);
    }

    #[test]
    fn to_abstract_bool() {
        assert_eq!(
            AbstractF32::from_constant(2.5).to_abstract_bool(),
            AbstractBool::Const(true)
        );
        assert_eq!(
            AbstractF32::from_constant(0.0).to_abstract_bool(),
            AbstractBool::Const(false)
        );
        assert_eq!(
            AbstractF32::from_constant(-0.0).to_abstract_bool(),
            AbstractBool::Const(false)
        );
        assert_eq!(
            AbstractF32::Zero.to_abstract_bool(),
            AbstractBool::Const(false)
        );
        assert_eq!(
            AbstractF32::from_constant(f32::MIN_POSITIVE / 2.0).to_abstract_bool(),
            AbstractBool::Top
        );
        assert_eq!(
            AbstractF32::from_constant(NAN_A).to_abstract_bool(),
            AbstractBool::Top
        );
        assert_eq!(AbstractF32::Top.to_abstract_bool(), AbstractBool::Top);
        assert_eq!(AbstractF32::Bottom.to_abstract_bool(), AbstractBool::Bottom);
    }

    #[test]
    fn to_abstract_i32() {
        assert_eq!(
            AbstractF32::Zero.to_abstract_i32(),
            AbstractI32::from_constant(0)
        );
        assert_eq!(
            AbstractF32::from_constant(-2.5).to_abstract_i32(),
            AbstractI32::from_constant(-2)
        );
        assert_eq!(
            AbstractF32::from_constant(f32::MAX).to_abstract_i32(),
            AbstractI32::from_constant(i32::MAX)
        );
        assert_eq!(
            AbstractF32::from_constant(f32::MIN_POSITIVE / 2.0).to_abstract_i32(),
            AbstractI32::top()
        );
        assert_eq!(
            AbstractF32::from_constant(NAN_A).to_abstract_i32(),
            AbstractI32::top()
        );
        assert_eq!(AbstractF32::Top.to_abstract_i32(), AbstractI32::top());
        assert_eq!(AbstractF32::Bottom.to_abstract_i32(), AbstractI32::bottom());
    }

    #[test]
    fn to_abstract_u32() {
        assert_eq!(
            AbstractF32::Zero.to_abstract_u32(),
            AbstractU32::from_constant(0)
        );
        assert_eq!(
            AbstractF32::from_constant(2.5).to_abstract_u32(),
            AbstractU32::from_constant(2)
        );
        assert_eq!(
            AbstractF32::from_constant(-1.0).to_abstract_u32(),
            AbstractU32::from_constant(0)
        );
        assert_eq!(
            AbstractF32::from_constant(f32::MAX).to_abstract_u32(),
            AbstractU32::from_constant(u32::MAX)
        );
        assert_eq!(
            AbstractF32::from_constant(f32::MIN_POSITIVE / 2.0).to_abstract_u32(),
            AbstractU32::top()
        );
        assert_eq!(
            AbstractF32::from_constant(NAN_A).to_abstract_u32(),
            AbstractU32::top()
        );
        assert_eq!(AbstractF32::Top.to_abstract_u32(), AbstractU32::top());
        assert_eq!(AbstractF32::Bottom.to_abstract_u32(), AbstractU32::bottom());
    }

    #[test]
    fn is_top() {
        assert!(AbstractF32::Top.is_top());
        assert!(!AbstractF32::Zero.is_top());
        assert!(!AbstractF32::Const(0.0).is_top());
        assert!(!AbstractF32::Bottom.is_top());
    }

    #[test]
    fn is_bottom() {
        assert!(AbstractF32::Bottom.is_bottom());
        assert!(!AbstractF32::Zero.is_bottom());
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
        assert!(AbstractF32::Zero.contains(0.0));
        assert!(AbstractF32::Zero.contains(-0.0));
        assert!(!AbstractF32::Zero.contains(1.0));
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
            AbstractF32::Zero
        );
        assert_eq!(
            AbstractF32::Zero.join(&AbstractF32::Const(0.0)),
            AbstractF32::Zero
        );
        assert_eq!(
            AbstractF32::Const(-0.0).join(&AbstractF32::Zero),
            AbstractF32::Zero
        );
        assert_eq!(
            AbstractF32::Zero.join(&AbstractF32::Const(1.0)),
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
        assert_eq!(
            AbstractF32::Zero.refine(&AbstractF32::Zero),
            AbstractF32::Zero
        );
        assert_eq!(
            AbstractF32::Zero.refine(&AbstractF32::Const(-0.0)),
            AbstractF32::Const(-0.0)
        );
        assert_eq!(
            AbstractF32::Const(0.0).refine(&AbstractF32::Zero),
            AbstractF32::Const(0.0)
        );
        assert_eq!(
            AbstractF32::Zero.refine(&AbstractF32::Const(1.0)),
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
        assert!(!AbstractF32::Zero.is_disjoint(&AbstractF32::Const(0.0)));
        assert!(!AbstractF32::Zero.is_disjoint(&AbstractF32::Const(-0.0)));
        assert!(AbstractF32::Zero.is_disjoint(&AbstractF32::Const(1.0)));
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
        assert!(AbstractF32::Const(0.0).is_subset(&AbstractF32::Zero));
        assert!(AbstractF32::Const(-0.0).is_subset(&AbstractF32::Zero));
        assert!(AbstractF32::Zero.is_subset(&AbstractF32::Zero));
        assert!(!AbstractF32::Zero.is_subset(&AbstractF32::Const(0.0)));
        assert!(AbstractF32::Const(NAN_A).is_subset(&AbstractF32::Top));
        assert!(!AbstractF32::Top.is_subset(&AbstractF32::Const(NAN_A)));
    }

    #[test]
    fn abstract_neg() {
        assert_eq!(AbstractF32::Zero.abstract_neg(), AbstractF32::Zero);
        assert_eq!(
            AbstractF32::Const(2.0).abstract_neg(),
            AbstractF32::Const(-2.0)
        );
        assert_eq!(
            AbstractF32::Const(0.0).abstract_neg(),
            AbstractF32::Const(-0.0)
        );
        assert_eq!(
            AbstractF32::Const(f32::INFINITY).abstract_neg(),
            AbstractF32::Top
        );
        assert_eq!(AbstractF32::Top.abstract_neg(), AbstractF32::Top);
        assert_eq!(AbstractF32::Bottom.abstract_neg(), AbstractF32::Bottom);
    }

    #[test]
    fn abstract_add() {
        assert_eq!(
            AbstractF32::Zero.abstract_add(&AbstractF32::Zero),
            AbstractF32::Zero
        );
        assert_eq!(
            AbstractF32::Const(1.5).abstract_add(&AbstractF32::Const(2.0)),
            AbstractF32::Const(3.5)
        );
        assert_eq!(
            AbstractF32::Const(1.5).abstract_add(&AbstractF32::Zero),
            AbstractF32::Const(1.5)
        );
        assert_eq!(
            AbstractF32::Const(f32::MAX).abstract_add(&AbstractF32::Const(f32::MAX)),
            AbstractF32::Top
        );
        assert_eq!(
            AbstractF32::Const(1.0).abstract_add(&AbstractF32::Top),
            AbstractF32::Top
        );
        assert_eq!(
            AbstractF32::Bottom.abstract_add(&AbstractF32::Const(1.0)),
            AbstractF32::Bottom
        );
    }

    #[test]
    fn abstract_sub() {
        assert_eq!(
            AbstractF32::Zero.abstract_sub(&AbstractF32::Zero),
            AbstractF32::Zero
        );
        assert_eq!(
            AbstractF32::Const(3.5).abstract_sub(&AbstractF32::Const(2.0)),
            AbstractF32::Const(1.5)
        );
        assert_eq!(
            AbstractF32::Const(3.5).abstract_sub(&AbstractF32::Zero),
            AbstractF32::Const(3.5)
        );

        let next_normal = f32::from_bits(f32::MIN_POSITIVE.to_bits() + 1);

        assert_eq!(
            AbstractF32::Const(f32::MIN_POSITIVE).abstract_sub(&AbstractF32::Const(next_normal)),
            AbstractF32::Top
        );
        assert_eq!(
            AbstractF32::Top.abstract_sub(&AbstractF32::Const(1.0)),
            AbstractF32::Top
        );
        assert_eq!(
            AbstractF32::Const(1.0).abstract_sub(&AbstractF32::Bottom),
            AbstractF32::Bottom
        );
    }

    #[test]
    fn abstract_mul() {
        assert_eq!(
            AbstractF32::Zero.abstract_mul(&AbstractF32::Zero),
            AbstractF32::Zero
        );
        assert_eq!(
            AbstractF32::Zero.abstract_mul(&AbstractF32::Const(-2.0)),
            AbstractF32::Zero
        );
        assert_eq!(
            AbstractF32::Const(1.5).abstract_mul(&AbstractF32::Const(2.0)),
            AbstractF32::Const(3.0)
        );
        assert_eq!(
            AbstractF32::Const(f32::MIN_POSITIVE).abstract_mul(&AbstractF32::Const(0.5)),
            AbstractF32::Top
        );
        assert_eq!(
            AbstractF32::Top.abstract_mul(&AbstractF32::Const(0.0)),
            AbstractF32::Top
        );
        assert_eq!(
            AbstractF32::Bottom.abstract_mul(&AbstractF32::Const(0.0)),
            AbstractF32::Bottom
        );
    }

    #[test]
    fn abstract_div() {
        assert_eq!(
            AbstractF32::Zero.abstract_div(&AbstractF32::Const(-2.0)),
            AbstractF32::Zero
        );
        assert_eq!(
            AbstractF32::Zero.abstract_div(&AbstractF32::Zero),
            AbstractF32::Top
        );
        assert_eq!(
            AbstractF32::Const(3.0).abstract_div(&AbstractF32::Const(2.0)),
            AbstractF32::Const(1.5)
        );
        assert_eq!(
            AbstractF32::Const(f32::MIN_POSITIVE).abstract_div(&AbstractF32::Const(2.0)),
            AbstractF32::Top
        );
        assert_eq!(
            AbstractF32::Const(1.0).abstract_div(&AbstractF32::Const(0.0)),
            AbstractF32::Top
        );
        assert_eq!(
            AbstractF32::Const(1.0).abstract_div(&AbstractF32::Const(-0.0)),
            AbstractF32::Top
        );
        assert_eq!(
            AbstractF32::Top.abstract_div(&AbstractF32::Const(1.0)),
            AbstractF32::Top
        );
        assert_eq!(
            AbstractF32::Const(1.0).abstract_div(&AbstractF32::Bottom),
            AbstractF32::Bottom
        );
    }

    #[test]
    fn abstract_mod() {
        assert_eq!(
            AbstractF32::Zero.abstract_mod(&AbstractF32::Const(2.0)),
            AbstractF32::Zero
        );
        assert_eq!(
            AbstractF32::Const(1.0).abstract_mod(&AbstractF32::Zero),
            AbstractF32::Top
        );
        assert_eq!(
            AbstractF32::Const(5.5).abstract_mod(&AbstractF32::Const(2.0)),
            AbstractF32::Const(1.5)
        );

        let next_normal = f32::from_bits(f32::MIN_POSITIVE.to_bits() + 1);

        assert_eq!(
            AbstractF32::Const(next_normal).abstract_mod(&AbstractF32::Const(f32::MIN_POSITIVE)),
            AbstractF32::Top
        );
        assert_eq!(
            AbstractF32::Const(1.0).abstract_mod(&AbstractF32::Const(0.0)),
            AbstractF32::Top
        );
        assert_eq!(
            AbstractF32::Const(1.0).abstract_mod(&AbstractF32::Const(-0.0)),
            AbstractF32::Top
        );
        assert_eq!(
            AbstractF32::Top.abstract_mod(&AbstractF32::Const(1.0)),
            AbstractF32::Top
        );
        assert_eq!(
            AbstractF32::Bottom.abstract_mod(&AbstractF32::Const(1.0)),
            AbstractF32::Bottom
        );
    }

    #[test]
    fn abstract_eq() {
        assert_eq!(
            AbstractF32::Zero.abstract_eq(&AbstractF32::Zero),
            AbstractBool::Const(true)
        );
        assert_eq!(
            AbstractF32::Zero.abstract_eq(&AbstractF32::Const(0.0)),
            AbstractBool::Const(true)
        );
        assert_eq!(
            AbstractF32::Zero.abstract_eq(&AbstractF32::Const(-0.0)),
            AbstractBool::Const(true)
        );
        assert_eq!(
            AbstractF32::Zero.abstract_eq(&AbstractF32::Const(1.0)),
            AbstractBool::Const(false)
        );
        assert_eq!(
            AbstractF32::Const(1.0).abstract_eq(&AbstractF32::Const(1.0)),
            AbstractBool::Const(true)
        );
        assert_eq!(
            AbstractF32::Const(1.0).abstract_eq(&AbstractF32::Const(2.0)),
            AbstractBool::Const(false)
        );
        assert_eq!(
            AbstractF32::Const(0.0).abstract_eq(&AbstractF32::Const(-0.0)),
            AbstractBool::Const(true)
        );
        assert_eq!(
            AbstractF32::Const(NAN_A).abstract_eq(&AbstractF32::Const(NAN_A)),
            AbstractBool::Top
        );
        assert_eq!(
            AbstractF32::Top.abstract_eq(&AbstractF32::Const(1.0)),
            AbstractBool::Top
        );
        assert_eq!(
            AbstractF32::Bottom.abstract_eq(&AbstractF32::Const(1.0)),
            AbstractBool::Bottom
        );
    }

    #[test]
    fn abstract_eq_inv() {
        assert_eq!(
            AbstractF32::Top.abstract_eq_inv(&AbstractF32::Const(1.0), &AbstractBool::Const(true)),
            (AbstractF32::Const(1.0), AbstractF32::Const(1.0))
        );
        assert_eq!(
            AbstractF32::Const(1.0).abstract_eq_inv(&AbstractF32::Top, &AbstractBool::Const(false)),
            (AbstractF32::Const(1.0), AbstractF32::Top)
        );
        assert_eq!(
            AbstractF32::Const(1.0)
                .abstract_eq_inv(&AbstractF32::Const(2.0), &AbstractBool::Const(true)),
            (AbstractF32::Bottom, AbstractF32::Bottom)
        );
        assert_eq!(
            AbstractF32::Const(1.0)
                .abstract_eq_inv(&AbstractF32::Const(1.0), &AbstractBool::Const(false)),
            (AbstractF32::Bottom, AbstractF32::Bottom)
        );
        assert_eq!(
            AbstractF32::Top.abstract_eq_inv(&AbstractF32::Const(0.0), &AbstractBool::Const(true)),
            (AbstractF32::Zero, AbstractF32::Const(0.0))
        );
        assert_eq!(
            AbstractF32::Top.abstract_eq_inv(&AbstractF32::Const(-0.0), &AbstractBool::Const(true)),
            (AbstractF32::Zero, AbstractF32::Const(-0.0))
        );
        assert_eq!(
            AbstractF32::Top.abstract_eq_inv(&AbstractF32::Const(0.0), &AbstractBool::Const(false)),
            (AbstractF32::Top, AbstractF32::Const(0.0))
        );
        assert_eq!(
            AbstractF32::Top.abstract_eq_inv(&AbstractF32::Zero, &AbstractBool::Const(true)),
            (AbstractF32::Zero, AbstractF32::Zero)
        );
        assert_eq!(
            AbstractF32::Const(0.0)
                .abstract_eq_inv(&AbstractF32::Const(-0.0), &AbstractBool::Const(true)),
            (AbstractF32::Const(0.0), AbstractF32::Const(-0.0))
        );
        assert_eq!(
            AbstractF32::Const(0.0)
                .abstract_eq_inv(&AbstractF32::Const(-0.0), &AbstractBool::Const(false)),
            (AbstractF32::Bottom, AbstractF32::Bottom)
        );
        assert_eq!(
            AbstractF32::Zero
                .abstract_eq_inv(&AbstractF32::Const(0.0), &AbstractBool::Const(false)),
            (AbstractF32::Bottom, AbstractF32::Bottom)
        );
        assert_eq!(
            AbstractF32::Top
                .abstract_eq_inv(&AbstractF32::Const(NAN_A), &AbstractBool::Const(true)),
            (AbstractF32::Top, AbstractF32::Const(NAN_A))
        );
        assert_eq!(
            AbstractF32::Top.abstract_eq_inv(&AbstractF32::Top, &AbstractBool::Bottom),
            (AbstractF32::Bottom, AbstractF32::Bottom)
        );
        assert_eq!(
            AbstractF32::Top.abstract_eq_inv(&AbstractF32::Bottom, &AbstractBool::Top),
            (AbstractF32::Bottom, AbstractF32::Bottom)
        );
    }

    #[test]
    fn abstract_not_eq() {
        assert_eq!(
            AbstractF32::Zero.abstract_not_eq(&AbstractF32::Const(0.0)),
            AbstractBool::Const(false)
        );
        assert_eq!(
            AbstractF32::Zero.abstract_not_eq(&AbstractF32::Const(-0.0)),
            AbstractBool::Const(false)
        );
        assert_eq!(
            AbstractF32::Zero.abstract_not_eq(&AbstractF32::Const(1.0)),
            AbstractBool::Const(true)
        );
        assert_eq!(
            AbstractF32::Const(1.0).abstract_not_eq(&AbstractF32::Const(1.0)),
            AbstractBool::Const(false)
        );
        assert_eq!(
            AbstractF32::Const(1.0).abstract_not_eq(&AbstractF32::Const(2.0)),
            AbstractBool::Const(true)
        );
        assert_eq!(
            AbstractF32::Const(0.0).abstract_not_eq(&AbstractF32::Const(-0.0)),
            AbstractBool::Const(false)
        );
        assert_eq!(
            AbstractF32::Const(f32::INFINITY).abstract_not_eq(&AbstractF32::Const(f32::INFINITY)),
            AbstractBool::Top
        );
        assert_eq!(
            AbstractF32::Const(1.0).abstract_not_eq(&AbstractF32::Top),
            AbstractBool::Top
        );
        assert_eq!(
            AbstractF32::Const(1.0).abstract_not_eq(&AbstractF32::Bottom),
            AbstractBool::Bottom
        );
    }

    #[test]
    fn abstract_not_eq_inv() {
        assert_eq!(
            AbstractF32::Top
                .abstract_not_eq_inv(&AbstractF32::Const(1.0), &AbstractBool::Const(false)),
            (AbstractF32::Const(1.0), AbstractF32::Const(1.0))
        );
        assert_eq!(
            AbstractF32::Const(1.0)
                .abstract_not_eq_inv(&AbstractF32::Top, &AbstractBool::Const(true)),
            (AbstractF32::Const(1.0), AbstractF32::Top)
        );
        assert_eq!(
            AbstractF32::Const(1.0)
                .abstract_not_eq_inv(&AbstractF32::Const(2.0), &AbstractBool::Const(false)),
            (AbstractF32::Bottom, AbstractF32::Bottom)
        );
        assert_eq!(
            AbstractF32::Top
                .abstract_not_eq_inv(&AbstractF32::Const(-0.0), &AbstractBool::Const(false)),
            (AbstractF32::Zero, AbstractF32::Const(-0.0))
        );
        assert_eq!(
            AbstractF32::Const(0.0)
                .abstract_not_eq_inv(&AbstractF32::Const(-0.0), &AbstractBool::Const(true)),
            (AbstractF32::Bottom, AbstractF32::Bottom)
        );
        assert_eq!(
            AbstractF32::Top.abstract_not_eq_inv(&AbstractF32::Top, &AbstractBool::Bottom),
            (AbstractF32::Bottom, AbstractF32::Bottom)
        );
    }

    #[test]
    fn abstract_lt() {
        assert_eq!(
            AbstractF32::Zero.abstract_lt(&AbstractF32::Zero),
            AbstractBool::Const(false)
        );
        assert_eq!(
            AbstractF32::Zero.abstract_lt(&AbstractF32::Const(1.0)),
            AbstractBool::Const(true)
        );
        assert_eq!(
            AbstractF32::Const(1.0).abstract_lt(&AbstractF32::Const(2.0)),
            AbstractBool::Const(true)
        );
        assert_eq!(
            AbstractF32::Const(2.0).abstract_lt(&AbstractF32::Const(1.0)),
            AbstractBool::Const(false)
        );
        assert_eq!(
            AbstractF32::Const(f32::MIN_POSITIVE / 2.0).abstract_lt(&AbstractF32::Const(1.0)),
            AbstractBool::Top
        );
        assert_eq!(
            AbstractF32::Top.abstract_lt(&AbstractF32::Const(1.0)),
            AbstractBool::Top
        );
        assert_eq!(
            AbstractF32::Bottom.abstract_lt(&AbstractF32::Const(1.0)),
            AbstractBool::Bottom
        );
    }

    #[test]
    fn abstract_lt_eq() {
        assert_eq!(
            AbstractF32::Zero.abstract_lt_eq(&AbstractF32::Const(-0.0)),
            AbstractBool::Const(true)
        );
        assert_eq!(
            AbstractF32::Const(1.0).abstract_lt_eq(&AbstractF32::Const(1.0)),
            AbstractBool::Const(true)
        );
        assert_eq!(
            AbstractF32::Const(2.0).abstract_lt_eq(&AbstractF32::Const(1.0)),
            AbstractBool::Const(false)
        );
        assert_eq!(
            AbstractF32::Const(-0.0).abstract_lt_eq(&AbstractF32::Const(0.0)),
            AbstractBool::Const(true)
        );
        assert_eq!(
            AbstractF32::Const(1.0).abstract_lt_eq(&AbstractF32::Const(NAN_A)),
            AbstractBool::Top
        );
        assert_eq!(
            AbstractF32::Const(1.0).abstract_lt_eq(&AbstractF32::Top),
            AbstractBool::Top
        );
        assert_eq!(
            AbstractF32::Const(1.0).abstract_lt_eq(&AbstractF32::Bottom),
            AbstractBool::Bottom
        );
    }

    #[test]
    fn abstract_gt() {
        assert_eq!(
            AbstractF32::Zero.abstract_gt(&AbstractF32::Const(1.0)),
            AbstractBool::Const(false)
        );
        assert_eq!(
            AbstractF32::Const(2.0).abstract_gt(&AbstractF32::Const(1.0)),
            AbstractBool::Const(true)
        );
        assert_eq!(
            AbstractF32::Const(1.0).abstract_gt(&AbstractF32::Const(2.0)),
            AbstractBool::Const(false)
        );
        assert_eq!(
            AbstractF32::Const(0.0).abstract_gt(&AbstractF32::Const(-0.0)),
            AbstractBool::Const(false)
        );
        assert_eq!(
            AbstractF32::Const(NAN_A).abstract_gt(&AbstractF32::Const(1.0)),
            AbstractBool::Top
        );
        assert_eq!(
            AbstractF32::Top.abstract_gt(&AbstractF32::Const(1.0)),
            AbstractBool::Top
        );
        assert_eq!(
            AbstractF32::Bottom.abstract_gt(&AbstractF32::Const(1.0)),
            AbstractBool::Bottom
        );
    }

    #[test]
    fn abstract_gt_eq() {
        assert_eq!(
            AbstractF32::Zero.abstract_gt_eq(&AbstractF32::Const(0.0)),
            AbstractBool::Const(true)
        );
        assert_eq!(
            AbstractF32::Const(1.0).abstract_gt_eq(&AbstractF32::Const(1.0)),
            AbstractBool::Const(true)
        );
        assert_eq!(
            AbstractF32::Const(1.0).abstract_gt_eq(&AbstractF32::Const(2.0)),
            AbstractBool::Const(false)
        );
        assert_eq!(
            AbstractF32::Const(0.0).abstract_gt_eq(&AbstractF32::Const(-0.0)),
            AbstractBool::Const(true)
        );
        assert_eq!(
            AbstractF32::Const(f32::NEG_INFINITY).abstract_gt_eq(&AbstractF32::Const(1.0)),
            AbstractBool::Top
        );
        assert_eq!(
            AbstractF32::Const(1.0).abstract_gt_eq(&AbstractF32::Top),
            AbstractBool::Top
        );
        assert_eq!(
            AbstractF32::Const(1.0).abstract_gt_eq(&AbstractF32::Bottom),
            AbstractBool::Bottom
        );
    }
}
