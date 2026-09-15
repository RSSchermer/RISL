mod abstract_bool;
mod abstract_f32;
mod abstract_i32;
mod abstract_predicate;
mod abstract_u32;

pub use self::abstract_bool::AbstractBool;
pub use self::abstract_f32::AbstractF32;
pub use self::abstract_i32::AbstractI32;
pub use self::abstract_predicate::AbstractPredicate;
pub use self::abstract_u32::AbstractU32;
use crate::rvsdg::analyse::scalar_constant::ScalarConstant;
use crate::ty::{TY_BOOL, TY_F32, TY_I32, TY_PREDICATE, TY_U32, Type};

/// The maximum number of disjoint intervals retained by an integer-type abstract value.
pub const MAX_INTEGER_INTERVALS: usize = 4;

/// A possibly constrained typed value.
///
/// We use these values when reasoning about value constraints. We call the value "abstract" as
/// opposed to "concrete": we cannot always constrain a value to a single constant concrete value
/// (although we sometimes can).
///
/// We only reason about value constraints on a limit set of supported scalar types: `bool`, `f32`,
/// `i32`, `u32`, and `predicate`. All other types use the [`Unsupported`](Self::Unsupported)
/// variant. This variant always represents an unconstrained value of its type (the "top" value in a
/// lattice-theory sense).
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum AbstractValue {
    Bool(AbstractBool),
    F32(AbstractF32),
    I32(AbstractI32),
    U32(AbstractU32),
    Predicate(AbstractPredicate),
    Unsupported(Type),
}

impl AbstractValue {
    /// Returns an unconstrained value of the given type.
    ///
    /// This is the "top" value in the lattice-theory sense: constraint-reasoning cannot use the
    /// fact represented by this [AbstractValue] to rule out any values of the given type.
    ///
    /// If the type is not one of the types with explicit support (see [AbstractValue]), then this
    /// will return the [`Unsupported`](Self::Unsupported) variant; all values of this variant are
    /// always considered "top".
    pub fn top(ty: Type) -> Self {
        match ty {
            TY_BOOL => Self::Bool(AbstractBool::Top),
            TY_F32 => Self::F32(AbstractF32::Top),
            TY_I32 => Self::I32(AbstractI32::top()),
            TY_U32 => Self::U32(AbstractU32::top()),
            TY_PREDICATE => Self::Predicate(AbstractPredicate::Top),
            _ => Self::Unsupported(ty),
        }
    }

    /// Returns an "impossible" value of the given type.
    ///
    /// This is the "bottom" value in the lattice-theory sense. It represents the fact that the
    /// accumulated constraints on a value have led to a contradiction.
    ///
    /// "Bottom" values can validly occur. For example, encountering a "bottom" value inside a
    /// switch branch may indicate that the branch is unreachable, potentially allowing us to
    /// simplify the switch node.
    ///
    /// # Panics
    ///
    /// Panics if `ty` is not one of the supported abstract value types (see [AbstractValue]).
    /// Unsupported types are always unconstrained, so any constraint-reasoning that arrives at a
    /// contradiction that implies the value is "impossible" must contain a bug.
    pub fn bottom(ty: Type) -> Self {
        match ty {
            TY_BOOL => Self::Bool(AbstractBool::Bottom),
            TY_F32 => Self::F32(AbstractF32::Bottom),
            TY_I32 => Self::I32(AbstractI32::bottom()),
            TY_U32 => Self::U32(AbstractU32::bottom()),
            TY_PREDICATE => Self::Predicate(AbstractPredicate::Set(Vec::new())),
            _ => panic!("unsupported type cannot be 'bottom'"),
        }
    }

    /// Returns the singleton abstract value containing `constant`.
    ///
    /// This represents the fact that the abstract value is constrained all the way to a single
    /// value.
    pub fn from_scalar_constant(constant: ScalarConstant) -> Self {
        match constant {
            ScalarConstant::Bool(value) => Self::Bool(AbstractBool::from_constant(value)),
            ScalarConstant::F32(bits) => {
                Self::F32(AbstractF32::from_constant(f32::from_bits(bits)))
            }
            ScalarConstant::I32(value) => Self::I32(AbstractI32::from_constant(value)),
            ScalarConstant::U32(value) => Self::U32(AbstractU32::from_constant(value)),
            ScalarConstant::Predicate(value) => {
                Self::Predicate(AbstractPredicate::from_constant(value))
            }
        }
    }

    /// Returns the value's type.
    pub fn ty(&self) -> Type {
        match self {
            Self::Bool(_) => TY_BOOL,
            Self::F32(_) => TY_F32,
            Self::I32(_) => TY_I32,
            Self::U32(_) => TY_U32,
            Self::Predicate(_) => TY_PREDICATE,
            Self::Unsupported(ty) => *ty,
        }
    }

    /// Returns this value as an [`AbstractBool`].
    ///
    /// # Panics
    ///
    /// Panics if this is not a [`Bool`](Self::Bool) value.
    pub fn expect_bool(&self) -> &AbstractBool {
        let Self::Bool(value) = self else {
            panic!("expected bool abstract value");
        };

        value
    }

    /// Returns this value as an [`AbstractF32`].
    ///
    /// # Panics
    ///
    /// Panics if this is not an [`F32`](Self::F32) value.
    pub fn expect_f32(&self) -> &AbstractF32 {
        let Self::F32(value) = self else {
            panic!("expected f32 abstract value");
        };

        value
    }

    /// Returns this value as an [`AbstractI32`].
    ///
    /// # Panics
    ///
    /// Panics if this is not an [`I32`](Self::I32) value.
    pub fn expect_i32(&self) -> &AbstractI32 {
        let Self::I32(value) = self else {
            panic!("expected i32 abstract value");
        };

        value
    }

    /// Returns this value as an [`AbstractU32`].
    ///
    /// # Panics
    ///
    /// Panics if this is not a [`U32`](Self::U32) value.
    pub fn expect_u32(&self) -> &AbstractU32 {
        let Self::U32(value) = self else {
            panic!("expected u32 abstract value");
        };

        value
    }

    /// Returns whether the value is an unconstrained "top" value.
    ///
    /// If a value is unconstrained, then reasoning cannot use this fact to rule out any values
    /// compatible with the type. For example, a boolean value can still be both `true` and `false`.
    ///
    /// If this value is of the [Unsupported](Self::Unsupported) variant, then the value is always
    /// considered unconstrained, and therefore this method always returns `true` for such values.
    pub fn is_top(&self) -> bool {
        match self {
            Self::Bool(value) => value.is_top(),
            Self::F32(value) => value.is_top(),
            Self::I32(value) => value.is_top(),
            Self::U32(value) => value.is_top(),
            Self::Predicate(value) => value.is_top(),
            Self::Unsupported(_) => true,
        }
    }

    /// Returns whether the value is an impossible "bottom" value.
    ///
    /// This represents that constraint-reasoning has led to an "impossible" contradiction. For
    /// example, we may conclude that a boolean-type value can neither be `true` nor `false`.
    ///
    /// "Bottom" values can validly occur. For example, encountering a "bottom" value inside a
    /// switch branch may indicate that the branch is unreachable, potentially allowing us to
    /// simplify the switch node.
    ///
    /// Values of the [Unsupported](Self::Unsupported) variant are always considered unconstrained
    /// and thus can never lead to a contradiction; this method therefore always returns `false`
    /// for such values.
    pub fn is_bottom(&self) -> bool {
        match self {
            Self::Bool(value) => value.is_bottom(),
            Self::F32(value) => value.is_bottom(),
            Self::I32(value) => value.is_bottom(),
            Self::U32(value) => value.is_bottom(),
            Self::Predicate(value) => value.is_bottom(),
            Self::Unsupported(_) => false,
        }
    }

    /// Returns `true` if this abstract value's constraints are compatible with `constant`, `false`
    /// otherwise.
    ///
    /// If the constant type does not match the abstract value's type, this will always return
    /// `false`.
    ///
    /// Floating-point comparisons conservatively use exact bit-pattern matching rather than IEEE
    /// equality, as GPU equality is not guaranteed to be fully IEEE-compliant.
    pub fn contains(&self, constant: ScalarConstant) -> bool {
        match (self, constant) {
            (Self::Bool(value), ScalarConstant::Bool(constant)) => value.contains(constant),
            (Self::F32(value), ScalarConstant::F32(bits)) => value.contains(f32::from_bits(bits)),
            (Self::I32(value), ScalarConstant::I32(constant)) => value.contains(constant),
            (Self::U32(value), ScalarConstant::U32(constant)) => value.contains(constant),
            (Self::Predicate(value), ScalarConstant::Predicate(constant)) => {
                value.contains(constant)
            }
            _ => false,
        }
    }

    /// Returns a new abstract value representing at least the union of `self` and `other`.
    ///
    /// The exact union is returned where possible. Integer unions requiring more than
    /// [`MAX_INTEGER_INTERVALS`] intervals and unions of differing `f32` constants conservatively
    /// produce "top".
    ///
    /// # Panics
    ///
    /// Panics if the operands have different types.
    pub fn join(&self, other: &Self) -> Self {
        assert_eq!(self.ty(), other.ty(), "abstract value types must match");

        match (self, other) {
            (Self::Bool(left), Self::Bool(right)) => Self::Bool(left.join(right)),
            (Self::F32(left), Self::F32(right)) => Self::F32(left.join(right)),
            (Self::I32(left), Self::I32(right)) => Self::I32(left.join(right)),
            (Self::U32(left), Self::U32(right)) => Self::U32(left.join(right)),
            (Self::Predicate(left), Self::Predicate(right)) => Self::Predicate(left.join(right)),
            (Self::Unsupported(ty), Self::Unsupported(_)) => Self::Unsupported(*ty),
            _ => unreachable!("equal types have matching abstract value variants"),
        }
    }

    /// Returns a new abstract value approximating the intersection of `self` and `other`.
    ///
    /// The exact intersection is returned where possible. For integer-type values, if an exact
    /// intersection requires more than [`MAX_INTEGER_INTERVALS`] intervals, a clone of `self` is
    /// returned as a conservative overapproximation. As a consequence, the resulting abstract value
    /// will be "underconstrained". This should never lead to incorrect compilation, only
    /// (potentially) to missed simplifications/optimizations. Note that this means that this
    /// operation is not commutative for integer-type abstract values; swapping `self` and `other`
    /// may give different results.
    ///
    /// # Panics
    ///
    /// Panics if the operands have different types.
    pub fn refine(&self, other: &Self) -> Self {
        assert_eq!(self.ty(), other.ty(), "abstract value types must match");

        match (self, other) {
            (Self::Bool(left), Self::Bool(right)) => Self::Bool(left.refine(right)),
            (Self::F32(left), Self::F32(right)) => Self::F32(left.refine(right)),
            (Self::I32(left), Self::I32(right)) => Self::I32(left.refine(right)),
            (Self::U32(left), Self::U32(right)) => Self::U32(left.refine(right)),
            (Self::Predicate(left), Self::Predicate(right)) => Self::Predicate(left.refine(right)),
            (Self::Unsupported(ty), Self::Unsupported(_)) => Self::Unsupported(*ty),
            _ => unreachable!("equal types have matching abstract value variants"),
        }
    }

    /// Returns `true` if there is no overlap between value sets representable by both operands,
    /// `false` otherwise.
    ///
    /// Unsupported values are never considered disjoint.
    ///
    /// # Panics
    ///
    /// Panics if the operands have different types.
    pub fn is_disjoint(&self, other: &Self) -> bool {
        assert_eq!(self.ty(), other.ty(), "abstract value types must match");

        match (self, other) {
            (Self::Bool(left), Self::Bool(right)) => left.is_disjoint(right),
            (Self::F32(left), Self::F32(right)) => left.is_disjoint(right),
            (Self::I32(left), Self::I32(right)) => left.is_disjoint(right),
            (Self::U32(left), Self::U32(right)) => left.is_disjoint(right),
            (Self::Predicate(left), Self::Predicate(right)) => left.is_disjoint(right),
            (Self::Unsupported(_), Self::Unsupported(_)) => false,
            _ => unreachable!("equal types have matching abstract value variants"),
        }
    }

    /// Returns `true` if the value set representable by `self` is a subset of the value set
    /// representable by `other`, `false` otherwise.
    ///
    /// # Panics
    ///
    /// Panics if the operands have different types.
    pub fn is_subset(&self, other: &Self) -> bool {
        assert_eq!(self.ty(), other.ty(), "abstract value types must match");

        match (self, other) {
            (Self::Bool(left), Self::Bool(right)) => left.is_subset(right),
            (Self::F32(left), Self::F32(right)) => left.is_subset(right),
            (Self::I32(left), Self::I32(right)) => left.is_subset(right),
            (Self::U32(left), Self::U32(right)) => left.is_subset(right),
            (Self::Predicate(left), Self::Predicate(right)) => left.is_subset(right),
            (Self::Unsupported(_), Self::Unsupported(_)) => true,
            _ => unreachable!("equal types have matching abstract value variants"),
        }
    }

    /// Returns the represented scalar constant if this abstract value can be
    /// represented as one, or `None` otherwise.
    ///
    /// Top, bottom, non-singleton values, and unsupported values return `None`.
    pub fn to_scalar_constant(&self) -> Option<ScalarConstant> {
        match self {
            Self::Bool(AbstractBool::Const(value)) => Some(ScalarConstant::Bool(*value)),
            Self::F32(AbstractF32::Const(value)) => Some(ScalarConstant::F32(value.to_bits())),
            Self::I32(value) => value.to_singleton().map(ScalarConstant::I32),
            Self::U32(value) => value.to_singleton().map(ScalarConstant::U32),
            Self::Predicate(value) => value.to_singleton().map(ScalarConstant::Predicate),
            _ => None,
        }
    }
}

macro_rules! impl_from_abstract_value {
    ($type:ty, $variant:ident) => {
        impl From<$type> for AbstractValue {
            fn from(value: $type) -> Self {
                Self::$variant(value)
            }
        }
    };
}

impl_from_abstract_value!(AbstractBool, Bool);
impl_from_abstract_value!(AbstractF32, F32);
impl_from_abstract_value!(AbstractI32, I32);
impl_from_abstract_value!(AbstractU32, U32);
impl_from_abstract_value!(AbstractPredicate, Predicate);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_constant_round_trip() {
        let constants = [
            ScalarConstant::Bool(true),
            ScalarConstant::F32(1.0f32.to_bits()),
            ScalarConstant::I32(-1),
            ScalarConstant::U32(1),
            ScalarConstant::Predicate(1),
        ];
        let values = constants.map(AbstractValue::from_scalar_constant);

        for (constant, value) in constants.into_iter().zip(values) {
            assert_eq!(value.to_scalar_constant(), Some(constant));
        }
    }
}
