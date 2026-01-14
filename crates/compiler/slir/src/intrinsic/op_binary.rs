use serde::{Deserialize, Serialize};

use crate::BinaryOperator;
use crate::BinaryOperator::{
    Add, And, BitAnd, BitOr, BitXor, Div, Eq, Gt, GtEq, Lt, LtEq, Mod, Mul, NotEq, Or, Shl, Shr,
    Sub,
};
use crate::intrinsic::Intrinsic;
use crate::ty::{Matrix, ScalarKind, TY_BOOL, TY_U32, Type, TypeKind, TypeRegistry, Vector};

#[derive(Clone, PartialEq, Serialize, Deserialize, Debug)]
pub struct OpBinary {
    pub operator: BinaryOperator,
}

impl Intrinsic for OpBinary {
    fn process_args(
        &self,
        ty_registry: &TypeRegistry,
        args: impl IntoIterator<Item = Type>,
    ) -> Result<Option<Type>, String> {
        use BinaryOperator::*;

        let op = self.operator;

        let mut args = args.into_iter();

        let Some(lhs) = args.next() else {
            return Err("a binary operation expects exactly two arguments, found none".to_string());
        };
        let Some(rhs) = args.next() else {
            return Err("a binary operation expects exactly two arguments, found one".to_string());
        };
        if args.next().is_some() {
            return Err("a binary operation expects exactly two arguments, found more".to_string());
        }

        let return_ty = match op {
            And | Or => check_logic_op(ty_registry, op, lhs, rhs)?,
            Add | Sub => check_add_or_sub_op(ty_registry, op, lhs, rhs)?,
            Mul => check_mul_op(ty_registry, lhs, rhs)?,
            Div | Mod => check_div_or_mod_op(ty_registry, op, lhs, rhs)?,
            Shl | Shr => check_shift_op(ty_registry, op, lhs, rhs)?,
            Eq | NotEq => check_eq_op(ty_registry, op, lhs, rhs)?,
            Gt | GtEq | Lt | LtEq => check_ord_op(ty_registry, op, lhs, rhs)?,
            BitOr | BitAnd | BitXor => check_bit_op(ty_registry, op, lhs, rhs)?,
        };

        Ok(Some(return_ty))
    }
}

fn check_logic_op(
    ty: &TypeRegistry,
    op: BinaryOperator,
    lhs: Type,
    rhs: Type,
) -> Result<Type, String> {
    if lhs == TY_BOOL && rhs == TY_BOOL {
        Ok(TY_BOOL)
    } else {
        Err(format!(
            "the `{}` operator expects boolean operands (lhs: `{}`, rhs: `{}`)",
            op,
            lhs.to_string(ty),
            rhs.to_string(ty)
        ))
    }
}

fn check_add_or_sub_op(
    ty: &TypeRegistry,
    op: BinaryOperator,
    lhs: Type,
    rhs: Type,
) -> Result<Type, String> {
    match &*ty.kind(lhs) {
        TypeKind::Scalar(kind) if kind.is_numeric() => match &*ty.kind(rhs) {
            TypeKind::Scalar(kind) if kind == kind => Ok(lhs),
            TypeKind::Vector(v) if &v.scalar == kind => Ok(rhs),
            TypeKind::Matrix(m) if &m.scalar == kind => Ok(rhs),
            _ => Err(format!(
                "if the left-hand-side operand to the `{}` operator is a `{}` value, then the \
                right-hand-side value must be a numeric scalar, vector or matrix of the same \
                (element) type (got `{}`)",
                op,
                lhs.to_string(ty),
                rhs.to_string(ty)
            )),
        },
        TypeKind::Vector(v) if v.scalar.is_numeric() => match &*ty.kind(rhs) {
            TypeKind::Scalar(kind) if &v.scalar == kind => Ok(lhs),
            TypeKind::Vector(other) if v.scalar == other.scalar && v.size == other.size => Ok(lhs),
            _ => Err(format!(
                "if the left-hand-side operand to the `{}` operator is a `{}` vector, then the \
                right-hand-side value must be a vector of the same type, or a numeric scalar that \
                matches the element type of the vector (got `{}`)",
                op,
                lhs.to_string(ty),
                rhs.to_string(ty)
            )),
        },
        TypeKind::Matrix(m) if m.scalar.is_numeric() => match &*ty.kind(rhs) {
            TypeKind::Scalar(kind) if &m.scalar == kind => Ok(lhs),
            TypeKind::Matrix(other)
                if m.rows == other.rows
                    && m.columns == other.columns
                    && m.scalar == other.scalar =>
            {
                Ok(lhs)
            }
            _ => Err(format!(
                "if the left-hand-side operand to the `{}` operator is a `{}` matrix, then the \
                right-hand-side value must be a matrix of the same size and type, or a numeric \
                scalar that matches the element type of the matrix (got `{}`)",
                op,
                lhs.to_string(ty),
                rhs.to_string(ty)
            )),
        },
        _ => Err(format!(
            "the `{}` operator expects a numeric scalar, a numeric vector or a matrix as its \
            left-hand-side operand (got `{}`)",
            op,
            lhs.to_string(ty)
        )),
    }
}

fn check_mul_op(ty: &TypeRegistry, lhs: Type, rhs: Type) -> Result<Type, String> {
    match &*ty.kind(lhs) {
        TypeKind::Scalar(kind) if kind.is_numeric() => match &*ty.kind(rhs) {
            TypeKind::Scalar(kind) if kind == kind => Ok(lhs),
            TypeKind::Vector(v) if &v.scalar == kind => Ok(rhs),
            TypeKind::Matrix(m) if &m.scalar == kind => Ok(rhs),
            _ => Err(format!(
                "if the left-hand-side operand to the `*` operator is a `{}` value, then the \
                right-hand-side value must be a numeric scalar, vector or matrix of the same \
                (element) type (got `{}`)",
                lhs.to_string(ty),
                rhs.to_string(ty)
            )),
        },
        TypeKind::Vector(v) if v.scalar.is_numeric() => match &*ty.kind(rhs) {
            TypeKind::Scalar(kind) if &v.scalar == kind => Ok(lhs),
            TypeKind::Vector(other) if v.scalar == other.scalar && v.size == other.size => Ok(lhs),
            TypeKind::Matrix(m) if v.scalar == m.scalar && v.size == m.rows => Ok(ty.register(
                Vector {
                    scalar: v.scalar,
                    size: m.columns,
                }
                .into(),
            )),
            _ => Err(format!(
                "if the left-hand-side operand to the `*` operator is a `{}` vector, then the \
                right-hand-side value must be a vector of the same type, or a numeric scalar that \
                matches the element type of the vector, or matrix with a matching element type and \
                a row-size equal to the vector's size (got `{}`)",
                lhs.to_string(ty),
                rhs.to_string(ty)
            )),
        },
        TypeKind::Matrix(m) if m.scalar.is_numeric() => match &*ty.kind(rhs) {
            TypeKind::Scalar(kind) if &m.scalar == kind => Ok(lhs),
            TypeKind::Vector(v) if m.scalar == v.scalar && m.columns == v.size => Ok(ty.register(
                Vector {
                    scalar: m.scalar,
                    size: m.rows,
                }
                .into(),
            )),
            TypeKind::Matrix(other) if m.columns == other.rows && m.scalar == other.scalar => {
                Ok(ty.register(
                    Matrix {
                        rows: m.rows,
                        columns: other.columns,
                        scalar: m.scalar,
                    }
                    .into(),
                ))
            }
            _ => Err(format!(
                "if the left-hand-side operand to the `*` operator is a `{}` matrix, then the \
                right-hand-side value must be a matrix of a matching element type and a  row-size \
                that matches the matrix's column-size, a vector of a matching element type and a \
                size that matches the matrix's column-size, or a numeric  scalar that matches the \
                element type of the matrix (got `{}`)",
                lhs.to_string(ty),
                rhs.to_string(ty)
            )),
        },
        _ => Err(format!(
            "the `*` operator expects a numeric scalar, a numeric vector or a matrix as its \
            left-hand-side operand (got `{}`)",
            lhs.to_string(ty)
        )),
    }
}

fn check_div_or_mod_op(
    ty: &TypeRegistry,
    op: BinaryOperator,
    lhs: Type,
    rhs: Type,
) -> Result<Type, String> {
    match &*ty.kind(lhs) {
        TypeKind::Scalar(kind) if kind.is_numeric() => match &*ty.kind(rhs) {
            TypeKind::Scalar(kind) if kind == kind => Ok(lhs),
            TypeKind::Vector(v) if &v.scalar == kind => Ok(rhs),
            TypeKind::Matrix(m) if &m.scalar == kind => Ok(rhs),
            _ => Err(format!(
                "if the left-hand-side operand to the `{}` operator is a `{}` value, then the \
                right-hand-side value must be a scalar or a vector of the same (element) type (got \
                `{}`)",
                op,
                lhs.to_string(ty),
                rhs.to_string(ty)
            )),
        },
        TypeKind::Vector(v) if v.scalar.is_numeric() => match &*ty.kind(rhs) {
            TypeKind::Scalar(kind) if &v.scalar == kind => Ok(lhs),
            TypeKind::Vector(other) if v.scalar == other.scalar && v.size == other.size => Ok(lhs),
            _ => Err(format!(
                "if the left-hand-side operand to the `{}` operator is a `{}` vector, then the \
                right-hand-side value must be a vector of the same type, or a numeric scalar that \
                matches the element type of the vector (got `{}`)",
                op,
                lhs.to_string(ty),
                rhs.to_string(ty)
            )),
        },
        _ => Err(format!(
            "the `{}` operator expects a numeric scalar or a numeric vector as its left-hand-side \
            operand (got `{}`)",
            op,
            lhs.to_string(ty)
        )),
    }
}

fn check_shift_op(
    ty: &TypeRegistry,
    op: BinaryOperator,
    lhs: Type,
    rhs: Type,
) -> Result<Type, String> {
    match &*ty.kind(lhs) {
        TypeKind::Scalar(kind) if kind.is_integer() => {
            if rhs == TY_U32 {
                Ok(lhs)
            } else {
                Err(format!(
                    "if the left-hand-side operand to the `{}` operator is a `{}` value, then the \
                    right-hand-side value must be a `{}` (got `{}`)",
                    op,
                    lhs.to_string(ty),
                    TY_U32.to_string(ty),
                    rhs.to_string(ty)
                ))
            }
        }
        TypeKind::Vector(v) if v.scalar.is_integer() => {
            let expected_rhs = TypeKind::Vector(Vector {
                size: v.size,
                scalar: ScalarKind::U32,
            });

            if &*ty.kind(rhs) == &expected_rhs {
                Ok(lhs)
            } else {
                Err(format!(
                    "if the left-hand-side operand to the `{}` operator is a `{}` vector, then the \
                    right-hand-side value must be a vector of equal size and element type `{}` \
                    (got `{}`)",
                    op,
                    lhs.to_string(ty),
                    TY_U32.to_string(ty),
                    rhs.to_string(ty)
                ))
            }
        }
        _ => Err(format!(
            "the `{}` operator expects a integer scalar or an integer vector as its left-hand-side \
            operand (got `{}`)",
            op,
            lhs.to_string(ty)
        )),
    }
}

fn check_eq_op(
    ty: &TypeRegistry,
    op: BinaryOperator,
    lhs: Type,
    rhs: Type,
) -> Result<Type, String> {
    if lhs.is_scalar() && lhs == rhs {
        Ok(TY_BOOL)
    } else {
        Err(format!(
            "the `{}` operator expects its operand to be scalar values of the same type (got `{}` \
            and `{}`)",
            op,
            lhs.to_string(ty),
            rhs.to_string(ty)
        ))
    }
}

fn check_ord_op(
    ty: &TypeRegistry,
    op: BinaryOperator,
    lhs: Type,
    rhs: Type,
) -> Result<Type, String> {
    if lhs.is_numeric_scalar() && lhs == rhs {
        Ok(TY_BOOL)
    } else {
        Err(format!(
            "the `{}` operator expects its operand to be numeric scalar values of the same type \
            (got `{}` and `{}`)",
            op,
            lhs.to_string(ty),
            rhs.to_string(ty)
        ))
    }
}

fn check_bit_op(
    ty: &TypeRegistry,
    op: BinaryOperator,
    lhs: Type,
    rhs: Type,
) -> Result<Type, String> {
    match &*ty.kind(lhs) {
        TypeKind::Scalar(ScalarKind::U32 | ScalarKind::I32)
        | TypeKind::Vector(Vector {
            scalar: ScalarKind::I32 | ScalarKind::U32,
            ..
        }) => {
            if lhs == rhs {
                Ok(lhs)
            } else {
                Err(format!(
                    "the `{}` operator expects both operand to have the same type (got `{}` and \
                    `{}`)",
                    op,
                    lhs.to_string(ty),
                    rhs.to_string(ty)
                ))
            }
        }
        _ => Err(format!(
            "the `{}` operator expects its operands to be signed or unsigned integers or vectors \
            of signed or unsigned integers (got `{}` and `{}`)",
            op,
            lhs.to_string(ty),
            rhs.to_string(ty)
        )),
    }
}
