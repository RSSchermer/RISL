use crate::UnaryOperator;
use crate::intrinsic::{Intrinsic, expect_one_arg};
use crate::ty::{TY_BOOL, TY_F32, TY_I32, Type, TypeRegistry};

pub struct OpUnary {
    pub operator: UnaryOperator,
}

impl Intrinsic for OpUnary {
    fn process_args(
        &self,
        ty_registry: &TypeRegistry,
        args: impl IntoIterator<Item = Type>,
    ) -> Result<Option<Type>, String> {
        let arg = expect_one_arg!("op-unary", args);

        match self.operator {
            UnaryOperator::Not => {
                if arg != TY_BOOL {
                    return Err(format!(
                        "the `not` operator expects a argument of type `bool`, found `{}`",
                        arg.to_string(ty_registry)
                    ));
                }
            }
            UnaryOperator::Neg => {
                if !matches!(arg, TY_I32 | TY_F32) {
                    return Err(format!(
                        "the `negate` operator expects a argument of type `i32` or `f32`, found \
                        `{}`",
                        arg.to_string(ty_registry)
                    ));
                }
            }
        }

        Ok(Some(arg))
    }
}
