use serde::{Deserialize, Serialize};

use crate::intrinsic::Intrinsic;
use crate::ty::{Type, TypeKind, TypeRegistry, Vector};

#[derive(Clone, PartialEq, Serialize, Deserialize, Debug)]
pub struct OpVector {
    pub ty: Vector,
}

impl Intrinsic for OpVector {
    fn process_args(
        &self,
        ty_registry: &TypeRegistry,
        args: impl IntoIterator<Item = Type>,
    ) -> Result<Option<Type>, String> {
        let size = self.ty.size.to_u32();
        let element_ty = self.ty.scalar.ty();

        let mut args = args.into_iter();

        for i in 0..size {
            let Some(arg) = args.next() else {
                return Err(format!(
                    "vector operation expected {size} arguments, found {i}"
                ));
            };

            if arg != element_ty {
                return Err(format!(
                    "vector operation expected argument `{i}` to be of type `{}`, but found `{}`",
                    element_ty.to_string(ty_registry),
                    arg.to_string(ty_registry)
                ));
            }
        }

        if args.next().is_some() {
            return Err(format!(
                "vector operation expected {size} arguments, found more"
            ));
        }

        Ok(Some(ty_registry.register(TypeKind::Vector(self.ty))))
    }

    fn affects_state(&self) -> bool {
        false
    }
}
