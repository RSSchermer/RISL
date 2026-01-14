use crate::intrinsic::Intrinsic;
use crate::ty::{Matrix, Type, TypeKind, TypeRegistry};

pub struct OpMatrix {
    pub ty: Matrix,
}

impl Intrinsic for OpMatrix {
    fn process_args(
        &self,
        ty_registry: &TypeRegistry,
        args: impl IntoIterator<Item = Type>,
    ) -> Result<Option<Type>, String> {
        let columns = self.ty.columns.to_u32();
        let column_ty = self.ty.column_ty();

        let mut args = args.into_iter();

        for i in 0..columns {
            let Some(arg) = args.next() else {
                return Err(format!(
                    "matrix operation expected {columns} arguments, found {i}"
                ));
            };

            if arg != column_ty {
                return Err(format!(
                    "matrix operation expected argument `{i}` to be of type `{}`, but found `{}`",
                    column_ty.to_string(ty_registry),
                    arg.to_string(ty_registry)
                ));
            }
        }

        if args.next().is_some() {
            return Err(format!(
                "vector operation expected {columns} arguments, found more"
            ));
        }

        Ok(Some(ty_registry.register(TypeKind::Matrix(self.ty))))
    }
}
