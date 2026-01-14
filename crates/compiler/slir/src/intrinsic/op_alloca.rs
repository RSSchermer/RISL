use crate::intrinsic::Intrinsic;
use crate::ty::{Type, TypeKind};

pub struct OpAlloca {
    pub ty: Type,
}

impl Intrinsic for OpAlloca {
    fn process_args(
        &self,
        ty_registry: &crate::ty::TypeRegistry,
        args: impl IntoIterator<Item = Type>,
    ) -> Result<Option<Type>, String> {
        if args.into_iter().next().is_some() {
            return Err("the alloca operation does not take any arguments".to_string());
        }

        let ptr_ty = ty_registry.register(TypeKind::Ptr(self.ty));

        Ok(Some(ptr_ty))
    }
}
