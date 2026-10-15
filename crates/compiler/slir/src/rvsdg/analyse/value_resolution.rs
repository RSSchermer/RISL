use crate::rvsdg::analyse::ValueKey;
use crate::rvsdg::analyse::scalar_constant::ScalarConstant;
use crate::rvsdg::{NodeKind, Region, Rvsdg, SimpleNode, ValueOrigin};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ResolvedValue {
    Constant(ScalarConstant),
    Fallback,
    Opaque { region: Region, origin: ValueOrigin },
}

pub fn resolve_value(rvsdg: &Rvsdg, region: Region, origin: ValueOrigin) -> ResolvedValue {
    let key = ValueKey::new(region, origin).canonicalize(rvsdg);

    if let ValueOrigin::Output { producer, output } = key.origin {
        if let Some(constant) = ScalarConstant::from_node(rvsdg, producer, output) {
            return ResolvedValue::Constant(constant);
        }

        if output == 0
            && matches!(
                rvsdg[producer].kind(),
                NodeKind::Simple(SimpleNode::ConstFallback(_))
            )
        {
            return ResolvedValue::Fallback;
        }
    }

    ResolvedValue::Opaque {
        region: key.region,
        origin: key.origin,
    }
}

#[cfg(test)]
mod tests {
    use std::iter;

    use super::*;
    use crate::rvsdg::ValueInput;
    use crate::ty::{TY_DUMMY, TY_U32};
    use crate::{FnSig, Function, Module, Symbol};

    #[test]
    fn resolves_scalar_constants_and_fallback() {
        let mut module = Module::new(Symbol::from_ref(""));
        let function = Function {
            name: Symbol::from_ref("test"),
            module: Symbol::from_ref(""),
        };

        module.fn_sigs.register(
            function,
            FnSig {
                name: function.name,
                ty: TY_DUMMY,
                args: vec![],
                ret_ty: None,
            },
        );

        let mut rvsdg = Rvsdg::new(module.ty.clone());
        let (_, body) = rvsdg.register_function(&module, function, iter::empty());

        let u32_constant = rvsdg.add_const_u32(body, 7);
        let i32_constant = rvsdg.add_const_i32(body, -3);
        let f32_constant = rvsdg.add_const_f32(body, 1.25);
        let bool_constant = rvsdg.add_const_bool(body, true);
        let predicate_constant = rvsdg.add_const_predicate(body, 2);
        let fallback = rvsdg.add_const_fallback(body, TY_U32);

        assert_eq!(
            resolve_value(
                &rvsdg,
                body,
                ValueOrigin::Output {
                    producer: u32_constant,
                    output: 0,
                },
            ),
            ResolvedValue::Constant(ScalarConstant::U32(7))
        );
        assert_eq!(
            resolve_value(
                &rvsdg,
                body,
                ValueOrigin::Output {
                    producer: i32_constant,
                    output: 0,
                },
            ),
            ResolvedValue::Constant(ScalarConstant::I32(-3))
        );
        assert_eq!(
            resolve_value(
                &rvsdg,
                body,
                ValueOrigin::Output {
                    producer: f32_constant,
                    output: 0,
                },
            ),
            ResolvedValue::Constant(ScalarConstant::F32(1.25_f32.to_bits()))
        );
        assert_eq!(
            resolve_value(
                &rvsdg,
                body,
                ValueOrigin::Output {
                    producer: bool_constant,
                    output: 0,
                },
            ),
            ResolvedValue::Constant(ScalarConstant::Bool(true))
        );
        assert_eq!(
            resolve_value(
                &rvsdg,
                body,
                ValueOrigin::Output {
                    producer: predicate_constant,
                    output: 0,
                },
            ),
            ResolvedValue::Constant(ScalarConstant::Predicate(2))
        );
        assert_eq!(
            resolve_value(
                &rvsdg,
                body,
                ValueOrigin::Output {
                    producer: fallback,
                    output: 0,
                },
            ),
            ResolvedValue::Fallback
        );
    }
}
