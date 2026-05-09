use leptos::prelude::*;
use leptos::{component, view, IntoView};
use slir::scf::{ExpressionKind, GlobalPtr};
use slotmap::Key;

use crate::module::functions::detail::scf_explorer::local_binding::LocalBinding;
use crate::module::use_module_data;

#[component]
pub fn Expression(expr_binding: slir::scf::Statement) -> impl IntoView {
    let module_data = use_module_data();
    let scf = module_data.expect_scf().read_value();
    let expr_data = &scf[expr_binding].expect_expr_binding().expression();

    match expr_data.kind() {
        ExpressionKind::ConstU32(v) => format!("{}u32", v).into_any(),
        ExpressionKind::ConstI32(v) => format!("{}i32", v).into_any(),
        ExpressionKind::ConstF32(v) => format!("{}f32", v).into_any(),
        ExpressionKind::ConstBool(v) => v.to_string().into_any(),
        ExpressionKind::GlobalPtr(ptr) => match ptr {
            GlobalPtr::Uniform(binding) => {
                format!("&U{}", binding.data().as_ffi())
            }
            GlobalPtr::Storage(binding) => {
                format!("&S{}", binding.data().as_ffi())
            }
            GlobalPtr::Workgroup(binding) => {
                format!("&W{}", binding.data().as_ffi())
            }
            GlobalPtr::Constant(constant) => format!("&{}", constant.name),
        }
        .into_any(),
        ExpressionKind::OpUnary(op) => view! {
            {op.operator().to_string()}<LocalBinding binding=op.operand()/>
        }
        .into_any(),
        ExpressionKind::OpBinary(op) => view! {
            <LocalBinding binding=op.lhs()/>
            {format!(" {} ", op.operator())}
            <LocalBinding binding=op.rhs()/>
        }
        .into_any(),
        ExpressionKind::OpMax(op) => view! {
            "max("<LocalBinding binding=op.lhs()/>", "<LocalBinding binding=op.rhs()/>")"
        }
        .into_any(),
        ExpressionKind::OpMin(op) => view! {
            "min("<LocalBinding binding=op.lhs()/>", "<LocalBinding binding=op.rhs()/>")"
        }
        .into_any(),
        ExpressionKind::OpRoundToEven(op) => view! {
            "round-to-even("<LocalBinding binding=op.value()/>")"
        }
        .into_any(),
        ExpressionKind::OpFloor(op) => view! {
            "floor("<LocalBinding binding=op.value()/>")"
        }
        .into_any(),
        ExpressionKind::OpCeil(op) => view! {
            "ceil("<LocalBinding binding=op.value()/>")"
        }
        .into_any(),
        ExpressionKind::OpClamp(op) => view! {
            "clamp("<LocalBinding binding=op.value()/>", "<LocalBinding binding=op.min()/>", "<LocalBinding binding=op.max()/>")"
        }
        .into_any(),
        ExpressionKind::OpFract(op) => view! {
            "fract("<LocalBinding binding=op.value()/>")"
        }
        .into_any(),
        ExpressionKind::OpTrunc(op) => view! {
            "trunc("<LocalBinding binding=op.value()/>")"
        }
        .into_any(),
        ExpressionKind::OpSqrt(op) => view! {
            "sqrt("<LocalBinding binding=op.value()/>")"
        }
        .into_any(),
        ExpressionKind::OpInverseSqrt(op) => view! {
            "inverse_sqrt("<LocalBinding binding=op.value()/>")"
        }
        .into_any(),
        ExpressionKind::OpExp(op) => view! {
            "exp("<LocalBinding binding=op.value()/>")"
        }
        .into_any(),
        ExpressionKind::OpExp2(op) => view! {
            "exp2("<LocalBinding binding=op.value()/>")"
        }
        .into_any(),
        ExpressionKind::OpLog(op) => view! {
            "log("<LocalBinding binding=op.value()/>")"
        }
        .into_any(),
        ExpressionKind::OpLog2(op) => view! {
            "log2("<LocalBinding binding=op.value()/>")"
        }
        .into_any(),
        ExpressionKind::OpCos(op) => view! {
            "cos("<LocalBinding binding=op.value()/>")"
        }
        .into_any(),
        ExpressionKind::OpAcos(op) => view! {
            "acos("<LocalBinding binding=op.value()/>")"
        }
        .into_any(),
        ExpressionKind::OpCosh(op) => view! {
            "cosh("<LocalBinding binding=op.value()/>")"
        }
        .into_any(),
        ExpressionKind::OpAcosh(op) => view! {
            "acosh("<LocalBinding binding=op.value()/>")"
        }
        .into_any(),
        ExpressionKind::OpSin(op) => view! {
            "sin("<LocalBinding binding=op.value()/>")"
        }
        .into_any(),
        ExpressionKind::OpAsin(op) => view! {
            "asin("<LocalBinding binding=op.value()/>")"
        }
        .into_any(),
        ExpressionKind::OpSinh(op) => view! {
            "sinh("<LocalBinding binding=op.value()/>")"
        }
        .into_any(),
        ExpressionKind::OpAsinh(op) => view! {
            "asinh("<LocalBinding binding=op.value()/>")"
        }
        .into_any(),
        ExpressionKind::OpTan(op) => view! {
            "tan("<LocalBinding binding=op.value()/>")"
        }
        .into_any(),
        ExpressionKind::OpAtan(op) => view! {
            "atan("<LocalBinding binding=op.value()/>")"
        }
        .into_any(),
        ExpressionKind::OpTanh(op) => view! {
            "tanh("<LocalBinding binding=op.value()/>")"
        }
        .into_any(),
        ExpressionKind::OpAtanh(op) => view! {
            "atanh("<LocalBinding binding=op.value()/>")"
        }
        .into_any(),
        ExpressionKind::OpVector(op) => view! {
            {op.vector_ty().to_string()}"("{
                op.elements().iter().copied().map(|binding| view! {
                    <LocalBinding binding/>
                }.into_any())
                .intersperse_with(|| view! {", "}.into_any())
                .collect_view()
            }")"
        }
        .into_any(),
        ExpressionKind::OpMatrix(op) => view! {
            {op.matrix_ty().to_string()}"("{
                op.columns().iter().copied().map(|binding| view! {
                    <LocalBinding binding/>
                }.into_any())
                .intersperse_with(|| view! {", "}.into_any())
                .collect_view()
            }")"
        }
        .into_any(),
        ExpressionKind::OpConvertToU32(op) => view! {
            "u32("<LocalBinding binding=op.value()/>")"
        }
        .into_any(),
        ExpressionKind::OpConvertToI32(op) => view! {
            "i32("<LocalBinding binding=op.value()/>")"
        }
        .into_any(),
        ExpressionKind::OpConvertToF32(op) => view! {
            "f32("<LocalBinding binding=op.value()/>")"
        }
        .into_any(),
        ExpressionKind::OpConvertToBool(op) => view! {
            "bool("<LocalBinding binding=op.value()/>")"
        }
        .into_any(),
        ExpressionKind::OpFieldPtr(op) => view! {
            "&"<LocalBinding binding=op.ptr()/>"._"{op.field_index()}
        }
        .into_any(),
        ExpressionKind::OpElementPtr(op) => view! {
            "&"<LocalBinding binding=op.ptr()/>"["<LocalBinding binding=op.index()/>"]"
        }
        .into_any(),
        ExpressionKind::OpExtractField(op) => view! {
            <LocalBinding binding=op.value()/>"._"{op.field_index()}
        }
        .into_any(),
        ExpressionKind::OpExtractElement(op) => view! {
            <LocalBinding binding=op.value()/>"["<LocalBinding binding=op.index()/>"]"
        }
        .into_any(),
        ExpressionKind::OpLoad(op) => view! {
            "*"<LocalBinding binding=op.ptr()/>
        }
        .into_any(),
        ExpressionKind::OpArrayLength(op) => view! {
            "array-length("<LocalBinding binding=op.ptr()/>")"
        }
        .into_any(),
    }
}
