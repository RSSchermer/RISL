use leptos::prelude::*;
use leptos::{component, IntoView};
use slir::scf::{LoopControl, StatementKind};

use crate::module::functions::detail::scf_explorer::block::Block;
use crate::module::functions::detail::scf_explorer::expression::Expression;
use crate::module::functions::detail::scf_explorer::local_binding::LocalBinding;
use crate::module::ty::Type;
use crate::module::use_module_data;

#[component]
pub fn Statement(statement: slir::scf::Statement) -> impl IntoView {
    let module_data = use_module_data().read_value();
    let stmt_data = &module_data.scf.as_ref().unwrap()[statement];

    match stmt_data.kind() {
        StatementKind::If(_) => view! {
            <If statement/><br/>
        }
        .into_any(),
        StatementKind::Switch(_) => view! {
            <Switch statement/><br/>
        }
        .into_any(),
        StatementKind::Loop(_) => view! {
            <Loop statement/><br/>
        }
        .into_any(),
        StatementKind::Return(_) => view! {
            <Return statement/><br/>
        }
        .into_any(),
        StatementKind::Alloca(_) => view! {
            <Alloca statement/><br/>
        }
        .into_any(),
        StatementKind::ExprBinding(_) => view! {
            <ExprBinding statement/><br/>
        }
        .into_any(),
        StatementKind::OpStore(_) => view! {
            <Store statement/><br/>
        }
        .into_any(),
    }
}

#[component]
pub fn If(statement: slir::scf::Statement) -> impl IntoView {
    let module_data = use_module_data();
    let m = module_data.read_value();
    let stmt = m.expect_scf()[statement].kind().expect_if();

    view! {
        {move || {
            let m = module_data.read_value();
            let stmt = m.expect_scf()[statement].kind().expect_if();

            stmt.out_vars().iter().map(|var| {
                let ty = m.scf.as_ref().unwrap()[*var].ty();

                view!{
                    "var "<LocalBinding binding=*var/>": "<Type ty/>";"<br/>
                }
            }).collect_view()
        }}
        <br/>
        "if " <LocalBinding binding=stmt.condition()/> " {"<br/>
            <Block block=stmt.then_block()/>
        "} "
        {move || {
            let m = module_data.read_value();
            let stmt = m.expect_scf()[statement].kind().expect_if();

            stmt.else_block().map(|else_block| {
                view! {
                    "else {"<br/>
                        <Block block=else_block/>
                    "}"
                }
            })
        }}
    }
}

#[component]
pub fn Switch(statement: slir::scf::Statement) -> impl IntoView {
    let module_data = use_module_data();
    let m = module_data.read_value();
    let stmt = m.expect_scf()[statement].kind().expect_switch();

    view! {
        {move || {
            let m = module_data.read_value();
            let stmt = m.expect_scf()[statement].kind().expect_switch();

            stmt.out_vars().iter().map(|var| {
                let ty = m.scf.as_ref().unwrap()[*var].ty();

                view!{
                    "var "<LocalBinding binding=*var/>": "<Type ty/>";"<br/>
                }
            }).collect_view()
        }}
        <br/>
        "switch " <LocalBinding binding=stmt.on()/> " {"<br/>
            <div class="scf-indent">
                {move || {
                    let m = module_data.read_value();
                    let stmt = m.expect_scf()[statement].kind().expect_switch();

                    stmt.cases().iter().map(|case| {
                        view! {
                            "case "{case.case()}": {"<br/>
                                <Block block=case.block()/>
                            "}"<br/>
                        }
                    }).collect_view()
                }}
                "default: {"<br/>
                    <Block block=stmt.default()/>
                "}"
            </div>
        "}"
    }
}

#[component]
pub fn Loop(statement: slir::scf::Statement) -> impl IntoView {
    let module_data = use_module_data();

    view! {
        {move || {
            let m = module_data.read_value();
            let stmt = m.expect_scf()[statement].kind().expect_loop();

            stmt.loop_vars().iter().map(|var| {
                view!{
                    "var "<LocalBinding binding=var.binding()/>
                    " = "<LocalBinding binding=var.initial_value()/>";"<br/>
                }
            }).collect_view()
        }}
        <br/>
        {move || {
            let m = module_data.read_value();
            let stmt = m.expect_scf()[statement].kind().expect_loop();

            match stmt.control() {
                LoopControl::Head(binding) => view!{
                    "while "<LocalBinding binding/>" {"<br/>
                        <Block block=stmt.block()/>
                    "}"<br/>
                }.into_any(),
                LoopControl::Tail(binding) => view!{
                    "do {"<br/>
                        <Block block=stmt.block()/>
                    "} while "<LocalBinding binding/>";"<br/>
                }.into_any(),
                LoopControl::Infinite => view!{
                    "loop {"<br/>
                        <Block block=stmt.block()/>
                    "}"<br/>
                }.into_any()
            }
        }}
    }
}

#[component]
pub fn Return(statement: slir::scf::Statement) -> impl IntoView {
    let m = use_module_data().read_value();
    let stmt = m.expect_scf()[statement].kind().expect_return();

    if let Some(binding) = stmt.value() {
        view! {
            "return "<LocalBinding binding/>";"
        }
        .into_any()
    } else {
        view! {
            "return;"
        }
        .into_any()
    }
}

#[component]
pub fn Alloca(statement: slir::scf::Statement) -> impl IntoView {
    let module_data = use_module_data().read_value();
    let stmt = module_data.expect_scf()[statement].kind().expect_alloca();

    view! {
        "let "<LocalBinding binding=stmt.binding()/>" = alloca;"
    }
}

#[component]
pub fn ExprBinding(statement: slir::scf::Statement) -> impl IntoView {
    let module_data = use_module_data().read_value();
    let stmt = module_data.expect_scf()[statement]
        .kind()
        .expect_expr_binding();

    view! {
        "let "<LocalBinding binding=stmt.binding()/>
        " = "<Expression expr_binding=statement/>";"
    }
}

#[component]
pub fn Store(statement: slir::scf::Statement) -> impl IntoView {
    let module_data = use_module_data().read_value();
    let stmt = module_data.expect_scf()[statement].kind().expect_op_store();

    view! {
        "*"<LocalBinding binding=stmt.ptr()/>
        " = "
        <LocalBinding binding=stmt.value()/>";"
    }
}
