use std::io::Write;

use anyhow::Result;
use slir::rvsdg::{Connectivity, Rvsdg, StateUser};

use crate::id_resolution;
use crate::renderer::Renderer;

pub fn render_trace_state_mode<W: Write>(
    rvsdg: &Rvsdg,
    renderer: &Renderer,
    writer: &mut W,
    region_id_str: &str,
) -> Result<()> {
    let region_id = id_resolution::parse_region_id(region_id_str)?;
    let region_data = &rvsdg[region_id];

    writeln!(writer, "State Trace for {}:", region_id_str)?;
    writeln!(writer, "  [State Arg]")?;

    let mut current_user = *region_data.state_argument();

    loop {
        match current_user {
            StateUser::Node(node) => {
                write!(writer, "    -> ")?;
                renderer.write_node_id(writer, node)?;
                writeln!(writer)?;

                if let Some(state) = rvsdg[node].state() {
                    current_user = state.user;
                } else {
                    writeln!(writer, "    !! State chain break detected !!")?;

                    break;
                }
            }
            StateUser::Result => {
                writeln!(writer, "    -> [State Result]")?;

                break;
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::iter;

    use slir::rvsdg::{Rvsdg, StateOrigin, ValueInput};
    use slir::ty::{TY_DUMMY, TY_PTR_U32, TY_U32};
    use slir::{FnArg, FnSig, Function, Module, Symbol};

    use super::*;

    #[test]
    fn test_render_trace_state_empty() {
        let mut module = Module::new(Symbol::from_ref("test_module"));
        let function = Function {
            name: Symbol::from_ref("test_func"),
            module: Symbol::from_ref("test_module"),
        };

        module.fn_sigs.register(
            function,
            FnSig {
                name: Default::default(),
                ty: TY_DUMMY,
                args: vec![],
                ret_ty: None,
            },
        );

        let mut rvsdg = Rvsdg::new(module.ty.clone());
        let (_func_node, _region) = rvsdg.register_function(&module, function, iter::empty());

        let renderer = Renderer::new(&rvsdg, 0, 0);
        let mut writer = Vec::new();

        render_trace_state_mode(&rvsdg, &renderer, &mut writer, "Region(2v1)").unwrap();

        let output = String::from_utf8(writer).unwrap();
        let expected = "\
State Trace for Region(2v1):
  [State Arg]
    -> [State Result]
";
        assert_eq!(output, expected);
    }

    #[test]
    fn test_render_trace_state_chain() {
        let mut module = Module::new(Symbol::from_ref("test_module"));
        let function = Function {
            name: Symbol::from_ref("test_func"),
            module: Symbol::from_ref("test_module"),
        };

        module.fn_sigs.register(
            function,
            FnSig {
                name: Default::default(),
                ty: TY_DUMMY,
                args: vec![FnArg {
                    ty: TY_PTR_U32,
                    shader_io_binding: None,
                }],
                ret_ty: Some(TY_U32),
            },
        );

        let mut rvsdg = Rvsdg::new(module.ty.clone());
        let (_func_node, region) = rvsdg.register_function(&module, function, iter::empty());

        // Body region:
        // a0: function param (ptr<u32>)
        // s: state

        // 1. Load from function param
        let load_node = rvsdg.add_op_load(
            region,
            ValueInput::argument(TY_PTR_U32, 0),
            StateOrigin::Argument,
        );

        // 2. Store to function param
        let _store_node = rvsdg.add_op_store(
            region,
            ValueInput::argument(TY_PTR_U32, 0),
            ValueInput::output(TY_U32, load_node, 0),
            StateOrigin::Node(load_node),
        );

        let renderer = Renderer::new(&rvsdg, 0, 0);
        let mut writer = Vec::new();

        render_trace_state_mode(&rvsdg, &renderer, &mut writer, "Region(2v1)").unwrap();

        let output = String::from_utf8(writer).unwrap();

        // Node(1v1): function
        // Region(2v1): body
        // Node(2v1): load
        // Node(3v1): store

        let expected = "\
State Trace for Region(2v1):
  [State Arg]
    -> Node(2v1)
    -> Node(3v1)
    -> [State Result]
";
        assert_eq!(output, expected);
    }
}
