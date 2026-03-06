use std::io::Write;

use anyhow::Result;
use slir::rvsdg::{Connectivity, Region, Rvsdg, ValueOrigin, ValueUser};

use crate::id_resolution;
use crate::renderer::Renderer;

pub fn render_trace_value_mode<W: Write>(
    rvsdg: &Rvsdg,
    renderer: &Renderer,
    writer: &mut W,
    value_id_str: &str,
) -> Result<()> {
    let parsed = id_resolution::parse_value_id(value_id_str)?;

    writeln!(writer, "Trace for: {}", value_id_str)?;

    match parsed {
        id_resolution::ParsedValueId::NodeOutput(node, output) => {
            let node_data = &rvsdg[node];

            render_forward_trace(
                writer,
                renderer,
                node_data.value_outputs()[output as usize]
                    .users
                    .iter()
                    .copied(),
                node_data.region().into(),
            )?;
        }
        id_resolution::ParsedValueId::NodeInput(node, input_idx) => {
            let node_data = &rvsdg[node];
            let input = &node_data.value_inputs()[input_idx as usize];

            render_backward_trace(writer, renderer, input.origin, node_data.region().into())?;
        }
        id_resolution::ParsedValueId::RegionArgument(region, arg_idx) => {
            let region_data = &rvsdg[region];

            render_forward_trace(
                writer,
                renderer,
                region_data.value_arguments()[arg_idx as usize]
                    .users
                    .iter()
                    .copied(),
                Some(region),
            )?;
        }
        id_resolution::ParsedValueId::RegionResult(region, res_idx) => {
            let res = &rvsdg[region].value_results()[res_idx as usize];

            render_backward_trace(writer, renderer, res.origin, Some(region))?;
        }
    }
    Ok(())
}

fn render_forward_trace<W: Write>(
    writer: &mut W,
    renderer: &Renderer,
    users: impl IntoIterator<Item = ValueUser>,
    region: Option<Region>,
) -> Result<()> {
    writeln!(writer, "  Forward Trace:")?;

    for user in users {
        write!(writer, "    - User: ")?;
        renderer.write_value_user(writer, user, region)?;
        writeln!(writer)?;
    }

    Ok(())
}

fn render_backward_trace<W: Write>(
    writer: &mut W,
    renderer: &Renderer,
    origin: ValueOrigin,
    region: Option<Region>,
) -> Result<()> {
    writeln!(writer, "  Backward Trace:")?;
    write!(writer, "    - Origin: ")?;
    renderer.write_value_origin(writer, origin, region)?;
    writeln!(writer)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::iter;

    use slir::rvsdg::{Rvsdg, ValueInput};
    use slir::ty::TY_U32;
    use slir::{BinaryOperator, FnArg, FnSig, Function, Module, Symbol};

    use super::*;

    #[test]
    fn test_render_trace_value() {
        let mut module = Module::new(Symbol::from_ref("test_module"));
        let function = Function {
            name: Symbol::from_ref("test_func"),
            module: Symbol::from_ref("test_module"),
        };

        module.fn_sigs.register(
            function,
            FnSig {
                name: Default::default(),
                ty: slir::ty::TY_DUMMY,
                args: vec![FnArg {
                    ty: TY_U32,
                    shader_io_binding: None,
                }],
                ret_ty: Some(TY_U32),
            },
        );

        let mut rvsdg = Rvsdg::new(module.ty.clone());

        // Body region:
        // a0: function parameter (u32)
        // s: state
        let (_func_node, region) = rvsdg.register_function(&module, function, iter::empty());

        // 1. Add node1: OpBinary(Add)(a0, a0) -> e0
        let node1 = rvsdg.add_op_binary(
            region,
            BinaryOperator::Add,
            ValueInput::argument(TY_U32, 0),
            ValueInput::argument(TY_U32, 0),
        );

        // 2. Add node2: OpBinary(Add)(node1:e0, a0) -> e0
        let node2 = rvsdg.add_op_binary(
            region,
            BinaryOperator::Add,
            ValueInput::output(TY_U32, node1, 0),
            ValueInput::argument(TY_U32, 0),
        );

        // Reconnect result to node2:e0
        rvsdg.reconnect_region_result(region, 0, ValueInput::output(TY_U32, node2, 0).origin);

        let renderer = Renderer::new(&rvsdg, 0, 0);

        // IDs:
        // Node(1v1): Function
        // Region(2v1): Body Region
        // Node(2v1): node1
        // Node(3v1): node2

        // Case 1: Node output (Forward Trace)
        {
            let mut writer = Vec::new();
            let value_id_str = "Node(2v1)e0";
            render_trace_value_mode(&rvsdg, &renderer, &mut writer, value_id_str).unwrap();
            let output = String::from_utf8(writer).unwrap();
            let expected = "\
Trace for: Node(2v1)e0
  Forward Trace:
    - User: Node(3v1)i0
";
            assert_eq!(output, expected);
        }

        // Case 2: Node input (Backward Trace)
        {
            let mut writer = Vec::new();
            let value_id_str = "Node(3v1)i0";
            render_trace_value_mode(&rvsdg, &renderer, &mut writer, value_id_str).unwrap();
            let output = String::from_utf8(writer).unwrap();
            let expected = "\
Trace for: Node(3v1)i0
  Backward Trace:
    - Origin: Node(2v1)e0
";
            assert_eq!(output, expected);
        }

        // Case 3: Region argument (Forward Trace)
        {
            let mut writer = Vec::new();
            let value_id_str = "Region(2v1)a0";
            render_trace_value_mode(&rvsdg, &renderer, &mut writer, value_id_str).unwrap();
            let output = String::from_utf8(writer).unwrap();
            let expected = "\
Trace for: Region(2v1)a0
  Forward Trace:
    - User: Node(2v1)i0
    - User: Node(2v1)i1
    - User: Node(3v1)i1
";
            assert_eq!(output, expected);
        }

        // Case 4: Region result (Backward Trace)
        {
            let mut writer = Vec::new();
            let value_id_str = "Region(2v1)r0";
            render_trace_value_mode(&rvsdg, &renderer, &mut writer, value_id_str).unwrap();
            let output = String::from_utf8(writer).unwrap();
            let expected = "\
Trace for: Region(2v1)r0
  Backward Trace:
    - Origin: Node(3v1)e0
";
            assert_eq!(output, expected);
        }
    }
}
