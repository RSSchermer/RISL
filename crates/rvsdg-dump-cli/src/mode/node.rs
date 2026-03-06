use std::io::Write;

use anyhow::Result;
use slir::rvsdg::Rvsdg;

use crate::id_resolution;
use crate::renderer::Renderer;

pub fn render_node_mode<W: Write>(
    _rvsdg: &Rvsdg,
    renderer: &Renderer,
    writer: &mut W,
    node_id_str: &str,
) -> Result<()> {
    let node_id = id_resolution::parse_node_id(node_id_str)?;
    renderer.write_node(writer, node_id, 0, 0)?;
    writeln!(writer)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::iter;

    use slir::rvsdg::{Rvsdg, ValueInput, ValueOutput};
    use slir::ty::{TY_DUMMY, TY_PREDICATE, TY_U32};
    use slir::{FnArg, FnSig, Function, Module, Symbol};

    use super::*;

    #[test]
    fn test_render_node() {
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
                args: vec![
                    FnArg {
                        ty: TY_PREDICATE,
                        shader_io_binding: None,
                    },
                    FnArg {
                        ty: TY_U32,
                        shader_io_binding: None,
                    },
                ],
                ret_ty: Some(TY_U32),
            },
        );

        let mut rvsdg = Rvsdg::new(module.ty.clone());

        // Register function and get body region
        let (_func_node, region) = rvsdg.register_function(&module, function, iter::empty());

        // body region:
        // arg0: function param 0 (predicate)
        // arg1: function param 1 (u32)
        // s: state

        // Add a Switch node
        let switch_node = rvsdg.add_switch(
            region,
            vec![
                ValueInput::argument(TY_PREDICATE, 0),
                ValueInput::argument(TY_U32, 1),
            ],
            vec![ValueOutput::new(TY_U32)],
            None,
        );

        // Branch 1: Simple (one node) -> should be inlined
        let branch1 = rvsdg.add_switch_branch(switch_node);
        let node1 = rvsdg.add_op_binary(
            branch1,
            slir::BinaryOperator::Add,
            ValueInput::argument(TY_U32, 0),
            ValueInput::argument(TY_U32, 0),
        );
        rvsdg.reconnect_region_result(branch1, 0, ValueInput::output(TY_U32, node1, 0).origin);

        // Branch 2: Complex (three nodes) -> should be collapsed
        let branch2 = rvsdg.add_switch_branch(switch_node);
        let node_a = rvsdg.add_op_binary(
            branch2,
            slir::BinaryOperator::Add,
            ValueInput::argument(TY_U32, 0),
            ValueInput::argument(TY_U32, 0),
        );
        let node_b = rvsdg.add_op_binary(
            branch2,
            slir::BinaryOperator::Add,
            ValueInput::output(TY_U32, node_a, 0),
            ValueInput::argument(TY_U32, 0),
        );
        let node_c = rvsdg.add_op_binary(
            branch2,
            slir::BinaryOperator::Add,
            ValueInput::output(TY_U32, node_b, 0),
            ValueInput::argument(TY_U32, 0),
        );
        rvsdg.reconnect_region_result(branch2, 0, ValueInput::output(TY_U32, node_c, 0).origin);

        let renderer = Renderer::new(&rvsdg, 2, 1);
        let mut writer = Vec::new();

        render_node_mode(&rvsdg, &renderer, &mut writer, "Node(2v1)").unwrap();

        let output = String::from_utf8(writer).unwrap();

        // Retracing IDs:
        // Node(1v1): Function
        // Region(1v1): Global Region (always present)
        // Region(2v1): Body Region
        // Node(2v1): Switch
        // Region(3v1): Branch 1
        // Node(3v1): node1 (in Branch 1)
        // Region(4v1): Branch 2
        // Node(4v1): node_a
        // Node(5v1): node_b
        // Node(6v1): node_c

        let expected = "\
[Node(2v1)] Switch(Region(2v1)a0, Region(2v1)a1) -> Node(2v1)e0 : u32
  Branch 0 (Region(3v1)):
    Arguments: [Region(3v1)a0: u32, Region(3v1)s: State]
    [Node(3v1)] OpBinary{operator: +}(Region(3v1)a0, Region(3v1)a0) -> Node(3v1)e0 : u32
    Results: [Node(3v1)e0, Region(3v1)s]
  Branch 1 (Region(4v1)): 3 child nodes
";
        assert_eq!(output, expected);
    }
}
