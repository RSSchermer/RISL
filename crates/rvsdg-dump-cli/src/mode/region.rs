use std::io::Write;

use anyhow::Result;
use slir::rvsdg::Rvsdg;

use crate::id_resolution;
use crate::renderer::Renderer;

pub fn render_region_mode<W: Write>(
    _rvsdg: &Rvsdg,
    renderer: &Renderer,
    writer: &mut W,
    region_id_str: &str,
) -> Result<()> {
    let region_id = id_resolution::parse_region_id(region_id_str)?;

    renderer.write_region(writer, region_id, "Region", 0, 0)?;
    writeln!(writer)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::iter;

    use slir::rvsdg::{Rvsdg, StateOrigin, ValueInput};
    use slir::ty::{TY_DUMMY, TY_PTR_U32, TY_U32};
    use slir::{
        Constant, FnArg, FnSig, Function, Module, ResourceBinding, Symbol, UniformBindingData,
    };

    use super::*;

    #[test]
    fn test_render_region_simple() {
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

        // Register a uniform binding (u32)
        let uniform_binding = module.uniform_bindings.register(UniformBindingData {
            ty: TY_U32,
            resource_binding: ResourceBinding {
                group: 0,
                binding: 0,
            },
        });
        let uniform_node = rvsdg.register_uniform_binding(&module, uniform_binding);

        // Register function and get body region
        let (_func_node, region) =
            rvsdg.register_function(&module, function, iter::once(uniform_node));

        // In the body region:
        // arg0: uniform_node (u32)
        // arg1: function param (ptr<u32>)
        // s: state

        // 1. Load from function param
        let load_node = rvsdg.add_op_load(
            region,
            ValueInput::argument(TY_PTR_U32, 1),
            StateOrigin::Argument,
        );

        // 2. Store to function param
        let _store_node = rvsdg.add_op_store(
            region,
            ValueInput::argument(TY_PTR_U32, 1),
            ValueInput::output(TY_U32, load_node, 0),
            StateOrigin::Node(load_node),
        );

        // Reconnect result to the value we loaded
        rvsdg.reconnect_region_result(region, 0, ValueInput::output(TY_U32, load_node, 0).origin);

        let renderer = Renderer::new(&rvsdg, 0, 0, true);
        let mut writer = Vec::new();

        render_region_mode(&rvsdg, &renderer, &mut writer, "Region(2v1)").unwrap();

        let output = String::from_utf8(writer).unwrap();
        let expected = "\
Region (Region(2v1)):
  Arguments: [Region(2v1)a0: u32, Region(2v1)a1: ptr<u32>, Region(2v1)s: State]
  [Node(3v1)] OpLoad(Region(2v1)a1) (state: Arg) -> Node(3v1)e0 : u32, Node(3v1)s : State
  [Node(4v1)] OpStore(Region(2v1)a1, Node(3v1)e0) (state: Node(3v1)) -> Node(4v1)s : State
  Results: [Node(3v1)e0, Region(2v1)s]
";
        assert_eq!(output, expected);
    }

    #[test]
    fn test_render_region_nested() {
        use slir::rvsdg::ValueOutput;
        use slir::ty::TY_PREDICATE;

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
                ret_ty: Some(TY_U32),
            },
        );

        let mut rvsdg = Rvsdg::new(module.ty.clone());

        // Global constants
        let const_true = Constant {
            name: Symbol::from_ref("true"),
            module: Symbol::from_ref("test_module"),
        };
        module
            .constants
            .register_byte_data(const_true, TY_PREDICATE, vec![1]);
        let predicate_global_node = rvsdg.register_constant(&module, const_true);

        let const_u32 = Constant {
            name: Symbol::from_ref("c_u32"),
            module: Symbol::from_ref("test_module"),
        };
        module
            .constants
            .register_byte_data(const_u32, TY_U32, vec![10, 0, 0, 0]);
        let u32_global_node = rvsdg.register_constant(&module, const_u32);

        // Register function with dependencies
        let (_func_node, region) = rvsdg.register_function(
            &module,
            function,
            vec![predicate_global_node, u32_global_node],
        );

        // body region:
        // arg0: predicate
        // arg1: u32
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
        // branch1:
        // arg0: u32 (from switch input 1)
        // s: state
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

        // Reconnect function result
        rvsdg.reconnect_region_result(region, 0, ValueInput::output(TY_U32, switch_node, 0).origin);

        let renderer = Renderer::new(&rvsdg, 2, 1, true);
        let mut writer = Vec::new();

        render_region_mode(&rvsdg, &renderer, &mut writer, "Region(2v1)").unwrap();

        let output = String::from_utf8(writer).unwrap();

        // Retracing IDs:
        // Node(1v1): Function
        // Region(1v1): Body
        // Node(2v1): predicate const
        // Node(3v1): u32 const
        // Node(4v1): Switch
        // Region(2v1): Branch 1
        // Node(5v1): node1 (in Branch 1)
        // Region(3v1): Branch 2
        // Node(6v1): node_a
        // Node(7v1): node_b
        // Node(8v1): node_c

        let expected = "\
Region (Region(2v1)):
  Arguments: [Region(2v1)a0: predicate, Region(2v1)a1: u32, Region(2v1)s: State]
  [Node(4v1)] Switch(Region(2v1)a0, Region(2v1)a1) -> Node(4v1)e0 : u32
    Branch 0 (Region(3v1)):
      Arguments: [Region(3v1)a0: u32, Region(3v1)s: State]
      [Node(5v1)] OpBinary{operator: +}(Region(3v1)a0, Region(3v1)a0) -> Node(5v1)e0 : u32
      Results: [Node(5v1)e0, Region(3v1)s]
    Branch 1 (Region(4v1)): 3 child nodes
  Results: [Node(4v1)e0, Region(2v1)s]
";
        assert_eq!(output, expected);
    }
}
