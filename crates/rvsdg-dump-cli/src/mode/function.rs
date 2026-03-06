use std::io::Write;

use anyhow::Result;
use slir::rvsdg::Rvsdg;

use crate::id_resolution;
use crate::renderer::Renderer;

pub fn render_function_mode<W: Write>(
    rvsdg: &Rvsdg,
    renderer: &Renderer,
    writer: &mut W,
    func_name: &str,
) -> Result<()> {
    let mut found_node = None;

    for (function, node) in rvsdg.registered_functions() {
        if function.name.as_str() == func_name {
            found_node = Some(node);
            break;
        }
    }

    if found_node.is_none() {
        // Try as ID
        if let Ok(node_id) = id_resolution::parse_node_id(func_name) {
            if rvsdg.is_live_node(node_id) {
                let node_data = &rvsdg[node_id];
                if node_data.is_function() {
                    found_node = Some(node_id);
                }
            }
        }
    }

    if let Some(node) = found_node {
        renderer.write_node(writer, node, 0, 0)?;
        writeln!(writer)?;

        Ok(())
    } else {
        anyhow::bail!("Function '{}' not found", func_name);
    }
}

#[cfg(test)]
mod tests {
    use slir::rvsdg::{Rvsdg, ValueInput};
    use slir::ty::{TY_DUMMY, TY_U32};
    use slir::{BinaryOperator, FnArg, FnSig, Function, Module, Symbol};

    use super::*;

    #[test]
    fn test_render_function() {
        use slir::{Constant, ResourceBinding, UniformBindingData};

        let mut module = Module::new(Symbol::from_ref("test_module"));

        // Global constant
        let constant = Constant {
            name: Symbol::from_ref("test_const"),
            module: Symbol::from_ref("test_module"),
        };
        module
            .constants
            .register_byte_data(constant, TY_U32, vec![42, 0, 0, 0]);

        // Global uniform binding
        let uniform_binding = module.uniform_bindings.register(UniformBindingData {
            ty: TY_U32,
            resource_binding: ResourceBinding {
                group: 0,
                binding: 0,
            },
        });

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
                        ty: TY_U32,
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

        // Register globals
        let const_node = rvsdg.register_constant(&module, constant);
        let uniform_node = rvsdg.register_uniform_binding(&module, uniform_binding);

        // Register function with dependencies
        let (_func_node, region) =
            rvsdg.register_function(&module, function, vec![const_node, uniform_node]);

        // In the body region:
        // arg0: test_const (u32)
        // arg1: uniform_node (u32)
        // arg2: function param 0 (u32)
        // arg3: function param 1 (u32)
        // s: state

        // 1. Add an addition node (adding param 0 and param 1)
        let add_node = rvsdg.add_op_binary(
            region,
            BinaryOperator::Add,
            ValueInput::argument(TY_U32, 2),
            ValueInput::argument(TY_U32, 3),
        );

        // Reconnect result to the addition result
        rvsdg.reconnect_region_result(
            region,
            0,
            slir::rvsdg::ValueOrigin::Output {
                producer: add_node,
                output: 0,
            },
        );

        let renderer = Renderer::new(&rvsdg, 0, 0);
        let mut writer = Vec::new();

        render_function_mode(&rvsdg, &renderer, &mut writer, "test_func").unwrap();

        let output = String::from_utf8(writer).unwrap();
        let expected = "\
[Node(3v1)] test_func(arg0: u32, arg1: u32) -> u32
  Dependencies:
    - Node(1v1) -> u32
    - Node(2v1) -> u32

  Body Region (Region(2v1)):
    Arguments: [Region(2v1)a0: u32, Region(2v1)a1: u32, Region(2v1)a2: u32, Region(2v1)a3: u32, Region(2v1)s: State]
    [Node(4v1)] OpBinary{operator: +}(Region(2v1)a2, Region(2v1)a3) -> Node(4v1)e0 : u32
    Results: [Node(4v1)e0, Region(2v1)s]
";
        assert_eq!(output, expected);
    }
}
