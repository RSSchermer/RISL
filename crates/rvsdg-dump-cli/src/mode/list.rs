use std::io::Write;

use anyhow::Result;
use slir::rvsdg::{Connectivity, NodeKind, Rvsdg};

use crate::renderer::Renderer;
pub fn render_list_mode<W: Write>(
    rvsdg: &Rvsdg,
    renderer: &Renderer,
    writer: &mut W,
) -> Result<()> {
    writeln!(writer, "Functions:")?;
    for (function, node) in rvsdg.registered_functions() {
        writeln!(
            writer,
            "  - {}: {}{}",
            renderer.format_node_id(node),
            function.name.as_str(),
            renderer.format_function_signature(node)
        )?;
    }

    writeln!(writer, "\nGlobal Bindings:")?;
    let global_region = rvsdg.global_region();
    for &node in rvsdg[global_region].nodes() {
        let node_data = &rvsdg[node];
        match node_data.kind() {
            NodeKind::UniformBinding(_) => {
                writeln!(
                    writer,
                    "  - Uniform: {} -> {}",
                    renderer.format_node_id(node),
                    renderer.format_type(node_data.value_outputs()[0].ty)
                )?;
            }
            NodeKind::StorageBinding(_) => {
                writeln!(
                    writer,
                    "  - Storage: {} -> {}",
                    renderer.format_node_id(node),
                    renderer.format_type(node_data.value_outputs()[0].ty)
                )?;
            }
            NodeKind::WorkgroupBinding(_) => {
                writeln!(
                    writer,
                    "  - Workgroup: {} -> {}",
                    renderer.format_node_id(node),
                    renderer.format_type(node_data.value_outputs()[0].ty)
                )?;
            }
            _ => {}
        }
    }

    writeln!(writer, "\nGlobal Constants:")?;
    for &node in rvsdg[global_region].nodes() {
        let node_data = &rvsdg[node];
        if let NodeKind::Constant(c) = node_data.kind() {
            writeln!(
                writer,
                "  - {} -> {} (\"{}\")",
                renderer.format_node_id(node),
                renderer.format_type(node_data.value_outputs()[0].ty),
                c.constant().name.as_str()
            )?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::iter;

    use slir::ty::{TY_DUMMY, TY_U32};
    use slir::{Constant, FnArg, FnSig, Function, Module, ResourceBinding, Symbol};

    use super::*;

    #[test]
    fn test_render_list() {
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
                    ty: TY_U32,
                    shader_io_binding: None,
                }],
                ret_ty: Some(TY_U32),
            },
        );

        let mut rvsdg = Rvsdg::new(module.ty.clone());

        // Register function
        rvsdg.register_function(&module, function, iter::empty());

        // Register a uniform binding
        use slir::{StorageBindingData, UniformBindingData, WorkgroupBindingData};
        let uniform_binding = module.uniform_bindings.register(UniformBindingData {
            ty: TY_U32,
            resource_binding: ResourceBinding {
                group: 0,
                binding: 0,
            },
        });
        rvsdg.register_uniform_binding(&module, uniform_binding);

        // Register a storage binding
        let storage_binding = module.storage_bindings.register(StorageBindingData {
            ty: TY_U32,
            resource_binding: ResourceBinding {
                group: 0,
                binding: 1,
            },
            writable: false,
        });
        rvsdg.register_storage_binding(&module, storage_binding);

        // Register a workgroup binding
        let workgroup_binding = module
            .workgroup_bindings
            .register(WorkgroupBindingData { ty: TY_U32 });
        rvsdg.register_workgroup_binding(&module, workgroup_binding);

        // Register a constant
        let constant = Constant {
            name: Symbol::from_ref("test_const"),
            module: Symbol::from_ref("test_module"),
        };
        module
            .constants
            .register_byte_data(constant, TY_U32, vec![42, 0, 0, 0]);
        rvsdg.register_constant(&module, constant);

        let renderer = Renderer::new(&rvsdg, 0, 0, true);
        let mut writer = Vec::new();
        render_list_mode(&rvsdg, &renderer, &mut writer).unwrap();

        let output = String::from_utf8(writer).unwrap();
        let expected = "\
Functions:
  - Node(1v1): test_func(arg0: u32) -> u32

Global Bindings:
  - Uniform: Node(2v1) -> u32
  - Storage: Node(3v1) -> u32
  - Workgroup: Node(4v1) -> u32

Global Constants:
  - Node(5v1) -> u32 (\"test_const\")
";
        assert_eq!(output, expected);
    }
}
