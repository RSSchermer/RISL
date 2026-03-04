use std::io::Write;

use anyhow::Result;
use slir::ty::Type;

use crate::id_resolution;
use crate::renderer::Renderer;

pub fn render_type_inspect_mode<W: Write>(
    renderer: &Renderer,
    writer: &mut W,
    ty_id_str: &str,
) -> Result<()> {
    let ty_id = id_resolution::parse_type_id(ty_id_str)?;
    let ty = Type::from_registration_id(ty_id as usize);

    renderer.write_type_detail(writer, ty)?;
    writeln!(writer)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use slir::rvsdg::Rvsdg;
    use slir::ty::{Enum, Struct, StructField, TY_F32, TY_U32, TypeKind, TypeRegistry};

    use super::*;
    use crate::renderer::Renderer;

    #[test]
    fn test_render_type_struct() {
        let registry = TypeRegistry::default();
        let struct_ty = registry.register(TypeKind::Struct(Struct {
            fields: vec![
                StructField {
                    ty: TY_U32,
                    offset: 0,
                    io_binding: None,
                },
                StructField {
                    ty: TY_F32,
                    offset: 4,
                    io_binding: None,
                },
            ],
        }));

        let rvsdg = Rvsdg::new(registry.into());
        let renderer = Renderer::new(&rvsdg, 0, 0, true);
        let mut writer = Vec::new();

        let ty_id_str = format!("struct({})", struct_ty.registration_id().unwrap());
        render_type_inspect_mode(&renderer, &mut writer, &ty_id_str).unwrap();

        let output = String::from_utf8(writer).unwrap();
        let expected = "\
struct(0):
  - field 0: u32 (offset: 0)
  - field 1: f32 (offset: 4)
";
        assert_eq!(output, expected);
    }

    #[test]
    fn test_render_type_enum() {
        let registry = TypeRegistry::default();
        let enum_ty = registry.register(TypeKind::Enum(Enum {
            variants: vec![TY_U32, TY_F32],
        }));

        let rvsdg = Rvsdg::new(registry.into());
        let renderer = Renderer::new(&rvsdg, 0, 0, true);
        let mut writer = Vec::new();

        let ty_id_str = format!("enum({})", enum_ty.registration_id().unwrap());
        render_type_inspect_mode(&renderer, &mut writer, &ty_id_str).unwrap();

        let output = String::from_utf8(writer).unwrap();
        let expected = "\
enum(0):
  - variant 0: u32
  - variant 1: f32
";
        assert_eq!(output, expected);
    }
}
