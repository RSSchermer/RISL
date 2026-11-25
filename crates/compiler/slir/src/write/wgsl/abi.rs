//! Map SLIR struct layouts to matching WGSL layouts using `@align` and `@size` struct-member
//! attributes, and field ordering.
//!
//! This currently only builds layouts for structs that are used across the ABI (transitively
//! reachable from uniform or storage bindings). We currently don't support any type casting other
//! than for the basic scalar types, so for structs that are not part of the ABI, the layout is not
//! observable. If a struct's layout is not observable, we use WGSL default layout rules.

use rustc_hash::FxHashMap;

use crate::Module;
use crate::ty::{ScalarKind, Struct, Type, TypeKind, TypeRegistry};

fn add(a: u64, b: u64, path: &str) -> u64 {
    let Some(size) = a.checked_add(b) else {
        panic!("WGSL buffer layout at {path}: layout size overflow");
    };

    size
}

fn round_up(align: u64, size: u64, path: &str) -> u64 {
    assert!(
        align.is_power_of_two(),
        "WGSL buffer layout at {path}: invalid alignment {align}"
    );

    add(size, align - 1, path) & !(align - 1)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TypeLayout {
    pub align: u64,
    pub size: Option<u64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemberKind {
    Padding,
    Data(usize),
}

#[derive(Clone, Debug)]
pub struct MemberLayout {
    pub kind: MemberKind,
    pub offset: u64,
    pub align: u64,
    pub size: Option<u64>,
    pub type_layout: TypeLayout,
}

#[derive(Clone, Debug)]
pub struct StructLayout {
    pub members: Vec<MemberLayout>,
    pub layout: TypeLayout,
}

#[derive(Debug, Default)]
pub struct AbiLayouts {
    pub structs: FxHashMap<Type, StructLayout>,
    types: FxHashMap<Type, TypeLayout>,
}

impl AbiLayouts {
    pub fn build(module: &Module) -> Self {
        let mut layouts = Self::default();

        for binding in module.uniform_bindings.values() {
            let path = format!(
                "uniform @group({}) @binding({})",
                binding.resource_binding.group, binding.resource_binding.binding
            );

            layouts.compute_layout(&module.ty, binding.ty, &path);
        }

        for binding in module.storage_bindings.values() {
            let path = format!(
                "storage @group({}) @binding({})",
                binding.resource_binding.group, binding.resource_binding.binding
            );

            layouts.compute_layout(&module.ty, binding.ty, &path);
        }

        layouts
    }

    fn compute_layout(&mut self, registry: &TypeRegistry, ty: Type, path: &str) -> TypeLayout {
        if let Some(layout) = self.types.get(&ty) {
            return *layout;
        }

        let layout = match &*registry.kind(ty) {
            TypeKind::Scalar(ScalarKind::I32)
            | TypeKind::Scalar(ScalarKind::U32)
            | TypeKind::Scalar(ScalarKind::F32)
            | TypeKind::Scalar(ScalarKind::Bool)
            | TypeKind::Atomic(ScalarKind::I32)
            | TypeKind::Atomic(ScalarKind::U32)
            | TypeKind::Atomic(ScalarKind::F32)
            | TypeKind::Atomic(ScalarKind::Bool) => TypeLayout {
                align: 4,
                size: Some(4),
            },
            TypeKind::Vector(vector) => {
                let count = vector.size.to_usize() as u64;

                TypeLayout {
                    align: if count == 2 { 8 } else { 16 },
                    size: Some(4 * count),
                }
            }
            TypeKind::Matrix(matrix) => {
                assert_eq!(
                    matrix.scalar,
                    ScalarKind::F32,
                    "WGSL buffer layout at {path}: native WGSL matrices require f32 components"
                );

                let rows = matrix.rows.to_usize() as u64;
                let columns = matrix.columns.to_usize() as u64;
                let stride = if rows == 2 { 8 } else { 16 };

                TypeLayout {
                    align: stride,
                    size: Some(columns * stride),
                }
            }
            TypeKind::Array {
                element_ty,
                count,
                stride,
            } => {
                assert_ne!(
                    *count, 0,
                    "WGSL buffer layout at {path}: zero-length arrays have no native WGSL \
                    representation"
                );

                let element = self.compute_element_layout(registry, *element_ty, *stride, path);
                let Some(size) = count.checked_mul(*stride) else {
                    panic!("WGSL buffer layout at {path}: array size overflow");
                };

                TypeLayout {
                    align: element.align,
                    size: Some(size),
                }
            }
            TypeKind::Slice { element_ty, stride } => {
                let element = self.compute_element_layout(registry, *element_ty, *stride, path);

                TypeLayout {
                    align: element.align,
                    size: None,
                }
            }
            TypeKind::Struct(data) => {
                let struct_layout = self.compute_struct_layout(registry, data, path);
                let layout = struct_layout.layout;

                self.structs.insert(ty, struct_layout);

                layout
            }
            _ => {
                panic!(
                    "WGSL buffer layout at {path}: type {ty:?} has no native WGSL buffer layout"
                );
            }
        };

        self.types.insert(ty, layout);

        layout
    }

    fn compute_element_layout(
        &mut self,
        registry: &TypeRegistry,
        ty: Type,
        stride: u64,
        path: &str,
    ) -> TypeLayout {
        let path = format!("{path}[]");
        let element = self.compute_layout(registry, ty, &path);
        let Some(size) = element.size else {
            panic!("WGSL buffer layout at {path}: array elements must be sized");
        };
        let calculated = round_up(element.align, size, &path);

        assert_eq!(
            calculated, stride,
            "WGSL array stride mismatch at {path}: element {ty:?}, emitted size {size}, alignment \
            {}, calculated stride {calculated}, recorded stride {stride}",
            element.align
        );

        element
    }

    fn compute_struct_layout(
        &mut self,
        registry: &TypeRegistry,
        data: &Struct,
        path: &str,
    ) -> StructLayout {
        let Some(source_align) = data.align else {
            panic!("WGSL buffer layout at {path}: buffer struct lacks source size/alignment");
        };

        assert!(
            !data.fields.is_empty() && data.size != Some(0),
            "WGSL buffer layout at {path}: zero-byte structs have no native WGSL representation"
        );

        assert!(
            data.fields.iter().all(|field| field.io_binding.is_none()),
            "WGSL buffer layout at {path}: stage IO members cannot carry buffer layout attributes"
        );

        round_up(source_align, 0, path);

        let mut members = Vec::new();
        let mut indices: Vec<_> = (0..data.fields.len()).collect();
        indices.sort_by_key(|index| (data.fields[*index].offset, *index));

        let first_offset = data.fields[indices[0]].offset;

        if first_offset != 0 {
            // We'll have to add a padding member.

            assert!(
                first_offset >= 4 && first_offset % 4 == 0,
                "WGSL buffer layout at {path}: cannot represent leading gap of {first_offset} bytes"
            );

            members.push(MemberLayout {
                kind: MemberKind::Padding,
                offset: 0,
                align: source_align.max(4),
                size: Some(first_offset),
                type_layout: TypeLayout {
                    align: 4,
                    size: Some(4),
                },
            });
        }

        for index in indices {
            let field = &data.fields[index];
            let field_path = format!("{path}._{index}");
            let type_layout = self.compute_layout(registry, field.ty, &field_path);

            let align = if members.is_empty() {
                source_align.max(type_layout.align)
            } else {
                type_layout.align
            };

            assert_eq!(
                field.offset % align,
                0,
                "WGSL buffer layout at {field_path}: target offset {} is not a multiple of emitted \
                alignment {align}",
                field.offset
            );

            if let Some(previous) = members.last_mut() {
                let Some(size) = previous.size else {
                    panic!(
                        "WGSL buffer layout at {field_path}: only the last struct member may be \
                        unsized"
                    );
                };
                let end = add(previous.offset, size, &field_path);
                let offset = round_up(align, end, &field_path);

                assert!(
                    offset <= field.offset,
                    "WGSL buffer layout at {field_path}: overlapping or unrepresentable target \
                    offset {}; previous member ends at {end}, emitted offset would be {offset}",
                    field.offset
                );

                if offset < field.offset {
                    previous.size = Some(field.offset - previous.offset);
                }
            }

            members.push(MemberLayout {
                kind: MemberKind::Data(index),
                offset: field.offset,
                align,
                size: type_layout.size,
                type_layout,
            });
        }

        let align = members
            .iter()
            .map(|member| member.align)
            .max()
            .expect("SLIR structs should have at least one member");
        let last = members
            .last_mut()
            .expect("SLIR structs should have at least one member");

        match (data.size, last.size) {
            (Some(struct_size), Some(last_size)) => {
                // The size of the struct layout we've constructed so far may not yet match the
                // expected struct size. If it is smaller, then we compensate by increasing the
                // size of the final member, to introduce the necessary padding at the end.

                let end = add(last.offset, last_size, path);
                let calculated = round_up(align, end, path);

                assert!(
                    calculated <= struct_size && struct_size % align == 0,
                    "WGSL buffer layout at {path}: source size {struct_size} cannot contain WGSL \
                    extent {calculated} with alignment {align}"
                );

                if calculated < struct_size {
                    last.size = Some(struct_size - last.offset);
                }
            }
            (None, None) => {}
            _ => {
                panic!("WGSL buffer layout at {path}: source and WGSL disagree on size");
            }
        }

        StructLayout {
            members,
            layout: TypeLayout {
                align,
                size: data.size,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ty::{StructField, TY_U32, TY_VEC3_F32};
    use crate::{ResourceBinding, StorageBindingData, Symbol};

    #[test]
    fn members_follow_physical_offsets_without_changing_logical_fields() {
        let mut module = Module::new(Symbol::new("layout_tests".into()));

        let array = module.ty.register(TypeKind::Array {
            element_ty: TY_U32,
            count: 2,
            stride: 4,
        });
        let ty = module.ty.register(TypeKind::Struct(Struct {
            fields: vec![
                StructField {
                    offset: 8,
                    ty: TY_U32,
                    io_binding: None,
                },
                StructField {
                    offset: 0,
                    ty: array,
                    io_binding: None,
                },
                StructField {
                    offset: 12,
                    ty: TY_U32,
                    io_binding: None,
                },
            ],
            align: Some(4),
            size: Some(16),
        }));

        module.storage_bindings.register(StorageBindingData {
            ty,
            resource_binding: ResourceBinding {
                group: 2,
                binding: 3,
            },
            writable: true,
        });

        let layouts = AbiLayouts::build(&module);
        let members = &layouts.structs[&ty].members;

        assert_eq!(
            members.iter().map(|m| m.kind).collect::<Vec<_>>(),
            [
                MemberKind::Data(1),
                MemberKind::Data(0),
                MemberKind::Data(2)
            ]
        );
        assert_eq!(module.ty.kind(ty).expect_struct().fields[0].offset, 8);
        assert!(
            members
                .iter()
                .all(|m| m.size == m.type_layout.size && m.align == m.type_layout.align)
        );
    }

    #[test]
    fn alignment_supplies_tail_padding_without_redundant_size() {
        let mut module = Module::new(Symbol::new("layout_tests".into()));

        let inner = module.ty.register(TypeKind::Struct(Struct {
            fields: vec![
                StructField {
                    offset: 0,
                    ty: TY_U32,
                    io_binding: None,
                },
                StructField {
                    offset: 4,
                    ty: TY_U32,
                    io_binding: None,
                },
            ],
            align: Some(32),
            size: Some(32),
        }));
        let outer = module.ty.register(TypeKind::Struct(Struct {
            fields: vec![
                StructField {
                    offset: 0,
                    ty: inner,
                    io_binding: None,
                },
                StructField {
                    offset: 32,
                    ty: TY_U32,
                    io_binding: None,
                },
            ],
            align: Some(32),
            size: Some(64),
        }));
        let array = module.ty.register(TypeKind::Array {
            element_ty: outer,
            count: 2,
            stride: 64,
        });

        module.storage_bindings.register(StorageBindingData {
            ty: array,
            resource_binding: ResourceBinding {
                group: 2,
                binding: 3,
            },
            writable: true,
        });

        let layouts = AbiLayouts::build(&module);

        assert_eq!(layouts.structs[&inner].members[0].align, 32);
        assert_eq!(layouts.structs[&inner].members[1].size, Some(4));
        assert_eq!(layouts.structs[&outer].members[1].offset, 32);
        assert_eq!(
            layouts.types[&array],
            TypeLayout {
                size: Some(128),
                align: 32
            }
        );
    }

    #[test]
    fn accounts_for_inter_member_padding_and_tail_padding() {
        let mut module = Module::new(Symbol::new("layout_tests".into()));

        let ty = module.ty.register(TypeKind::Struct(Struct {
            fields: vec![
                StructField {
                    offset: 0,
                    ty: TY_U32,
                    io_binding: None,
                },
                StructField {
                    offset: 12,
                    ty: TY_U32,
                    io_binding: None,
                },
            ],
            align: Some(4),
            size: Some(24),
        }));

        module.storage_bindings.register(StorageBindingData {
            ty,
            resource_binding: ResourceBinding {
                group: 2,
                binding: 3,
            },
            writable: true,
        });

        let layouts = AbiLayouts::build(&module);
        let members = &layouts.structs[&ty].members;

        assert_eq!(members[0].size, Some(12));
        assert_eq!(members[1].size, Some(12));
    }

    #[test]
    fn add_leading_padding_if_the_smallest_offset_is_not_zero() {
        let mut module = Module::new(Symbol::new("layout_tests".into()));

        let ty = module.ty.register(TypeKind::Struct(Struct {
            fields: vec![StructField {
                offset: 8,
                ty: TY_U32,
                io_binding: None,
            }],
            align: Some(8),
            size: Some(16),
        }));

        module.storage_bindings.register(StorageBindingData {
            ty,
            resource_binding: ResourceBinding {
                group: 2,
                binding: 3,
            },
            writable: true,
        });

        let layouts = AbiLayouts::build(&module);
        let members = &layouts.structs[&ty].members;

        assert_eq!(members[0].kind, MemberKind::Padding);
        assert_eq!(members[0].size, Some(8));
        assert_eq!(members[0].align, 8);
        assert_eq!(members[1].kind, MemberKind::Data(0));
    }

    #[test]
    fn member_size_is_only_adjusted_when_sibling_alignment_would_not_produce_the_correct_offset() {
        let mut module = Module::new(Symbol::new("layout_tests".into()));

        let scalar = module.ty.register(TypeKind::Struct(Struct {
            fields: vec![
                StructField {
                    offset: 0,
                    ty: TY_VEC3_F32,
                    io_binding: None,
                },
                StructField {
                    offset: 16,
                    ty: TY_U32,
                    io_binding: None,
                },
            ],
            align: Some(16),
            size: Some(32),
        }));
        let vector = module.ty.register(TypeKind::Struct(Struct {
            fields: vec![
                StructField {
                    offset: 0,
                    ty: TY_VEC3_F32,
                    io_binding: None,
                },
                StructField {
                    offset: 16,
                    ty: TY_VEC3_F32,
                    io_binding: None,
                },
            ],
            align: Some(16),
            size: Some(32),
        }));

        module.storage_bindings.register(StorageBindingData {
            ty: scalar,
            resource_binding: ResourceBinding {
                group: 2,
                binding: 3,
            },
            writable: true,
        });
        module.storage_bindings.register(StorageBindingData {
            ty: vector,
            resource_binding: ResourceBinding {
                group: 2,
                binding: 3,
            },
            writable: true,
        });

        let layouts = AbiLayouts::build(&module);

        assert_eq!(layouts.structs[&scalar].members[0].size, Some(16));
        assert_eq!(layouts.structs[&vector].members[0].size, Some(12));
        assert_eq!(layouts.structs[&vector].members[1].size, Some(12));
    }

    #[test]
    fn nested_arrays_and_runtime_slice_use_native_stride() {
        let mut module = Module::new(Symbol::new("layout_tests".into()));

        let ty = module.ty.register(TypeKind::Struct(Struct {
            fields: vec![
                StructField {
                    offset: 0,
                    ty: TY_U32,
                    io_binding: None,
                },
                StructField {
                    offset: 4,
                    ty: TY_U32,
                    io_binding: None,
                },
            ],
            align: Some(32),
            size: Some(32),
        }));
        let array = module.ty.register(TypeKind::Array {
            element_ty: ty,
            count: 2,
            stride: 32,
        });
        let nested = module.ty.register(TypeKind::Array {
            element_ty: array,
            count: 3,
            stride: 64,
        });
        let slice = module.ty.register(TypeKind::Slice {
            element_ty: nested,
            stride: 192,
        });

        module.storage_bindings.register(StorageBindingData {
            ty: slice,
            resource_binding: ResourceBinding {
                group: 2,
                binding: 3,
            },
            writable: true,
        });

        let layouts = AbiLayouts::build(&module);

        assert_eq!(layouts.types[&nested].size, Some(192));
        assert_eq!(
            layouts.types[&slice],
            TypeLayout {
                size: None,
                align: 32
            }
        );
    }

    #[test]
    fn unsized_struct_keeps_tail_offset_and_has_no_fixed_size() {
        let mut module = Module::new(Symbol::new("layout_tests".into()));

        let slice = module.ty.register(TypeKind::Slice {
            element_ty: TY_U32,
            stride: 4,
        });
        let ty = module.ty.register(TypeKind::Struct(Struct {
            fields: vec![
                StructField {
                    offset: 0,
                    ty: TY_U32,
                    io_binding: None,
                },
                StructField {
                    offset: 16,
                    ty: slice,
                    io_binding: None,
                },
            ],
            align: Some(4),
            size: None,
        }));

        module.storage_bindings.register(StorageBindingData {
            ty,
            resource_binding: ResourceBinding {
                group: 2,
                binding: 3,
            },
            writable: true,
        });

        let layouts = AbiLayouts::build(&module);

        assert_eq!(layouts.structs[&ty].layout.size, None);
        assert_eq!(layouts.structs[&ty].members[0].size, Some(16));
        assert_eq!(layouts.structs[&ty].members[1].size, None);
    }
}
