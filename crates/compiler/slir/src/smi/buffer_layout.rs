use risl_smi::{ArrayLayout, MemoryUnit, MemoryUnitLayout, UnsizedTailLayout};
use rustc_hash::FxHashMap;

use crate::ty;
use crate::ty::{ScalarKind, Struct, Type, TypeKind, TypeRegistry, VectorSize};

#[derive(Clone, PartialEq, Debug)]
pub struct Layout {
    pub head: Vec<MemoryUnit>,
    pub tail: Option<UnsizedTailLayout>,
}

pub struct BufferLayoutProvider {
    cache: FxHashMap<Type, Layout>,
}

impl BufferLayoutProvider {
    pub fn new() -> Self {
        Self {
            cache: Default::default(),
        }
    }

    pub fn layout(&mut self, ty_registry: &TypeRegistry, ty: Type) -> &Layout {
        self.cache.entry(ty).or_insert_with(|| {
            let mut layout_builder = MemoryLayoutBuilder::new();

            layout_builder.visit_ty(ty_registry, ty, 0);

            Layout {
                head: layout_builder.head,
                tail: layout_builder.tail,
            }
        });

        &self.cache[&ty]
    }
}

struct MemoryLayoutBuilder {
    head: Vec<MemoryUnit>,
    tail: Option<UnsizedTailLayout>,
}

impl MemoryLayoutBuilder {
    fn new() -> Self {
        Self {
            head: Vec::new(),
            tail: None,
        }
    }

    fn visit_ty(&mut self, ty_registry: &TypeRegistry, ty: Type, offset: u64) {
        match &*ty_registry.kind(ty) {
            TypeKind::Scalar(kind) => self.visit_scalar(*kind, offset),
            TypeKind::Atomic(kind) => self.visit_scalar(*kind, offset),
            TypeKind::Vector(v) => self.visit_vector(v, offset),
            TypeKind::Matrix(m) => self.visit_matrix(m, offset),
            TypeKind::Array {
                element_ty,
                count,
                stride,
            } => self.visit_array(ty_registry, *element_ty, *count, *stride, offset),
            TypeKind::Slice { element_ty, stride } => {
                self.visit_slice(ty_registry, *element_ty, *stride, offset)
            }
            TypeKind::Struct(decl) => self.visit_struct(ty_registry, decl, offset),
            TypeKind::Enum(_)
            | TypeKind::Ptr(_)
            | TypeKind::Function(_)
            | TypeKind::Predicate
            | TypeKind::Dummy => panic!("type is not ABI compatible ({:?})", ty),
        }
    }

    fn visit_scalar(&mut self, kind: ScalarKind, offset: u64) {
        let layout = match kind {
            ScalarKind::I32 => MemoryUnitLayout::Integer,
            ScalarKind::U32 => MemoryUnitLayout::UnsignedInteger,
            ScalarKind::F32 => MemoryUnitLayout::Float,
            ScalarKind::Bool => panic!("`bool` type is not ABI compatible"),
        };

        self.head.push(MemoryUnit { offset, layout });
    }

    fn visit_vector(&mut self, ty: &ty::Vector, offset: u64) {
        use ScalarKind::*;
        use VectorSize::*;

        let layout = match (ty.scalar, ty.size) {
            (I32, Two) => MemoryUnitLayout::IntegerVector2,
            (I32, Three) => MemoryUnitLayout::IntegerVector3,
            (I32, Four) => MemoryUnitLayout::IntegerVector4,
            (U32, Two) => MemoryUnitLayout::UnsignedIntegerVector2,
            (U32, Three) => MemoryUnitLayout::UnsignedIntegerVector3,
            (U32, Four) => MemoryUnitLayout::UnsignedIntegerVector4,
            (F32, Two) => MemoryUnitLayout::FloatVector2,
            (F32, Three) => MemoryUnitLayout::FloatVector3,
            (F32, Four) => MemoryUnitLayout::FloatVector4,
            (Bool, _) => panic!("`bool` type is not ABI compatible"),
        };

        self.head.push(MemoryUnit { offset, layout });
    }

    fn visit_matrix(&mut self, ty: &ty::Matrix, offset: u64) {
        use ScalarKind::*;
        use VectorSize::*;

        let layout = match (ty.scalar, ty.columns, ty.rows) {
            (F32, Two, Two) => MemoryUnitLayout::Matrix2x2,
            (F32, Two, Three) => MemoryUnitLayout::Matrix2x3,
            (F32, Two, Four) => MemoryUnitLayout::Matrix2x4,
            (F32, Three, Two) => MemoryUnitLayout::Matrix3x2,
            (F32, Three, Three) => MemoryUnitLayout::Matrix3x3,
            (F32, Three, Four) => MemoryUnitLayout::Matrix3x4,
            (F32, Four, Two) => MemoryUnitLayout::Matrix4x2,
            (F32, Four, Three) => MemoryUnitLayout::Matrix4x3,
            (F32, Four, Four) => MemoryUnitLayout::Matrix4x4,
            _ => panic!("only float-type matrices are ABI compatible"),
        };

        self.head.push(MemoryUnit { offset, layout });
    }

    fn visit_array(
        &mut self,
        ty_registry: &TypeRegistry,
        element_ty: Type,
        len: u64,
        stride: u64,
        offset: u64,
    ) {
        let mut element_layout_builder = MemoryLayoutBuilder::new();

        element_layout_builder.visit_ty(ty_registry, element_ty, 0);

        assert!(
            element_layout_builder.tail.is_none(),
            "the element type of an array must be sized"
        );

        let layout = ArrayLayout {
            element_layout: element_layout_builder.head.into(),
            stride,
            len,
        };

        self.head.push(MemoryUnit {
            offset,
            layout: MemoryUnitLayout::Array(layout),
        });
    }

    fn visit_slice(
        &mut self,
        ty_registry: &TypeRegistry,
        element_ty: Type,
        stride: u64,
        offset: u64,
    ) {
        let mut element_layout_builder = MemoryLayoutBuilder::new();

        element_layout_builder.visit_ty(&ty_registry, element_ty, 0);

        assert!(
            element_layout_builder.tail.is_none(),
            "the element type of an array must be sized"
        );

        self.tail = Some(UnsizedTailLayout {
            offset,
            element_layout: element_layout_builder.head.into(),
            stride,
        });
    }

    fn visit_struct(&mut self, ty_registry: &TypeRegistry, decl: &Struct, offset: u64) {
        for field in &decl.fields {
            assert!(
                self.tail.is_none(),
                "if a layout contains an unsized array, it must be the last element in the layout"
            );

            self.visit_ty(&ty_registry, field.ty, offset + field.offset);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ty::{StructField, TY_F32, TY_U32, TY_VEC4_F32};
    use crate::{Module, Symbol};

    #[test]
    fn test_sized_struct_layout() {
        let mut module = Module::new(Symbol::from_ref(""));

        let element_ty = module.ty.register(TypeKind::Struct(Struct {
            fields: vec![
                StructField {
                    offset: 0,
                    ty: TY_U32,
                    io_binding: None,
                },
                StructField {
                    offset: 4,
                    ty: TY_F32,
                    io_binding: None,
                },
            ],
        }));
        let array_ty = module.ty.register(TypeKind::Array {
            element_ty,
            count: 4,
            stride: 8,
        });
        let ty = module.ty.register(TypeKind::Struct(Struct {
            fields: vec![
                StructField {
                    offset: 0,
                    ty: TY_VEC4_F32,
                    io_binding: None,
                },
                StructField {
                    offset: 16,
                    ty: TY_F32,
                    io_binding: None,
                },
                StructField {
                    offset: 20,
                    ty: array_ty,
                    io_binding: None,
                },
            ],
        }));

        let mut layout_provider = BufferLayoutProvider::new();

        let layout = layout_provider.layout(&module.ty, ty);

        assert_eq!(
            layout,
            &Layout {
                head: vec![
                    MemoryUnit {
                        offset: 0,
                        layout: MemoryUnitLayout::FloatVector4
                    },
                    MemoryUnit {
                        offset: 16,
                        layout: MemoryUnitLayout::Float
                    },
                    MemoryUnit {
                        offset: 20,
                        layout: MemoryUnitLayout::Array(ArrayLayout {
                            element_layout: vec![
                                MemoryUnit {
                                    offset: 0,
                                    layout: MemoryUnitLayout::UnsignedInteger
                                },
                                MemoryUnit {
                                    offset: 4,
                                    layout: MemoryUnitLayout::Float
                                },
                            ]
                            .into(),
                            stride: 8,
                            len: 4
                        })
                    },
                ],
                tail: None,
            }
        )
    }

    #[test]
    fn test_unsized_struct_layout() {
        let mut module = Module::new(Symbol::from_ref(""));

        let element_ty = module.ty.register(TypeKind::Struct(Struct {
            fields: vec![
                StructField {
                    offset: 0,
                    ty: TY_U32,
                    io_binding: None,
                },
                StructField {
                    offset: 4,
                    ty: TY_F32,
                    io_binding: None,
                },
            ],
        }));
        let slice_ty = module.ty.register(TypeKind::Slice {
            element_ty,
            stride: 8,
        });
        let ty = module.ty.register(TypeKind::Struct(Struct {
            fields: vec![
                StructField {
                    offset: 0,
                    ty: TY_VEC4_F32,
                    io_binding: None,
                },
                StructField {
                    offset: 16,
                    ty: TY_F32,
                    io_binding: None,
                },
                StructField {
                    offset: 20,
                    ty: slice_ty,
                    io_binding: None,
                },
            ],
        }));

        let mut layout_provider = BufferLayoutProvider::new();

        let layout = layout_provider.layout(&module.ty, ty);

        assert_eq!(
            layout,
            &Layout {
                head: vec![
                    MemoryUnit {
                        offset: 0,
                        layout: MemoryUnitLayout::FloatVector4
                    },
                    MemoryUnit {
                        offset: 16,
                        layout: MemoryUnitLayout::Float
                    },
                ],
                tail: Some(UnsizedTailLayout {
                    offset: 20,
                    element_layout: vec![
                        MemoryUnit {
                            offset: 0,
                            layout: MemoryUnitLayout::UnsignedInteger
                        },
                        MemoryUnit {
                            offset: 4,
                            layout: MemoryUnitLayout::Float
                        },
                    ]
                    .into(),
                    stride: 8,
                }),
            }
        )
    }
}
