use internment::Intern;
use rustc_middle::bug;
use rustc_public::abi::{Layout as LayoutToken, LayoutShape, Primitive, Scalar, VariantsShape};
use rustc_public::target::{MachineInfo, MachineSize};
use rustc_public::ty::{Region, RegionKind, RigidTy, Ty, TyKind, UintTy, VariantIdx};
use rustc_public_bridge::IndexedVal;

pub type ShapeToken = Intern<ShapeWrapper>;

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct ShapeWrapper {
    shape: LayoutShape,
}

unsafe impl Send for ShapeWrapper {}
unsafe impl Sync for ShapeWrapper {}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Layout {
    Layout(LayoutToken),
    Shape(ShapeToken),
}

impl From<LayoutToken> for Layout {
    fn from(token: LayoutToken) -> Self {
        Layout::Layout(token)
    }
}

impl From<ShapeToken> for Layout {
    fn from(token: ShapeToken) -> Self {
        Layout::Shape(token)
    }
}

impl From<LayoutShape> for Layout {
    fn from(shape: LayoutShape) -> Self {
        Layout::Shape(Intern::new(ShapeWrapper {
            shape
        }))
    }
}

impl Layout {
    pub fn shape(&self) -> LayoutShape {
        match self {
            Layout::Layout(t) => t.shape(),
            Layout::Shape(s) => s.as_ref().shape.clone(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TyAndLayout {
    pub ty: Ty,
    pub layout: Layout,
}

impl TyAndLayout {
    pub fn expect_from_ty(ty: Ty) -> Self {
        let layout = ty
            .layout()
            .expect("type should have valid layout during codegen");

        TyAndLayout {
            ty,
            layout: layout.into(),
        }
    }

    pub fn field(&self, i: usize) -> TyAndLayout {
        enum TyMaybeWithLayout {
            Ty(Ty),
            TyAndLayout(TyAndLayout),
        }

        fn field_ty_or_layout(this: &TyAndLayout, i: usize) -> TyMaybeWithLayout {
            let TyKind::RigidTy(ty) = this.ty.kind() else {
                bug!("only rigid types can project to fields")
            };

            match ty {
                // Types that don't have fields.
                RigidTy::Bool
                | RigidTy::Char
                | RigidTy::Int(..)
                | RigidTy::Uint(..)
                | RigidTy::Float(..)
                | RigidTy::FnPtr(..)
                | RigidTy::Never
                | RigidTy::FnDef(..)
                | RigidTy::CoroutineWitness(..)
                | RigidTy::Foreign(..)
                | RigidTy::Pat(..)
                | RigidTy::Dynamic(..) => {
                    bug!("TyAndLayout::field({:?}): not applicable", this)
                }

                // Types that are not supported by RISL.
                RigidTy::Dynamic(..) | RigidTy::Coroutine(..) | RigidTy::CoroutineClosure(..) => {
                    bug!("Type field projection not supported by RISL for {:?}", ty)
                }

                // Potentially-wide pointers.
                RigidTy::Ref(_, pointee, mutability) | RigidTy::RawPtr(pointee, mutability) => {
                    let shape = this.layout.shape();

                    assert!(i < shape.fields.count());

                    if i == 0 {
                        let unit_ty = Ty::new_tuple(&[]);

                        let unit_ptr_ty = if matches!(ty, RigidTy::RawPtr(..)) {
                            Ty::new_ptr(unit_ty, mutability)
                        } else {
                            Ty::new_ref(
                                Region {
                                    kind: RegionKind::ReStatic,
                                },
                                unit_ty,
                                mutability,
                            )
                        };
                        let layout = unit_ptr_ty.layout().unwrap();

                        TyMaybeWithLayout::TyAndLayout(TyAndLayout {
                            ty: this.ty,
                            layout: layout.into(),
                        })
                    } else if i == 1 {
                        // We assume the second field is the size slice, as other wide pointer types
                        // are not supported by RISL, and should have triggered errors earlier.
                        let ty = Ty::usize_ty();

                        TyMaybeWithLayout::TyAndLayout(TyAndLayout::expect_from_ty(ty))
                    } else {
                        bug!()
                    }
                }

                // Arrays and slices.
                RigidTy::Array(element, _) | RigidTy::Slice(element) => {
                    TyMaybeWithLayout::Ty(element)
                }
                RigidTy::Str => {
                    TyMaybeWithLayout::Ty(Ty::from_rigid_kind(RigidTy::Uint(UintTy::U8)))
                }

                // Closures.
                RigidTy::Closure(_, args) => {
                    // The last closure argument should be a tuple type representing its up-vars
                    let Some(tupled_upvars_ty) = args.0.last().and_then(|arg| arg.ty()) else {
                        bug!(
                            "Closure args ({:?}) should declare a tuple type for its up-vars",
                            args
                        )
                    };

                    field_ty_or_layout(
                        &TyAndLayout {
                            ty: *tupled_upvars_ty,
                            layout: this.layout.clone(),
                        },
                        i,
                    )
                }

                // Tuples.
                RigidTy::Tuple(tys) => TyMaybeWithLayout::Ty(tys[i]),

                // ADTs.
                RigidTy::Adt(def, args) => {
                    match this.layout.shape().variants {
                        VariantsShape::Single { index } => {
                            let field = &def.variant(index).unwrap().fields()[i];

                            TyMaybeWithLayout::Ty(field.ty_with_args(&args))
                        }
                        VariantsShape::Empty => bug!("there is no field in Variants::Empty types"),

                        // Discriminant field for enums (where applicable).
                        VariantsShape::Multiple { tag, .. } => {
                            assert_eq!(i, 0);

                            todo!()
                            //     TyAndLayout {
                            //         layout: tcx.mk_layout(LayoutData::scalar(cx, tag)),
                            //         ty: tag.primitive().to_ty(tcx),
                            //     }
                        }
                    }
                }
            }
        }

        match field_ty_or_layout(self, i) {
            TyMaybeWithLayout::Ty(field_ty) => {
                let Ok(layout) = field_ty.layout() else {
                    bug!(
                        "failed to get layout for `{field_ty}`,\n\
                         despite it being a field (#{i}) of an existing layout: {self:#?}",
                    )
                };

                TyAndLayout {
                    ty: field_ty,
                    layout: layout.into(),
                }
            }
            TyMaybeWithLayout::TyAndLayout(field_layout) => field_layout,
        }
    }

    pub fn for_variant(self, variant_idx: VariantIdx) -> Self {
        match self.layout.shape().variants {
            VariantsShape::Single { index } if index == variant_idx => self,
            VariantsShape::Multiple { variants, .. } => {
                let variant_shape = &variants[variant_idx.to_index()];

                TyAndLayout {
                    ty: self.ty,
                    layout: variant_shape.clone().into(),
                }
            }
            _ => bug!("invalid variant"),
        }
    }
}

pub trait ScalarExt {
    fn primitive(&self) -> &Primitive;

    fn size(&self, target: &MachineInfo) -> MachineSize;
}

impl ScalarExt for Scalar {
    fn primitive(&self) -> &Primitive {
        match self {
            Scalar::Initialized { value, .. } => value,
            Scalar::Union { value } => value,
        }
    }

    fn size(&self, target: &MachineInfo) -> MachineSize {
        self.primitive().size(target)
    }
}
