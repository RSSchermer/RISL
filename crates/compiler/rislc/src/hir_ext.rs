use indexmap::IndexMap;
use rustc_ast::{IsAuto, Mutability};
use rustc_hir::{
    BodyId, ConstItemRhs, Constness, EnumDef, FnSig, GenericBounds, Generics, HirId, Impl, Item,
    ItemId, ItemKind, Mod, Safety, TraitItemId, Ty, VariantData,
};
use rustc_span::def_id::{DefId, LocalDefId, LocalModDefId};
use rustc_span::source_map::Spanned;
use rustc_span::{Ident, Span, Symbol};

pub struct HirExt {
    pub shader_requests: Vec<ShaderRequest>,
    pub mod_ext: IndexMap<LocalModDefId, ModExt>,
    pub fn_ext: IndexMap<LocalDefId, FnExt>,
    pub impl_fn_ext: IndexMap<HirId, GpuFnExt>,
    pub trait_fn_ext: IndexMap<HirId, GpuFnExt>,
    pub struct_ext: IndexMap<ItemId, StructExt>,
    pub enum_ext: IndexMap<ItemId, EnumExt>,
    pub trait_ext: IndexMap<ItemId, TraitExt>,
    pub const_ext: IndexMap<ItemId, ConstExt>,
    pub static_ext: IndexMap<ItemId, StaticExt>,
    pub impl_ext: IndexMap<ItemId, ImplExt>,
    pub param_ext: IndexMap<HirId, ParamExt>,
    pub generic_param_ext: IndexMap<HirId, GenericParamExt>,
    pub field_ext: IndexMap<LocalDefId, FieldExt>,
}

impl HirExt {
    pub fn new() -> Self {
        HirExt {
            shader_requests: vec![],
            mod_ext: Default::default(),
            fn_ext: Default::default(),
            impl_fn_ext: Default::default(),
            trait_fn_ext: Default::default(),
            struct_ext: Default::default(),
            enum_ext: Default::default(),
            trait_ext: Default::default(),
            const_ext: Default::default(),
            static_ext: Default::default(),
            impl_ext: Default::default(),
            param_ext: Default::default(),
            generic_param_ext: Default::default(),
            field_ext: Default::default(),
        }
    }

    pub fn extend_item<'ext, 'hir>(
        &'ext self,
        item: &'hir Item,
    ) -> Option<ExtendedItem<'hir, 'ext>> {
        match &item.kind {
            ItemKind::Static(mutability, ident, ty, body_id) => self
                .static_ext
                .get(&item.item_id())
                .map(|ext| ExtendedItem {
                    item,
                    kind: ExtendedItemKind::Static(*mutability, *ident, ty, *body_id, ext),
                }),
            ItemKind::Const(ident, generics, ty, rhs) => {
                self.const_ext.get(&item.item_id()).map(|ext| ExtendedItem {
                    item,
                    kind: ExtendedItemKind::Const(*ident, generics, ty, *rhs, ext),
                })
            }
            ItemKind::Fn {
                sig,
                generics,
                body,
                ..
            } => self
                .fn_ext
                .get(&item.item_id().owner_id.def_id)
                .map(|ext| ExtendedItem {
                    item,
                    kind: ExtendedItemKind::Fn(sig, generics, *body, ext),
                }),
            ItemKind::Mod(ident, m) => {
                let id = LocalModDefId::new_unchecked(item.owner_id.def_id);

                self.mod_ext.get(&id).map(|ext| ExtendedItem {
                    item,
                    kind: ExtendedItemKind::Mod(*ident, m, ext),
                })
            }
            ItemKind::Struct(ident, generics, variant_data) => self
                .struct_ext
                .get(&item.item_id())
                .map(|ext| ExtendedItem {
                    item,
                    kind: ExtendedItemKind::Struct(*ident, generics, variant_data, ext),
                }),
            ItemKind::Enum(ident, generics, enum_def) => {
                self.enum_ext.get(&item.item_id()).map(|ext| ExtendedItem {
                    item,
                    kind: ExtendedItemKind::Enum(*ident, generics, enum_def, ext),
                })
            }
            ItemKind::Trait(constness, is_auto, safety, ident, generics, bounds, items) => {
                self.trait_ext.get(&item.item_id()).map(|ext| ExtendedItem {
                    item,
                    kind: ExtendedItemKind::Trait(
                        *constness, *is_auto, *safety, *ident, generics, bounds, items, ext,
                    ),
                })
            }
            ItemKind::Impl(i) => self.impl_ext.get(&item.item_id()).map(|ext| ExtendedItem {
                item,
                kind: ExtendedItemKind::Impl(i, ext),
            }),
            _ => None,
        }
    }

    pub fn get_mod_ext(&self, mod_: LocalModDefId) -> Option<&ModExt> {
        self.mod_ext.get(&mod_)
    }

    pub fn param_ext(&self, id: HirId) -> Option<&ParamExt> {
        self.param_ext.get(&id)
    }

    pub fn expect_param_ext(&self, id: HirId) -> &ParamExt {
        self.param_ext(id).expect("node does not have a `ParamExt`")
    }

    pub fn generic_param_ext(&self, id: HirId) -> Option<&GenericParamExt> {
        self.generic_param_ext.get(&id)
    }

    pub fn expect_generic_param_ext(&self, id: HirId) -> &GenericParamExt {
        self.generic_param_ext(id)
            .expect("node does not have a `GenericParamExt`")
    }

    pub fn fn_ext(&self, id: LocalDefId) -> Option<&FnExt> {
        self.fn_ext.get(&id)
    }

    pub fn expect_fn_ext(&self, id: LocalDefId) -> &FnExt {
        self.fn_ext(id).expect("node does not have a `FnExt`")
    }

    pub fn static_ext(&self, id: ItemId) -> Option<&StaticExt> {
        self.static_ext.get(&id)
    }

    pub fn expect_static_ext(&self, id: ItemId) -> &StaticExt {
        self.static_ext(id)
            .expect("node does not have a `StaticExt`")
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ShaderRequestKind {
    Wgsl,
    ShaderModuleInterface,
}

#[derive(Clone, Debug)]
pub struct ShaderRequest {
    pub shader_mod: DefId,
    pub span: Span,
    pub request_id: Symbol,
    pub kind: ShaderRequestKind,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum BlendSrc {
    Zero,
    One,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum ShaderIOBinding {
    VertexIndex,
    InstanceIndex,
    Position {
        invariant: Option<Span>,
    },
    FrontFacing,
    FragDepth,
    SampleIndex,
    SampleMask,
    LocalInvocationId,
    LocalInvocationIndex,
    GlobalInvocationId,
    WorkgroupId,
    NumWorkgroups,
    Location {
        location: Spanned<u32>,
        blend_src: Option<Spanned<BlendSrc>>,
        interpolation: Option<Interpolation>,
    },
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Interpolation {
    pub tpe: InterpolationType,
    pub sampling: Option<InterpolationSampling>,
    pub span: Span,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum InterpolationType {
    Perspective,
    Linear,
    Flat,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum InterpolationSampling {
    Center,
    Centroid,
    Sample,
    First,
    Either,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct ParamExt {
    pub shader_io_binding: Option<ShaderIOBinding>,
}

#[derive(Debug)]
pub struct GenericParamExt {
    pub specialize: bool,
}

#[derive(Debug)]
pub struct ModExt {
    pub is_shader_module: bool,
}

#[derive(Debug)]
pub struct WorkgroupSize {
    pub x: u32,
    pub y: u32,
    pub z: u32,
    pub span: Span,
}

#[derive(Debug)]
pub struct ExtendedItem<'hir, 'ext> {
    pub item: &'hir Item<'hir>,
    pub kind: ExtendedItemKind<'hir, 'ext>,
}

impl<'hir, 'ext> ExtendedItem<'hir, 'ext> {
    pub fn expect_fn(self) -> (&'hir FnSig<'hir>, &'hir Generics<'hir>, BodyId, &'ext FnExt) {
        if let ExtendedItemKind::Fn(sig, generics, body, ext) = self.kind {
            (sig, generics, body, ext)
        } else {
            panic!("expected fn")
        }
    }

    pub fn expect_impl(self) -> (&'hir Impl<'hir>, &'ext ImplExt) {
        if let ExtendedItemKind::Impl(i, ext) = self.kind {
            (i, ext)
        } else {
            panic!("expected impl")
        }
    }

    pub fn expect_trait(
        self,
    ) -> (
        Constness,
        IsAuto,
        Safety,
        Ident,
        &'hir Generics<'hir>,
        GenericBounds<'hir>,
        &'hir [TraitItemId],
        &'ext TraitExt,
    ) {
        if let ExtendedItemKind::Trait(
            constness,
            is_auto,
            safety,
            ident,
            generics,
            bounds,
            items,
            ext,
        ) = self.kind
        {
            (
                constness, is_auto, safety, ident, generics, bounds, items, ext,
            )
        } else {
            panic!("expected trait")
        }
    }

    pub fn expect_struct(
        self,
    ) -> (
        Ident,
        &'hir Generics<'hir>,
        &'hir VariantData<'hir>,
        &'ext StructExt,
    ) {
        if let ExtendedItemKind::Struct(ident, generics, variant_data, ext) = self.kind {
            (ident, generics, variant_data, ext)
        } else {
            panic!("expected struct")
        }
    }

    pub fn expect_enum(
        self,
    ) -> (
        Ident,
        &'hir Generics<'hir>,
        &'hir EnumDef<'hir>,
        &'ext EnumExt,
    ) {
        if let ExtendedItemKind::Enum(ident, generics, variant_data, ext) = self.kind {
            (ident, generics, variant_data, ext)
        } else {
            panic!("expected enum")
        }
    }

    pub fn expect_const(
        self,
    ) -> (
        Ident,
        &'hir Generics<'hir>,
        &'hir Ty<'hir>,
        ConstItemRhs<'hir>,
        &'ext ConstExt,
    ) {
        if let ExtendedItemKind::Const(ident, generics, ty, rhs, ext) = self.kind {
            (ident, generics, ty, rhs, ext)
        } else {
            panic!("expected const")
        }
    }

    pub fn expect_static(self) -> (Mutability, Ident, &'hir Ty<'hir>, BodyId, &'ext StaticExt) {
        if let ExtendedItemKind::Static(mutability, ident, ty, body_id, ext) = self.kind {
            (mutability, ident, ty, body_id, ext)
        } else {
            panic!("expected static")
        }
    }

    pub fn expect_mod(self) -> (Ident, &'hir Mod<'hir>, &'ext ModExt) {
        if let ExtendedItemKind::Mod(ident, m, ext) = self.kind {
            (ident, m, ext)
        } else {
            panic!("expected mod")
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum ExtendedItemKind<'hir, 'ext> {
    Fn(&'hir FnSig<'hir>, &'hir Generics<'hir>, BodyId, &'ext FnExt),
    Impl(&'hir Impl<'hir>, &'ext ImplExt),
    Trait(
        Constness,
        IsAuto,
        Safety,
        Ident,
        &'hir Generics<'hir>,
        GenericBounds<'hir>,
        &'hir [TraitItemId],
        &'ext TraitExt,
    ),
    Struct(
        Ident,
        &'hir Generics<'hir>,
        &'hir VariantData<'hir>,
        &'ext StructExt,
    ),
    Enum(
        Ident,
        &'hir Generics<'hir>,
        &'hir EnumDef<'hir>,
        &'ext EnumExt,
    ),
    Const(
        Ident,
        &'hir Generics<'hir>,
        &'hir Ty<'hir>,
        ConstItemRhs<'hir>,
        &'ext ConstExt,
    ),
    Static(Mutability, Ident, &'hir Ty<'hir>, BodyId, &'ext StaticExt),
    Mod(Ident, &'hir Mod<'hir>, &'ext ModExt),
}

#[derive(Debug)]
pub struct GpuFnExt {
    pub core_shim_for: Option<Symbol>,
}

#[derive(Debug)]
pub struct EntryPoint {
    pub name: Symbol,
    pub kind: EntryPointKind,
}

#[derive(Debug)]
pub enum EntryPointKind {
    Compute(WorkgroupSize),
    Vertex,
    Fragment,
}

#[derive(Debug)]
pub enum FnExt {
    GpuFn(GpuFnExt),
    EntryPoint(EntryPoint),
}

impl From<EntryPoint> for FnExt {
    fn from(value: EntryPoint) -> Self {
        FnExt::EntryPoint(value)
    }
}

#[derive(Debug)]
pub struct ImplExt {}

#[derive(Debug)]
pub struct TraitExt {}

#[derive(Debug)]
pub struct StructExt {}

#[derive(Debug)]
pub struct EnumExt {}

#[derive(Debug)]
pub struct FieldExt {
    pub shader_io_binding: Option<ShaderIOBinding>,
}

#[derive(Debug)]
pub struct OverrideId {
    pub value: u32,
    pub span: Span,
}

#[derive(Debug)]
pub struct ConstExt {
    pub id: OverrideId,
    pub required: bool,
}

#[derive(Debug)]
pub struct ResourceBinding {
    pub group: u32,
    pub binding: u32,
    pub span: Span,
}

#[derive(Debug)]
pub enum StaticExt {
    Uniform(ResourceBinding),
    Storage(ResourceBinding),
    StorageMut(ResourceBinding),
    Workgroup,
}
