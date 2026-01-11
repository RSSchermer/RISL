use std::cell::RefCell;

use rustc_hash::FxHashMap;
use rustc_middle::bug;
use rustc_public::abi::{
    FieldsShape, FloatLength, FnAbi, IntegerLength, PassMode, Primitive, ValueAbi, VariantsShape,
    WrappingRange,
};
use rustc_public::mir::alloc::GlobalAlloc;
use rustc_public::mir::mono::{Instance, StaticDef};
use rustc_public::rustc_internal::internal;
use rustc_public::target::{MachineInfo, MachineSize};
use rustc_public::ty::{Align, Allocation, GenericArgKind, RigidTy, TyKind, VariantIdx};
use rustc_public::{CrateDef, abi};
use rustc_public_bridge::IndexedVal;

use crate::context::RislContext;
use crate::hir_ext::{
    BlendSrc, EntryPointKind, FnExt, Interpolation, InterpolationSampling, InterpolationType,
    ResourceBinding, ShaderIOBinding, StaticExt,
};
use crate::slir_build::risl_primitive_ty::RislPrimitiveTy;
use crate::slir_build::ty::Type;
use crate::slir_build::value::Value;
use crate::stable_cg::traits::{
    BackendTypes, BaseTypeCodegenMethods, ConstCodegenMethods, LayoutTypeCodegenMethods,
    MiscCodegenMethods, PreDefineCodegenMethods, StaticCodegenMethods,
};
use crate::stable_cg::{Scalar, ScalarExt, TyAndLayout, TypeKind};

fn blend_src_to_slir(blend_src: BlendSrc) -> slir::BlendSrc {
    match blend_src {
        BlendSrc::Zero => slir::BlendSrc::Zero,
        BlendSrc::One => slir::BlendSrc::One,
    }
}

fn interpolation_tpe_to_slir(ty: InterpolationType) -> slir::InterpolationType {
    match ty {
        InterpolationType::Perspective => slir::InterpolationType::Perspective,
        InterpolationType::Linear => slir::InterpolationType::Linear,
        InterpolationType::Flat => slir::InterpolationType::Flat,
    }
}

fn interpolation_sampling_to_slir(sampling: InterpolationSampling) -> slir::InterpolationSampling {
    match sampling {
        InterpolationSampling::Center => slir::InterpolationSampling::Center,
        InterpolationSampling::Centroid => slir::InterpolationSampling::Centroid,
        InterpolationSampling::Sample => slir::InterpolationSampling::Sample,
        InterpolationSampling::First => slir::InterpolationSampling::First,
        InterpolationSampling::Either => slir::InterpolationSampling::Either,
    }
}

fn interpolation_to_slir(interpolation: Interpolation) -> slir::Interpolation {
    slir::Interpolation {
        tpe: interpolation_tpe_to_slir(interpolation.tpe),
        sampling: interpolation.sampling.map(interpolation_sampling_to_slir),
    }
}

fn shader_io_binding_to_slir(shader_io_binding: ShaderIOBinding) -> slir::ShaderIOBinding {
    match shader_io_binding {
        ShaderIOBinding::VertexIndex => slir::ShaderIOBinding::VertexIndex,
        ShaderIOBinding::InstanceIndex => slir::ShaderIOBinding::InstanceIndex,
        ShaderIOBinding::Position { invariant } => slir::ShaderIOBinding::Position {
            invariant: invariant.is_some(),
        },
        ShaderIOBinding::FrontFacing => slir::ShaderIOBinding::FrontFacing,
        ShaderIOBinding::FragDepth => slir::ShaderIOBinding::FragDepth,
        ShaderIOBinding::SampleIndex => slir::ShaderIOBinding::SampleIndex,
        ShaderIOBinding::SampleMask => slir::ShaderIOBinding::SampleMask,
        ShaderIOBinding::LocalInvocationId => slir::ShaderIOBinding::LocalInvocationId,
        ShaderIOBinding::LocalInvocationIndex => slir::ShaderIOBinding::LocalInvocationIndex,
        ShaderIOBinding::GlobalInvocationId => slir::ShaderIOBinding::GlobalInvocationId,
        ShaderIOBinding::WorkgroupId => slir::ShaderIOBinding::WorkgroupId,
        ShaderIOBinding::NumWorkgroups => slir::ShaderIOBinding::NumWorkgroups,
        ShaderIOBinding::Location {
            location,
            blend_src,
            interpolation,
        } => slir::ShaderIOBinding::Location {
            location: location.node,
            blend_src: blend_src.map(|s| blend_src_to_slir(s.node)),
            interpolation: interpolation.map(interpolation_to_slir),
        },
    }
}

fn resource_binding_to_slir(resource_binding: &ResourceBinding) -> slir::ResourceBinding {
    slir::ResourceBinding {
        group: resource_binding.group,
        binding: resource_binding.binding,
    }
}

fn primitive_ty_to_slir(primitive_ty: RislPrimitiveTy) -> slir::ty::Type {
    match primitive_ty {
        RislPrimitiveTy::Vec2F32 => slir::ty::TY_VEC2_F32,
        RislPrimitiveTy::Vec2U32 => slir::ty::TY_VEC2_U32,
        RislPrimitiveTy::Vec2I32 => slir::ty::TY_VEC2_I32,
        RislPrimitiveTy::Vec2Bool => slir::ty::TY_VEC2_BOOL,
        RislPrimitiveTy::Vec3F32 => slir::ty::TY_VEC3_F32,
        RislPrimitiveTy::Vec3U32 => slir::ty::TY_VEC3_U32,
        RislPrimitiveTy::Vec3I32 => slir::ty::TY_VEC3_I32,
        RislPrimitiveTy::Vec3Bool => slir::ty::TY_VEC3_BOOL,
        RislPrimitiveTy::Vec4F32 => slir::ty::TY_VEC4_F32,
        RislPrimitiveTy::Vec4U32 => slir::ty::TY_VEC4_U32,
        RislPrimitiveTy::Vec4I32 => slir::ty::TY_VEC4_I32,
        RislPrimitiveTy::Vec4Bool => slir::ty::TY_VEC4_BOOL,
        RislPrimitiveTy::Mat2x2F32 => slir::ty::TY_MAT2X2,
        RislPrimitiveTy::Mat2x3F32 => slir::ty::TY_MAT2X3,
        RislPrimitiveTy::Mat2x4F32 => slir::ty::TY_MAT2X4,
        RislPrimitiveTy::Mat3x2F32 => slir::ty::TY_MAT3X2,
        RislPrimitiveTy::Mat3x3F32 => slir::ty::TY_MAT3X3,
        RislPrimitiveTy::Mat3x4F32 => slir::ty::TY_MAT3X4,
        RislPrimitiveTy::Mat4x2F32 => slir::ty::TY_MAT4X2,
        RislPrimitiveTy::Mat4x3F32 => slir::ty::TY_MAT4X3,
        RislPrimitiveTy::Mat4x4F32 => slir::ty::TY_MAT4X4,
    }
}

#[derive(Clone, Copy)]
pub enum SlirStatic {
    Uniform(slir::UniformBinding),
    Storage(slir::StorageBinding),
    Workgroup(slir::WorkgroupBinding),
}

pub struct CodegenContext<'a, 'tcx> {
    pub rcx: &'a RislContext<'tcx>,
    pub module_name: slir::Symbol,
    pub module: RefCell<slir::Module>,
    pub cfg: RefCell<slir::cfg::Cfg>,
    pub scalar_to_slir: RefCell<FxHashMap<rustc_abi::Scalar, slir::ty::Type>>,
    pub ty_to_slir: RefCell<FxHashMap<TyAndLayout, slir::ty::Type>>,
    pub instance_to_slir: RefCell<FxHashMap<Instance, slir::Function>>,
    pub static_to_slir: RefCell<FxHashMap<StaticDef, SlirStatic>>,
}

impl<'a, 'tcx> CodegenContext<'a, 'tcx> {
    pub fn new(rcx: &'a RislContext<'tcx>, module_name: slir::Symbol) -> Self {
        let module = slir::Module::new(module_name);
        let cfg = slir::cfg::Cfg::new(module.ty.clone());

        CodegenContext {
            rcx,
            module_name,
            module: RefCell::new(module),
            cfg: RefCell::new(cfg),
            scalar_to_slir: RefCell::new(Default::default()),
            ty_to_slir: RefCell::new(Default::default()),
            instance_to_slir: RefCell::new(Default::default()),
            static_to_slir: RefCell::new(Default::default()),
        }
    }

    fn workgroup_binding_initializer_const_name(&self, def: StaticDef) -> slir::Constant {
        let name = slir::Symbol::new(format!("W{}", def.def_id().to_index()));

        slir::Constant {
            name,
            module: self.module_name,
        }
    }

    pub fn ty_and_layout_resolve(&self, layout: &TyAndLayout) -> slir::ty::Type {
        // Note: this cannot be inlined into the if-let expression, because then the borrow guard would
        // not drop in time, which would result in an "already borrowed" error.
        let ty = self.ty_to_slir.borrow().get(layout).copied();

        if let Some(ty) = ty {
            ty
        } else {
            let ty = self.ty_and_layout_register(layout);

            self.ty_to_slir.borrow_mut().insert(layout.clone(), ty);

            ty
        }
    }

    fn ty_and_layout_register(&self, layout: &TyAndLayout) -> slir::ty::Type {
        if let Some(primitive_ty) = RislPrimitiveTy::from_ty(layout.ty) {
            return primitive_ty_to_slir(primitive_ty);
        }

        if let Some(ty) = self.try_register_as_rislc_mem_resource_ty(&layout) {
            return ty;
        }

        // Handle enums before scalar primitives, as there enum types that cam present as scalar
        // primitives (e.g. an Option<T> where T has a "niche"), but we want to explicitly pass
        // those to SLIR as enums regardless.
        if let VariantsShape::Multiple { variants, .. } = &layout.layout.variants {
            return self.register_enum(variants, layout);
        }

        match layout.layout.abi.clone() {
            ValueAbi::Scalar(scalar) => {
                return self.resolve_scalar_ty(scalar, layout);
            }
            ValueAbi::ScalarPair(s0, s1) => {
                return self.register_scalar_pair_ty(s0, s1, layout);
            }
            ValueAbi::Vector { element, count } => {
                return self.register_vector_ty(element, count, layout);
            }
            _ => {}
        }

        match layout.layout.fields {
            FieldsShape::Primitive => bug!("primitive should have been handled earlier"),
            FieldsShape::Union(_) => bug!("union types are not supported by RISL"),
            FieldsShape::Array { count, stride } => self.register_array_ty(count, stride, layout),
            FieldsShape::Arbitrary { ref offsets, .. } => {
                self.register_struct(&layout.layout, offsets, layout)
            }
        }
    }

    /// Unpacks memory container types.
    ///
    /// Certain types is the RISL standard library acts as proxies for device-controlled memory
    /// resources (e.g., the `risl::resource::Storage` type). Such wrappers help us achieve the
    /// interface we want to expose, but end up being noise at lower levels of compilation. We
    /// therefore unpack these types here.
    fn try_register_as_rislc_mem_resource_ty(
        &self,
        layout: &TyAndLayout,
    ) -> Option<slir::ty::Type> {
        let ty = layout.ty;

        if let TyKind::RigidTy(RigidTy::Adt(def, generics)) = ty.kind() {
            if !def
                .tool_attrs(&["rislc".into(), "mem_resource_ty".into()])
                .is_empty()
            {
                let Some(GenericArgKind::Type(ty)) = generics.0.first() else {
                    panic!("`risl::mem_resource_ty` must always have a type parameter");
                };

                let layout = ty
                    .layout()
                    .expect("type must have a known layout during codegen");
                let slir_ty = self.ty_and_layout_resolve(&TyAndLayout::expect_from_ty(*ty));

                return Some(slir_ty);
            }
        }

        None
    }

    fn resolve_scalar_ty(&self, scalar: abi::Scalar, layout: &TyAndLayout) -> slir::ty::Type {
        if matches!(
            scalar,
            abi::Scalar::Initialized {
                value: Primitive::Int {
                    length: IntegerLength::I8,
                    signed: false
                },
                valid_range: WrappingRange { start: 0, end: 1 }
            }
        ) {
            return slir::ty::TY_BOOL;
        }

        let primitive = match scalar {
            abi::Scalar::Initialized { value, .. } => value,
            abi::Scalar::Union { value, .. } => value,
        };

        match primitive {
            Primitive::Int { signed: true, .. } => slir::ty::TY_I32,
            Primitive::Int { signed: false, .. } => slir::ty::TY_U32,
            Primitive::Float {
                length: FloatLength::F32,
            } => slir::ty::TY_F32,
            Primitive::Pointer(_) => {
                let TyKind::RigidTy(ty) = layout.ty.kind() else {
                    bug!("primitive type must be rigid")
                };

                match ty {
                    RigidTy::Ref(_, pointee_ty, _) | RigidTy::RawPtr(pointee_ty, _) => {
                        let pointee_ty =
                            self.ty_and_layout_resolve(&TyAndLayout::expect_from_ty(pointee_ty));

                        self.module
                            .borrow()
                            .ty
                            .register(slir::ty::TypeKind::Ptr(pointee_ty))
                    }
                    _ => bug!("pointer primitive type must be a ref or raw ptr"),
                }
            }
            _ => bug!("primitive type not supported by SLIR"),
        }
    }

    fn register_scalar_pair_ty(
        &self,
        s0: abi::Scalar,
        s1: abi::Scalar,
        layout: &TyAndLayout,
    ) -> slir::ty::Type {
        let t0 = self.resolve_scalar_ty(s0, &layout.field(0));
        let t1 = self.resolve_scalar_ty(s1, &layout.field(1));

        // My understanding from rustc_codegen_llvm is that there is never any padding between
        // the values of a scalar pair (as they are never meant to be stored to memory).
        let fields = vec![
            slir::ty::StructField {
                offset: 0,
                ty: t0,
                io_binding: None,
            },
            slir::ty::StructField {
                offset: s0.size(&MachineInfo::target()).bytes() as u64,
                ty: t1,
                io_binding: None,
            },
        ];

        let ty = self
            .module
            .borrow()
            .ty
            .register(slir::ty::Struct { fields }.into());

        self.ty_to_slir.borrow_mut().insert(layout.clone(), ty);

        ty
    }

    fn register_vector_ty(
        &self,
        element: abi::Scalar,
        count: u64,
        layout: &TyAndLayout,
    ) -> slir::ty::Type {
        let base = self.resolve_scalar_ty(element, &layout.field(0));
        let ty = self.module.borrow().ty.register(slir::ty::TypeKind::Array {
            element_ty: base,
            count,
            stride: 4,
        });

        self.ty_to_slir.borrow_mut().insert(layout.clone(), ty);

        ty
    }

    fn register_array_ty(
        &self,
        count: u64,
        stride: MachineSize,
        layout: &TyAndLayout,
    ) -> slir::ty::Type {
        let element_layout = layout.field(0);
        let base = self.ty_and_layout_resolve(&element_layout);

        let kind = if layout.ty.kind().is_slice() {
            slir::ty::TypeKind::Slice {
                element_ty: base,
                stride: stride.bytes() as u64,
            }
        } else {
            slir::ty::TypeKind::Array {
                element_ty: base,
                count,
                stride: stride.bytes() as u64,
            }
        };

        let ty = self.module.borrow().ty.register(kind);

        self.ty_to_slir.borrow_mut().insert(layout.clone(), ty);

        ty
    }

    fn register_struct(
        &self,
        shape: &abi::LayoutShape,
        field_offsets: &[MachineSize],
        layout: &TyAndLayout,
    ) -> slir::ty::Type {
        let TyKind::RigidTy(RigidTy::Adt(def, _)) = layout.ty.kind() else {
            panic!("expected a struct type")
        };

        let field_defs = def
            .variant(VariantIdx::to_val(0))
            .expect("expected a struct variant")
            .fields();

        let fields = shape
            .fields
            .fields_by_offset_order()
            .into_iter()
            .map(|i| {
                let layout = layout.field(i);
                let ty = self.ty_and_layout_resolve(&layout);

                // Not all types have an equal number of field-defs in the THIR representation as
                // they do in the ABI shape (notably fat pointers won't), but types that have
                // io-binding attributes must have io-binding attributes on all fields and therefore
                // all fields must be of io-bindable types (which does not include fat pointers);
                // for such types this should always resolve to the correct field-def.
                let io_binding = if let Some(field_def) = field_defs.get(i) {
                    internal(self.rcx.tcx(), field_def.def)
                        .as_local()
                        .and_then(|id| self.rcx.hir_ext().field_ext.get(&id))
                        .and_then(|ext| ext.shader_io_binding)
                        .map(shader_io_binding_to_slir)
                } else {
                    None
                };

                slir::ty::StructField {
                    offset: field_offsets[i].bytes() as u64,
                    ty,
                    io_binding,
                }
            })
            .collect();

        let ty = self
            .module
            .borrow()
            .ty
            .register(slir::ty::Struct { fields }.into());

        self.ty_to_slir.borrow_mut().insert(layout.clone(), ty);

        ty
    }

    fn register_enum(
        &self,
        variant_shapes: &[abi::LayoutShape],
        layout: &TyAndLayout,
    ) -> slir::ty::Type {
        let variants = variant_shapes
            .iter()
            .map(|shape| {
                let variant_layout = TyAndLayout {
                    ty: layout.ty,
                    layout: shape.clone(),
                };

                let FieldsShape::Arbitrary { offsets, .. } = &shape.fields else {
                    bug!("enum variant should have an arbitrary field shape")
                };

                self.register_struct(shape, offsets, &variant_layout)
            })
            .collect();

        let ty = self
            .module
            .borrow()
            .ty
            .register(slir::ty::Enum { variants }.into());

        self.ty_to_slir.borrow_mut().insert(layout.clone(), ty);

        ty
    }

    pub fn finish(self) -> (slir::Module, slir::cfg::Cfg) {
        (self.module.into_inner(), self.cfg.into_inner())
    }
}

impl<'a, 'tcx> PreDefineCodegenMethods for CodegenContext<'a, 'tcx> {
    fn predefine_static(&self, def: StaticDef, symbol_name: &str) {
        let def_id = internal(self.rcx.tcx(), def.0);
        let item = self.rcx.tcx().hir_expect_item(def_id.expect_local());
        let (mutability, _, _, _, ext) = self
            .rcx
            .hir_ext()
            .extend_item(item)
            .expect("statics referenced by RISL functions must be extended")
            .expect_static();

        let ty = self.ty_and_layout_resolve(&TyAndLayout::expect_from_ty(def.ty()));

        let mut module = self.module.borrow_mut();

        match ext {
            StaticExt::Uniform(resource_binding) => {
                let b = module.uniform_bindings.register(slir::UniformBindingData {
                    ty,
                    resource_binding: resource_binding_to_slir(resource_binding),
                });

                self.static_to_slir
                    .borrow_mut()
                    .insert(def, SlirStatic::Uniform(b));
            }
            StaticExt::Storage(resource_binding) => {
                let b = module.storage_bindings.register(slir::StorageBindingData {
                    ty,
                    resource_binding: resource_binding_to_slir(resource_binding),
                    writable: false,
                });

                self.static_to_slir
                    .borrow_mut()
                    .insert(def, SlirStatic::Storage(b));
            }
            StaticExt::StorageMut(resource_binding) => {
                let b = module.storage_bindings.register(slir::StorageBindingData {
                    ty,
                    resource_binding: resource_binding_to_slir(resource_binding),
                    writable: true,
                });

                self.static_to_slir
                    .borrow_mut()
                    .insert(def, SlirStatic::Storage(b));
            }
            StaticExt::Workgroup => {
                let b = module
                    .workgroup_bindings
                    .register(slir::WorkgroupBindingData { ty });

                self.static_to_slir
                    .borrow_mut()
                    .insert(def, SlirStatic::Workgroup(b));
            }
        }
    }

    fn predefine_fn(&self, instance: Instance, symbol_name: &str) {
        let function_name = slir::Symbol::from_ref(symbol_name);
        let function = slir::Function {
            name: function_name,
            module: self.module_name,
        };
        let def_id = internal(self.rcx.tcx(), instance.def.def_id());

        let abi = instance.fn_abi().unwrap();

        let mut arg_io_bindings = vec![None; abi.args.len()];

        if let Some(local_id) = def_id.as_local()
            && let Some(fn_ext) = self.rcx.hir_ext().fn_ext(local_id)
        {
            let body = self.rcx.tcx().hir_body_owned_by(local_id);

            for (i, param) in body.params.iter().enumerate() {
                if let Some(param_ext) = self.rcx.hir_ext().param_ext(param.hir_id)
                    && let Some(shader_io_binding) = param_ext.shader_io_binding
                {
                    arg_io_bindings[i] = Some(shader_io_binding)
                }
            }

            if let FnExt::EntryPoint(ep) = fn_ext {
                let kind = match &ep.kind {
                    EntryPointKind::Compute(size) => {
                        slir::EntryPointKind::Compute(size.x, size.y, size.z)
                    }
                    EntryPointKind::Vertex => slir::EntryPointKind::Vertex,
                    EntryPointKind::Fragment => slir::EntryPointKind::Fragment,
                };

                self.module.borrow_mut().entry_points.register(
                    function,
                    slir::EntryPoint {
                        name: slir::Symbol::from_ref(ep.name.as_str()),
                        kind,
                    },
                );
            }
        }

        let mut args = Vec::new();

        if matches!(abi.ret.mode, PassMode::Indirect { .. }) {
            let ret_ty = self.ty_and_layout_resolve(&TyAndLayout {
                ty: abi.ret.ty,
                layout: abi.ret.layout.shape(),
            });

            args.push(slir::FnArg {
                ty: self
                    .module
                    .borrow_mut()
                    .ty
                    .register(slir::ty::TypeKind::Ptr(ret_ty)),
                shader_io_binding: None,
            })
        }

        for (arg, binding) in abi.args.iter().zip(arg_io_bindings) {
            match arg.mode {
                PassMode::Ignore => {}
                PassMode::Direct(_) => {
                    let ty = self.ty_and_layout_resolve(&TyAndLayout {
                        ty: arg.ty,
                        layout: arg.layout.shape(),
                    });

                    args.push(slir::FnArg {
                        ty,
                        shader_io_binding: binding.map(shader_io_binding_to_slir),
                    });
                }
                PassMode::Pair(..) => {
                    let ValueAbi::ScalarPair(a, b) = arg.layout.shape().abi else {
                        bug!("value ABI does not match pass-mode")
                    };

                    let layout = TyAndLayout {
                        ty: arg.ty,
                        layout: arg.layout.shape(),
                    };

                    let a_ty = self.resolve_scalar_ty(a, &layout.field(0));
                    let b_ty = self.resolve_scalar_ty(b, &layout.field(1));

                    args.push(slir::FnArg {
                        ty: a_ty,
                        shader_io_binding: binding.map(shader_io_binding_to_slir),
                    });
                    args.push(slir::FnArg {
                        ty: b_ty,
                        shader_io_binding: None,
                    });
                }
                PassMode::Indirect { .. } => {
                    let ty = self.ty_and_layout_resolve(&TyAndLayout {
                        ty: arg.ty,
                        layout: arg.layout.shape(),
                    });

                    args.push(slir::FnArg {
                        ty: self
                            .module
                            .borrow_mut()
                            .ty
                            .register(slir::ty::TypeKind::Ptr(ty)),
                        shader_io_binding: binding.map(shader_io_binding_to_slir),
                    });
                }
                PassMode::Cast { .. } => bug!("not supported by RISL"),
            }
        }

        let ret_ty = if matches!(abi.ret.mode, PassMode::Ignore | PassMode::Indirect { .. }) {
            None
        } else {
            Some(self.ty_and_layout_resolve(&TyAndLayout {
                ty: abi.ret.ty,
                layout: abi.ret.layout.shape(),
            }))
        };

        let function_ty = self
            .module
            .borrow_mut()
            .ty
            .register(slir::ty::TypeKind::Function(function));

        let sig = slir::FnSig {
            name: function_name,
            ty: function_ty,
            args,
            ret_ty,
        };

        let mut module = self.module.borrow_mut();

        module.fn_sigs.register(function, sig);
        self.cfg.borrow_mut().register_function(&module, function);

        self.instance_to_slir
            .borrow_mut()
            .insert(instance, function);
    }
}

impl<'a, 'tcx> BackendTypes for CodegenContext<'a, 'tcx> {
    type Value = Value;
    type Local = slir::cfg::LocalBinding;
    type Function = slir::Function;
    type BasicBlock = slir::cfg::BasicBlock;
    type Type = Type;
}

impl<'a, 'tcx> StaticCodegenMethods for CodegenContext<'a, 'tcx> {
    fn static_addr_of(&self, cv: Self::Value, align: Align, kind: Option<&str>) -> Self::Value {
        todo!()
    }

    fn codegen_static(&self, def: StaticDef) {}
}

impl<'a, 'tcx> ConstCodegenMethods for CodegenContext<'a, 'tcx> {
    fn const_null(&self, t: Self::Type) -> Self::Value {
        todo!()
    }

    fn const_undef(&self, t: Self::Type) -> Self::Value {
        todo!()
    }

    fn is_undef(&self, v: Self::Value) -> bool {
        todo!()
    }

    fn const_poison(&self, t: Self::Type) -> Self::Value {
        todo!()
    }

    fn const_bool(&self, val: bool) -> Self::Value {
        todo!()
    }

    fn const_i8(&self, i: i8) -> Self::Value {
        todo!()
    }

    fn const_i16(&self, i: i16) -> Self::Value {
        todo!()
    }

    fn const_i32(&self, i: i32) -> Self::Value {
        todo!()
    }

    fn const_int(&self, t: Self::Type, i: i64) -> Self::Value {
        todo!()
    }

    fn const_u8(&self, i: u8) -> Self::Value {
        todo!()
    }

    fn const_u32(&self, i: u32) -> Self::Value {
        slir::cfg::Value::InlineConst(slir::cfg::InlineConst::U32(i as u32)).into()
    }

    fn const_u64(&self, i: u64) -> Self::Value {
        todo!()
    }

    fn const_u128(&self, i: u128) -> Self::Value {
        todo!()
    }

    fn const_usize(&self, i: u64) -> Self::Value {
        slir::cfg::Value::InlineConst(slir::cfg::InlineConst::U32(i as u32)).into()
    }

    fn const_uint(&self, t: Self::Type, i: u64) -> Self::Value {
        slir::cfg::Value::InlineConst(slir::cfg::InlineConst::U32(i as u32)).into()
    }

    fn const_uint_big(&self, t: Self::Type, u: u128) -> Self::Value {
        slir::cfg::Value::InlineConst(slir::cfg::InlineConst::U32(u as u32)).into()
    }

    fn const_real(&self, t: Self::Type, val: f64) -> Self::Value {
        todo!()
    }

    fn const_str(&self, s: &str) -> (Self::Value, Self::Value) {
        todo!()
    }

    fn const_struct(&self, elts: &[Self::Value], packed: bool) -> Self::Value {
        todo!()
    }

    fn const_vector(&self, elts: &[Self::Value]) -> Self::Value {
        todo!()
    }

    fn const_to_opt_uint(&self, v: Self::Value) -> Option<u64> {
        todo!()
    }

    fn const_to_opt_u128(&self, v: Self::Value, _sign_ext: bool) -> Option<u128> {
        match v.expect_value() {
            slir::cfg::Value::InlineConst(v) => match v {
                slir::cfg::InlineConst::U32(v) => Some(v as u128),
                _ => unimplemented!(),
            },
            _ => None,
        }
    }

    fn const_data_from_alloc(&self, alloc: &Allocation, layout: &TyAndLayout) -> Self::Value {
        todo!()
    }

    fn scalar_to_backend(&self, scalar: Scalar) -> Self::Value {
        let inline_const = match scalar {
            Scalar::U32(v) => slir::cfg::InlineConst::U32(v),
            Scalar::I32(v) => slir::cfg::InlineConst::I32(v),
            Scalar::F32(v) => slir::cfg::InlineConst::from(v),
            Scalar::Bool(v) => slir::cfg::InlineConst::Bool(v),
            Scalar::Pointer(ptr) => {
                let ident = match GlobalAlloc::from(ptr.alloc_id) {
                    GlobalAlloc::Function(_) => bug!("function pointers are not supported by SLIR"),
                    GlobalAlloc::VTable(_, _) => bug!("V-tables are not supported by SLIR"),
                    GlobalAlloc::Static(def) => {
                        let slir_static = *self
                            .static_to_slir
                            .borrow()
                            .get(&def)
                            .expect("static should have been pre-defined");

                        match slir_static {
                            SlirStatic::Uniform(b) => slir::cfg::RootIdentifier::Uniform(b),
                            SlirStatic::Storage(b) => slir::cfg::RootIdentifier::Storage(b),
                            SlirStatic::Workgroup(b) => slir::cfg::RootIdentifier::Workgroup(b),
                        }
                    }
                    GlobalAlloc::Memory(alloc) => {
                        let bytes = alloc
                            .raw_bytes()
                            .expect("could not read constant memory allocation");
                        let const_id = ptr.alloc_id.to_index();
                        let const_name = slir::Symbol::new(format!("c{}", const_id));
                        let constant = slir::Constant {
                            name: const_name,
                            module: self.module_name,
                        };
                        let ty = self.ty_and_layout_resolve(&ptr.pointee_layout);

                        self.module
                            .borrow_mut()
                            .constants
                            .register_byte_data(constant, ty, bytes);

                        slir::cfg::RootIdentifier::Constant(constant)
                    }
                    GlobalAlloc::TypeId { .. } => todo!(),
                };

                slir::cfg::InlineConst::Ptr(slir::cfg::ConstPtr::new(
                    &*self.module.borrow(),
                    &*self.cfg.borrow(),
                    ident,
                ))
            }
        };

        inline_const.into()
    }
}

impl<'a, 'tcx> LayoutTypeCodegenMethods for CodegenContext<'a, 'tcx> {
    fn backend_type(&self, layout: &TyAndLayout) -> Self::Type {
        self.ty_and_layout_resolve(layout).into()
    }

    fn fn_decl_backend_type(&self, fn_abi: &FnAbi) -> Self::Type {
        let ret_ty = if matches!(
            fn_abi.ret.mode,
            PassMode::Ignore | PassMode::Indirect { .. }
        ) {
            None
        } else {
            Some(self.ty_and_layout_resolve(&TyAndLayout {
                ty: fn_abi.ret.ty,
                layout: fn_abi.ret.layout.shape(),
            }))
        };

        Type::FnDecl { ret_ty }
    }

    fn immediate_backend_type(&self, layout: &TyAndLayout) -> Self::Type {
        self.ty_and_layout_resolve(layout).into()
    }

    fn is_backend_immediate(&self, layout: &TyAndLayout) -> bool {
        match layout.layout.abi {
            ValueAbi::Scalar(_) | ValueAbi::Vector { .. } => true,
            ValueAbi::ScalarPair(..) => false,
            ValueAbi::Aggregate { .. } => layout.layout.is_1zst(),
        }
    }

    fn is_backend_scalar_pair(&self, layout: &TyAndLayout) -> bool {
        matches!(layout.layout.abi, ValueAbi::ScalarPair(..))
    }

    fn scalar_pair_element_backend_type(
        &self,
        layout: &TyAndLayout,
        index: usize,
        immediate: bool,
    ) -> Self::Type {
        todo!()
    }
}

impl<'a, 'tcx> MiscCodegenMethods for CodegenContext<'a, 'tcx> {
    fn get_fn(&self, instance: &Instance) -> Self::Function {
        let name = slir::Symbol::from_ref(instance.mangled_name().as_str());

        // To facilitate function resolution in SLIR, we store the module name as part of the
        // function identifier. We base the module name on the name of the crate from which the
        // module was built. For many instances we can therefore construct the module name to be
        // used by the function identifier, based on the crate associated with the instance
        // definition. However, for instances of generic monomorphized functions, the SLIR function
        // will be stored in the current module, while the crate name associated with the instance
        // will identify the crate that contains the originating function definition. If that
        // originating crate happens to not be the current local crate, then basing the function
        // identifier's module name on the instance's associated crate name would produce an
        // incorrect identifier. Fortunately, such a function will have been "predefined" earlier,
        // so we can simply first check we've already registered a signature with the module and
        // make sure we use the current module's name if we find a matching function.
        if self.module.borrow().fn_sigs.contains(slir::Function {
            name,
            module: self.module_name,
        }) {
            slir::Function {
                name,
                module: self.module_name,
            }
        } else {
            let krate = instance.def.krate();

            let module_name = if krate.is_local {
                self.module_name
            } else {
                let crate_num = internal(self.rcx.tcx(), krate.id);

                slir::Symbol::new(self.rcx.crate_slir_module_name(crate_num))
            };

            slir::Function {
                module: module_name,
                name,
            }
        }
    }

    fn get_fn_addr(&self, instance: &Instance) -> Self::Value {
        self.get_fn(instance).into()
    }
}

impl<'a, 'tcx> BaseTypeCodegenMethods for CodegenContext<'a, 'tcx> {
    fn type_i8(&self) -> Self::Type {
        slir::ty::TY_I32.into()
    }

    fn type_i16(&self) -> Self::Type {
        todo!()
    }

    fn type_i32(&self) -> Self::Type {
        todo!()
    }

    fn type_i64(&self) -> Self::Type {
        todo!()
    }

    fn type_i128(&self) -> Self::Type {
        todo!()
    }

    fn type_isize(&self) -> Self::Type {
        todo!()
    }

    fn type_f16(&self) -> Self::Type {
        todo!()
    }

    fn type_f32(&self) -> Self::Type {
        todo!()
    }

    fn type_f64(&self) -> Self::Type {
        todo!()
    }

    fn type_f128(&self) -> Self::Type {
        todo!()
    }

    fn type_array(&self, ty: Self::Type, len: u64) -> Self::Type {
        todo!()
    }

    fn type_func(&self, args: &[Self::Type], ret: Self::Type) -> Self::Type {
        todo!()
    }

    fn type_kind(&self, ty: Self::Type) -> TypeKind {
        todo!()
    }

    fn type_ptr(&self) -> Self::Type {
        todo!()
    }

    fn element_type(&self, ty: Self::Type) -> Self::Type {
        todo!()
    }

    fn vector_length(&self, ty: Self::Type) -> usize {
        todo!()
    }

    fn float_width(&self, ty: Self::Type) -> usize {
        todo!()
    }

    fn int_width(&self, ty: Self::Type) -> u64 {
        todo!()
    }

    fn val_ty(&self, v: Self::Value) -> Self::Type {
        todo!()
    }
}
