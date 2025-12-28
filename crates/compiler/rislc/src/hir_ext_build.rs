use std::convert::identity;

use rustc_hir::def::{DefKind, Res};
use rustc_hir::intravisit::Visitor;
use rustc_hir::{
    Arm, ConstBlock, Expr, ExprField, ExprKind, FieldDef, ForeignItem, GenericParam, Impl,
    ImplItem, ImplItemKind, Item, ItemKind, MethodKind, Param, PatField, QPath, Stmt, StmtKind,
    Target, TyKind, UseKind, Variant, intravisit,
};
use rustc_middle::hir::nested_filter;
use rustc_middle::ty::TyCtxt;
use rustc_span::def_id::{DefId, LocalModDefId};
use rustc_span::source_map::{Spanned, respan};
use rustc_span::{ErrorGuaranteed, Span, Symbol};

use crate::attr::{
    AttrBlendSrc, AttrInterpolate, Attributes, BuiltinName, InterpolationSamplingName,
    InterpolationTypeName, collect_risl_attributes,
};
use crate::hir_ext::{
    BlendSrc, ConstExt, EntryPoint, EntryPointKind, EnumExt, FieldExt, FnExt, HirExt, ImplExt,
    Interpolation, InterpolationSampling, InterpolationType, ModExt, OverrideId, ParamExt,
    ResourceBinding, ShaderIOBinding, ShaderRequest, ShaderRequestKind, StaticExt, StructExt,
    TraitExt, WorkgroupSize,
};

// Borrowed from https://github.com/Rust-GPU/rust-gpu
fn target_from_impl_item(tcx: TyCtxt<'_>, impl_item: &ImplItem<'_>) -> Target {
    match impl_item.kind {
        ImplItemKind::Const(..) => Target::AssocConst,
        ImplItemKind::Fn(..) => {
            let parent_owner_id = tcx.hir_get_parent_item(impl_item.hir_id());
            let containing_item = tcx.hir_expect_item(parent_owner_id.def_id);
            let containing_impl_is_for_trait = match &containing_item.kind {
                ItemKind::Impl(Impl { of_trait, .. }) => of_trait.is_some(),
                _ => unreachable!("parent of an ImplItem must be an Impl"),
            };
            if containing_impl_is_for_trait {
                Target::Method(MethodKind::Trait { body: true })
            } else {
                Target::Method(MethodKind::Inherent)
            }
        }
        ImplItemKind::Type(..) => Target::AssocTy,
    }
}

fn try_resolve_stmt_to_mod_id(tcx: TyCtxt<'_>, stmt: &Stmt) -> Result<DefId, ErrorGuaranteed> {
    if let StmtKind::Item(item_id) = &stmt.kind {
        let item = tcx.hir_item(*item_id);

        if let ItemKind::Use(path, UseKind::Single(_)) = &item.kind {
            if let Some(Res::Def(DefKind::Mod, id)) = path.res.type_ns {
                return Ok(id);
            }
        }
    }

    Err(tcx
        .sess
        .dcx()
        .err("expected a `use` statement that points to a shader module"))
}

fn try_build_shader_request(
    tcx: TyCtxt<'_>,
    block: &ConstBlock,
    span: Span,
    request_id: Symbol,
    kind: ShaderRequestKind,
) -> Result<ShaderRequest, ErrorGuaranteed> {
    let body = tcx.hir_body(block.body);

    if let ExprKind::Block(e, _) = body.value.kind {
        if let Some(stmt) = e.stmts.first() {
            let shader_mod = try_resolve_stmt_to_mod_id(tcx, stmt)?;

            Ok(ShaderRequest {
                shader_mod,
                span,
                request_id,
                kind,
            })
        } else {
            Err(tcx.dcx().span_err(
                body.value.span,
                "expected a `use` statement that points to a shader module",
            ))
        }
    } else {
        Err(tcx.dcx().span_err(
            body.value.span,
            "expected a block expression for a shader source request",
        ))
    }
}

fn try_blend_src_from_attr(
    tcx: TyCtxt<'_>,
    attr: &AttrBlendSrc,
) -> Result<Spanned<BlendSrc>, ErrorGuaranteed> {
    let value = match attr.blend_src {
        0 => BlendSrc::Zero,
        1 => BlendSrc::One,
        _ => {
            return Err(tcx.dcx().span_err(
                attr.span,
                "the value of a `blend_src` attribute must be either `0` or `1`",
            ));
        }
    };

    Ok(respan(attr.span, value))
}

fn interpolation_type_from_name(name: InterpolationTypeName) -> InterpolationType {
    match name {
        InterpolationTypeName::Perspective => InterpolationType::Perspective,
        InterpolationTypeName::Linear => InterpolationType::Linear,
        InterpolationTypeName::Flat => InterpolationType::Flat,
    }
}

fn interpolation_sampling_from_name(name: InterpolationSamplingName) -> InterpolationSampling {
    match name {
        InterpolationSamplingName::Center => InterpolationSampling::Center,
        InterpolationSamplingName::Centroid => InterpolationSampling::Centroid,
        InterpolationSamplingName::Sample => InterpolationSampling::Sample,
        InterpolationSamplingName::First => InterpolationSampling::First,
        InterpolationSamplingName::Either => InterpolationSampling::Either,
    }
}

fn interpolation_from_attr(attr: &AttrInterpolate) -> Interpolation {
    Interpolation {
        tpe: interpolation_type_from_name(attr.type_name),
        sampling: attr.sampling_name.map(interpolation_sampling_from_name),
        span: attr.span,
    }
}

fn try_maybe_shader_io_binding(
    tcx: TyCtxt<'_>,
    attrs: &Attributes,
    span: Span,
) -> Result<Option<ShaderIOBinding>, ErrorGuaranteed> {
    if attrs.location.is_some() && attrs.builtin.is_some() {
        return Err(tcx.dcx().span_err(
            span,
            "the `location` and `builtin` attributes are mutually exclusive",
        ));
    }

    let has_builtin_position_attr = attrs
        .builtin
        .as_ref()
        .map(|b| b.is_position())
        .unwrap_or(false);

    if attrs.invariant.is_some() && !has_builtin_position_attr {
        return Err(tcx.dcx().span_err(
            span,
            "the `invariant` attribute requires the presence of a `#[builtin(position)]` attribute",
        ));
    }

    if attrs.interpolate.is_some() && attrs.location.is_none() {
        return Err(tcx.dcx().span_err(
            span,
            "the `interpolate` attribute requires the presence of a `location` attribute",
        ));
    }

    if attrs.blend_src.is_some() && attrs.location.is_none() {
        return Err(tcx.dcx().span_err(
            span,
            "the `blend_src` attribute requires the presence of a `location` attribute",
        ));
    }

    if let Some(location) = &attrs.location {
        let blend_src = attrs
            .blend_src
            .as_ref()
            .map(|a| try_blend_src_from_attr(tcx, a))
            .transpose()?;
        let interpolation = attrs.interpolate.as_ref().map(interpolation_from_attr);

        return Ok(Some(ShaderIOBinding::Location {
            location: respan(location.span, location.location),
            blend_src,
            interpolation,
        }));
    }

    Ok(attrs.builtin.as_ref().map(|attr| match attr.builtin_name {
        BuiltinName::VertexIndex => ShaderIOBinding::VertexIndex,
        BuiltinName::InstanceIndex => ShaderIOBinding::InstanceIndex,
        BuiltinName::Position => ShaderIOBinding::Position {
            invariant: attrs.invariant.as_ref().map(|a| a.span),
        },
        BuiltinName::FrontFacing => ShaderIOBinding::FrontFacing,
        BuiltinName::FragDepth => ShaderIOBinding::FragDepth,
        BuiltinName::SampleIndex => ShaderIOBinding::SampleIndex,
        BuiltinName::SampleMask => ShaderIOBinding::SampleMask,
        BuiltinName::LocalInvocationId => ShaderIOBinding::LocalInvocationId,
        BuiltinName::LocalInvocationIndex => ShaderIOBinding::LocalInvocationIndex,
        BuiltinName::GlobalInvocationId => ShaderIOBinding::GlobalInvocationId,
        BuiltinName::WorkgroupId => ShaderIOBinding::WorkgroupId,
        BuiltinName::NumWorkgroups => ShaderIOBinding::NumWorkgroups,
    }))
}

pub struct Locator<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    hir_ext: &'a mut HirExt,
}

impl<'a, 'tcx> Locator<'a, 'tcx> {
    fn visit_item_const(&mut self, item: &Item<'tcx>, attrs: &Attributes) {
        if let Some(attr) = &attrs.override_id
            && attrs.override_.is_none()
        {
            self.tcx.dcx().span_err(
                attr.span,
                "`id` attributes may only be declared on overridable constants",
            );

            return;
        }

        if let Some(attr) = &attrs.override_required
            && attrs.override_.is_none()
        {
            // `override_required` attributes are not declared by the user, but added in the macro
            // expansion stage, so this would indicate a bug the macro(s).
            self.tcx.dcx().span_bug(
                attr.span,
                "`override_required` attributes may only be declared on overridable constants",
            );
        }

        if attrs.override_.is_some() {
            if let Some(id_attr) = &attrs.override_id {
                self.hir_ext.const_ext.insert(
                    item.item_id(),
                    ConstExt {
                        id: OverrideId {
                            value: id_attr.override_id,
                            span: id_attr.span,
                        },
                        required: attrs.override_required.is_some(),
                    },
                );
            } else {
                self.tcx.dcx().span_err(
                    item.span,
                    "overridable constants must specify an `id` attribute",
                );
            }
        }
    }

    fn visit_item_static(&mut self, item: &Item<'tcx>, attrs: &Attributes) {
        let count = [attrs.resource.is_some(), attrs.workgroup_shared.is_some()]
            .into_iter()
            .filter(|v| *v)
            .count();

        if count > 1 {
            self.tcx.dcx().span_bug(
                item.span,
                "the `resource` and `workgroup_shared` attributes are mutually exclusive",
            );
        }

        if let Some(attr) = attrs.resource.as_ref() {
            let (_, _, ty, _) = item.expect_static();

            let TyKind::Path(path) = ty.kind else {
                self.tcx
                    .dcx()
                    .span_bug(item.span, "expected a static with a path type");
            };
            let QPath::Resolved(None, path) = path else {
                self.tcx
                    .dcx()
                    .span_bug(item.span, "expected a static with a path type");
            };

            let resource_binding = ResourceBinding {
                group: attr.group,
                binding: attr.binding,
                span: attr.span,
            };
            let resource_kind = path.segments.last().unwrap().ident.name;

            // Note: this is a bit of a hack to resolve the resource kind, but it should work as
            // we assert in the `#[resource(...)]` attribute macro that the static's type implements
            // the `risl::resource::Resource` trait, which is a sealed trait only implemented by a
            // known set of types.

            let static_ext = match resource_kind.as_str() {
                "Uniform" => StaticExt::Uniform(resource_binding),
                "Storage" => StaticExt::Storage(resource_binding),
                "StorageMut" => StaticExt::StorageMut(resource_binding),
                _ => self.tcx.dcx().span_bug(
                    attr.span,
                    "the `static` item's type is not a valid resource type",
                ),
            };

            self.hir_ext.static_ext.insert(item.item_id(), static_ext);
        } else if attrs.workgroup_shared.is_some() {
            self.hir_ext
                .static_ext
                .insert(item.item_id(), StaticExt::Workgroup);
        }
    }

    fn visit_item_fn(&mut self, item: &Item<'tcx>, attrs: &Attributes) {
        let count = [
            attrs.gpu.is_some(),
            attrs.compute.is_some(),
            attrs.vertex.is_some(),
            attrs.fragment.is_some(),
        ]
        .into_iter()
        .filter(|v| *v)
        .count();

        if count > 1 {
            self.tcx.dcx().span_err(
                item.span,
                "the `gpu`, `compute`, `vertex` and `fragment` attributes are mutually exclusive",
            );

            return;
        }

        let def_id = item.owner_id.def_id;

        if attrs.gpu.is_some() {
            self.hir_ext.fn_ext.insert(def_id, FnExt::GpuFn);
        }

        if attrs.vertex.is_some() {
            self.hir_ext.fn_ext.insert(
                def_id,
                EntryPoint {
                    name: item.expect_fn().0.name,
                    kind: EntryPointKind::Vertex,
                }
                .into(),
            );
        }

        if attrs.fragment.is_some() {
            self.hir_ext.fn_ext.insert(
                def_id,
                EntryPoint {
                    name: item.expect_fn().0.name,
                    kind: EntryPointKind::Fragment,
                }
                .into(),
            );
        }

        if let Some(attr) = &attrs.compute {
            self.hir_ext.fn_ext.insert(
                def_id,
                EntryPoint {
                    name: item.expect_fn().0.name,
                    kind: EntryPointKind::Compute(WorkgroupSize {
                        x: attr.workgroup_size.0,
                        y: attr.workgroup_size.1,
                        z: attr.workgroup_size.2,
                        span: attr.span,
                    }),
                }
                .into(),
            );
        }
    }

    fn visit_item_mod(&mut self, item: &Item<'tcx>, attrs: &Attributes) {
        if attrs.shader_module.is_some() {
            let id = LocalModDefId::new_unchecked(item.owner_id.def_id);

            self.hir_ext.mod_ext.insert(
                id,
                ModExt {
                    is_shader_module: true,
                },
            );
        }
    }

    fn visit_item_struct(&mut self, item: &Item<'tcx>, attrs: &Attributes) {
        if attrs.gpu.is_some() {
            self.hir_ext.struct_ext.insert(item.item_id(), StructExt {});
        }
    }

    fn visit_item_enum(&mut self, item: &Item<'tcx>, attrs: &Attributes) {
        if attrs.gpu.is_some() {
            self.hir_ext.enum_ext.insert(item.item_id(), EnumExt {});
        }
    }

    fn visit_item_trait(&mut self, item: &Item<'tcx>, attrs: &Attributes) {
        if attrs.gpu.is_some() {
            self.hir_ext.trait_ext.insert(item.item_id(), TraitExt {});
        }
    }

    fn visit_item_impl(&mut self, item: &Item<'tcx>, attrs: &Attributes) {
        if attrs.gpu.is_some() {
            self.hir_ext.impl_ext.insert(item.item_id(), ImplExt {});
        }
    }
}

impl<'a, 'tcx> Visitor<'tcx> for Locator<'a, 'tcx> {
    type NestedFilter = nested_filter::OnlyBodies;

    fn maybe_tcx(&mut self) -> Self::MaybeTyCtxt {
        self.tcx
    }

    fn visit_item(&mut self, item: &'tcx Item<'tcx>) {
        let attrs = self.tcx.hir_attrs(item.hir_id());
        let attrs = collect_risl_attributes(self.tcx, attrs);

        attrs.check_target(self.tcx, &Target::from_item(item));

        match &item.kind {
            ItemKind::Static(_, _, _, _) => self.visit_item_static(item, &attrs),
            ItemKind::Const(_, _, _, _) => self.visit_item_const(item, &attrs),
            ItemKind::Fn { .. } => self.visit_item_fn(item, &attrs),
            ItemKind::Mod(_, _) => self.visit_item_mod(item, &attrs),
            ItemKind::Struct(_, _, _) => self.visit_item_struct(item, &attrs),
            ItemKind::Enum(_, _, _) => self.visit_item_enum(item, &attrs),
            ItemKind::Trait(_, _, _, _, _, _, _) => self.visit_item_trait(item, &attrs),
            ItemKind::Impl(_) => self.visit_item_impl(item, &attrs),
            _ => (),
        }

        intravisit::walk_item(self, item)
    }

    fn visit_expr(&mut self, ex: &'tcx Expr<'tcx>) {
        let attrs = self.tcx.hir_attrs(ex.hir_id);
        let attrs = collect_risl_attributes(self.tcx, attrs);

        attrs.check_target(self.tcx, &Target::Expression);

        if let ExprKind::ConstBlock(block) = &ex.kind {
            if let Some(attr) = &attrs.shader_wgsl {
                if let Ok(request) = try_build_shader_request(
                    self.tcx,
                    block,
                    ex.span,
                    attr.request_id,
                    ShaderRequestKind::Wgsl,
                ) {
                    self.hir_ext.shader_requests.push(request);
                }
            }

            if let Some(attr) = &attrs.shader_module_interface {
                if let Ok(request) = try_build_shader_request(
                    self.tcx,
                    block,
                    ex.span,
                    attr.request_id,
                    ShaderRequestKind::ShaderModuleInterface,
                ) {
                    self.hir_ext.shader_requests.push(request);
                }
            }
        }

        intravisit::walk_expr(self, ex)
    }

    fn visit_param(&mut self, param: &'tcx Param<'tcx>) {
        let attrs = self.tcx.hir_attrs(param.hir_id);
        let attrs = collect_risl_attributes(self.tcx, attrs);

        attrs.check_target(self.tcx, &Target::Param);

        if let Ok(shader_io_binding) = try_maybe_shader_io_binding(self.tcx, &attrs, param.span) {
            self.hir_ext
                .param_ext
                .insert(param.hir_id, ParamExt { shader_io_binding });
        }

        intravisit::walk_param(self, param)
    }

    fn visit_field_def(&mut self, s: &'tcx FieldDef<'tcx>) {
        let attrs = self.tcx.hir_attrs(s.hir_id);
        let attrs = collect_risl_attributes(self.tcx, attrs);

        attrs.check_target(self.tcx, &Target::Field);

        if let Ok(shader_io_binding) = try_maybe_shader_io_binding(self.tcx, &attrs, s.span) {
            self.hir_ext
                .field_ext
                .insert(s.def_id, FieldExt { shader_io_binding });
        }

        intravisit::walk_field_def(self, s)
    }

    fn visit_generic_param(&mut self, p: &'tcx GenericParam<'tcx>) {
        let attrs = self.tcx.hir_attrs(p.hir_id);
        let attrs = collect_risl_attributes(self.tcx, attrs);

        attrs.check_target(self.tcx, &Target::from_generic_param(p));

        intravisit::walk_generic_param(self, p)
    }

    // Visit all remaining node types that can be attribute targets to ensure we emit errors for
    // missplaced attributes.
    fn visit_expr_field(&mut self, field: &'tcx ExprField<'tcx>) {
        let attrs = self.tcx.hir_attrs(field.hir_id);
        let attrs = collect_risl_attributes(self.tcx, attrs);

        attrs.check_target(self.tcx, &Target::ExprField);

        intravisit::walk_expr_field(self, field)
    }

    fn visit_arm(&mut self, a: &'tcx Arm<'tcx>) {
        let attrs = self.tcx.hir_attrs(a.hir_id);
        let attrs = collect_risl_attributes(self.tcx, attrs);

        attrs.check_target(self.tcx, &Target::Arm);

        intravisit::walk_arm(self, a)
    }

    fn visit_pat_field(&mut self, f: &'tcx PatField<'tcx>) {
        let attrs = self.tcx.hir_attrs(f.hir_id);
        let attrs = collect_risl_attributes(self.tcx, attrs);

        attrs.check_target(self.tcx, &Target::PatField);

        intravisit::walk_pat_field(self, f)
    }

    fn visit_variant(&mut self, v: &'tcx Variant<'tcx>) {
        let attrs = self.tcx.hir_attrs(v.hir_id);
        let attrs = collect_risl_attributes(self.tcx, attrs);

        attrs.check_target(self.tcx, &Target::Variant);

        intravisit::walk_variant(self, v)
    }

    fn visit_stmt(&mut self, s: &'tcx Stmt<'tcx>) {
        let attrs = self.tcx.hir_attrs(s.hir_id);
        let attrs = collect_risl_attributes(self.tcx, attrs);

        attrs.check_target(self.tcx, &Target::Statement);

        intravisit::walk_stmt(self, s)
    }

    fn visit_foreign_item(&mut self, i: &'tcx ForeignItem<'tcx>) -> Self::Result {
        let attrs = self.tcx.hir_attrs(i.hir_id());
        let attrs = collect_risl_attributes(self.tcx, attrs);

        attrs.check_target(self.tcx, &Target::from_foreign_item(i));

        intravisit::walk_foreign_item(self, i)
    }
}

pub fn build(hir_ext: &mut HirExt, tcx: TyCtxt<'_>) {
    let mut locator = Locator { tcx, hir_ext };

    tcx.hir_visit_all_item_likes_in_crate(&mut locator);
}
