use rustc_ast::{AttrKind, ast};
use rustc_hir::Target;
use rustc_middle::ty::TyCtxt;
use rustc_span::{ErrorGuaranteed, Span, Symbol};

use crate::compiler::ATTRIBUTE_NAMESPACE;

trait Attr: Sized {
    fn try_from_ast(
        tcx: TyCtxt<'_>,
        attr: &rustc_hir::Attribute,
    ) -> Result<Option<Self>, ErrorGuaranteed>;

    fn valid_target(&self, target: &Target) -> bool;
}

fn is_risl_attribute(attr: &rustc_hir::Attribute) -> bool {
    attr.path()
        .first()
        .map(|segment| segment.as_str() == ATTRIBUTE_NAMESPACE)
        .unwrap_or(false)
}

/// Helper for testing that an attribute matches an expected name.
///
/// Assumes that we've already verified that the first segment matches the [ATTRIBUTE_NAMESPACE]
/// (with [is_risl_attribute]).
fn attr_matches_name(attr: &rustc_hir::Attribute, name: &str) -> bool {
    attr.get_normal_item().path.segments[1].as_str() == name
}

fn expect_u32(tcx: TyCtxt<'_>, item: &ast::MetaItemInner) -> Result<u32, ErrorGuaranteed> {
    let v = item.lit().and_then(|lit| {
        if let ast::LitKind::Int(n, _) = &lit.kind {
            u32::try_from(n.get()).ok()
        } else {
            None
        }
    });

    v.ok_or_else(|| {
        tcx.sess
            .dcx()
            .span_err(item.span(), "expected an integer literal")
    })
}

fn expect_string(tcx: TyCtxt<'_>, item: &ast::MetaItemInner) -> Result<Symbol, ErrorGuaranteed> {
    let v = item.lit().and_then(|lit| {
        if let ast::LitKind::Str(n, _) = &lit.kind {
            Some(*n)
        } else {
            None
        }
    });

    v.ok_or_else(|| {
        tcx.sess
            .dcx()
            .span_err(item.span(), "expected a string literal")
    })
}

macro_rules! impl_attr_from_ast_no_args {
    ($T:ident, $name:literal) => {
        fn try_from_ast(
            tcx: TyCtxt<'_>,
            attr: &rustc_hir::Attribute,
        ) -> Result<Option<Self>, ErrorGuaranteed> {
            if attr_matches_name(attr, $name) {
                // We check the attribute is "normal" before invoking this function
                if !matches!(attr.get_normal_item().args, rustc_hir::AttrArgs::Empty) {
                    Err(tcx.dcx().span_err(
                        attr.span(),
                        format!("`{}` attribute does not take any arguments", $name),
                    ))
                } else {
                    Ok(Some($T { span: attr.span() }))
                }
            } else {
                Ok(None)
            }
        }
    };
}

macro_rules! impl_attr_from_ast_single_int_arg {
    ($T:ident, $field:ident, $name:literal) => {
        fn try_from_ast(
            tcx: TyCtxt<'_>,
            attr: &rustc_hir::Attribute,
        ) -> Result<Option<Self>, ErrorGuaranteed> {
            if attr_matches_name(attr, $name) {
                if let Some(meta_item_list) = attr.meta_item_list()
                    && meta_item_list.len() == 1
                {
                    return Ok(Some($T {
                        $field: expect_u32(tcx, &meta_item_list[0])?,
                        span: Default::default(),
                    }));
                }

                Err(tcx.dcx().span_err(
                    attr.span(),
                    format!("`{}` attribute expected one argument", $name),
                ))
            } else {
                Ok(None)
            }
        }
    };
}

macro_rules! impl_attr_from_ast_single_string_arg {
    ($T:ident, $field:ident, $name:literal) => {
        fn try_from_ast(
            tcx: TyCtxt<'_>,
            attr: &rustc_hir::Attribute,
        ) -> Result<Option<Self>, ErrorGuaranteed> {
            if attr_matches_name(attr, $name) {
                if let Some(meta_item_list) = attr.meta_item_list()
                    && meta_item_list.len() == 1
                {
                    return Ok(Some($T {
                        $field: expect_string(tcx, &meta_item_list[0])?,
                        span: Default::default(),
                    }));
                }

                Err(tcx.dcx().span_err(
                    attr.span(),
                    format!("`{}` attribute expected one argument", $name),
                ))
            } else {
                Ok(None)
            }
        }
    };
}

/// Decorates requests for a compiled shader made with the [risl::shader_wgsl] macro.
#[derive(Debug)]
pub struct AttrShaderWgsl {
    pub request_id: Symbol,
    pub span: Span,
}

impl Attr for AttrShaderWgsl {
    impl_attr_from_ast_single_string_arg!(AttrShaderWgsl, request_id, "shader_wgsl");

    fn valid_target(&self, target: &Target) -> bool {
        match target {
            Target::Expression => true,
            _ => false,
        }
    }
}

/// Decorates `mod` items that describe a shader module.
///
/// See also [risl::shader_module].
#[derive(Debug)]
pub struct AttrShaderModule {
    pub span: Span,
}

impl Attr for AttrShaderModule {
    impl_attr_from_ast_no_args!(AttrShaderModule, "shader_module");

    fn valid_target(&self, target: &Target) -> bool {
        match target {
            Target::Mod => true,
            _ => false,
        }
    }
}

/// Decorates `fn` or `impl` items to indicate that the function(s) are GPU compatible.
///
/// See also [risl::gpu].
#[derive(Debug)]
pub struct AttrGpu {
    pub span: Span,
}

impl Attr for AttrGpu {
    impl_attr_from_ast_no_args!(AttrGpu, "gpu");

    fn valid_target(&self, target: &Target) -> bool {
        match target {
            Target::Fn
            | Target::Method(_)
            | Target::Trait
            | Target::Impl { .. }
            | Target::Struct
            | Target::Enum => true,
            _ => false,
        }
    }
}

/// Decorates "uniform" and "storage" resource bindings on `static` items (that must be owned by a
/// `mod` with the `#[shader_module]` attribute).
///
/// These are initially written as:
///
/// ```
/// #[buffer_bound(group = 0, binding = 0)]
/// static X: Uniform<u32>;
/// ```
///
/// The `#[buffer_bound]` attribute macro rewrites these to:
///
/// ```
/// #[risl::buffer_bound(0, 0)]
/// static X: Uniform<u32> = unsafe { core::mem::zeroed() };
/// ```
#[derive(Debug)]
pub struct AttrResource {
    pub group: u32,
    pub binding: u32,
    pub span: Span,
}

impl Attr for AttrResource {
    fn try_from_ast(
        tcx: TyCtxt<'_>,
        attr: &rustc_hir::Attribute,
    ) -> Result<Option<Self>, ErrorGuaranteed> {
        if attr_matches_name(attr, "resource") {
            if let Some(meta_item_list) = attr.meta_item_list()
                && meta_item_list.len() == 2
            {
                return Ok(Some(AttrResource {
                    group: expect_u32(tcx, &meta_item_list[0])?,
                    binding: expect_u32(tcx, &meta_item_list[1])?,
                    span: attr.span(),
                }));
            }

            Err(tcx
                .dcx()
                .span_err(attr.span(), "`resource` attribute expected two arguments"))
        } else {
            Ok(None)
        }
    }

    fn valid_target(&self, target: &Target) -> bool {
        match target {
            Target::Static => true,
            _ => false,
        }
    }
}

/// Decorates workgroup-shared `static` items (that must be owned by a `mod` with the
/// `#[shader_module]` attribute).
///
/// These are initially written as:
///
/// ```
/// workgroup x: u32 = 0;
/// ```
///
/// The `#[shader_module]` attribute macro rewrites these to:
///
/// ```
/// #[risl::workgroup]
/// static x: u32 = 0;
/// ```
#[derive(Debug)]
pub struct AttrWorkgroupShared {
    pub span: Span,
}

impl Attr for AttrWorkgroupShared {
    impl_attr_from_ast_no_args!(AttrWorkgroupShared, "workgroup_shared");

    fn valid_target(&self, target: &Target) -> bool {
        match target {
            Target::Static => true,
            _ => false,
        }
    }
}

/// Decorates `const` items (that must be owned by a `mod` with the `#[shader_module]` attribute) to
/// indicate that they are specializable/overridable.
///
/// These are initially written as:
///
/// ```
/// override X: u32;
/// ```
///
/// The `#[shader_module]` attribute macro rewrites these to:
///
/// ```
/// #[risl::override]
/// const X: u32 = ...;
/// ```
///
/// (where the initializer in a dummy value).
#[derive(Debug)]
pub struct AttrOverride {
    pub span: Span,
}

impl Attr for AttrOverride {
    impl_attr_from_ast_no_args!(AttrOverride, "override");

    fn valid_target(&self, target: &Target) -> bool {
        match target {
            Target::Const => true,
            _ => false,
        }
    }
}

/// Decorates `const` items (that must also be decorated with an `#[override]` attribute) to set an
/// override ID.
///
/// See also [risl::id].
#[derive(Debug)]
pub struct AttrOverrideId {
    pub override_id: u32,
    pub span: Span,
}

impl Attr for AttrOverrideId {
    impl_attr_from_ast_single_int_arg!(AttrOverrideId, override_id, "id");

    fn valid_target(&self, target: &Target) -> bool {
        match target {
            Target::Const => true,
            _ => false,
        }
    }
}

/// Decorates overridable `const` items to indicate that the constant does not have a default value.
#[derive(Debug)]
pub struct AttrOverrideRequired {
    pub span: Span,
}

impl Attr for AttrOverrideRequired {
    impl_attr_from_ast_no_args!(AttrOverrideRequired, "override_required");

    fn valid_target(&self, target: &Target) -> bool {
        match target {
            Target::Const => true,
            _ => false,
        }
    }
}

/// Decorates `fn` items (that must be owned by a `mod` with the `#[shader_module]` attribute) to
/// indicate that the function may be used as a "compute" shader entry-point.
///
/// Takes 3 arguments for the x, y and z dimensions of the workgroup thread grid.
///
/// See also [risl::compute].
#[derive(Debug)]
pub struct AttrCompute {
    pub workgroup_size: (u32, u32, u32),
    pub span: Span,
}

impl Attr for AttrCompute {
    fn try_from_ast(
        tcx: TyCtxt<'_>,
        attr: &rustc_hir::Attribute,
    ) -> Result<Option<Self>, ErrorGuaranteed> {
        if attr_matches_name(attr, "compute") {
            if let Some(meta_item_list) = attr.meta_item_list()
                && meta_item_list.len() == 3
            {
                let x = expect_u32(tcx, &meta_item_list[0])?;
                let y = expect_u32(tcx, &meta_item_list[1])?;
                let z = expect_u32(tcx, &meta_item_list[2])?;

                return Ok(Some(AttrCompute {
                    workgroup_size: (x, y, z),
                    span: attr.span(),
                }));
            }

            return Err(tcx.dcx().span_err(
                attr.span(),
                "the `workgroup_size` attribute expects at least one and at most three arguments",
            ));
        }

        Ok(None)
    }

    fn valid_target(&self, target: &Target) -> bool {
        match target {
            Target::Fn => true,
            _ => false,
        }
    }
}

/// Decorates `fn` items (that must be owned by a `mod` with the `#[shader_module]` attribute) to
/// indicate that the function may be used as a "vertex" shader entry-point.
///
/// See also [risl::vertex].
#[derive(Debug)]
pub struct AttrVertex {
    pub span: Span,
}

impl Attr for AttrVertex {
    impl_attr_from_ast_no_args!(AttrVertex, "vertex");

    fn valid_target(&self, target: &Target) -> bool {
        match target {
            Target::Fn => true,
            _ => false,
        }
    }
}

/// Decorates `fn` items (that must be owned by a `mod` with the `#[shader_module]` attribute) to
/// indicate that the function may be used as a "fragment" shader entry-point.
///
/// See also [risl::fragment].
#[derive(Debug)]
pub struct AttrFragment {
    pub span: Span,
}

impl Attr for AttrFragment {
    impl_attr_from_ast_no_args!(AttrFragment, "fragment");

    fn valid_target(&self, target: &Target) -> bool {
        match target {
            Target::Fn => true,
            _ => false,
        }
    }
}

/// Decorates function parameters or struct fields (for `fn` or `struct` items that must be owned by
/// a `mod` with the `#[shader_module]` attribute) to specify a shader IO binding location.
///
/// See also [risl::location]
#[derive(Debug)]
pub struct AttrLocation {
    pub location: u32,
    pub span: Span,
}

impl Attr for AttrLocation {
    impl_attr_from_ast_single_int_arg!(AttrLocation, location, "id");

    fn valid_target(&self, target: &Target) -> bool {
        match target {
            Target::Param | Target::Field => true,
            _ => false,
        }
    }
}

/// Enumerates the available "builtin" shader IO bindings.
///
/// See also [AttrBuiltin] and [risl::builtin].
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum BuiltinName {
    VertexIndex,
    InstanceIndex,
    Position,
    FrontFacing,
    FragDepth,
    SampleIndex,
    SampleMask,
    LocalInvocationId,
    LocalInvocationIndex,
    GlobalInvocationId,
    WorkgroupId,
    NumWorkgroups,
}

fn expect_builtin_name(
    tcx: TyCtxt<'_>,
    item: &ast::MetaItemInner,
) -> Result<BuiltinName, ErrorGuaranteed> {
    let builtin_name = item.ident().and_then(|ident| match ident.as_str() {
        "vertex_index" => Some(BuiltinName::VertexIndex),
        "instance_index" => Some(BuiltinName::InstanceIndex),
        "position" => Some(BuiltinName::Position),
        "front_facing" => Some(BuiltinName::FrontFacing),
        "frag_depth" => Some(BuiltinName::FragDepth),
        "sample_index" => Some(BuiltinName::SampleIndex),
        "sample_mask" => Some(BuiltinName::SampleMask),
        "local_invocation_id" => Some(BuiltinName::LocalInvocationId),
        "local_invocation_index" => Some(BuiltinName::LocalInvocationIndex),
        "global_invocation_id" => Some(BuiltinName::GlobalInvocationId),
        "workgroup_id" => Some(BuiltinName::WorkgroupId),
        "num_workgroups" => Some(BuiltinName::NumWorkgroups),
        _ => None,
    });

    builtin_name.ok_or_else(|| {
        tcx.dcx().span_err(
            item.span(),
            "expected an identifier for a builtin name token",
        )
    })
}

/// Decorates function parameters or struct fields (for `fn` or `struct` items that must be owned by
/// a `mod` with the `#[shader_module]` attribute) to specify a shader "builtin" IO binding.
///
/// See also [risl::builtin]
#[derive(Debug)]
pub struct AttrBuiltin {
    pub builtin_name: BuiltinName,
    pub span: Span,
}

impl AttrBuiltin {
    pub fn is_position(&self) -> bool {
        matches!(self.builtin_name, BuiltinName::Position)
    }
}

impl Attr for AttrBuiltin {
    fn try_from_ast(
        tcx: TyCtxt<'_>,
        attr: &rustc_hir::Attribute,
    ) -> Result<Option<Self>, ErrorGuaranteed> {
        if attr_matches_name(attr, "builtin") {
            if let Some(meta_item_list) = attr.meta_item_list()
                && meta_item_list.len() == 1
            {
                let builtin_name = expect_builtin_name(tcx, &meta_item_list[0])?;

                return Ok(Some(AttrBuiltin {
                    builtin_name,
                    span: attr.span(),
                }));
            }

            return Err(tcx
                .sess
                .dcx()
                .span_err(attr.span(), "the `builtin` attribute expects one argument"));
        }

        Ok(None)
    }

    fn valid_target(&self, target: &Target) -> bool {
        match target {
            Target::Param | Target::Field => true,
            _ => false,
        }
    }
}

/// Decorates function parameters or struct fields (that must also be decorated with a
/// `#[builtin(position)]` attribute) to specify that the computation of the result is invariant
/// across different programs and different invocations of the same entry point.
///
/// See also [risl::invariant]
#[derive(Debug)]
pub struct AttrInvariant {
    pub span: Span,
}

impl Attr for AttrInvariant {
    impl_attr_from_ast_no_args!(AttrInvariant, "invariant");

    fn valid_target(&self, target: &Target) -> bool {
        match target {
            Target::Param | Target::Field => true,
            _ => false,
        }
    }
}

/// Enumerates the available interpolation types for the `#[interpolate(...)]` attribute.
///
/// See also [AttrInterpolate] and [risl::interpolate].
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum InterpolationTypeName {
    Perspective,
    Linear,
    Flat,
}

fn expect_interpolation_type_name(
    tcx: TyCtxt<'_>,
    item: &ast::MetaItemInner,
) -> Result<InterpolationTypeName, ErrorGuaranteed> {
    let type_name = item.ident().and_then(|ident| match ident.as_str() {
        "perspective" => Some(InterpolationTypeName::Perspective),
        "linear" => Some(InterpolationTypeName::Linear),
        "flat" => Some(InterpolationTypeName::Flat),
        _ => None,
    });

    type_name.ok_or_else(|| {
        tcx.dcx().span_err(
            item.span(),
            "expected an identifier for an interpolation type name token",
        )
    })
}

/// Enumerates the available interpolation sampling methods for the `#[interpolate(...)]` attribute.
///
/// See also [AttrInterpolate] and [risl::interpolate].
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum InterpolationSamplingName {
    Center,
    Centroid,
    Sample,
    First,
    Either,
}

fn expect_interpolation_sampling_name(
    tcx: TyCtxt<'_>,
    item: &ast::MetaItemInner,
) -> Result<InterpolationSamplingName, ErrorGuaranteed> {
    let sampling_name = item.ident().and_then(|ident| match ident.as_str() {
        "center" => Some(InterpolationSamplingName::Center),
        "centroid" => Some(InterpolationSamplingName::Centroid),
        "sample" => Some(InterpolationSamplingName::Sample),
        "first" => Some(InterpolationSamplingName::First),
        "either" => Some(InterpolationSamplingName::Either),
        _ => None,
    });

    sampling_name.ok_or_else(|| {
        tcx.dcx().span_err(
            item.span(),
            "expected an identifier for an interpolation sampling name token",
        )
    })
}

/// Decorates function parameters or struct fields (that must also be decorated with a
/// `#[builtin(position)]` attribute) to specify that the computation of the result is invariant
/// across different programs and different invocations of the same entry point.
///
/// See also [risl::invariant]
#[derive(Debug)]
pub struct AttrInterpolate {
    pub type_name: InterpolationTypeName,
    pub sampling_name: Option<InterpolationSamplingName>,
    pub span: Span,
}

impl Attr for AttrInterpolate {
    fn try_from_ast(
        tcx: TyCtxt<'_>,
        attr: &rustc_hir::Attribute,
    ) -> Result<Option<Self>, ErrorGuaranteed> {
        if attr_matches_name(attr, "interpolate") {
            if let Some(meta_item_list) = attr.meta_item_list()
                && (1..=2).contains(&meta_item_list.len())
            {
                let type_name = expect_interpolation_type_name(tcx, &meta_item_list[0])?;
                let sampling_name = meta_item_list
                    .get(1)
                    .map(|item| expect_interpolation_sampling_name(tcx, item))
                    .transpose()?;

                return Ok(Option::from(AttrInterpolate {
                    type_name,
                    sampling_name,
                    span: Default::default(),
                }));
            }

            return Err(tcx.dcx().span_err(
                attr.span(),
                "the `builtin` attribute expects at least one and at most two arguments",
            ));
        }

        Ok(None)
    }

    fn valid_target(&self, target: &Target) -> bool {
        match target {
            Target::Param | Target::Field => true,
            _ => false,
        }
    }
}

/// Decorates function parameters or struct fields (for `fn` or `struct` items that must be owned by
/// a `mod` with the `#[shader_module]` attribute) to specify a shader IO binding location.
///
/// See also [risl::location]
#[derive(Debug)]
pub struct AttrBlendSrc {
    pub blend_src: u32,
    pub span: Span,
}

impl Attr for AttrBlendSrc {
    impl_attr_from_ast_single_int_arg!(AttrBlendSrc, blend_src, "blend_src");

    fn valid_target(&self, target: &Target) -> bool {
        match target {
            Target::Field => true,
            _ => false,
        }
    }
}

macro_rules! register_attributes {
    ($($field:ident => $Ty:ident,)*) => {
        #[derive(Debug)]
        pub struct Attributes {
            $(pub $field: Option<$Ty>,)*
        }

        impl Attributes {
            fn new() -> Self {
                Self {
                    $($field: None,)*
                }
            }

            fn try_insert(&mut self, tcx: TyCtxt<'_>, attr: &rustc_hir::Attribute) {
                $(
                    if let Ok(Some(attr)) = $Ty::try_from_ast(tcx, attr) {
                        if self.$field.is_some() {
                            tcx.dcx().span_err(attr.span, "duplicate attribute definition");
                        }

                        self.$field = Some(attr);
                    }
                )*
            }

            pub fn check_target(&self, tcx: TyCtxt<'_>, target: &Target) {
                $(
                    if let Some(attr) = &self.$field {
                        if !attr.valid_target(target) {
                            tcx.dcx().span_err(attr.span, "not a valid location for this attribute");
                        }
                    }
                )*
            }
        }
    }
}

register_attributes!(
    shader_wgsl => AttrShaderWgsl,
    shader_module => AttrShaderModule,
    gpu => AttrGpu,
    resource => AttrResource,
    workgroup_shared => AttrWorkgroupShared,
    override_ => AttrOverride,
    override_id => AttrOverrideId,
    override_required => AttrOverrideRequired,
    compute => AttrCompute,
    vertex => AttrVertex,
    fragment => AttrFragment,
    location => AttrLocation,
    builtin => AttrBuiltin,
    invariant => AttrInvariant,
    interpolate => AttrInterpolate,
    blend_src => AttrBlendSrc,
);

pub fn collect_risl_attributes(tcx: TyCtxt<'_>, attributes: &[rustc_hir::Attribute]) -> Attributes {
    let mut result = Attributes::new();

    for attr in attributes {
        if is_risl_attribute(attr) {
            result.try_insert(tcx, attr);
        }
    }

    result
}
