use std::cell::OnceCell;
use std::sync::RwLock;

use indexmap::{IndexMap, IndexSet};
use rustc_hash::{FxHashMap, FxHashSet};
use rustc_hir as hir;
use rustc_hir::attrs::InlineAttr;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::{LocalDefId, LocalModDefId};
use rustc_hir::lang_items::LangItem;
use rustc_hir::limit::Limit;
use rustc_middle::mir::ConstValue;
use rustc_middle::mir::interpret::Scalar;
use rustc_middle::mir::mono::{CollectionMode, MonoItem as InternalMonoItem};
use rustc_middle::ty::{
    self, GenericArgs, GenericParamDefKind, Instance as InternalInstance, TyCtxt,
};
use rustc_middle::util::Providers;
use rustc_middle::{bug, span_bug};
use rustc_public::mir::alloc::{AllocId, GlobalAlloc};
use rustc_public::mir::mono::{Instance, InstanceKind, MonoItem};
use rustc_public::rustc_internal::{internal, stable};
use rustc_public::ty::Ty;
use rustc_public::{CrateDef, DefId};
use rustc_span::source_map::{Spanned, dummy_spanned, respan};
use rustc_span::{DUMMY_SP, Span};
use tracing::{debug, instrument, trace};

use crate::context::RislContext as Cx;
use crate::monomorphize::errors::{
    EncounteredErrorWhileInstantiating, NoOptimizedMir, RecursionLimit,
};

#[derive(Clone, Copy, PartialEq)]
pub(crate) enum MonoItemCollectionStrategy {
    Eager,
    Lazy,
}

/// The state that is shared across the concurrent threads that are doing collection.
struct SharedState {
    /// Items that have been or are currently being recursively collected.
    visited: RwLock<FxHashSet<MonoItem>>,
    /// Items that have been or are currently being recursively treated as "mentioned", i.e., their
    /// consts are evaluated but nothing is added to the collection.
    mentioned: RwLock<FxHashSet<MonoItem>>,
    /// Which items are being used where, for better errors.
    usage_map: RwLock<UsageMap>,
}

pub(crate) struct UsageMap {
    // Maps every mono item to the mono items used by it.
    used_map: FxHashMap<MonoItem, Vec<MonoItem>>,

    // Maps every mono item to the mono items that use it.
    user_map: FxHashMap<MonoItem, Vec<MonoItem>>,
}

impl UsageMap {
    fn new() -> UsageMap {
        UsageMap {
            used_map: Default::default(),
            user_map: Default::default(),
        }
    }

    fn record_used(&mut self, user_item: MonoItem, used_items: &MonoItems) {
        for used_item in used_items.items() {
            self.user_map
                .entry(used_item)
                .or_default()
                .push(user_item.clone());
        }

        assert!(
            self.used_map
                .insert(user_item, used_items.items().collect())
                .is_none()
        );
    }
}

struct MonoItems {
    // We want a set of MonoItem + Span where trying to re-insert a MonoItem with a different Span
    // is ignored. Map does that, but it looks odd.
    items: IndexMap<MonoItem, Span>,
}

impl MonoItems {
    fn new() -> Self {
        Self {
            items: IndexMap::default(),
        }
    }

    fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    fn push(&mut self, item: Spanned<MonoItem>) {
        // Insert only if the entry does not exist. A normal insert would stomp the first span that
        // got inserted.
        self.items.entry(item.node).or_insert(item.span);
    }

    fn items(&self) -> impl Iterator<Item = MonoItem> + '_ {
        self.items.keys().cloned()
    }
}

impl IntoIterator for MonoItems {
    type Item = Spanned<MonoItem>;
    type IntoIter = impl Iterator<Item = Spanned<MonoItem>>;

    fn into_iter(self) -> Self::IntoIter {
        self.items
            .into_iter()
            .map(|(item, span)| respan(span, item))
    }
}

impl Extend<Spanned<MonoItem>> for MonoItems {
    fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = Spanned<MonoItem>>,
    {
        for item in iter {
            self.push(item)
        }
    }
}

fn collect_items_root(
    cx: &Cx,
    starting_item: Spanned<MonoItem>,
    state: &SharedState,
    recursion_limit: Limit,
) {
    if !state
        .visited
        .write()
        .unwrap()
        .insert(starting_item.node.clone())
    {
        // We've been here already, no need to search again.
        return;
    }

    let mut recursion_depths = FxHashMap::default();

    collect_items_rec(
        cx,
        starting_item,
        state,
        &mut recursion_depths,
        recursion_limit,
        CollectionMode::UsedItems,
    );
}

fn maybe_shimmed(cx: &Cx, item: MonoItem) -> MonoItem {
    if let MonoItem::Fn(instance) = item {
        MonoItem::Fn(cx.shim_def_lookup().maybe_shimmed(cx.tcx(), instance))
    } else {
        // We only do shims for functions, so for other mono-item kinds we always return the
        // original item.

        item
    }
}

/// Collect all monomorphized items reachable from `starting_point`, and emit a note diagnostic if a
/// post-monomorphization error is encountered during a collection step.
///
/// `mode` determined whether we are scanning for [used items][CollectionMode::UsedItems]
/// or [mentioned items][CollectionMode::MentionedItems].
fn collect_items_rec(
    cx: &Cx,
    starting_item: Spanned<MonoItem>,
    state: &SharedState,
    recursion_depths: &mut FxHashMap<DefId, usize>,
    recursion_limit: Limit,
    mode: CollectionMode,
) {
    let mut used_items = MonoItems::new();
    let mut mentioned_items = MonoItems::new();
    let recursion_depth_reset;

    // Post-monomorphization errors MVP
    //
    // We can encounter errors while monomorphizing an item, but we don't have a good way of
    // showing a complete stack of spans ultimately leading to collecting the erroneous one yet.
    // (It's also currently unclear exactly which diagnostics and information would be interesting
    // to report in such cases)
    //
    // This leads to suboptimal error reporting: a post-monomorphization error (PME) will be
    // shown with just a spanned piece of code causing the error, without information on where
    // it was called from. This is especially obscure if the erroneous mono item is in a
    // dependency. See for example issue #85155, where, before minimization, a PME happened two
    // crates downstream from libcore's stdarch, without a way to know which dependency was the
    // cause.
    //
    // If such an error occurs in the current crate, its span will be enough to locate the
    // source. If the cause is in another crate, the goal here is to quickly locate which mono
    // item in the current crate is ultimately responsible for causing the error.
    //
    // To give at least _some_ context to the user: while collecting mono items, we check the
    // error count. If it has changed, a PME occurred, and we trigger some diagnostics about the
    // current step of mono items collection.
    //
    // FIXME: don't rely on global state, instead bubble up errors. Note: this is very hard to do.
    let error_count = cx.tcx().dcx().err_count();

    // In `mentioned_items` we collect items that were mentioned in this MIR but possibly do not
    // need to be monomorphized. This is done to ensure that optimizing away function calls does not
    // hide const-eval errors that those calls would otherwise have triggered.
    match starting_item.node {
        MonoItem::Static(def) => {
            recursion_depth_reset = None;

            // RISL does not allow functions to access regular Rust statics, we only allow functions
            // inside a shader-module to access special "uniform", "storage" and "workgroup-shared"
            // statics. These special statics may not have initializers ("uniform" and "storage"
            // statics are initialized by the device based on pipeline resource bindings and
            // "workgroup-shared" statics are always zero-initialized). To satisfy rustc we give
            // these statics a dummy initializer, but we don't need to evaluate that initializer
            // here.
        }
        MonoItem::Fn(instance) => {
            // Keep track of the monomorphization recursion depth
            recursion_depth_reset = Some(check_recursion_limit(
                cx.tcx(),
                instance,
                starting_item.span,
                recursion_depths,
                recursion_limit,
            ));

            rustc_data_structures::stack::ensure_sufficient_stack(|| {
                let instance = internal(cx.tcx(), instance);

                let Ok((used, mentioned)) = cx.tcx().items_of_instance((instance, mode)) else {
                    // Normalization errors here are usually due to trait solving overflow.
                    // FIXME: I assume that there are few type errors at post-analysis stage, but not
                    // entirely sure.
                    // We have to emit the error outside of `items_of_instance` to access the
                    // span of the `starting_item`.
                    let def_id = instance.def_id();
                    let def_span = cx.tcx().def_span(def_id);
                    let def_path_str = cx.tcx().def_path_str(def_id);
                    cx.tcx().dcx().emit_fatal(RecursionLimit {
                        span: starting_item.span,
                        instance: instance.to_string(),
                        def_span,
                        def_path_str,
                    });
                };

                used_items.extend(used.iter().map(|spanned| Spanned {
                    span: spanned.span,
                    node: maybe_shimmed(cx, stable(&spanned.node)),
                }));
                mentioned_items.extend(mentioned.iter().map(|spanned| Spanned {
                    span: spanned.span,
                    node: stable(&spanned.node),
                }));
            });
        }
        MonoItem::GlobalAsm(_) => {
            bug!(
                "RISL does not support inline ASM; should have been caught during the analysis \
                phase"
            );
        }
    };

    let node = internal(cx.tcx(), &starting_item.node);

    // Check for PMEs and emit a diagnostic if one happened. To try to show relevant edges of the
    // mono item graph.
    if cx.tcx().dcx().err_count() > error_count && node.is_generic_fn() && node.is_user_defined() {
        match starting_item.node {
            MonoItem::Fn(instance) => {
                cx.tcx()
                    .dcx()
                    .emit_note(EncounteredErrorWhileInstantiating {
                        span: starting_item.span,
                        kind: "fn",
                        instance: instance.name().to_string(),
                    })
            }
            MonoItem::Static(def_id) => {
                cx.tcx()
                    .dcx()
                    .emit_note(EncounteredErrorWhileInstantiating {
                        span: starting_item.span,
                        kind: "static",
                        instance: def_id.name().to_string(),
                    })
            }
            MonoItem::GlobalAsm(_) => {
                bug!(
                    "RISL does not support inline ASM; should have been caught during the analysis \
                    phase"
                );
            }
        }
    }
    // Only updating `usage_map` for used items as otherwise we may be inserting the same item
    // multiple times (if it is first 'mentioned' and then later actually used), and the usage map
    // logic does not like that.
    // This is part of the output of collection and hence only relevant for "used" items.
    // ("Mentioned" items are only considered internally during collection.)
    if mode == CollectionMode::UsedItems {
        state
            .usage_map
            .write()
            .unwrap()
            .record_used(starting_item.node, &used_items);
    }

    {
        let mut visited = OnceCell::default();
        if mode == CollectionMode::UsedItems {
            used_items.items.retain(|k, _| {
                visited
                    .get_mut_or_init(|| state.visited.write().unwrap())
                    .insert(k.clone())
            });
        }

        let mut mentioned = OnceCell::default();
        mentioned_items.items.retain(|k, _| {
            !visited
                .get_or_init(|| state.visited.write().unwrap())
                .contains(k)
                && mentioned
                    .get_mut_or_init(|| state.mentioned.write().unwrap())
                    .insert(k.clone())
        });
    }
    if mode == CollectionMode::MentionedItems {
        assert!(
            used_items.is_empty(),
            "'mentioned' collection should never encounter used items"
        );
    } else {
        for used_item in used_items {
            collect_items_rec(
                cx,
                used_item,
                state,
                recursion_depths,
                recursion_limit,
                CollectionMode::UsedItems,
            );
        }
    }

    // Walk over mentioned items *after* used items, so that if an item is both mentioned and used then
    // the loop above has fully collected it, so this loop will skip it.
    for mentioned_item in mentioned_items {
        collect_items_rec(
            cx,
            mentioned_item,
            state,
            recursion_depths,
            recursion_limit,
            CollectionMode::MentionedItems,
        );
    }

    if let Some((def_id, depth)) = recursion_depth_reset {
        recursion_depths.insert(def_id, depth);
    }
}

fn check_recursion_limit(
    tcx: TyCtxt,
    instance: Instance,
    span: Span,
    recursion_depths: &mut FxHashMap<DefId, usize>,
    recursion_limit: Limit,
) -> (DefId, usize) {
    let def_id = instance.def.def_id();
    let recursion_depth = recursion_depths.get(&def_id).cloned().unwrap_or(0);
    debug!(" => recursion depth={}", recursion_depth);

    let adjusted_recursion_depth = if tcx.is_lang_item(internal(tcx, def_id), LangItem::DropInPlace)
    {
        // HACK: drop_in_place creates tight monomorphization loops. Give
        // it more margin.
        recursion_depth / 4
    } else {
        recursion_depth
    };

    // Code that needs to instantiate the same function recursively
    // more than the recursion limit is assumed to be causing an
    // infinite expansion.
    if !recursion_limit.value_within_limit(adjusted_recursion_depth) {
        let instance_name = instance.name();
        let def_span = internal(tcx, instance.def.span());
        let def_path_str = instance.def.name();

        tcx.dcx().emit_fatal(RecursionLimit {
            span,
            instance: instance_name,
            def_span,
            def_path_str,
        });
    }

    recursion_depths.insert(def_id, recursion_depth + 1);

    (def_id, recursion_depth)
}

fn visit_drop_use(tcx: TyCtxt, ty: Ty, is_direct_call: bool, source: Span, output: &mut MonoItems) {
    let instance = Instance::resolve_drop_in_place(ty);

    visit_instance_use(tcx, instance, is_direct_call, source, output);
}

fn visit_instance_use(
    tcx: TyCtxt,
    instance: Instance,
    is_direct_call: bool,
    source: Span,
    output: &mut MonoItems,
) {
    debug!(
        "visit_item_use({:?}, is_direct_call={:?})",
        instance, is_direct_call
    );

    if !tcx.should_codegen_locally(internal(tcx, instance)) {
        return;
    }

    match instance.kind {
        InstanceKind::Virtual { .. } | InstanceKind::Intrinsic => {
            if !is_direct_call {
                bug!("{:?} being reified", instance);
            }
        }
        | InstanceKind::Item
        // TODO: the `Shim` kind includes DropGlue, for which rustc::monomorphize only collects the
        // item if the call is not direct. rustc_public does not allow to easily make the
        // distinction. Is this a problem, or does this only add some noise? Do we need to drop
        // to the `internal` representation here?
        | InstanceKind::Shim => {
            output.push(create_fn_mono_item(tcx, instance, source));
        }
    }
}

fn create_fn_mono_item(tcx: TyCtxt, instance: Instance, source: Span) -> Spanned<MonoItem> {
    respan(source, MonoItem::Fn(instance))
}

/// Scans the CTFE alloc in order to find function pointers and statics that must be monomorphized.
fn collect_alloc<'tcx>(tcx: TyCtxt<'tcx>, alloc_id: AllocId, output: &mut MonoItems) {
    match GlobalAlloc::from(alloc_id) {
        GlobalAlloc::Static(def) => {
            output.push(dummy_spanned(MonoItem::Static(def)));
        }
        GlobalAlloc::Memory(alloc) => {
            trace!("collecting {:?} with {:#?}", alloc_id, alloc);

            let ptrs = &alloc.provenance.ptrs;
            // avoid `ensure_sufficient_stack` in the common case of "no pointers"
            if !ptrs.is_empty() {
                rustc_data_structures::stack::ensure_sufficient_stack(move || {
                    for &prov in ptrs.iter().map(|(_, prov)| prov) {
                        collect_alloc(tcx, prov.0, output);
                    }
                });
            }
        }
        GlobalAlloc::Function(instance) => {
            if tcx.should_codegen_locally(internal(tcx, instance)) {
                trace!("collecting {:?} with {:#?}", alloc_id, instance);
                output.push(create_fn_mono_item(tcx, instance, DUMMY_SP));
            }
        }
        GlobalAlloc::VTable(ty, dyn_ty) => {
            bug!("v-tables are not supported by RISL")
        }
        GlobalAlloc::TypeId { .. } => {}
    }
}

#[instrument(skip(tcx, output), level = "debug")]
fn collect_const_value(tcx: TyCtxt, value: ConstValue, output: &mut MonoItems) {
    match value {
        ConstValue::Scalar(Scalar::Ptr(ptr, _size)) => {
            collect_alloc(tcx, stable(ptr.provenance.alloc_id()), output)
        }
        ConstValue::Indirect { alloc_id, .. } => collect_alloc(tcx, stable(alloc_id), output),
        ConstValue::Slice { alloc_id, .. } => {
            collect_alloc(tcx, stable(alloc_id), output);
        }
        _ => {}
    }
}

//=-----------------------------------------------------------------------------
// Root Collection
//=-----------------------------------------------------------------------------

fn is_shader_module_item<'tcx>(cx: &Cx<'tcx>, id: hir::HirId) -> bool {
    // We currently don't allow sub-modules in shader modules, so we only have to check the
    // direct parent.

    let mod_id = cx.tcx().parent_module(id);

    if let Some((_, ext)) = cx.extended_module(mod_id) {
        ext.is_shader_module
    } else {
        false
    }
}

fn is_instantiable(tcx: TyCtxt, mono_item: &MonoItem) -> bool {
    let internal = internal(tcx, mono_item);

    internal.is_instantiable(tcx)
}

// Find all "free" (not in a shader module) GPU root items.
fn collect_free_roots(cx: &Cx, mode: MonoItemCollectionStrategy) -> Vec<MonoItem> {
    let mut roots = MonoItems::new();
    let mut collector = RootCollector {
        cx,
        is_shader_module: false,
        strategy: mode,
        output: &mut roots,
    };

    let crate_items = cx.tcx().hir_crate_items(());

    for id in crate_items.free_items() {
        if !is_shader_module_item(cx, id.hir_id()) {
            collector.process_item(id);
        }
    }

    for id in crate_items.impl_items() {
        if !is_shader_module_item(cx, id.hir_id()) {
            collector.process_impl_item(id);
        }
    }

    // We can only stable_cg items that are instantiable - items all of
    // whose predicates hold. Luckily, items that aren't instantiable
    // can't actually be used, so we can just skip codegenning them.
    roots
        .into_iter()
        .filter_map(
            |Spanned {
                 node: mono_item, ..
             }| { is_instantiable(cx.tcx(), &mono_item).then_some(mono_item) },
        )
        .collect()
}

// Find all root items in the given shader module.
fn collect_shader_module_roots(
    cx: &Cx,
    shader_mod: LocalModDefId,
    mode: MonoItemCollectionStrategy,
) -> Vec<MonoItem> {
    let mut roots = MonoItems::new();
    let mut collector = RootCollector {
        cx,
        is_shader_module: true,
        strategy: mode,
        output: &mut roots,
    };

    let mod_items = cx.tcx().hir_module_items(shader_mod);

    for id in mod_items.free_items() {
        collector.process_item(id);
    }

    for id in mod_items.impl_items() {
        collector.process_impl_item(id);
    }

    // We can only stable_cg items that are instantiable - items all of
    // whose predicates hold. Luckily, items that aren't instantiable
    // can't actually be used, so we can just skip codegenning them.
    roots
        .into_iter()
        .filter_map(
            |Spanned {
                 node: mono_item, ..
             }| { is_instantiable(cx.tcx(), &mono_item).then_some(mono_item) },
        )
        .collect()
}

struct RootCollector<'a, 'tcx> {
    cx: &'a Cx<'tcx>,
    is_shader_module: bool,
    strategy: MonoItemCollectionStrategy,
    output: &'a mut MonoItems,
}

impl<'v> RootCollector<'_, 'v> {
    fn process_item(&mut self, id: hir::ItemId) {
        if !self.is_shader_module && self.cx.extended_item(id).is_none() {
            return;
        }

        match self.cx.tcx().def_kind(id.owner_id) {
            DefKind::Enum | DefKind::Struct | DefKind::Union => {
                if self.strategy == MonoItemCollectionStrategy::Eager
                    && self.cx.tcx().generics_of(id.owner_id).is_empty()
                {
                    debug!("RootCollector: ADT drop-glue for `{id:?}`",);

                    // This type is impossible to instantiate, so we should not try to
                    // generate a `drop_in_place` instance for it.
                    if self.cx.tcx().instantiate_and_check_impossible_predicates((
                        id.owner_id.to_def_id(),
                        ty::List::empty(),
                    )) {
                        return;
                    }

                    let ty = self
                        .cx
                        .tcx()
                        .type_of(id.owner_id.to_def_id())
                        .no_bound_vars()
                        .unwrap();
                    visit_drop_use(self.cx.tcx(), stable(ty), true, DUMMY_SP, self.output);
                }
            }
            DefKind::Static { .. } => {
                let def_id = id.owner_id.to_def_id();

                debug!(
                    "RootCollector: ItemKind::Static({})",
                    self.cx.tcx().def_path_str(def_id)
                );

                let mono_item = InternalMonoItem::Static(def_id);

                self.output.push(dummy_spanned(stable(mono_item)));
            }
            DefKind::Const => {
                // const items only generate mono items if they are
                // actually used somewhere. Just declaring them is insufficient.

                // but even just declaring them must collect the items they refer to
                if let Ok(val) = self.cx.tcx().const_eval_poly(id.owner_id.to_def_id()) {
                    collect_const_value(self.cx.tcx(), val, self.output);
                }
            }
            DefKind::Impl { .. } => {
                if self.strategy == MonoItemCollectionStrategy::Eager {
                    create_mono_items_for_default_impls(self.cx.tcx(), id, self.output);
                }
            }
            DefKind::Fn => {
                self.push_if_root(id.owner_id.def_id);
            }
            _ => {}
        }
    }

    fn process_impl_item(&mut self, id: hir::ImplItemId) {
        let hir_id = self.cx.tcx().hir_impl_item(id).hir_id();
        let parent_id = self.cx.tcx().hir_get_parent_item(hir_id);
        let parent = self.cx.tcx().hir_expect_item(parent_id.def_id);
        let parent_ext = self.cx.hir_ext().extend_item(parent);

        if !self.is_shader_module && parent_ext.is_none() {
            return;
        }

        if matches!(self.cx.tcx().def_kind(id.owner_id), DefKind::AssocFn) {
            self.push_if_root(id.owner_id.def_id);
        }
    }

    fn is_root(&self, def_id: LocalDefId) -> bool {
        !self
            .cx
            .tcx()
            .generics_of(def_id)
            .requires_monomorphization(self.cx.tcx())
            && match self.strategy {
                MonoItemCollectionStrategy::Eager => true,
                MonoItemCollectionStrategy::Lazy => {
                    self.cx.tcx().is_reachable_non_generic(def_id) || self.is_shader_module
                }
            }
    }

    /// If `def_id` represents a root, pushes it onto the list of
    /// outputs. (Note that all roots must be monomorphic.)
    fn push_if_root(&mut self, def_id: LocalDefId) {
        if self.is_root(def_id) {
            debug!("found root");

            let instance = InternalInstance::mono(self.cx.tcx(), def_id.to_def_id());

            let mono = self.output.push(create_fn_mono_item(
                self.cx.tcx(),
                stable(instance),
                DUMMY_SP,
            ));
        }
    }
}

fn create_mono_items_for_default_impls(tcx: TyCtxt, item: hir::ItemId, output: &mut MonoItems) {
    let impl_ = tcx.impl_trait_header(item.owner_id);

    if matches!(impl_.polarity, ty::ImplPolarity::Negative) {
        return;
    }

    if tcx
        .generics_of(item.owner_id)
        .own_requires_monomorphization()
    {
        return;
    }

    // Lifetimes never affect trait selection, so we are allowed to eagerly
    // instantiate an instance of an impl method if the impl (and method,
    // which we check below) is only parameterized over lifetime. In that case,
    // we use the ReErased, which has no lifetime information associated with
    // it, to validate whether or not the impl is legal to instantiate at all.
    let only_region_params = |param: &ty::GenericParamDef, _: &_| match param.kind {
        GenericParamDefKind::Lifetime => tcx.lifetimes.re_erased.into(),
        GenericParamDefKind::Type { .. } | GenericParamDefKind::Const { .. } => {
            unreachable!(
                "`own_requires_monomorphization` check means that \
                we should have no type/const params"
            )
        }
    };
    let impl_args = GenericArgs::for_item(tcx, item.owner_id.to_def_id(), only_region_params);
    let trait_ref = impl_.trait_ref.instantiate(tcx, impl_args);

    // Unlike 'lazy' monomorphization that begins by collecting items transitively
    // called by `main` or other global items, when eagerly monomorphizing impl
    // items, we never actually check that the predicates of this impl are satisfied
    // in a empty param env (i.e. with no assumptions).
    //
    // Even though this impl has no type or const generic parameters, because we don't
    // consider higher-ranked predicates such as `for<'a> &'a mut [u8]: Copy` to
    // be trivially false. We must now check that the impl has no impossible-to-satisfy
    // predicates.
    if tcx.instantiate_and_check_impossible_predicates((item.owner_id.to_def_id(), impl_args)) {
        return;
    }

    let typing_env = ty::TypingEnv::fully_monomorphized();
    let trait_ref = tcx.normalize_erasing_regions(typing_env, trait_ref);
    let overridden_methods = tcx.impl_item_implementor_ids(item.owner_id);
    for method in tcx.provided_trait_methods(trait_ref.def_id) {
        if overridden_methods.contains_key(&method.def_id) {
            continue;
        }

        if tcx
            .generics_of(method.def_id)
            .own_requires_monomorphization()
        {
            continue;
        }

        // As mentioned above, the method is legal to eagerly instantiate if it
        // only has lifetime generic parameters. This is validated by calling
        // `own_requires_monomorphization` on both the impl and method.
        let args = trait_ref
            .args
            .extend_to(tcx, method.def_id, only_region_params);
        let instance = ty::Instance::expect_resolve(tcx, typing_env, method.def_id, args, DUMMY_SP);
        let mono_item = InternalMonoItem::Fn(instance);

        if mono_item.is_instantiable(tcx) && tcx.should_codegen_locally(instance) {
            output.push(dummy_spanned(stable(mono_item)));
        }
    }
}

/// Finds all `shader_module` mod items in the current crate that are candidates for compilation.
///
/// These are all `shader_module` mod items with public visibility, and any non-public
/// `shader_module` mod items for which there exists a compilation request inside the current crate.
fn find_candidate_modules(cx: &Cx) -> IndexSet<LocalModDefId> {
    let mut modules = IndexSet::new();

    for (mod_id, ext) in &cx.hir_ext().mod_ext {
        if ext.is_shader_module {
            let vis = cx.tcx().local_visibility(mod_id.to_local_def_id());

            if vis.is_public() {
                modules.insert(*mod_id);
            }
        }
    }

    for shader_source_request in &cx.hir_ext().shader_requests {
        if let Some(local_id) = shader_source_request.shader_mod.as_local() {
            modules.insert(LocalModDefId::new_unchecked(local_id));
        }
    }

    modules
}

fn collect_mono_items(
    cx: &Cx,
    strategy: MonoItemCollectionStrategy,
) -> (
    Vec<MonoItem>,
    IndexMap<LocalModDefId, Vec<MonoItem>>,
    UsageMap,
) {
    debug!("collecting mono items");

    let tcx = cx.tcx();

    let free_roots = collect_free_roots(cx, strategy);

    let mut shader_module_root_items = IndexMap::new();
    let mut roots = free_roots.clone();

    for mod_ in find_candidate_modules(cx) {
        let mod_roots = collect_shader_module_roots(cx, mod_, strategy);

        roots.extend_from_slice(&mod_roots);

        shader_module_root_items.insert(mod_, mod_roots);
    }

    let state = SharedState {
        visited: RwLock::new(FxHashSet::default()),
        mentioned: RwLock::new(FxHashSet::default()),
        usage_map: RwLock::new(UsageMap::new()),
    };
    let recursion_limit = tcx.recursion_limit();

    tcx.sess.time("monomorphization_collector_graph_walk", || {
        // TODO: rustc::monomorphize runs a parallel iterator here. Since we modify the
        // implementation to use rustc_public wherever possible, and since rustc_public values are
        // not thread-safe, this is currently not an option here. Can we come up with a thread-safe
        // way of getting that parallelism back with rustc_public?

        for root in roots {
            collect_items_root(cx, dummy_spanned(root), &state, recursion_limit);
        }
    });

    (
        free_roots,
        shader_module_root_items,
        state.usage_map.into_inner().unwrap(),
    )
}

fn collect_used<'tcx>(
    user: &MonoItem,
    usage: &FxHashMap<MonoItem, Vec<MonoItem>>,
    output: &mut IndexSet<MonoItem>,
) {
    if let Some(used) = usage.get(user) {
        for item in used {
            if output.insert(item.clone()) {
                collect_used(item, usage, output);
            }
        }
    }
}

#[derive(Debug)]
pub struct ShaderModuleCodegenUnit {
    pub def_id: LocalModDefId,
    pub items: IndexSet<MonoItem>,
}

pub fn collect_shader_module_codegen_units(
    cx: &Cx,
) -> (IndexSet<MonoItem>, Vec<ShaderModuleCodegenUnit>) {
    let (free_roots, modules_roots, usage) =
        collect_mono_items(cx, MonoItemCollectionStrategy::Lazy);

    let mut free_items = IndexSet::new();

    for root in free_roots {
        free_items.insert(root.clone());
        collect_used(&root, &usage.used_map, &mut free_items);
    }

    let mut modules = Vec::new();

    for (def_id, roots) in modules_roots {
        let mut items = IndexSet::new();

        for root in roots {
            items.insert(root.clone());
            collect_used(&root, &usage.used_map, &mut items);
        }

        modules.push(ShaderModuleCodegenUnit { def_id, items });
    }

    (free_items, modules)
}

/// A replacement query provider for [TyCtxt::should_codegen_locally].
///
/// Modified version of `rustc_monomorphize::collector::should_codegen_locally`. Specifically, the
/// default rustc query provider will return `false` if the mono-item has already been codegenned in
/// an upstream crate; in this case rustc wants to link the upstream mono-item, rather than create a
/// duplicate.
///
/// However, this will also resolve non-GPU upstream mono-items, in which case there is no actual
/// SLIR-CFG representation available for the item. It would currently also not resolve to the
/// correct SLIR module (it would resolve to the module associated with the crate that originally
/// defines the item, not the module associated with the crate that provides the monomorphization).
/// We therefore opt to always codegen generic items locally and accept potential duplicate IR
/// items.
///
/// We may in the future want to implement our own version of upstream mono-item resolution to
/// prevent duplications in the IR. However, since we currently exhaustively inline the SLIR, these
/// duplicates do not affect the final output, and thus deduplication is not a high priority.
fn should_codegen_locally<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: rustc_middle::ty::Instance<'tcx>,
) -> bool {
    let Some(def_id) = instance.def.def_id_if_not_guaranteed_local_codegen() else {
        return true;
    };

    if tcx.is_foreign_item(def_id) {
        // Foreign items are always linked against, there's no way of instantiating them.
        return false;
    }

    if tcx.def_kind(def_id).has_codegen_attrs()
        && matches!(
            tcx.codegen_fn_attrs(def_id).inline,
            InlineAttr::Force { .. }
        )
    {
        // `#[rustc_force_inline]` items should never be codegened. This should be caught by
        // the MIR validator.
        tcx.dcx()
            .delayed_bug("attempt to codegen `#[rustc_force_inline]` item");
    }

    if def_id.is_local() {
        // Local items cannot be referred to locally without monomorphizing them locally.
        return true;
    }

    if tcx.is_reachable_non_generic(def_id) {
        // We can link to the item in question, no instance needed in this crate.
        return false;
    }

    if let DefKind::Static { .. } = tcx.def_kind(def_id) {
        // We cannot monomorphize statics from upstream crates.
        return false;
    }

    // See comment in should_encode_mir in rustc_metadata for why we don't report
    // an error for constructors.
    if !tcx.is_mir_available(def_id) && !matches!(tcx.def_kind(def_id), DefKind::Ctor(..)) {
        tcx.dcx().emit_fatal(NoOptimizedMir {
            span: tcx.def_span(def_id),
            crate_name: tcx.crate_name(def_id.krate),
            instance: instance.to_string(),
        });
    }

    true
}

pub fn provide(providers: &mut Providers) {
    providers.hooks.should_codegen_locally = should_codegen_locally;
}
