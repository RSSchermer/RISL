use std::cell::OnceCell;
use std::path::PathBuf;

use indexmap::{IndexMap, IndexSet};
use rustc_data_structures::fx::FxIndexMap;
use rustc_data_structures::sync::{MTLock, par_for_each_in};
use rustc_data_structures::unord::{UnordMap, UnordSet};
use rustc_hir as hir;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::{DefId, DefIdMap, LocalDefId};
use rustc_hir::lang_items::LangItem;
use rustc_hir::limit::Limit;
use rustc_middle::middle::codegen_fn_attrs::CodegenFnAttrFlags;
use rustc_middle::mir::interpret::{AllocId, ErrorHandled, GlobalAlloc, Scalar};
use rustc_middle::mir::mono::{CollectionMode, InstantiationMode, MonoItem};
use rustc_middle::mir::visit::Visitor as MirVisitor;
use rustc_middle::mir::{self, Location, MentionedItem, traversal};
use rustc_middle::query::TyCtxtAt;
use rustc_middle::ty::adjustment::{CustomCoerceUnsized, PointerCoercion};
use rustc_middle::ty::layout::ValidityRequirement;
use rustc_middle::ty::print::with_no_trimmed_paths;
use rustc_middle::ty::{
    self, GenericArgs, GenericParamDefKind, Instance, InstanceKind, Ty, TyCtxt, TypeFoldable,
    TypeVisitableExt, VtblEntry,
};
use rustc_middle::util::Providers;
use rustc_middle::{bug, span_bug, traits};
use rustc_session::config::EntryFnType;
use rustc_span::def_id::LocalModDefId;
use rustc_span::source_map::{Spanned, dummy_spanned, respan};
use rustc_span::symbol::sym;
use rustc_span::{DUMMY_SP, ErrorGuaranteed, Span, Symbol};
use tracing::{debug, instrument, trace};

use crate::context::RislContext as Cx;
use crate::hir_ext::ExtendedItemKind;
use crate::monomorphize::errors::{
    self, EncounteredErrorWhileInstantiating, NoOptimizedMir, RecursionLimit,
};

#[derive(Clone, Copy, PartialEq)]
pub(crate) enum MonoItemCollectionStrategy {
    Eager,
    Lazy,
}

/// The state that is shared across the concurrent threads that are doing collection.
struct SharedState<'tcx> {
    /// Items that have been or are currently being recursively collected.
    visited: MTLock<UnordSet<MonoItem<'tcx>>>,
    /// Items that have been or are currently being recursively treated as "mentioned", i.e., their
    /// consts are evaluated but nothing is added to the collection.
    mentioned: MTLock<UnordSet<MonoItem<'tcx>>>,
    /// Which items are being used where, for better errors.
    usage_map: MTLock<UsageMap<'tcx>>,
}

pub(crate) struct UsageMap<'tcx> {
    // Maps every mono item to the mono items used by it.
    used_map: UnordMap<MonoItem<'tcx>, Vec<MonoItem<'tcx>>>,

    // Maps every mono item to the mono items that use it.
    user_map: UnordMap<MonoItem<'tcx>, Vec<MonoItem<'tcx>>>,
}

impl<'tcx> UsageMap<'tcx> {
    fn new() -> UsageMap<'tcx> {
        UsageMap {
            used_map: Default::default(),
            user_map: Default::default(),
        }
    }

    fn record_used<'a>(&mut self, user_item: MonoItem<'tcx>, used_items: &'a MonoItems<'tcx>)
    where
        'tcx: 'a,
    {
        for used_item in used_items.items() {
            self.user_map.entry(used_item).or_default().push(user_item);
        }

        assert!(
            self.used_map
                .insert(user_item, used_items.items().collect())
                .is_none()
        );
    }
}

struct MonoItems<'tcx> {
    // We want a set of MonoItem + Span where trying to re-insert a MonoItem with a different Span
    // is ignored. Map does that, but it looks odd.
    items: FxIndexMap<MonoItem<'tcx>, Span>,
}

impl<'tcx> MonoItems<'tcx> {
    fn new() -> Self {
        Self {
            items: FxIndexMap::default(),
        }
    }

    fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    fn push(&mut self, item: Spanned<MonoItem<'tcx>>) {
        // Insert only if the entry does not exist. A normal insert would stomp the first span that
        // got inserted.
        self.items.entry(item.node).or_insert(item.span);
    }

    fn items(&self) -> impl Iterator<Item = MonoItem<'tcx>> + '_ {
        self.items.keys().cloned()
    }
}

impl<'tcx> IntoIterator for MonoItems<'tcx> {
    type Item = Spanned<MonoItem<'tcx>>;
    type IntoIter = impl Iterator<Item = Spanned<MonoItem<'tcx>>>;

    fn into_iter(self) -> Self::IntoIter {
        self.items
            .into_iter()
            .map(|(item, span)| respan(span, item))
    }
}

impl<'tcx> Extend<Spanned<MonoItem<'tcx>>> for MonoItems<'tcx> {
    fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = Spanned<MonoItem<'tcx>>>,
    {
        for item in iter {
            self.push(item)
        }
    }
}

fn collect_items_root<'tcx>(
    tcx: TyCtxt<'tcx>,
    starting_item: Spanned<MonoItem<'tcx>>,
    state: &SharedState<'tcx>,
    recursion_limit: Limit,
) {
    if !state.visited.lock_mut().insert(starting_item.node) {
        // We've been here already, no need to search again.
        return;
    }
    let mut recursion_depths = DefIdMap::default();
    collect_items_rec(
        tcx,
        starting_item,
        state,
        &mut recursion_depths,
        recursion_limit,
        CollectionMode::UsedItems,
    );
}

/// Collect all monomorphized items reachable from `starting_point`, and emit a note diagnostic if a
/// post-monomorphization error is encountered during a collection step.
///
/// `mode` determined whether we are scanning for [used items][CollectionMode::UsedItems]
/// or [mentioned items][CollectionMode::MentionedItems].
#[instrument(skip(tcx, state, recursion_depths, recursion_limit), level = "debug")]
fn collect_items_rec<'tcx>(
    tcx: TyCtxt<'tcx>,
    starting_item: Spanned<MonoItem<'tcx>>,
    state: &SharedState<'tcx>,
    recursion_depths: &mut DefIdMap<usize>,
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
    let error_count = tcx.dcx().err_count();

    // In `mentioned_items` we collect items that were mentioned in this MIR but possibly do not
    // need to be monomorphized. This is done to ensure that optimizing away function calls does not
    // hide const-eval errors that those calls would otherwise have triggered.
    match starting_item.node {
        MonoItem::Static(def_id) => {
            recursion_depth_reset = None;

            // Statics always get evaluated (which is possible because they can't be generic), so for
            // `MentionedItems` collection there's nothing to do here.
            if mode == CollectionMode::UsedItems {
                let instance = Instance::mono(tcx, def_id);

                // Sanity check whether this ended up being collected accidentally
                debug_assert!(tcx.should_codegen_locally(instance));

                let DefKind::Static { nested, .. } = tcx.def_kind(def_id) else {
                    bug!()
                };
                // Nested statics have no type.
                if !nested {
                    let ty = instance.ty(tcx, ty::TypingEnv::fully_monomorphized());
                    visit_drop_use(tcx, ty, true, starting_item.span, &mut used_items);
                }

                if let Ok(alloc) = tcx.eval_static_initializer(def_id) {
                    for &prov in alloc.inner().provenance().ptrs().values() {
                        collect_alloc(tcx, prov.alloc_id(), &mut used_items);
                    }
                }

                if tcx.needs_thread_local_shim(def_id) {
                    used_items.push(respan(
                        starting_item.span,
                        MonoItem::Fn(Instance {
                            def: InstanceKind::ThreadLocalShim(def_id),
                            args: GenericArgs::empty(),
                        }),
                    ));
                }
            }

            // mentioned_items stays empty since there's no codegen for statics. statics don't get
            // optimized, and if they did then the const-eval interpreter would have to worry about
            // mentioned_items.
        }
        MonoItem::Fn(instance) => {
            // Sanity check whether this ended up being collected accidentally
            debug_assert!(tcx.should_codegen_locally(instance));

            // Keep track of the monomorphization recursion depth
            recursion_depth_reset = Some(check_recursion_limit(
                tcx,
                instance,
                starting_item.span,
                recursion_depths,
                recursion_limit,
            ));

            rustc_data_structures::stack::ensure_sufficient_stack(|| {
                let Ok((used, mentioned)) = tcx.items_of_instance((instance, mode)) else {
                    // Normalization errors here are usually due to trait solving overflow.
                    // FIXME: I assume that there are few type errors at post-analysis stage, but not
                    // entirely sure.
                    // We have to emit the error outside of `items_of_instance` to access the
                    // span of the `starting_item`.
                    let def_id = instance.def_id();
                    let def_span = tcx.def_span(def_id);
                    let def_path_str = tcx.def_path_str(def_id);
                    tcx.dcx().emit_fatal(RecursionLimit {
                        span: starting_item.span,
                        instance,
                        def_span,
                        def_path_str,
                    });
                };
                used_items.extend(used.into_iter().copied());
                mentioned_items.extend(mentioned.into_iter().copied());
            });
        }
        MonoItem::GlobalAsm(item_id) => {
            let item = tcx.hir_item(item_id);

            span_bug!(
                item.span,
                "RISL does not support inline ASM; should have been caught during the analysis \
                phase"
            );
        }
    };

    // Check for PMEs and emit a diagnostic if one happened. To try to show relevant edges of the
    // mono item graph.
    if tcx.dcx().err_count() > error_count
        && starting_item.node.is_generic_fn()
        && starting_item.node.is_user_defined()
    {
        match starting_item.node {
            MonoItem::Fn(instance) => tcx.dcx().emit_note(EncounteredErrorWhileInstantiating {
                span: starting_item.span,
                kind: "fn",
                instance,
            }),
            MonoItem::Static(def_id) => tcx.dcx().emit_note(EncounteredErrorWhileInstantiating {
                span: starting_item.span,
                kind: "static",
                instance: Instance::new_raw(def_id, GenericArgs::empty()),
            }),
            MonoItem::GlobalAsm(_) => {
                span_bug!(
                    starting_item.span,
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
            .lock_mut()
            .record_used(starting_item.node, &used_items);
    }

    {
        let mut visited = OnceCell::default();
        if mode == CollectionMode::UsedItems {
            used_items.items.retain(|k, _| {
                visited
                    .get_mut_or_init(|| state.visited.lock_mut())
                    .insert(*k)
            });
        }

        let mut mentioned = OnceCell::default();
        mentioned_items.items.retain(|k, _| {
            !visited.get_or_init(|| state.visited.lock()).contains(k)
                && mentioned
                    .get_mut_or_init(|| state.mentioned.lock_mut())
                    .insert(*k)
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
                tcx,
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
            tcx,
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

fn check_recursion_limit<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
    span: Span,
    recursion_depths: &mut DefIdMap<usize>,
    recursion_limit: Limit,
) -> (DefId, usize) {
    let def_id = instance.def_id();
    let recursion_depth = recursion_depths.get(&def_id).cloned().unwrap_or(0);
    debug!(" => recursion depth={}", recursion_depth);

    let adjusted_recursion_depth = if tcx.is_lang_item(def_id, LangItem::DropInPlace) {
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
        let def_span = tcx.def_span(def_id);
        let def_path_str = tcx.def_path_str(def_id);
        tcx.dcx().emit_fatal(RecursionLimit {
            span,
            instance,
            def_span,
            def_path_str,
        });
    }

    recursion_depths.insert(def_id, recursion_depth + 1);

    (def_id, recursion_depth)
}

fn visit_drop_use<'tcx>(
    tcx: TyCtxt<'tcx>,
    ty: Ty<'tcx>,
    is_direct_call: bool,
    source: Span,
    output: &mut MonoItems<'tcx>,
) {
    let instance = Instance::resolve_drop_in_place(tcx, ty);
    visit_instance_use(tcx, instance, is_direct_call, source, output);
}

fn visit_instance_use<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: ty::Instance<'tcx>,
    is_direct_call: bool,
    source: Span,
    output: &mut MonoItems<'tcx>,
) {
    debug!(
        "visit_item_use({:?}, is_direct_call={:?})",
        instance, is_direct_call
    );
    if !tcx.should_codegen_locally(instance) {
        return;
    }
    if let Some(intrinsic) = tcx.intrinsic(instance.def_id()) {
        if let Some(_requirement) = ValidityRequirement::from_intrinsic(intrinsic.name) {
            // The intrinsics assert_inhabited, assert_zero_valid, and assert_mem_uninitialized_valid will
            // be lowered in codegen to nothing or a call to panic_nounwind. So if we encounter any
            // of those intrinsics, we need to include a mono item for panic_nounwind, else we may try to
            // codegen a call to that function without generating code for the function itself.
            let def_id = tcx.require_lang_item(LangItem::PanicNounwind, source);
            let panic_instance = Instance::mono(tcx, def_id);
            if tcx.should_codegen_locally(panic_instance) {
                output.push(create_fn_mono_item(tcx, panic_instance, source));
            }
        } else if !intrinsic.must_be_overridden {
            // Codegen the fallback body of intrinsics with fallback bodies.
            // We explicitly skip this otherwise to ensure we get a linker error
            // if anyone tries to call this intrinsic and the codegen backend did not
            // override the implementation.
            let instance = ty::Instance::new_raw(instance.def_id(), instance.args);
            if tcx.should_codegen_locally(instance) {
                output.push(create_fn_mono_item(tcx, instance, source));
            }
        }
    }

    match instance.def {
        ty::InstanceKind::Virtual(..) | ty::InstanceKind::Intrinsic(_) => {
            if !is_direct_call {
                bug!("{:?} being reified", instance);
            }
        }
        ty::InstanceKind::ThreadLocalShim(..) => {
            bug!("{:?} being reified", instance);
        }
        ty::InstanceKind::DropGlue(_, None) => {
            // Don't need to emit noop drop glue if we are calling directly.
            //
            // Note that we also optimize away the call to visit_instance_use in vtable construction
            // (see create_mono_items_for_vtable_methods).
            if !is_direct_call {
                output.push(create_fn_mono_item(tcx, instance, source));
            }
        }
        ty::InstanceKind::DropGlue(_, Some(_))
        | ty::InstanceKind::FutureDropPollShim(..)
        | ty::InstanceKind::AsyncDropGlue(_, _)
        | ty::InstanceKind::AsyncDropGlueCtorShim(_, _)
        | ty::InstanceKind::VTableShim(..)
        | ty::InstanceKind::ReifyShim(..)
        | ty::InstanceKind::ClosureOnceShim { .. }
        | ty::InstanceKind::ConstructCoroutineInClosureShim { .. }
        | ty::InstanceKind::Item(..)
        | ty::InstanceKind::FnPtrShim(..)
        | ty::InstanceKind::CloneShim(..)
        | ty::InstanceKind::FnPtrAddrShim(..) => {
            output.push(create_fn_mono_item(tcx, instance, source));
        }
    }
}

fn create_fn_mono_item<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
    source: Span,
) -> Spanned<MonoItem<'tcx>> {
    respan(source, MonoItem::Fn(instance))
}

/// Scans the CTFE alloc in order to find function pointers and statics that must be monomorphized.
fn collect_alloc<'tcx>(tcx: TyCtxt<'tcx>, alloc_id: AllocId, output: &mut MonoItems<'tcx>) {
    match tcx.global_alloc(alloc_id) {
        GlobalAlloc::Static(def_id) => {
            assert!(!tcx.is_thread_local_static(def_id));
            let instance = Instance::mono(tcx, def_id);
            if tcx.should_codegen_locally(instance) {
                trace!("collecting static {:?}", def_id);
                output.push(dummy_spanned(MonoItem::Static(def_id)));
            }
        }
        GlobalAlloc::Memory(alloc) => {
            trace!("collecting {:?} with {:#?}", alloc_id, alloc);
            let ptrs = alloc.inner().provenance().ptrs();
            // avoid `ensure_sufficient_stack` in the common case of "no pointers"
            if !ptrs.is_empty() {
                rustc_data_structures::stack::ensure_sufficient_stack(move || {
                    for &prov in ptrs.values() {
                        collect_alloc(tcx, prov.alloc_id(), output);
                    }
                });
            }
        }
        GlobalAlloc::Function { instance, .. } => {
            if tcx.should_codegen_locally(instance) {
                trace!("collecting {:?} with {:#?}", alloc_id, instance);
                output.push(create_fn_mono_item(tcx, instance, DUMMY_SP));
            }
        }
        GlobalAlloc::VTable(ty, dyn_ty) => {
            let alloc_id = tcx.vtable_allocation((
                ty,
                dyn_ty
                    .principal()
                    .map(|principal| tcx.instantiate_bound_regions_with_erased(principal)),
            ));

            collect_alloc(tcx, alloc_id, output)
        }
        GlobalAlloc::TypeId { .. } => {}
    }
}

#[instrument(skip(tcx, output), level = "debug")]
fn collect_const_value<'tcx>(
    tcx: TyCtxt<'tcx>,
    value: mir::ConstValue,
    output: &mut MonoItems<'tcx>,
) {
    match value {
        mir::ConstValue::Scalar(Scalar::Ptr(ptr, _size)) => {
            collect_alloc(tcx, ptr.provenance.alloc_id(), output)
        }
        mir::ConstValue::Indirect { alloc_id, .. } => collect_alloc(tcx, alloc_id, output),
        mir::ConstValue::Slice { alloc_id, .. } => {
            collect_alloc(tcx, alloc_id, output);
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

// Find all "free" (not in a shader module) GPU root items.
fn collect_free_roots<'tcx>(
    cx: &Cx<'tcx>,
    mode: MonoItemCollectionStrategy,
) -> Vec<MonoItem<'tcx>> {
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
             }| { mono_item.is_instantiable(cx.tcx()).then_some(mono_item) },
        )
        .collect()
}

// Find all root items in the given shader module.
fn collect_shader_module_roots<'tcx>(
    cx: &Cx<'tcx>,
    shader_mod: LocalModDefId,
    mode: MonoItemCollectionStrategy,
) -> Vec<MonoItem<'tcx>> {
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
             }| { mono_item.is_instantiable(cx.tcx()).then_some(mono_item) },
        )
        .collect()
}

struct RootCollector<'a, 'tcx> {
    cx: &'a Cx<'tcx>,
    is_shader_module: bool,
    strategy: MonoItemCollectionStrategy,
    output: &'a mut MonoItems<'tcx>,
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
                    visit_drop_use(self.cx.tcx(), ty, true, DUMMY_SP, self.output);
                }
            }
            DefKind::Static { .. } => {
                let def_id = id.owner_id.to_def_id();
                debug!(
                    "RootCollector: ItemKind::Static({})",
                    self.cx.tcx().def_path_str(def_id)
                );
                self.output.push(dummy_spanned(MonoItem::Static(def_id)));
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

            let instance = Instance::mono(self.cx.tcx(), def_id.to_def_id());
            self.output
                .push(create_fn_mono_item(self.cx.tcx(), instance, DUMMY_SP));
        }
    }
}

fn create_mono_items_for_default_impls<'tcx>(
    tcx: TyCtxt<'tcx>,
    item: hir::ItemId,
    output: &mut MonoItems<'tcx>,
) {
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

        let mono_item = create_fn_mono_item(tcx, instance, DUMMY_SP);
        if mono_item.node.is_instantiable(tcx) && tcx.should_codegen_locally(instance) {
            output.push(mono_item);
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

fn collect_mono_items<'tcx>(
    cx: &Cx<'tcx>,
    strategy: MonoItemCollectionStrategy,
) -> (
    Vec<MonoItem<'tcx>>,
    IndexMap<LocalModDefId, Vec<MonoItem<'tcx>>>,
    UsageMap<'tcx>,
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
        visited: MTLock::new(UnordSet::default()),
        mentioned: MTLock::new(UnordSet::default()),
        usage_map: MTLock::new(UsageMap::new()),
    };
    let recursion_limit = tcx.recursion_limit();

    tcx.sess.time("monomorphization_collector_graph_walk", || {
        par_for_each_in(roots, |root| {
            collect_items_root(tcx, dummy_spanned(*root), &state, recursion_limit);
        });
    });

    (
        free_roots,
        shader_module_root_items,
        state.usage_map.into_inner(),
    )
}

fn collect_used<'tcx>(
    user: &MonoItem<'tcx>,
    usage: &UnordMap<MonoItem<'tcx>, Vec<MonoItem<'tcx>>>,
    output: &mut IndexSet<MonoItem<'tcx>>,
) {
    if let Some(used) = usage.get(user) {
        for item in used {
            if output.insert(*item) {
                collect_used(item, usage, output);
            }
        }
    }
}

pub struct ShaderModuleCodegenUnit<'tcx> {
    pub def_id: LocalModDefId,
    pub items: IndexSet<MonoItem<'tcx>>,
}

pub fn collect_shader_module_codegen_units<'tcx>(
    cx: &Cx<'tcx>,
) -> (IndexSet<MonoItem<'tcx>>, Vec<ShaderModuleCodegenUnit<'tcx>>) {
    let (free_roots, modules_roots, usage) =
        collect_mono_items(cx, MonoItemCollectionStrategy::Lazy);

    let mut free_items = IndexSet::new();

    for root in free_roots {
        free_items.insert(root);
        collect_used(&root, &usage.used_map, &mut free_items);
    }

    let mut modules = Vec::new();

    for (def_id, roots) in modules_roots {
        let mut items = IndexSet::new();

        for root in roots {
            items.insert(root);
            collect_used(&root, &usage.used_map, &mut items);
        }

        modules.push(ShaderModuleCodegenUnit { def_id, items });
    }

    (free_items, modules)
}
