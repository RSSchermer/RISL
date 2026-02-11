use indexmap::IndexSet;
use rustc_hash::{FxHashMap, FxHashSet};
use slotmap::KeyData;

use crate::Module;
use crate::cfg::OpAlloca;
use crate::rvsdg::NodeKind::{Loop, Switch};
use crate::rvsdg::transform::variable_pointer_emulation::EmulationContext;
use crate::rvsdg::visit::reverse_value_flow::ReverseValueFlowVisitor;
use crate::rvsdg::visit::value_flow::ValueFlowVisitor;
use crate::rvsdg::{
    Connectivity, Node, NodeKind, Region, Rvsdg, SimpleNode, StateOrigin, StateUser, ValueInput,
    ValueOrigin, ValueUser, visit,
};
use crate::ty::{TypeKind, TypeRegistry};

#[derive(Debug)]
struct AggregateAnalyzer {
    cache: FxHashMap<Node, bool>,
    visited: FxHashSet<(Region, ValueUser)>,
    is_aggregate: bool,
}

impl AggregateAnalyzer {
    fn new() -> Self {
        AggregateAnalyzer {
            cache: Default::default(),
            visited: Default::default(),
            is_aggregate: false,
        }
    }

    fn is_aggregate(&mut self, rvsdg: &Rvsdg, alloca_node: Node) -> bool {
        let alloca_ty = rvsdg[alloca_node].expect_op_alloca().ty();

        match &*rvsdg.ty().kind(alloca_ty) {
            TypeKind::Struct(_) | TypeKind::Array { .. } | TypeKind::Enum(_) => true,
            TypeKind::Scalar(_) | TypeKind::Ptr(_) | TypeKind::Predicate => false,
            TypeKind::Vector(_) | TypeKind::Matrix(_) => {
                // We treat vectors and matrices as promotable scalars if they are only stored to or
                // loaded from "in whole", that is, not via element pointers to individual elements.
                // Otherwise, we treat them as aggregates. We'll run a value-flow analysis to check
                // if the alloca pointer is only every used as the pointer input to a load or store,
                // or passed through switch or loop nodes. Since we may end up analyzing the same
                // alloca node multiple times, and this is a relatively expensive operation, we
                // cache the result. Note that the memory-promotion pass is iterative and each
                // subsequent iteration will invalidate the cache, so the cache must be cleared at
                // the start of each iteration.

                if let Some(cached) = self.cache.get(&alloca_node).copied() {
                    return cached;
                }

                self.visited.clear();
                self.is_aggregate = false;

                self.visit_value_output(rvsdg, alloca_node, 0);
                self.cache.insert(alloca_node, self.is_aggregate);

                self.is_aggregate
            }
            TypeKind::Atomic(_)
            | TypeKind::Slice { .. }
            | TypeKind::Function(_)
            | TypeKind::Dummy => unreachable!("type cannot be stored in an alloca"),
        }
    }

    fn clear_cache(&mut self) {
        self.cache.clear();
    }
}

impl ValueFlowVisitor for AggregateAnalyzer {
    fn should_visit(&mut self, region: Region, user: ValueUser) -> bool {
        self.visited.insert((region, user))
    }

    fn visit_value_input(&mut self, rvsdg: &Rvsdg, node: Node, input: u32) {
        use NodeKind::*;
        use SimpleNode::*;

        match rvsdg[node].kind() {
            Simple(OpLoad(_)) => {
                // Don't continue visiting the load node's output: we're only interested in the uses
                // of the alloca pointer, not in how its value is used.
            }
            Simple(OpStore(_)) if input == 0 => {
                // Store nodes don't output values.
            }
            Switch(_) | Loop(_) => {
                visit::value_flow::visit_value_input(self, rvsdg, node, input);
            }
            _ => {
                self.is_aggregate = true;
            }
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum PointerAction {
    /// A memory operation that uses the pointer can be promoted to value-flow.
    ///
    /// The pointer has a determinate access chain that resolves back to an originating [OpAlloca].
    /// Contains the [Node] that represents the originating [OpAlloca].
    Promotion(Node),

    /// Memory operations that use this pointer need variable-pointer-emulation to be legal on a
    /// target that does not support variable pointers.
    VariablePointerEmulation,

    /// Memory operations that use the pointer should be left as they are.
    Nothing,
}

struct PointerAnalyzer {
    cache: FxHashMap<(Region, ValueOrigin), PointerAction>,
    visitor: PointerOriginVisitor,
}

impl PointerAnalyzer {
    fn new() -> Self {
        PointerAnalyzer {
            cache: Default::default(),
            visitor: PointerOriginVisitor::new(),
        }
    }

    fn resolve_action(
        &mut self,
        rvsdg: &Rvsdg,
        region: Region,
        pointer_origin: ValueOrigin,
    ) -> PointerAction {
        *self
            .cache
            .entry((region, pointer_origin))
            .or_insert_with(|| {
                self.visitor
                    .analyze_pointer_origin(rvsdg, region, pointer_origin)
            })
    }

    fn clear_cache(&mut self) {
        self.cache.clear();
        self.visitor.aggregate_analyzer.clear_cache();
    }
}

#[derive(Debug)]
pub struct PointerOriginVisitor {
    visited: FxHashSet<(Region, ValueOrigin)>,
    aggregate_analyzer: AggregateAnalyzer,
    promotion_candidate: Option<Node>,
    can_promote: bool,
    needs_emulation: bool,
    can_emulate: bool,
}

impl PointerOriginVisitor {
    fn new() -> Self {
        PointerOriginVisitor {
            visited: Default::default(),
            aggregate_analyzer: AggregateAnalyzer::new(),
            promotion_candidate: None,
            can_promote: true,
            needs_emulation: false,
            can_emulate: true,
        }
    }

    fn analyze_pointer_origin(
        &mut self,
        rvsdg: &Rvsdg,
        region: Region,
        pointer_origin: ValueOrigin,
    ) -> PointerAction {
        self.visited.clear();
        self.promotion_candidate = None;
        self.can_promote = true;
        self.needs_emulation = false;
        self.can_emulate = true;

        self.visit_value_origin(rvsdg, region, pointer_origin);

        if self.needs_emulation && self.can_emulate {
            PointerAction::VariablePointerEmulation
        } else if let Some(candidate) = self.promotion_candidate
            && self.can_promote
        {
            PointerAction::Promotion(candidate)
        } else {
            PointerAction::Nothing
        }
    }
}

impl ReverseValueFlowVisitor for PointerOriginVisitor {
    fn should_visit(&mut self, region: Region, origin: ValueOrigin) -> bool {
        self.visited.insert((region, origin))
    }

    fn visit_value_output(&mut self, rvsdg: &Rvsdg, node: Node, output: u32) {
        use NodeKind::*;
        use SimpleNode::*;

        match rvsdg[node].kind() {
            Switch(_) | Loop(_) => {
                self.needs_emulation = true;
                self.can_promote = false;

                visit::reverse_value_flow::visit_value_output(self, rvsdg, node, output);
            }
            Simple(OpAlloca(_)) => {
                self.promotion_candidate = Some(node);
                self.can_promote &= !self.aggregate_analyzer.is_aggregate(rvsdg, node);
            }
            Simple(OpFieldPtr(_)) | Simple(OpElementPtr(_)) => {
                self.can_promote = false;

                visit::reverse_value_flow::visit_value_input(self, rvsdg, node, 0);
            }
            Simple(OpLoad(_) | OpVariantPtr(_) | OpDiscriminantPtr(_)) => {
                self.can_promote = false;
                self.can_emulate = false;
            }
            Simple(OpOffsetSlice(_) | ValueProxy(_)) => {
                visit::reverse_value_flow::visit_value_input(self, rvsdg, node, 0);
            }
            Simple(ConstPtr(_) | ConstFallback(_)) => (),
            _ => unreachable!("node kind cannot output a pointer"),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum PointerValue {
    Stored(ValueOrigin),
    Fallback,
}

struct TouchedOuterAllocaStack {
    stack: Vec<IndexSet<Node>>,
}

impl TouchedOuterAllocaStack {
    fn new() -> Self {
        TouchedOuterAllocaStack { stack: vec![] }
    }

    fn touch(&mut self, op_alloca: Node) {
        if let Some(touched_allocas) = self.stack.last_mut() {
            touched_allocas.insert(op_alloca);
        }
    }

    fn push(&mut self) {
        self.stack.push(IndexSet::new());
    }

    fn pop(&mut self) -> Option<IndexSet<Node>> {
        let touched_allocas = self.stack.pop();

        // When an alloca is touched in a region, it should also be considered touched in all of
        // that region's ancestor regions. Instead of moving up the entire stack every time
        // TouchedAllocaStack::touch is called, we instead add them all at once here when popping.
        if let Some(touched_allocas) = &touched_allocas
            && let Some(parent) = self.stack.last_mut()
        {
            for alloca in touched_allocas.iter().copied() {
                parent.insert(alloca);
            }
        }

        touched_allocas
    }

    fn clear(&mut self) {
        self.stack.clear();
    }
}

/// Promotes and legalizes all pointer-mediated memory operations (loads and stores) in a function
/// body region.
///
/// Memory promotion pertains to replacing a load of memory that originates from an [OpAlloca], with
/// the latest (as per the state chain) value stored into that memory. This kind of transform is
/// sometimes also referred to as a "memory to register" transform or "mem2reg". Our RVSDG
/// abstraction is not really explicitly concerned with the concept of registers, so here we'll
/// refer to this slightly more abstractly as a "memory to value-flow" transform. We only promote
/// a memory operation if its pointer resolves to a complete alloca, we don't promote operations on
/// part pointers (as produced by [OpFieldPtr] or an [OpElementPtr] node); such operations must
/// first be split by a scalar-replacement pass. We also do not promote memory operations on
/// "variable pointers".
///
/// For our purposes, variable pointers are defined as pointers that result from the output of
/// switch or loop nodes, or further refinements of such pointers (e.g. a pointer resulting from an
/// [OpElementPtr] for which the input pointer was a variable pointer). Not only will variable
/// pointers prevent memory promotion, our primary compilation target WGSL does not support variable
/// pointers at all. Therefore, we must be able to emulate *all* variable pointers if we are to
/// compile a legal program.
///
/// We can emulate a variable pointer if we can reconstruct the entire access chain and branching
/// history, all the way back to the root-identifier pointer(s) (the output of an [OpAlloca] node
/// or a [ConstPtr] node), see `variable_pointer_emulation.rs` for details. However, we cannot
/// reliably do this if the pointer was the output of an [OpLoad] node. Fortunately, we don't allow
/// uniform, storage or workgroup values to be pointer-typed or contain pointer-typed fields at any
/// level. That means [OpLoad] operations that produce pointer-typed outputs can only operate on
/// input pointers that originate from [OpAlloca] nodes. Consequently, if we can first promote such
/// [OpLoad] nodes to value-flow, then we can always emulate the variable pointer.
///
/// As noted earlier, promoting memory to value-flow can only be done for memory operations where
/// the input pointer is not a variable pointer. This creates a dependency graph between promotion
/// and emulation operations. If this graph were to contain a cycle, then we would not be able to
/// successfully construct an order of operations that can always emulate all variable pointers.
/// Fortunately, this graph is guaranteed to be acyclic, as all memory operations must be part of
/// the state chain, and the state chain must be acyclic. As such, if we process all memory
/// operations in state chain order, then it is guaranteed that we will have been able to emulate
/// all variable pointers into legal WGSL.
///
/// Because of this inter-dependence between memory to value-flow promotion and variable pointer
/// emulation, these transforms are combined into a single pass.
pub struct MemoryPromoterLegalizer {
    emulation_context: EmulationContext,
    pointer_analyzer: PointerAnalyzer,
    state_origin: (Region, StateOrigin),
    value_availability: FxHashMap<(Node, Region), ValueOrigin>,
    owner_stack: Vec<Node>,
    touched_outer_alloca_stack: TouchedOuterAllocaStack,
}

impl MemoryPromoterLegalizer {
    pub fn new() -> Self {
        MemoryPromoterLegalizer {
            emulation_context: EmulationContext::new(),
            pointer_analyzer: PointerAnalyzer::new(),
            state_origin: (Region::from(KeyData::from_ffi(0)), StateOrigin::Argument),
            value_availability: Default::default(),
            owner_stack: vec![],
            touched_outer_alloca_stack: TouchedOuterAllocaStack::new(),
        }
    }

    pub fn promote_and_legalize(&mut self, rvsdg: &mut Rvsdg, region: Region) {
        // Set the initial state origin to the region's state argument
        self.state_origin = (region, StateOrigin::Argument);

        // Clear any state set by a previous promote-and-legalize run
        self.emulation_context.clear();
        self.pointer_analyzer.clear_cache();
        self.value_availability.clear();
        self.owner_stack.clear();
        self.touched_outer_alloca_stack.clear();

        while self.visit_state_user(rvsdg) {}
    }

    fn visit_state_user(&mut self, rvsdg: &mut Rvsdg) -> bool {
        let (current_region, current_origin) = self.state_origin;

        let current_user = match current_origin {
            StateOrigin::Argument => *rvsdg[current_region].state_argument(),
            StateOrigin::Node(node) => {
                rvsdg[node]
                    .state()
                    .expect("state origin should be part of state chain")
                    .user
            }
        };

        match current_user {
            StateUser::Result => {
                // We've reached the end of the region's state chain: return `false` to indicate
                // that we're done visiting state users.
                false
            }
            StateUser::Node(node) => {
                use NodeKind::*;
                use SimpleNode::*;

                match rvsdg[node].kind() {
                    Switch(_) => self.visit_switch(rvsdg, node),
                    Loop(_) => self.visit_loop(rvsdg, node),
                    Simple(OpLoad(_)) => self.visit_op_load(rvsdg, node),
                    Simple(OpStore(_)) => self.visit_op_store(rvsdg, node),
                    _ => unreachable!("node kind cannot be part of the state chain"),
                }

                true
            }
        }
    }

    fn visit_switch(&mut self, rvsdg: &mut Rvsdg, switch_node: Node) {
        let region = rvsdg[switch_node].region();
        let switch_data = rvsdg[switch_node].expect_switch();
        let branch_count = switch_data.branches().len();

        self.touched_outer_alloca_stack.push();

        for i in 0..branch_count {
            let branch = rvsdg[switch_node].expect_switch().branches()[i];

            self.state_origin = (branch, StateOrigin::Argument);

            while self.visit_state_user(rvsdg) {}
        }

        let touched_allocas = self
            .touched_outer_alloca_stack
            .pop()
            .expect("we should be able to pop the set we pushed earlier");

        // Create an output for every input alloca (an alloca that originates outside the switch
        // node) that was touched (stored to) inside the switch node to make the pointer value
        // available in the outer region.
        for op_alloca in touched_allocas.iter().copied() {
            let output_ty = rvsdg[op_alloca].expect_op_alloca().ty();
            let output = rvsdg.add_switch_output(switch_node, output_ty);

            for i in 0..branch_count {
                let branch = rvsdg[switch_node].expect_switch().branches()[i];

                // Make sure the value is available in the branch. If the branch actually touched
                // the alloca (did a store), then the value will already be available. If the branch
                // did not tough the alloca, then this will find the latest available value in the
                // outer region and make it available as an argument.
                let origin = self.resolve_alloca_value(rvsdg, op_alloca, branch);

                rvsdg.reconnect_region_result(branch, output, origin);
            }

            // Mark the value as available in the outer region. Note that we must do this after
            // looping over the branches above, since for some branches we may need to make the
            // latest pointer value available as an argument; if we update this prematurely, then
            // we would incorrectly find the switch node's own new output to be the latest pointer
            // value available in the outer region.
            self.value_availability.insert(
                (op_alloca, region),
                ValueOrigin::Output {
                    producer: switch_node,
                    output,
                },
            );
        }

        self.state_origin = (region, StateOrigin::Node(switch_node));
    }

    fn visit_loop(&mut self, rvsdg: &mut Rvsdg, loop_node: Node) {
        let region = rvsdg[loop_node].region();
        let loop_region = rvsdg[loop_node].expect_loop().loop_region();

        self.touched_outer_alloca_stack.push();

        self.state_origin = (loop_region, StateOrigin::Argument);

        while self.visit_state_user(rvsdg) {}

        let touched_allocas = self
            .touched_outer_alloca_stack
            .pop()
            .expect("we should be able to pop the set we pushed earlier");

        // Create an output for every input alloca (an alloca that originates from outside the loop
        // node) that was touched (stored to) inside the switch node to make the pointer value
        // available in the outer region.
        for op_alloca in touched_allocas.iter().copied() {
            let origin = self.resolve_alloca_value(rvsdg, op_alloca, loop_region);

            let users = match origin {
                ValueOrigin::Argument(argument) => {
                    &rvsdg[loop_region].value_arguments()[argument as usize].users
                }
                ValueOrigin::Output { producer, output } => {
                    &rvsdg[producer].value_outputs()[output as usize].users
                }
            };

            let mut preexisting_output = None;

            for user in users {
                if let ValueUser::Result(result) = user {
                    preexisting_output = Some(*result - 1);
                }
            }

            let output = if let Some(preexisting_output) = preexisting_output {
                preexisting_output
            } else {
                let initial_value_origin = self.resolve_alloca_value(rvsdg, op_alloca, region);
                let ty = rvsdg[op_alloca].expect_op_alloca().ty();
                let input = rvsdg.add_loop_input(
                    loop_node,
                    ValueInput {
                        ty,
                        origin: initial_value_origin,
                    },
                );
                let result = input + 1;

                rvsdg.reconnect_region_result(loop_region, result, origin);

                input
            };

            self.value_availability.insert(
                (op_alloca, region),
                ValueOrigin::Output {
                    producer: loop_node,
                    output,
                },
            );
        }

        self.state_origin = (region, StateOrigin::Node(loop_node));
    }

    fn visit_op_store(&mut self, rvsdg: &mut Rvsdg, op_store: Node) {
        let region = rvsdg[op_store].region();
        let store_data = rvsdg[op_store].expect_op_store();
        let pointer_origin = store_data.ptr_input().origin;
        let value_origin = store_data.value_input().origin;
        let action = self
            .pointer_analyzer
            .resolve_action(rvsdg, region, pointer_origin);

        match action {
            PointerAction::Promotion(op_alloca) => {
                self.value_availability
                    .insert((op_alloca, region), value_origin);

                // If the alloca originated from an outer region, then "touch" the alloca so that
                // we can make the value available to the outer region later (see `visit_switch` and
                // `visit_loop`). Note that to test if the alloca originated from an outer region,
                // we only have to compare its region with the store node's region, since an alloca
                // could never come from a sub-region of the store node's region (an alloca cannot
                // outlive its region, so that would be UB); if the alloca's region is not equal to
                // the store node's region, then it must have originated from an outer region.
                if rvsdg[op_alloca].region() != region {
                    self.touched_outer_alloca_stack.touch(op_alloca);
                }

                rvsdg.remove_node(op_store);

                // Note that removing the node will adjust the state chain by connecting the
                // OpStore's state origin to the OpStore's state user, so we don't need to update
                // `self.state_origin`.
            }
            PointerAction::VariablePointerEmulation => {
                self.emulation_context.emulate_op_store(rvsdg, op_store);

                // Note that emulation will replace the OpStore in the state chain with the
                // emulation node, so the "visit loop" will automatically visit the emulation node
                // on the next iteration.
            }
            PointerAction::Nothing => {
                // Do nothing, except move the current state origin beyond the load
                self.state_origin = (region, StateOrigin::Node(op_store));
            }
        }
    }

    fn visit_op_load(&mut self, rvsdg: &mut Rvsdg, op_load: Node) {
        let region = rvsdg[op_load].region();
        let origin = rvsdg[op_load].expect_op_load().ptr_input().origin;
        let action = self.pointer_analyzer.resolve_action(rvsdg, region, origin);

        match action {
            PointerAction::Promotion(op_alloca) => {
                let origin = self.resolve_alloca_value(rvsdg, op_alloca, region);
                let user_count = rvsdg[op_load].expect_op_load().value_output().users.len();

                for i in (0..user_count).rev() {
                    let user = rvsdg[op_load].expect_op_load().value_output().users[i];

                    rvsdg.reconnect_value_user(region, user, origin);
                }

                rvsdg.remove_node(op_load);

                // Note that removing the node will adjust the state chain by connecting the
                // OpLoad's state origin to the OpLoad's state user, so we don't need to
                // update `self.state_origin`.
            }
            PointerAction::VariablePointerEmulation => {
                self.emulation_context.emulate_op_load(rvsdg, op_load);

                // Note that emulation will replace the OpLoad in the state chain with the emulation
                // node, so the "visit loop" will automatically visit the emulation node on the
                // next iteration.
            }
            PointerAction::Nothing => {
                // Do nothing, except move the current state origin beyond the load
                self.state_origin = (region, StateOrigin::Node(op_load));
            }
        }
    }

    fn resolve_alloca_value(
        &mut self,
        rvsdg: &mut Rvsdg,
        op_alloca: Node,
        mut region: Region,
    ) -> ValueOrigin {
        // Search outwards through parent regions until we find a region in which the value is
        // available. We record an "owner stack" of region owner-nodes in which the value will have
        // to be made available recursively.
        let mut origin = loop {
            if let Some(origin) = self.value_availability.get(&(op_alloca, region)) {
                // The value was previously recorded as being available in the region.
                break *origin;
            }

            if rvsdg[op_alloca].region() == region {
                // No value is available yet, but we're in the region that contains the OpAlloca for
                // the value: use the fallback value.
                let ty = rvsdg[op_alloca].expect_op_alloca().ty();
                let fallback_node = rvsdg.add_const_fallback(region, ty);
                let origin = ValueOrigin::Output {
                    producer: fallback_node,
                    output: 0,
                };

                self.value_availability.insert((op_alloca, region), origin);

                break origin;
            }

            let owner = rvsdg[region].owner();

            self.owner_stack.push(owner);
            region = rvsdg[owner].region();

            if region == rvsdg.global_region() {
                // We've arrived at the function node itself (whose owner region is the global
                // region) without encountering the region that contains the originating OpAlloca:
                // we're trying to make a value available that is not "in scope", which implies
                // something must have gone wrong.
                panic!(
                    "did not encounter the OpAlloca node (`{:?}`) on the region stack",
                    op_alloca
                );
            }
        };

        // Iterate over the region owners on the stack from the outside in, recursively making
        // the value available inside the owned regions until the stack is empty.
        while let Some(owner) = self.owner_stack.pop() {
            // Ensure the value is available to the region(s) inside the `owner`.

            let outer_region = rvsdg[owner].region();
            let ty = rvsdg.value_origin_ty(outer_region, origin);

            origin = match rvsdg[owner].kind() {
                NodeKind::Switch(_) => {
                    let input = rvsdg[owner]
                        .value_input_for_origin(origin)
                        .unwrap_or_else(|| {
                            rvsdg.add_switch_input(owner, ValueInput { ty, origin })
                        });

                    let argument = input - 1;
                    let inner_origin = ValueOrigin::Argument(argument);

                    for branch in rvsdg[owner].expect_switch().branches() {
                        // Make the value available in the branch, but only if it was not
                        // available already, as if a store operation already made a value
                        // available inside the branch, that value will represent the more
                        // recent value.
                        if !self.value_availability.contains_key(&(op_alloca, *branch)) {
                            self.value_availability
                                .insert((op_alloca, *branch), inner_origin);
                        }
                    }

                    inner_origin
                }
                NodeKind::Loop(loop_data) => {
                    let loop_region = loop_data.loop_region();

                    let argument =
                        rvsdg[owner]
                            .value_input_for_origin(origin)
                            .unwrap_or_else(|| {
                                let input = rvsdg.add_loop_input(owner, ValueInput { ty, origin });
                                let result = input + 1;

                                // Connect the new input to its corresponding result to make
                                // the value available to subsequent loop iterations.
                                rvsdg.reconnect_region_result(
                                    loop_region,
                                    result,
                                    ValueOrigin::Argument(input),
                                );

                                input
                            });

                    let inner_origin = ValueOrigin::Argument(argument);

                    self.value_availability
                        .insert((op_alloca, loop_region), inner_origin);

                    inner_origin
                }
                _ => panic!("not a valid region owner"),
            }
        }

        origin
    }
}

pub fn entry_points_memory_promotion_and_legalization(module: &mut Module, rvsdg: &mut Rvsdg) {
    let mut memory_promoter_legalizer = MemoryPromoterLegalizer::new();

    let entry_points = module
        .entry_points
        .iter()
        .map(|(f, _)| f)
        .collect::<Vec<_>>();

    for entry_point in entry_points {
        let node = rvsdg
            .get_function_node(entry_point)
            .expect("function should have RVSDG body");
        let body_region = rvsdg[node].expect_function().body_region();

        memory_promoter_legalizer.promote_and_legalize(rvsdg, body_region);
    }
}

#[cfg(test)]
mod tests {
    use std::iter;

    use super::*;
    use crate::rvsdg::ValueOutput;
    use crate::ty::{TY_DUMMY, TY_F32, TY_PREDICATE, TY_U32, TY_VEC2_F32, TypeKind};
    use crate::{BinaryOperator, FnArg, FnSig, Function, Module, Symbol, thin_set, ty};

    #[test]
    fn test_promote_store_then_load_then_store_then_load() {
        let mut module = Module::new(Symbol::from_ref(""));
        let function = Function {
            name: Symbol::from_ref(""),
            module: Symbol::from_ref(""),
        };

        module.fn_sigs.register(
            function,
            FnSig {
                name: Default::default(),
                ty: TY_DUMMY,
                args: vec![
                    FnArg {
                        ty: TY_U32,
                        shader_io_binding: None,
                    },
                    FnArg {
                        ty: TY_U32,
                        shader_io_binding: None,
                    },
                ],
                ret_ty: Some(TY_U32),
            },
        );

        let mut rvsdg = Rvsdg::new(module.ty.clone());

        let (_, region) = rvsdg.register_function(&module, function, iter::empty());

        let ptr_ty = module.ty.register(TypeKind::Ptr(TY_U32));

        let alloca_node = rvsdg.add_op_alloca(region, TY_U32);
        let store_0_node = rvsdg.add_op_store(
            region,
            ValueInput::output(ptr_ty, alloca_node, 0),
            ValueInput::argument(TY_U32, 0),
            StateOrigin::Argument,
        );
        let load_0_node = rvsdg.add_op_load(
            region,
            ValueInput::output(ptr_ty, alloca_node, 0),
            StateOrigin::Node(store_0_node),
        );
        let store_1_node = rvsdg.add_op_store(
            region,
            ValueInput::output(ptr_ty, alloca_node, 0),
            ValueInput::argument(TY_U32, 1),
            StateOrigin::Node(load_0_node),
        );
        let load_1_node = rvsdg.add_op_load(
            region,
            ValueInput::output(ptr_ty, alloca_node, 0),
            StateOrigin::Node(store_1_node),
        );
        let add_node = rvsdg.add_op_binary(
            region,
            BinaryOperator::Add,
            ValueInput::output(TY_U32, load_0_node, 0),
            ValueInput::output(TY_U32, load_1_node, 0),
        );

        rvsdg.reconnect_region_result(
            region,
            0,
            ValueOrigin::Output {
                producer: add_node,
                output: 0,
            },
        );

        let mut promoter_legalizer = MemoryPromoterLegalizer::new();

        promoter_legalizer.promote_and_legalize(&mut rvsdg, region);

        assert_eq!(
            &rvsdg[region].value_arguments()[0].users,
            &thin_set![ValueUser::Input {
                consumer: add_node,
                input: 0,
            }],
            "the first function argument should be used as the first input to the add node"
        );
        assert_eq!(
            &rvsdg[region].value_arguments()[1].users,
            &thin_set![ValueUser::Input {
                consumer: add_node,
                input: 1,
            }],
            "the second function argument should be used as the second input to the add node"
        );
        assert!(
            rvsdg[alloca_node]
                .expect_op_alloca()
                .value_output()
                .users
                .is_empty(),
            "the alloca node's output should no longer have any users after promotion"
        );
        assert!(
            !rvsdg.is_live_node(store_0_node),
            "the first store node should be removed after promotion"
        );
        assert!(
            !rvsdg.is_live_node(load_0_node),
            "the first load node should be removed after promotion"
        );
        assert!(
            !rvsdg.is_live_node(store_1_node),
            "the second store node should be removed after promotion"
        );
        assert!(
            !rvsdg.is_live_node(load_1_node),
            "the second load node should be removed after promotion"
        );
    }

    #[test]
    fn test_promote_store_then_store_then_load() {
        let mut module = Module::new(Symbol::from_ref(""));
        let function = Function {
            name: Symbol::from_ref(""),
            module: Symbol::from_ref(""),
        };

        module.fn_sigs.register(
            function,
            FnSig {
                name: Default::default(),
                ty: TY_DUMMY,
                args: vec![
                    FnArg {
                        ty: TY_U32,
                        shader_io_binding: None,
                    },
                    FnArg {
                        ty: TY_U32,
                        shader_io_binding: None,
                    },
                ],
                ret_ty: Some(TY_U32),
            },
        );

        let mut rvsdg = Rvsdg::new(module.ty.clone());

        let (_, region) = rvsdg.register_function(&module, function, iter::empty());

        let ptr_ty = module.ty.register(TypeKind::Ptr(TY_U32));

        let alloca_node = rvsdg.add_op_alloca(region, TY_U32);
        let store_0_node = rvsdg.add_op_store(
            region,
            ValueInput::output(ptr_ty, alloca_node, 0),
            ValueInput::argument(TY_U32, 0),
            StateOrigin::Argument,
        );
        let store_1_node = rvsdg.add_op_store(
            region,
            ValueInput::output(ptr_ty, alloca_node, 0),
            ValueInput::argument(TY_U32, 1),
            StateOrigin::Node(store_0_node),
        );
        let load_node = rvsdg.add_op_load(
            region,
            ValueInput::output(ptr_ty, alloca_node, 0),
            StateOrigin::Node(store_1_node),
        );

        rvsdg.reconnect_region_result(
            region,
            0,
            ValueOrigin::Output {
                producer: load_node,
                output: 0,
            },
        );

        let mut promoter_legalizer = MemoryPromoterLegalizer::new();

        promoter_legalizer.promote_and_legalize(&mut rvsdg, region);

        assert!(
            rvsdg[region].value_arguments()[0].users.is_empty(),
            "the first function argument should no longer have any users"
        );
        assert_eq!(
            &rvsdg[region].value_arguments()[1].users,
            &thin_set![ValueUser::Result(0)],
            "the second function argument should be used as the second input to function's result"
        );
        assert!(
            rvsdg[alloca_node]
                .expect_op_alloca()
                .value_output()
                .users
                .is_empty(),
            "the alloca node's output should no longer have any users after promotion"
        );
        assert!(
            !rvsdg.is_live_node(store_0_node),
            "the first store node should be removed after promotion"
        );
        assert!(
            !rvsdg.is_live_node(store_1_node),
            "the second store node should be removed after promotion"
        );
        assert!(
            !rvsdg.is_live_node(load_node),
            "the load node should be removed after promotion"
        );
    }

    #[test]
    fn test_promote_store_vector_then_load_vector() {
        let mut module = Module::new(Symbol::from_ref(""));
        let function = Function {
            name: Symbol::from_ref(""),
            module: Symbol::from_ref(""),
        };

        module.fn_sigs.register(
            function,
            FnSig {
                name: Default::default(),
                ty: TY_DUMMY,
                args: vec![
                    FnArg {
                        ty: TY_F32,
                        shader_io_binding: None,
                    },
                    FnArg {
                        ty: TY_F32,
                        shader_io_binding: None,
                    },
                ],
                ret_ty: Some(TY_VEC2_F32),
            },
        );

        let mut rvsdg = Rvsdg::new(module.ty.clone());

        let (_, region) = rvsdg.register_function(&module, function, iter::empty());

        let ptr_ty = module.ty.register(TypeKind::Ptr(TY_VEC2_F32));

        let alloca_node = rvsdg.add_op_alloca(region, TY_VEC2_F32);
        let vector_node = rvsdg.add_op_vector(
            region,
            ty::Vector::vec2_f32(),
            [
                ValueInput::argument(TY_F32, 0),
                ValueInput::argument(TY_F32, 1),
            ],
        );
        let store_node = rvsdg.add_op_store(
            region,
            ValueInput::output(ptr_ty, alloca_node, 0),
            ValueInput::output(TY_VEC2_F32, vector_node, 0),
            StateOrigin::Argument,
        );
        let load_node = rvsdg.add_op_load(
            region,
            ValueInput::output(ptr_ty, alloca_node, 0),
            StateOrigin::Node(store_node),
        );

        rvsdg.reconnect_region_result(
            region,
            0,
            ValueOrigin::Output {
                producer: load_node,
                output: 0,
            },
        );

        let mut promoter_legalizer = MemoryPromoterLegalizer::new();

        promoter_legalizer.promote_and_legalize(&mut rvsdg, region);

        assert_eq!(
            rvsdg[region].value_results()[0].origin,
            ValueOrigin::Output {
                producer: vector_node,
                output: 0
            },
            "the region result should connect directly to the vector node"
        );
        assert!(
            rvsdg[alloca_node]
                .expect_op_alloca()
                .value_output()
                .users
                .is_empty(),
            "the alloca node's output should no longer have any users after promotion"
        );
        assert!(
            !rvsdg.is_live_node(store_node),
            "the store node should be removed after promotion"
        );
        assert!(
            !rvsdg.is_live_node(load_node),
            "the load node should be removed after promotion"
        );
    }

    #[test]
    fn test_do_not_promote_store_vector_then_load_vector_with_partial_vector_store() {
        let mut module = Module::new(Symbol::from_ref(""));
        let function = Function {
            name: Symbol::from_ref(""),
            module: Symbol::from_ref(""),
        };

        module.fn_sigs.register(
            function,
            FnSig {
                name: Default::default(),
                ty: TY_DUMMY,
                args: vec![
                    FnArg {
                        ty: TY_F32,
                        shader_io_binding: None,
                    },
                    FnArg {
                        ty: TY_F32,
                        shader_io_binding: None,
                    },
                ],
                ret_ty: Some(TY_VEC2_F32),
            },
        );

        let mut rvsdg = Rvsdg::new(module.ty.clone());

        let (_, region) = rvsdg.register_function(&module, function, iter::empty());

        let ptr_ty = module.ty.register(TypeKind::Ptr(TY_VEC2_F32));
        let element_ptr_ty = module.ty.register(TypeKind::Ptr(TY_F32));

        let alloca_node = rvsdg.add_op_alloca(region, TY_VEC2_F32);
        let vector_node = rvsdg.add_op_vector(
            region,
            ty::Vector::vec2_f32(),
            [
                ValueInput::argument(TY_F32, 0),
                ValueInput::argument(TY_F32, 1),
            ],
        );
        let store_node = rvsdg.add_op_store(
            region,
            ValueInput::output(ptr_ty, alloca_node, 0),
            ValueInput::output(TY_VEC2_F32, vector_node, 0),
            StateOrigin::Argument,
        );
        let load_node = rvsdg.add_op_load(
            region,
            ValueInput::output(ptr_ty, alloca_node, 0),
            StateOrigin::Node(store_node),
        );
        let element_index_node = rvsdg.add_const_u32(region, 0);
        let element_ptr_node = rvsdg.add_op_element_ptr(
            region,
            ValueInput::output(ptr_ty, alloca_node, 0),
            ValueInput::output(TY_U32, element_index_node, 0),
        );
        let element_value_node = rvsdg.add_const_f32(region, 1.0);
        let element_store_node = rvsdg.add_op_store(
            region,
            ValueInput::output(element_ptr_ty, element_ptr_node, 0),
            ValueInput::output(TY_F32, element_value_node, 0),
            StateOrigin::Node(load_node),
        );

        rvsdg.reconnect_region_result(
            region,
            0,
            ValueOrigin::Output {
                producer: load_node,
                output: 0,
            },
        );

        let mut promoter_legalizer = MemoryPromoterLegalizer::new();

        promoter_legalizer.promote_and_legalize(&mut rvsdg, region);

        assert!(
            rvsdg.is_live_node(store_node),
            "the store node should still be alive"
        );
        assert!(
            rvsdg.is_live_node(load_node),
            "the load node should still be alive"
        );
        assert!(
            rvsdg.is_live_node(element_store_node),
            "the element-store node should still be alive"
        );
        assert_eq!(
            rvsdg[region].value_results()[0].origin,
            ValueOrigin::Output {
                producer: load_node,
                output: 0
            },
            "the region result should still connect to the load node"
        );
    }

    #[test]
    fn test_promote_store_then_load_inside_switch() {
        let mut module = Module::new(Symbol::from_ref(""));
        let function = Function {
            name: Symbol::from_ref(""),
            module: Symbol::from_ref(""),
        };

        module.fn_sigs.register(
            function,
            FnSig {
                name: Default::default(),
                ty: TY_DUMMY,
                args: vec![
                    FnArg {
                        ty: TY_U32,
                        shader_io_binding: None,
                    },
                    FnArg {
                        ty: TY_PREDICATE,
                        shader_io_binding: None,
                    },
                ],
                ret_ty: Some(TY_U32),
            },
        );

        let mut rvsdg = Rvsdg::new(module.ty.clone());

        let (_, region) = rvsdg.register_function(&module, function, iter::empty());

        let ptr_ty = module.ty.register(TypeKind::Ptr(TY_U32));

        let alloca_node = rvsdg.add_op_alloca(region, TY_U32);
        let store_node = rvsdg.add_op_store(
            region,
            ValueInput::output(ptr_ty, alloca_node, 0),
            ValueInput::argument(TY_U32, 0),
            StateOrigin::Argument,
        );
        let switch_node = rvsdg.add_switch(
            region,
            vec![
                ValueInput::argument(TY_PREDICATE, 1),
                ValueInput::output(ptr_ty, alloca_node, 0),
            ],
            vec![ValueOutput::new(TY_U32)],
            Some(StateOrigin::Node(store_node)),
        );

        let branch_0 = rvsdg.add_switch_branch(switch_node);

        let load_node = rvsdg.add_op_load(
            branch_0,
            ValueInput::argument(ptr_ty, 0),
            StateOrigin::Argument,
        );

        rvsdg.reconnect_region_result(
            branch_0,
            0,
            ValueOrigin::Output {
                producer: load_node,
                output: 0,
            },
        );

        let branch_1 = rvsdg.add_switch_branch(switch_node);

        let other_value_node = rvsdg.add_const_u32(branch_1, 0);

        rvsdg.reconnect_region_result(
            branch_1,
            0,
            ValueOrigin::Output {
                producer: other_value_node,
                output: 0,
            },
        );

        rvsdg.reconnect_region_result(
            region,
            0,
            ValueOrigin::Output {
                producer: switch_node,
                output: 0,
            },
        );

        let mut promoter_legalizer = MemoryPromoterLegalizer::new();

        promoter_legalizer.promote_and_legalize(&mut rvsdg, region);

        assert_eq!(
            &rvsdg[region].value_arguments()[0].users,
            &thin_set![ValueUser::Input {
                consumer: switch_node,
                input: 2,
            }],
            "the first function argument should now be used as the third input to the switch node"
        );
        assert_eq!(
            rvsdg[switch_node].expect_switch().value_inputs(),
            &[
                ValueInput::argument(TY_PREDICATE, 1),
                ValueInput::output(ptr_ty, alloca_node, 0),
                ValueInput::argument(TY_U32, 0),
            ],
            "the switch node should now use the first function argument as an additional input"
        );
        assert!(
            rvsdg[branch_0].value_arguments()[0].users.is_empty(),
            "the original pointer argument in the first branch should no longer have any users"
        );
        assert_eq!(
            &rvsdg[branch_0].value_arguments()[1].users,
            &thin_set![ValueUser::Result(0)],
            "the second region argument of the first branch should now connect to the first result \
            of the branch region"
        );
        assert_eq!(
            rvsdg[branch_0].value_results()[0].origin,
            ValueOrigin::Argument(1),
            "the result of the first branch should connect to the second region argument"
        );
        assert!(
            !rvsdg.is_live_node(store_node),
            "the store node should be removed after promotion"
        );
        assert!(
            !rvsdg.is_live_node(load_node),
            "the load node should be removed after promotion"
        );
    }

    #[test]
    fn test_promote_store_then_load_inside_nested_switch() {
        let mut module = Module::new(Symbol::from_ref(""));
        let function = Function {
            name: Symbol::from_ref(""),
            module: Symbol::from_ref(""),
        };

        module.fn_sigs.register(
            function,
            FnSig {
                name: Default::default(),
                ty: TY_DUMMY,
                args: vec![
                    FnArg {
                        ty: TY_U32,
                        shader_io_binding: None,
                    },
                    FnArg {
                        ty: TY_PREDICATE,
                        shader_io_binding: None,
                    },
                    FnArg {
                        ty: TY_PREDICATE,
                        shader_io_binding: None,
                    },
                ],
                ret_ty: Some(TY_U32),
            },
        );

        let mut rvsdg = Rvsdg::new(module.ty.clone());

        let (_, region) = rvsdg.register_function(&module, function, iter::empty());

        let ptr_ty = module.ty.register(TypeKind::Ptr(TY_U32));

        // Create alloca node and store the first function argument to it
        let alloca_node = rvsdg.add_op_alloca(region, TY_U32);
        let store_node = rvsdg.add_op_store(
            region,
            ValueInput::output(ptr_ty, alloca_node, 0),
            ValueInput::argument(TY_U32, 0),
            StateOrigin::Argument,
        );

        // Create outer switch node with the second argument as the condition
        let outer_switch_node = rvsdg.add_switch(
            region,
            vec![
                ValueInput::argument(TY_PREDICATE, 1),
                ValueInput::argument(TY_PREDICATE, 2),
                ValueInput::output(ptr_ty, alloca_node, 0),
            ],
            vec![ValueOutput::new(TY_U32)],
            Some(StateOrigin::Node(store_node)),
        );

        // Create the first branch of the outer switch node
        let outer_branch_0 = rvsdg.add_switch_branch(outer_switch_node);

        // Create a nested switch node inside the first branch of the outer switch
        let inner_switch_node = rvsdg.add_switch(
            outer_branch_0,
            vec![
                ValueInput::argument(TY_PREDICATE, 0),
                ValueInput::argument(ptr_ty, 1),
            ],
            vec![ValueOutput::new(TY_U32)],
            Some(StateOrigin::Argument),
        );

        // Create the first branch of the inner switch
        let inner_branch_0 = rvsdg.add_switch_branch(inner_switch_node);

        // Add a load node inside the first branch of the inner switch
        let load_node = rvsdg.add_op_load(
            inner_branch_0,
            ValueInput::argument(ptr_ty, 0),
            StateOrigin::Argument,
        );

        // Connect the load node's output to the inner branch's result
        rvsdg.reconnect_region_result(
            inner_branch_0,
            0,
            ValueOrigin::Output {
                producer: load_node,
                output: 0,
            },
        );

        // Create the second branch of the inner switch
        let inner_branch_1 = rvsdg.add_switch_branch(inner_switch_node);

        // Add a constant node in the second branch of the inner switch
        let inner_const_node = rvsdg.add_const_u32(inner_branch_1, 0);

        // Connect the constant node to the inner branch's result
        rvsdg.reconnect_region_result(
            inner_branch_1,
            0,
            ValueOrigin::Output {
                producer: inner_const_node,
                output: 0,
            },
        );

        // Connect the inner switch node's output to the outer branch's result
        rvsdg.reconnect_region_result(
            outer_branch_0,
            0,
            ValueOrigin::Output {
                producer: inner_switch_node,
                output: 0,
            },
        );

        // Create the second branch of the outer switch node
        let outer_branch_1 = rvsdg.add_switch_branch(outer_switch_node);

        // Add a constant node in the second branch of the outer switch
        let outer_const_node = rvsdg.add_const_u32(outer_branch_1, 42);

        // Connect the constant node to the outer branch's result
        rvsdg.reconnect_region_result(
            outer_branch_1,
            0,
            ValueOrigin::Output {
                producer: outer_const_node,
                output: 0,
            },
        );

        // Connect the outer switch node's output to the function's result
        rvsdg.reconnect_region_result(
            region,
            0,
            ValueOrigin::Output {
                producer: outer_switch_node,
                output: 0,
            },
        );

        let mut promoter_legalizer = MemoryPromoterLegalizer::new();

        promoter_legalizer.promote_and_legalize(&mut rvsdg, region);

        assert_eq!(
            &rvsdg[region].value_arguments()[0].users,
            &thin_set![ValueUser::Input {
                consumer: outer_switch_node,
                input: 3,
            }],
            "the first function argument should now be used as the fourth input to the outer \
            switch node"
        );

        assert_eq!(
            rvsdg[outer_switch_node].expect_switch().value_inputs(),
            &[
                ValueInput::argument(TY_PREDICATE, 1),
                ValueInput::argument(TY_PREDICATE, 2),
                ValueInput::output(ptr_ty, alloca_node, 0),
                ValueInput::argument(TY_U32, 0),
            ],
            "the outer switch node should now use the first function argument as an additional input"
        );

        assert_eq!(
            &rvsdg[outer_branch_0].value_arguments()[2].users,
            &thin_set![ValueUser::Input {
                consumer: inner_switch_node,
                input: 2
            }],
            "the third argument of the first outer branch should now connect to the third input of \
            the inner switch"
        );

        // Verify that the original pointer argument in the inner branch is no longer used
        assert!(
            rvsdg[inner_branch_0].value_arguments()[0].users.is_empty(),
            "the original pointer argument in the inner branch should no longer have any users"
        );
        assert_eq!(
            &rvsdg[inner_branch_0].value_arguments()[1].users,
            &thin_set![ValueUser::Result(0)],
            "the second region argument of the inner branch should now connect to the first result \
            of the branch region"
        );
        assert_eq!(
            rvsdg[inner_branch_0].value_results()[0].origin,
            ValueOrigin::Argument(1),
            "the result of the first inner branch should connect to the second region argument"
        );

        assert!(
            !rvsdg.is_live_node(load_node),
            "the load node should be removed after promotion"
        );
        assert!(
            !rvsdg.is_live_node(store_node),
            "the store node should be removed after promotion"
        );
    }

    #[test]
    fn test_store_then_store_argument_inside_switch_then_load_outside_switch() {
        let mut module = Module::new(Symbol::from_ref(""));
        let function = Function {
            name: Symbol::from_ref(""),
            module: Symbol::from_ref(""),
        };

        module.fn_sigs.register(
            function,
            FnSig {
                name: Default::default(),
                ty: TY_DUMMY,
                args: vec![
                    FnArg {
                        ty: TY_PREDICATE,
                        shader_io_binding: None,
                    },
                    FnArg {
                        ty: TY_U32,
                        shader_io_binding: None,
                    },
                    FnArg {
                        ty: TY_U32,
                        shader_io_binding: None,
                    },
                ],
                ret_ty: Some(TY_U32),
            },
        );

        let mut rvsdg = Rvsdg::new(module.ty.clone());

        let (_, region) = rvsdg.register_function(&module, function, iter::empty());

        let ptr_ty = module.ty.register(TypeKind::Ptr(TY_U32));

        let alloca_node = rvsdg.add_op_alloca(region, TY_U32);
        let store_node = rvsdg.add_op_store(
            region,
            ValueInput::output(ptr_ty, alloca_node, 0),
            ValueInput::argument(TY_U32, 1),
            StateOrigin::Argument,
        );
        let switch_node = rvsdg.add_switch(
            region,
            vec![
                ValueInput::argument(TY_PREDICATE, 0),
                ValueInput::output(ptr_ty, alloca_node, 0),
                ValueInput::argument(TY_U32, 2),
            ],
            vec![],
            Some(StateOrigin::Node(store_node)),
        );

        let branch_0 = rvsdg.add_switch_branch(switch_node);

        let switch_store_node = rvsdg.add_op_store(
            branch_0,
            ValueInput::argument(ptr_ty, 0),
            ValueInput::argument(TY_U32, 1),
            StateOrigin::Argument,
        );

        // The second branch does nothing
        let branch_1 = rvsdg.add_switch_branch(switch_node);

        let load_node = rvsdg.add_op_load(
            region,
            ValueInput::output(ptr_ty, alloca_node, 0),
            StateOrigin::Node(switch_node),
        );

        rvsdg.reconnect_region_result(
            region,
            0,
            ValueOrigin::Output {
                producer: load_node,
                output: 0,
            },
        );

        let mut promoter_legalizer = MemoryPromoterLegalizer::new();

        promoter_legalizer.promote_and_legalize(&mut rvsdg, region);

        assert_eq!(
            &rvsdg[region].value_arguments()[1].users,
            &thin_set![ValueUser::Input {
                consumer: switch_node,
                input: 3,
            }],
            "the second function argument should now be used as the fourth input to the switch node"
        );
        assert_eq!(
            &rvsdg[region].value_arguments()[2].users,
            &thin_set![ValueUser::Input {
                consumer: switch_node,
                input: 2,
            }],
            "the third function argument should now be used as the third input to the switch node"
        );
        assert_eq!(
            rvsdg[switch_node].expect_switch().value_inputs(),
            &[
                ValueInput::argument(TY_PREDICATE, 0),
                ValueInput::output(ptr_ty, alloca_node, 0),
                ValueInput::argument(TY_U32, 2),
                ValueInput::argument(TY_U32, 1),
            ],
            "in addition to the branch selector and the original pointer, the switch should now \
            use the third and second function arguments as inputs"
        );

        assert!(
            rvsdg[branch_0].value_arguments()[0].users.is_empty(),
            "the original pointer argument should no longer have any users"
        );
        assert_eq!(
            &rvsdg[branch_0].value_arguments()[1].users,
            &thin_set![ValueUser::Result(0)],
            "the second region argument of the first branch should now connect to the first result \
            of the branch region"
        );
        assert!(
            rvsdg[branch_0].value_arguments()[2].users.is_empty(),
            "the first branch should not use the third region argument"
        );
        assert_eq!(
            rvsdg[branch_0].value_results()[0].origin,
            ValueOrigin::Argument(1),
            "the result of the first branch should connect to the second region argument"
        );

        assert!(
            rvsdg[branch_1].value_arguments()[0].users.is_empty(),
            "the original pointer argument should still not have any users"
        );
        assert!(
            rvsdg[branch_1].value_arguments()[1].users.is_empty(),
            "the second branch should not use the second region argument"
        );
        assert_eq!(
            &rvsdg[branch_1].value_arguments()[2].users,
            &thin_set![ValueUser::Result(0)],
            "the third region argument of the second branch should now connect to the first result \
            of the branch region"
        );
        assert_eq!(
            rvsdg[branch_1].value_results()[0].origin,
            ValueOrigin::Argument(2),
            "the result of the second branch should connect to the third region argument"
        );

        assert_eq!(
            rvsdg[switch_node].value_outputs(),
            &[ValueOutput {
                ty: TY_U32,
                users: thin_set![ValueUser::Result(0)],
            }],
            "the switch node's output should be used by the function region's result"
        );
        assert_eq!(
            rvsdg[region].value_results()[0].origin,
            ValueOrigin::Output {
                producer: switch_node,
                output: 0,
            },
            "the function region's result should connect to the switch node's output"
        );

        assert!(
            !rvsdg.is_live_node(store_node),
            "the outer store node should be removed after promotion"
        );
        assert!(
            !rvsdg.is_live_node(switch_store_node),
            "the store node inside the switch branch should be removed after promotion"
        );
        assert!(
            !rvsdg.is_live_node(load_node),
            "the load node should be removed after promotion"
        );
    }

    #[test]
    fn test_store_then_store_argument_inside_nested_switch_then_load_outside_switch() {
        let mut module = Module::new(Symbol::from_ref(""));
        let function = Function {
            name: Symbol::from_ref(""),
            module: Symbol::from_ref(""),
        };

        module.fn_sigs.register(
            function,
            FnSig {
                name: Default::default(),
                ty: TY_DUMMY,
                args: vec![
                    FnArg {
                        ty: TY_PREDICATE,
                        shader_io_binding: None,
                    },
                    FnArg {
                        ty: TY_PREDICATE,
                        shader_io_binding: None,
                    },
                    FnArg {
                        ty: TY_U32,
                        shader_io_binding: None,
                    },
                    FnArg {
                        ty: TY_U32,
                        shader_io_binding: None,
                    },
                ],
                ret_ty: Some(TY_U32),
            },
        );

        let mut rvsdg = Rvsdg::new(module.ty.clone());

        let (_, region) = rvsdg.register_function(&module, function, iter::empty());

        let ptr_ty = module.ty.register(TypeKind::Ptr(TY_U32));

        // Create alloca node and initial store
        let alloca_node = rvsdg.add_op_alloca(region, TY_U32);
        let store_node = rvsdg.add_op_store(
            region,
            ValueInput::output(ptr_ty, alloca_node, 0),
            ValueInput::argument(TY_U32, 2),
            StateOrigin::Argument,
        );

        // Create the outer switch
        let outer_switch_node = rvsdg.add_switch(
            region,
            vec![
                ValueInput::argument(TY_PREDICATE, 0),
                ValueInput::argument(TY_PREDICATE, 1),
                ValueInput::output(ptr_ty, alloca_node, 0),
                ValueInput::argument(TY_U32, 3),
            ],
            vec![],
            Some(StateOrigin::Node(store_node)),
        );

        // First branch of outer switch containing a nested switch
        let outer_branch_0 = rvsdg.add_switch_branch(outer_switch_node);

        // Create inner switch inside the first branch of the outer switch
        let inner_switch_node = rvsdg.add_switch(
            outer_branch_0,
            vec![
                ValueInput::argument(TY_PREDICATE, 0),
                ValueInput::argument(ptr_ty, 1),
                ValueInput::argument(TY_U32, 2),
            ],
            vec![],
            Some(StateOrigin::Argument),
        );

        // First branch of inner switch - perform a store to the pointer
        let inner_branch_0 = rvsdg.add_switch_branch(inner_switch_node);

        let inner_switch_store_node = rvsdg.add_op_store(
            inner_branch_0,
            ValueInput::argument(ptr_ty, 0),
            ValueInput::argument(TY_U32, 1),
            StateOrigin::Argument,
        );

        // Second branch of inner switch - does nothing
        let inner_branch_1 = rvsdg.add_switch_branch(inner_switch_node);

        // Second branch of outer switch - does nothing
        let outer_branch_1 = rvsdg.add_switch_branch(outer_switch_node);

        // Load the pointer value after all the switches
        let load_node = rvsdg.add_op_load(
            region,
            ValueInput::output(ptr_ty, alloca_node, 0),
            StateOrigin::Node(outer_switch_node),
        );

        // Connect the loaded value to the function's result
        rvsdg.reconnect_region_result(
            region,
            0,
            ValueOrigin::Output {
                producer: load_node,
                output: 0,
            },
        );

        let mut promoter_legalizer = MemoryPromoterLegalizer::new();

        promoter_legalizer.promote_and_legalize(&mut rvsdg, region);

        // Verify that the third function argument (initial stored value) is correctly passed into
        // the outer switch
        assert_eq!(
            &rvsdg[region].value_arguments()[2].users,
            &thin_set![ValueUser::Input {
                consumer: outer_switch_node,
                input: 4,
            }],
            "the second function argument should now be used as the fifth input to the outer \
            switch node"
        );

        assert_eq!(
            rvsdg[outer_switch_node].value_outputs(),
            &[ValueOutput {
                ty: TY_U32,
                users: thin_set![ValueUser::Result(0)],
            }],
            "the outer switch should now have a single output that connects to the function result"
        );

        // Verify that the first outer branch passes its fourth argument (initial stored value) into
        // the inner switch node
        assert_eq!(
            &rvsdg[outer_branch_0].value_arguments()[3].users,
            &thin_set![ValueUser::Input {
                consumer: inner_switch_node,
                input: 3,
            }],
            "the first branch of the outer switch node should pass its fourth argument into the \
            inner switch node as its fourth input"
        );

        assert_eq!(
            &rvsdg[outer_branch_1].value_arguments()[3].users,
            &thin_set![ValueUser::Result(0)],
            "the second branch of the outer switch node should connect the fourth argument to the \
            region result"
        );

        // The inner switch node should now have a single output that connects the first outer
        // branch's result
        assert_eq!(
            rvsdg[inner_switch_node].value_outputs(),
            &[ValueOutput {
                ty: TY_U32,
                users: thin_set![ValueUser::Result(0)],
            }],
            "the outer switch should now have a single output that connects to the function result"
        );

        // Verify the connections for the first branch of the inner switch node
        assert!(
            rvsdg[inner_branch_0].value_arguments()[0].users.is_empty(),
            "the inner switch's first branch should no longer use the pointer argument"
        );
        assert_eq!(
            &rvsdg[inner_branch_0].value_arguments()[1].users,
            &thin_set![ValueUser::Result(0)],
            "the inner switch's first branch should connect its second argument to its result"
        );
        assert!(
            rvsdg[inner_branch_0].value_arguments()[2].users.is_empty(),
            "the inner switch's first branch should not use its third argument (the initial stored
            value)"
        );

        // Verify the connections for the second branch of the inner switch node
        assert_eq!(
            &rvsdg[inner_branch_1].value_arguments()[2].users,
            &thin_set![ValueUser::Result(0)],
            "the inner switch's second branch should connect its third argument to its result"
        );

        // Verify that all the memory operation nodes have been removed
        assert!(
            !rvsdg.is_live_node(store_node),
            "the outer store node should be removed after promotion"
        );
        assert!(
            !rvsdg.is_live_node(inner_switch_store_node),
            "the store node inside the inner switch branch should be removed after promotion"
        );
        assert!(
            !rvsdg.is_live_node(load_node),
            "the load node should be removed after promotion"
        );
    }

    #[test]
    fn test_promote_store_then_load_inside_loop() {
        let mut module = Module::new(Symbol::from_ref(""));
        let function = Function {
            name: Symbol::from_ref(""),
            module: Symbol::from_ref(""),
        };

        module.fn_sigs.register(
            function,
            FnSig {
                name: Default::default(),
                ty: TY_DUMMY,
                args: vec![
                    FnArg {
                        ty: TY_U32,
                        shader_io_binding: None,
                    },
                    FnArg {
                        ty: TY_U32,
                        shader_io_binding: None,
                    },
                ],
                ret_ty: Some(TY_U32),
            },
        );

        let mut rvsdg = Rvsdg::new(module.ty.clone());

        let (_, region) = rvsdg.register_function(&module, function, iter::empty());

        let ptr_ty = module.ty.register(TypeKind::Ptr(TY_U32));

        let alloca_node = rvsdg.add_op_alloca(region, TY_U32);
        let store_node = rvsdg.add_op_store(
            region,
            ValueInput::output(ptr_ty, alloca_node, 0),
            ValueInput::argument(TY_U32, 0),
            StateOrigin::Argument,
        );
        let iteration_count_node = rvsdg.add_const_u32(region, 0);
        let fallback_node = rvsdg.add_const_u32(region, 0);
        let (loop_node, loop_region) = rvsdg.add_loop(
            region,
            vec![
                ValueInput::output(TY_U32, iteration_count_node, 0),
                ValueInput::argument(TY_U32, 1),
                ValueInput::output(ptr_ty, alloca_node, 0),
                ValueInput::output(TY_U32, fallback_node, 0),
            ],
            Some(StateOrigin::Node(store_node)),
        );

        let test_node = rvsdg.add_op_binary(
            loop_region,
            BinaryOperator::LtEq,
            ValueInput::argument(TY_U32, 0),
            ValueInput::argument(TY_U32, 1),
        );
        let step_value_node = rvsdg.add_const_u32(loop_region, 1);
        let increment_node = rvsdg.add_op_binary(
            loop_region,
            BinaryOperator::Add,
            ValueInput::argument(TY_U32, 0),
            ValueInput::output(TY_U32, step_value_node, 0),
        );
        let load_node = rvsdg.add_op_load(
            loop_region,
            ValueInput::argument(ptr_ty, 2),
            StateOrigin::Argument,
        );

        rvsdg.reconnect_region_result(
            loop_region,
            0,
            ValueOrigin::Output {
                producer: test_node,
                output: 0,
            },
        );

        rvsdg.reconnect_region_result(
            loop_region,
            1,
            ValueOrigin::Output {
                producer: increment_node,
                output: 0,
            },
        );

        rvsdg.reconnect_region_result(loop_region, 2, ValueOrigin::Argument(1));

        rvsdg.reconnect_region_result(loop_region, 3, ValueOrigin::Argument(2));

        rvsdg.reconnect_region_result(
            loop_region,
            4,
            ValueOrigin::Output {
                producer: load_node,
                output: 0,
            },
        );

        rvsdg.reconnect_region_result(
            region,
            0,
            ValueOrigin::Output {
                producer: loop_node,
                output: 0,
            },
        );

        let mut promoter_legalizer = MemoryPromoterLegalizer::new();

        promoter_legalizer.promote_and_legalize(&mut rvsdg, region);

        assert_eq!(
            &rvsdg[region].value_arguments()[0].users,
            &thin_set![ValueUser::Input {
                consumer: loop_node,
                input: 4,
            }],
            "the first function argument should now be used as the fifth input to the loop node"
        );
        assert_eq!(
            rvsdg[loop_node].value_inputs(),
            &[
                ValueInput::output(TY_U32, iteration_count_node, 0),
                ValueInput::argument(TY_U32, 1),
                ValueInput::output(ptr_ty, alloca_node, 0),
                ValueInput::output(TY_U32, fallback_node, 0),
                ValueInput::argument(TY_U32, 0),
            ],
            "the loop node should now use the first function argument as an additional input"
        );
        assert_eq!(
            &rvsdg[loop_region].value_arguments()[2].users,
            &thin_set![ValueUser::Result(3)],
            "the original pointer argument in the loop region should no longer be used by any nodes"
        );

        // The stored value that was added as the fifth input should be made available to subsequent
        // iterations by connecting to the sixth result. It should also have replaced the OpLoad
        // node's connection by connecting to the fifth result.
        assert_eq!(
            &rvsdg[loop_region].value_arguments()[4].users,
            &thin_set![ValueUser::Result(4), ValueUser::Result(5)],
            "the fifth argument the loop region should connect to the fifth and sixth results"
        );

        assert!(
            !rvsdg.is_live_node(store_node),
            "the store node should be removed after promotion"
        );
        assert!(
            !rvsdg.is_live_node(load_node),
            "the load node should be removed after promotion"
        );
    }

    #[test]
    fn test_promote_store_then_store_inside_loop_then_load_outside_loop() {
        let mut module = Module::new(Symbol::from_ref(""));
        let function = Function {
            name: Symbol::from_ref(""),
            module: Symbol::from_ref(""),
        };

        module.fn_sigs.register(
            function,
            FnSig {
                name: Default::default(),
                ty: TY_DUMMY,
                args: vec![
                    FnArg {
                        ty: TY_U32,
                        shader_io_binding: None,
                    },
                    FnArg {
                        ty: TY_U32,
                        shader_io_binding: None,
                    },
                ],
                ret_ty: Some(TY_U32),
            },
        );

        let mut rvsdg = Rvsdg::new(module.ty.clone());

        let (_, region) = rvsdg.register_function(&module, function, iter::empty());

        let ptr_ty = module.ty.register(TypeKind::Ptr(TY_U32));

        let alloca_node = rvsdg.add_op_alloca(region, TY_U32);
        let outer_store_node = rvsdg.add_op_store(
            region,
            ValueInput::output(ptr_ty, alloca_node, 0),
            ValueInput::argument(TY_U32, 0),
            StateOrigin::Argument,
        );
        let iteration_count_node = rvsdg.add_const_u32(region, 0);
        let (loop_node, loop_region) = rvsdg.add_loop(
            region,
            vec![
                ValueInput::output(TY_U32, iteration_count_node, 0),
                ValueInput::argument(TY_U32, 1),
                ValueInput::output(ptr_ty, alloca_node, 0),
            ],
            Some(StateOrigin::Node(outer_store_node)),
        );

        let test_node = rvsdg.add_op_binary(
            loop_region,
            BinaryOperator::LtEq,
            ValueInput::argument(TY_U32, 0),
            ValueInput::argument(TY_U32, 1),
        );
        let step_value_node = rvsdg.add_const_u32(loop_region, 1);
        let increment_node = rvsdg.add_op_binary(
            loop_region,
            BinaryOperator::Add,
            ValueInput::argument(TY_U32, 0),
            ValueInput::output(TY_U32, step_value_node, 0),
        );
        let inner_stored_value_node = rvsdg.add_const_u32(loop_region, 1);
        let inner_store_node = rvsdg.add_op_store(
            loop_region,
            ValueInput::argument(ptr_ty, 2),
            ValueInput::output(TY_U32, inner_stored_value_node, 0),
            StateOrigin::Argument,
        );

        rvsdg.reconnect_region_result(
            loop_region,
            0,
            ValueOrigin::Output {
                producer: test_node,
                output: 0,
            },
        );

        rvsdg.reconnect_region_result(
            loop_region,
            1,
            ValueOrigin::Output {
                producer: increment_node,
                output: 0,
            },
        );

        rvsdg.reconnect_region_result(loop_region, 2, ValueOrigin::Argument(1));

        rvsdg.reconnect_region_result(loop_region, 3, ValueOrigin::Argument(2));

        let load_node = rvsdg.add_op_load(
            region,
            ValueInput::output(ptr_ty, alloca_node, 0),
            StateOrigin::Node(loop_node),
        );

        rvsdg.reconnect_region_result(
            region,
            0,
            ValueOrigin::Output {
                producer: load_node,
                output: 0,
            },
        );

        let mut promoter_legalizer = MemoryPromoterLegalizer::new();

        promoter_legalizer.promote_and_legalize(&mut rvsdg, region);

        assert_eq!(
            &rvsdg[region].value_arguments()[0].users,
            &thin_set![ValueUser::Input {
                consumer: loop_node,
                input: 3,
            }],
            "the first function argument should now be used as the fourth input to the loop node"
        );
        assert_eq!(
            rvsdg[loop_node].value_inputs(),
            &[
                ValueInput::output(TY_U32, iteration_count_node, 0),
                ValueInput::argument(TY_U32, 1),
                ValueInput::output(ptr_ty, alloca_node, 0),
                ValueInput::argument(TY_U32, 0),
            ],
            "the loop node should now use the first function argument as an additional input"
        );
        assert_eq!(
            &rvsdg[loop_region].value_arguments()[2].users,
            &thin_set![ValueUser::Result(3)],
            "the original pointer argument in the loop region should no longer be used by any nodes"
        );
        assert!(
            rvsdg[loop_region].value_arguments()[3].users.is_empty(),
            "the new argument should not be used by any nodes inside the loop region"
        );
        assert_eq!(
            &rvsdg[inner_stored_value_node]
                .expect_const_u32()
                .output()
                .users,
            &thin_set![ValueUser::Result(4)],
            "the inner-store-value node should be used by the new result"
        );
        assert_eq!(
            &rvsdg[loop_node].value_outputs()[3].users,
            &thin_set![ValueUser::Result(0)],
            "the new output should be used by the function result"
        );

        assert!(
            !rvsdg.is_live_node(outer_store_node),
            "the outer store node should be removed after promotion"
        );
        assert!(
            !rvsdg.is_live_node(inner_store_node),
            "the inner store node should be removed after promotion"
        );
        assert!(
            !rvsdg.is_live_node(load_node),
            "the load node should be removed after promotion"
        );
    }

    #[test]
    fn test_emulate_promote_variable_pointer_then_load() {
        let mut module = Module::new(Symbol::from_ref(""));
        let function = Function {
            name: Symbol::from_ref(""),
            module: Symbol::from_ref(""),
        };

        module.fn_sigs.register(
            function,
            FnSig {
                name: Default::default(),
                ty: TY_DUMMY,
                args: vec![FnArg {
                    ty: TY_PREDICATE,
                    shader_io_binding: None,
                }],
                ret_ty: Some(TY_U32),
            },
        );

        let mut rvsdg = Rvsdg::new(module.ty.clone());

        let (_, region) = rvsdg.register_function(&module, function, iter::empty());

        let ptr_ty = module.ty.register(TypeKind::Ptr(TY_U32));

        let alloca_0_node = rvsdg.add_op_alloca(region, TY_U32);
        let store_0_value = rvsdg.add_const_u32(region, 0);
        let store_0_node = rvsdg.add_op_store(
            region,
            ValueInput::output(ptr_ty, alloca_0_node, 0),
            ValueInput::output(TY_U32, store_0_value, 0),
            StateOrigin::Argument,
        );

        let alloca_1_node = rvsdg.add_op_alloca(region, TY_U32);
        let store_1_value = rvsdg.add_const_u32(region, 1);
        let store_1_node = rvsdg.add_op_store(
            region,
            ValueInput::output(ptr_ty, alloca_1_node, 0),
            ValueInput::output(TY_U32, store_1_value, 0),
            StateOrigin::Node(store_0_node),
        );

        let switch_node = rvsdg.add_switch(
            region,
            vec![
                ValueInput::argument(TY_PREDICATE, 0),
                ValueInput::output(ptr_ty, alloca_0_node, 0),
                ValueInput::output(ptr_ty, alloca_1_node, 0),
            ],
            vec![ValueOutput::new(ptr_ty)],
            None,
        );

        let branch_0 = rvsdg.add_switch_branch(switch_node);

        rvsdg.reconnect_region_result(branch_0, 0, ValueOrigin::Argument(0));

        let branch_1 = rvsdg.add_switch_branch(switch_node);

        rvsdg.reconnect_region_result(branch_1, 0, ValueOrigin::Argument(1));

        let load_node = rvsdg.add_op_load(
            region,
            ValueInput::output(ptr_ty, switch_node, 0),
            StateOrigin::Node(store_1_node),
        );

        rvsdg.reconnect_region_result(
            region,
            0,
            ValueOrigin::Output {
                producer: load_node,
                output: 0,
            },
        );

        let mut promoter_legalizer = MemoryPromoterLegalizer::new();

        promoter_legalizer.promote_and_legalize(&mut rvsdg, region);

        let ValueOrigin::Output {
            producer: emulation_node,
            output: 0,
        } = rvsdg[region].value_results()[0].origin
        else {
            panic!("the function result should connect to the first output of the emulation node");
        };

        let emulation_data = rvsdg[emulation_node].expect_switch();

        assert_eq!(
            &rvsdg[region].value_arguments()[0].users,
            &thin_set![
                ValueUser::Input {
                    consumer: switch_node,
                    input: 0,
                },
                ValueUser::Input {
                    consumer: emulation_node,
                    input: 0,
                }
            ],
            "the branch selector argument should now also be used by the emulation node"
        );
        assert_eq!(
            &rvsdg[alloca_0_node].value_outputs()[0].users,
            &thin_set![
                ValueUser::Input {
                    consumer: switch_node,
                    input: 1,
                },
                ValueUser::Input {
                    consumer: emulation_node,
                    input: 1,
                }
            ],
            "the first alloca should now also be used by the emulation node"
        );
        assert_eq!(
            &rvsdg[alloca_1_node].value_outputs()[0].users,
            &thin_set![
                ValueUser::Input {
                    consumer: switch_node,
                    input: 2,
                },
                ValueUser::Input {
                    consumer: emulation_node,
                    input: 2,
                }
            ],
            "the second alloca should now also be used by the emulation node"
        );
        assert_eq!(
            &rvsdg[store_0_value].value_outputs()[0].users,
            &thin_set![ValueUser::Input {
                consumer: emulation_node,
                input: 3,
            }],
            "the first store-value should now be used by the emulation node and should no longer \
            by used by the first store node"
        );
        assert_eq!(
            &rvsdg[store_1_value].value_outputs()[0].users,
            &thin_set![ValueUser::Input {
                consumer: emulation_node,
                input: 4,
            }],
            "the second store-value should now be used by the emulation node and should no longer \
            by used by the first store node"
        );
        assert!(
            rvsdg[switch_node].value_outputs()[0].users.is_empty(),
            "the variable pointer should no longer be used"
        );

        assert_eq!(
            emulation_data.value_inputs().len(),
            5,
            "the emulation node should have 5 inputs"
        );
        assert_eq!(
            emulation_data.branches().len(),
            2,
            "the emulation node should have 2 branches"
        );
        assert_eq!(
            emulation_data.value_outputs(),
            &[ValueOutput {
                ty: TY_U32,
                users: thin_set![ValueUser::Result(0)],
            }],
            "the emulation node should have single output that connects to the function result"
        );

        let branch_0 = emulation_data.branches()[0];
        let branch_1 = emulation_data.branches()[1];

        assert!(
            rvsdg[branch_0].value_arguments()[0].users.is_empty(),
            "the first branch of the emulation should not use the pointer to alloca `0`"
        );
        assert!(
            rvsdg[branch_0].value_arguments()[1].users.is_empty(),
            "the first branch of the emulation should not use the pointer to alloca `1`"
        );
        assert_eq!(
            &rvsdg[branch_0].value_arguments()[2].users,
            &thin_set![ValueUser::Result(0)],
            "the first branch of the emulation node should connect the store-value-0 argument to \
            the region result"
        );
        assert!(
            rvsdg[branch_0].value_arguments()[3].users.is_empty(),
            "the first branch of the emulation should not use the store-value-1 argument"
        );

        assert!(
            rvsdg[branch_1].value_arguments()[0].users.is_empty(),
            "the first branch of the emulation should not use the pointer to alloca `0`"
        );
        assert!(
            rvsdg[branch_1].value_arguments()[1].users.is_empty(),
            "the first branch of the emulation should not use the pointer to alloca `1`"
        );
        assert!(
            rvsdg[branch_1].value_arguments()[2].users.is_empty(),
            "the first branch of the emulation should not use the store-value-0 argument"
        );
        assert_eq!(
            &rvsdg[branch_1].value_arguments()[3].users,
            &thin_set![ValueUser::Result(0)],
            "the first branch of the emulation node should connect the store-value-1 argument to \
            the region result"
        );
    }
}
