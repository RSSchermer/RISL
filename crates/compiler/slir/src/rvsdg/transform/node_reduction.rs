use indexmap::IndexSet;

use crate::rvsdg::visit::region_nodes::{RegionNodesVisitor, visit_node};
use crate::rvsdg::visit::reverse_value_flow::{ReverseValueFlowVisitor, visit_region_argument};
use crate::rvsdg::{
    Connectivity, Node, NodeKind, Region, Rvsdg, SimpleNode, ValueInput, ValueOrigin, ValueUser,
};
use crate::ty::{Int, IntSize, TY_BOOL, TY_F32, TY_I32, TY_U32};
use crate::{BinaryOperator, Module, UnaryOperator};

#[derive(Clone, Copy, Debug, PartialEq)]
enum MaybeConstantValue {
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    Predicate(u32),
    Variable(ValueInput),
}

pub fn transform_entry_points(module: &Module, rvsdg: &mut Rvsdg) {
    let mut reducer = NodeReducer::new();

    for (entry_point, _) in module.entry_points.iter() {
        let fn_node = rvsdg
            .get_function_node(entry_point)
            .expect("function not registered");
        let body_region = rvsdg[fn_node].expect_function().body_region();

        reducer.process_region(rvsdg, body_region);
    }
}

pub struct NodeReducer {
    worklist: IndexSet<Node>,
}

impl NodeReducer {
    pub fn new() -> Self {
        Self {
            worklist: IndexSet::new(),
        }
    }

    pub fn process_region(&mut self, rvsdg: &mut Rvsdg, region: Region) {
        let mut collector = CandidateCollector {
            worklist: &mut self.worklist,
        };
        collector.visit_region(rvsdg, region);

        while let Some(node) = self.worklist.pop() {
            if !rvsdg.is_live_node(node) {
                continue;
            }

            self.try_reduce_node(rvsdg, node);
        }
    }

    fn try_reduce_node(&mut self, rvsdg: &mut Rvsdg, node: Node) {
        let node_data = &rvsdg[node];
        let region = node_data.region();

        use NodeKind::*;
        use SimpleNode::*;

        match node_data.kind() {
            Simple(OpBinary(op_binary)) => {
                let op = op_binary.operator();

                let lhs = self.resolve_value(rvsdg, node, 0);
                let rhs = self.resolve_value(rvsdg, node, 1);

                if let Some(reduced) = try_reduce_binary(op, lhs, rhs) {
                    self.apply_reduction(rvsdg, node, region, reduced);
                }
            }
            Simple(OpUnary(op_unary)) => {
                let op = op_unary.operator();
                let val = self.resolve_value(rvsdg, node, 0);

                if let Some(reduced) = try_reduce_unary(op, val) {
                    self.apply_reduction(rvsdg, node, region, reduced);
                }
            }
            Simple(OpConvertToU32(_)) => {
                let val = self.resolve_value(rvsdg, node, 0);

                if let Some(reduced) = try_reduce_convert_to_u32(val) {
                    self.apply_reduction(rvsdg, node, region, reduced);
                }
            }
            Simple(OpConvertToI32(_)) => {
                let val = self.resolve_value(rvsdg, node, 0);

                if let Some(reduced) = try_reduce_convert_to_i32(val) {
                    self.apply_reduction(rvsdg, node, region, reduced);
                }
            }
            Simple(OpConvertToF32(_)) => {
                let val = self.resolve_value(rvsdg, node, 0);

                if let Some(reduced) = try_reduce_convert_to_f32(val) {
                    self.apply_reduction(rvsdg, node, region, reduced);
                }
            }
            Simple(OpConvertToBool(_)) => {
                let val = self.resolve_value(rvsdg, node, 0);

                if let Some(reduced) = try_reduce_convert_to_bool(val) {
                    self.apply_reduction(rvsdg, node, region, reduced);
                }
            }
            Simple(OpCaseToBranchSelector(n)) => {
                let encoding = n.encoding();
                let cases = n.cases();
                let val = self.resolve_value(rvsdg, node, 0);

                println!(
                    "try reduce OpCaseToBranchSelector: encoding {:?} - cases {:?} - resolved_val {:?} ",
                    encoding, cases, val
                );

                if let Some(reduced) = try_reduce_op_case_to_branch_selector(encoding, cases, val) {
                    self.apply_reduction(rvsdg, node, region, reduced);
                }
            }
            _ => {}
        }
    }

    fn apply_reduction(
        &mut self,
        rvsdg: &mut Rvsdg,
        node: Node,
        region: Region,
        reduced: MaybeConstantValue,
    ) {
        let new_origin = match reduced {
            MaybeConstantValue::U32(v) => {
                let const_node = rvsdg.add_const_u32(region, v);

                ValueOrigin::Output {
                    producer: const_node,
                    output: 0,
                }
            }
            MaybeConstantValue::I32(v) => {
                let const_node = rvsdg.add_const_i32(region, v);

                ValueOrigin::Output {
                    producer: const_node,
                    output: 0,
                }
            }
            MaybeConstantValue::F32(v) => {
                let const_node = rvsdg.add_const_f32(region, v);

                ValueOrigin::Output {
                    producer: const_node,
                    output: 0,
                }
            }
            MaybeConstantValue::Bool(v) => {
                let const_node = rvsdg.add_const_bool(region, v);

                ValueOrigin::Output {
                    producer: const_node,
                    output: 0,
                }
            }
            MaybeConstantValue::Predicate(v) => {
                let const_node = rvsdg.add_const_predicate(region, v);

                ValueOrigin::Output {
                    producer: const_node,
                    output: 0,
                }
            }
            MaybeConstantValue::Variable(input) => input.origin,
        };

        let original_origin = ValueOrigin::Output {
            producer: node,
            output: 0,
        };

        for user in &rvsdg[node].value_outputs()[0].users {
            if let ValueUser::Input { consumer, .. } = user {
                if let NodeKind::Simple(simple) = rvsdg[*consumer].kind() {
                    use SimpleNode::*;

                    match simple {
                        OpBinary(_)
                        | OpUnary(_)
                        | OpConvertToU32(_)
                        | OpConvertToI32(_)
                        | OpConvertToF32(_)
                        | OpConvertToBool(_)
                        | OpCaseToBranchSelector(_) => {
                            self.worklist.insert(*consumer);
                        }
                        _ => {}
                    }
                }
            }
        }

        rvsdg.reconnect_value_users(region, original_origin, new_origin);
        rvsdg.remove_node(node);
    }

    fn resolve_value(&self, rvsdg: &Rvsdg, node: Node, input_index: u32) -> MaybeConstantValue {
        let mut resolver = ValueResolver { result: None };

        resolver.visit_value_input(rvsdg, node, input_index);

        resolver.result.unwrap_or_else(|| {
            MaybeConstantValue::Variable(rvsdg[node].value_inputs()[input_index as usize])
        })
    }
}

struct CandidateCollector<'a> {
    worklist: &'a mut IndexSet<Node>,
}

impl<'a> RegionNodesVisitor for CandidateCollector<'a> {
    fn visit_node(&mut self, rvsdg: &Rvsdg, node: Node) {
        use NodeKind::*;
        use SimpleNode::*;

        if let Simple(
            OpBinary(_)
            | OpUnary(_)
            | OpConvertToU32(_)
            | OpConvertToI32(_)
            | OpConvertToF32(_)
            | OpConvertToBool(_)
            | OpCaseToBranchSelector(_),
        ) = rvsdg[node].kind()
        {
            self.worklist.insert(node);
        }

        visit_node(self, rvsdg, node);
    }
}

struct ValueResolver {
    result: Option<MaybeConstantValue>,
}

impl ReverseValueFlowVisitor for ValueResolver {
    fn should_visit(&mut self, _region: Region, _origin: ValueOrigin) -> bool {
        self.result.is_none()
    }

    fn visit_region_argument(&mut self, rvsdg: &Rvsdg, region: Region, argument: u32) {
        let owner = rvsdg[region].owner();

        use NodeKind::*;

        match rvsdg[owner].kind() {
            Switch(_) => {
                visit_region_argument(self, rvsdg, region, argument);
            }
            Loop(loop_node) => {
                let loop_region = loop_node.loop_region();
                let result_origin =
                    rvsdg[loop_region].value_results()[argument as usize + 1].origin;

                // We only trace "up" out of a loop-region if the argument we traced to is a
                // loop-invariant loop-value, that is, the corresponding loop-region result
                // connects directly to the argument.
                if result_origin == ValueOrigin::Argument(argument) {
                    visit_region_argument(self, rvsdg, region, argument);
                }
            }
            Function(_) => {}
            _ => unreachable!(),
        }
    }

    fn visit_value_output(&mut self, rvsdg: &Rvsdg, node: Node, _output: u32) {
        use NodeKind::*;
        use SimpleNode::*;

        match rvsdg[node].kind() {
            Simple(ConstU32(c)) => {
                self.result = Some(MaybeConstantValue::U32(c.value()));
            }
            Simple(ConstI32(c)) => {
                self.result = Some(MaybeConstantValue::I32(c.value()));
            }
            Simple(ConstF32(c)) => {
                self.result = Some(MaybeConstantValue::F32(c.value()));
            }
            Simple(ConstBool(c)) => {
                self.result = Some(MaybeConstantValue::Bool(c.value()));
            }
            Simple(ConstPredicate(c)) => {
                self.result = Some(MaybeConstantValue::Predicate(c.value()));
            }
            Simple(ValueProxy(_)) => {
                self.visit_value_input(rvsdg, node, 0);
            }
            _ => {}
        }
    }
}

/// There are some discrepancies between floating point arithmetic in Rust and floating point
/// arithmetic in WGSL. To avoid such discrepancies, we use this helper function to conservatively
/// only apply reductions involving `f32` values when it is "safe" to do so.
fn is_safe_f32(f: f32) -> bool {
    f.is_normal() || f == 0.0
}

fn try_reduce_unary(op: UnaryOperator, val: MaybeConstantValue) -> Option<MaybeConstantValue> {
    use MaybeConstantValue::*;
    use UnaryOperator::*;

    match (op, val) {
        (Not, Bool(b)) => Some(Bool(!b)),
        (Neg, I32(i)) => Some(I32(i.wrapping_neg())),
        (Neg, F32(f)) if is_safe_f32(f) => {
            let res = -f;

            is_safe_f32(res).then(|| F32(res))
        }
        _ => None,
    }
}

fn try_reduce_binary(
    op: BinaryOperator,
    lhs: MaybeConstantValue,
    rhs: MaybeConstantValue,
) -> Option<MaybeConstantValue> {
    use BinaryOperator::*;

    match op {
        Add => try_reduce_add(lhs, rhs),
        Sub => try_reduce_sub(lhs, rhs),
        Mul => try_reduce_mul(lhs, rhs),
        Div => try_reduce_div(lhs, rhs),
        Mod => try_reduce_mod(lhs, rhs),
        BitAnd => try_reduce_bit_and(lhs, rhs),
        BitOr => try_reduce_bit_or(lhs, rhs),
        BitXor => try_reduce_bit_xor(lhs, rhs),
        Shl => try_reduce_shl(lhs, rhs),
        Shr => try_reduce_shr(lhs, rhs),
        Lt => try_reduce_lt(lhs, rhs),
        LtEq => try_reduce_lt_eq(lhs, rhs),
        Gt => try_reduce_gt(lhs, rhs),
        GtEq => try_reduce_gt_eq(lhs, rhs),
        Eq => try_reduce_eq(lhs, rhs),
        NotEq => try_reduce_not_eq(lhs, rhs),
        And => try_reduce_and(lhs, rhs),
        Or => try_reduce_or(lhs, rhs),
    }
}

fn try_reduce_add(lhs: MaybeConstantValue, rhs: MaybeConstantValue) -> Option<MaybeConstantValue> {
    use MaybeConstantValue::*;

    match (lhs, rhs) {
        (U32(l), U32(r)) => Some(U32(l.wrapping_add(r))),
        (I32(l), I32(r)) => Some(I32(l.wrapping_add(r))),
        (F32(l), F32(r)) if is_safe_f32(l) && is_safe_f32(r) => {
            let res = l + r;

            is_safe_f32(res).then(|| F32(res))
        }
        (Variable(v), U32(0)) | (U32(0), Variable(v)) => Some(Variable(v)),
        (Variable(v), I32(0)) | (I32(0), Variable(v)) => Some(Variable(v)),
        _ => None,
    }
}

fn try_reduce_sub(lhs: MaybeConstantValue, rhs: MaybeConstantValue) -> Option<MaybeConstantValue> {
    use MaybeConstantValue::*;

    match (lhs, rhs) {
        (U32(l), U32(r)) => Some(U32(l.wrapping_sub(r))),
        (I32(l), I32(r)) => Some(I32(l.wrapping_sub(r))),
        (F32(l), F32(r)) if is_safe_f32(l) && is_safe_f32(r) => {
            let res = l - r;

            is_safe_f32(res).then(|| F32(res))
        }
        (Variable(v), U32(0)) => Some(Variable(v)),
        (Variable(v), I32(0)) => Some(Variable(v)),
        _ => None,
    }
}

fn try_reduce_mul(lhs: MaybeConstantValue, rhs: MaybeConstantValue) -> Option<MaybeConstantValue> {
    use MaybeConstantValue::*;

    match (lhs, rhs) {
        (U32(l), U32(r)) => Some(U32(l.wrapping_mul(r))),
        (I32(l), I32(r)) => Some(I32(l.wrapping_mul(r))),
        (F32(l), F32(r)) if is_safe_f32(l) && is_safe_f32(r) => {
            let res = l * r;

            is_safe_f32(res).then(|| F32(res))
        }
        (Variable(_), U32(0)) | (U32(0), Variable(_)) => Some(U32(0)),
        (Variable(_), I32(0)) | (I32(0), Variable(_)) => Some(I32(0)),
        (Variable(v), U32(1)) | (U32(1), Variable(v)) => Some(Variable(v)),
        (Variable(v), I32(1)) | (I32(1), Variable(v)) => Some(Variable(v)),
        _ => None,
    }
}

fn try_reduce_div(lhs: MaybeConstantValue, rhs: MaybeConstantValue) -> Option<MaybeConstantValue> {
    use MaybeConstantValue::*;

    match (lhs, rhs) {
        (U32(l), U32(r)) if r != 0 => Some(U32(l.wrapping_div(r))),
        (I32(l), I32(r)) if r != 0 => Some(I32(l.wrapping_div(r))),
        (F32(l), F32(r)) if is_safe_f32(l) && is_safe_f32(r) && r != 0.0 => {
            let res = l / r;

            is_safe_f32(res).then(|| F32(res))
        }
        (Variable(v), U32(1)) | (Variable(v), I32(1)) => Some(Variable(v)),
        _ => None,
    }
}

fn try_reduce_mod(lhs: MaybeConstantValue, rhs: MaybeConstantValue) -> Option<MaybeConstantValue> {
    use MaybeConstantValue::*;

    match (lhs, rhs) {
        (U32(l), U32(r)) if r != 0 => Some(U32(l.wrapping_rem(r))),
        (I32(l), I32(r)) if r != 0 => Some(I32(l.wrapping_rem(r))),
        (F32(l), F32(r)) if is_safe_f32(l) && is_safe_f32(r) && r != 0.0 => {
            let res = l % r;

            is_safe_f32(res).then(|| F32(res))
        }
        _ => None,
    }
}

fn try_reduce_bit_and(
    lhs: MaybeConstantValue,
    rhs: MaybeConstantValue,
) -> Option<MaybeConstantValue> {
    use MaybeConstantValue::*;

    match (lhs, rhs) {
        (U32(l), U32(r)) => Some(U32(l & r)),
        (I32(l), I32(r)) => Some(I32(l & r)),
        (Variable(_), U32(0)) | (U32(0), Variable(_)) => Some(U32(0)),
        (Variable(_), I32(0)) | (I32(0), Variable(_)) => Some(I32(0)),
        _ => None,
    }
}

fn try_reduce_bit_or(
    lhs: MaybeConstantValue,
    rhs: MaybeConstantValue,
) -> Option<MaybeConstantValue> {
    use MaybeConstantValue::*;

    match (lhs, rhs) {
        (U32(l), U32(r)) => Some(U32(l | r)),
        (I32(l), I32(r)) => Some(I32(l | r)),
        (Variable(v), U32(0)) | (U32(0), Variable(v)) => Some(Variable(v)),
        (Variable(v), I32(0)) | (I32(0), Variable(v)) => Some(Variable(v)),
        _ => None,
    }
}

fn try_reduce_bit_xor(
    lhs: MaybeConstantValue,
    rhs: MaybeConstantValue,
) -> Option<MaybeConstantValue> {
    use MaybeConstantValue::*;

    match (lhs, rhs) {
        (U32(l), U32(r)) => Some(U32(l ^ r)),
        (I32(l), I32(r)) => Some(I32(l ^ r)),
        (Variable(v), U32(0)) | (U32(0), Variable(v)) => Some(Variable(v)),
        (Variable(v), I32(0)) | (I32(0), Variable(v)) => Some(Variable(v)),
        _ => None,
    }
}

fn try_reduce_shl(lhs: MaybeConstantValue, rhs: MaybeConstantValue) -> Option<MaybeConstantValue> {
    use MaybeConstantValue::*;

    // Note that the WGSL specification explicitly specifies that the rhs shift amount is module the
    // number of bits in the lhs.

    match (lhs, rhs) {
        (U32(l), U32(r)) => Some(U32(l << (r % 32))),
        (I32(l), I32(r)) => Some(I32(l << (r % 32))),
        (Variable(v), U32(0)) | (Variable(v), I32(0)) => Some(Variable(v)),
        (U32(0), Variable(_)) => Some(U32(0)),
        (I32(0), Variable(_)) => Some(I32(0)),
        _ => None,
    }
}

fn try_reduce_shr(lhs: MaybeConstantValue, rhs: MaybeConstantValue) -> Option<MaybeConstantValue> {
    use MaybeConstantValue::*;

    // Note that the WGSL specification explicitly specifies that the rhs shift amount is module the
    // number of bits in the lhs.

    match (lhs, rhs) {
        (U32(l), U32(r)) => Some(U32(l >> (r % 32))),
        (I32(l), I32(r)) => Some(I32(l >> (r % 32))),
        (Variable(v), U32(0)) | (Variable(v), I32(0)) => Some(Variable(v)),
        (U32(0), Variable(_)) => Some(U32(0)),
        (I32(0), Variable(_)) => Some(I32(0)),
        _ => None,
    }
}

fn try_reduce_lt(lhs: MaybeConstantValue, rhs: MaybeConstantValue) -> Option<MaybeConstantValue> {
    use MaybeConstantValue::*;

    match (lhs, rhs) {
        (U32(l), U32(r)) => Some(Bool(l < r)),
        (I32(l), I32(r)) => Some(Bool(l < r)),
        (F32(l), F32(r)) if is_safe_f32(l) && is_safe_f32(r) => Some(Bool(l < r)),
        (Variable(_), U32(0)) => Some(Bool(false)),
        _ => None,
    }
}

fn try_reduce_lt_eq(
    lhs: MaybeConstantValue,
    rhs: MaybeConstantValue,
) -> Option<MaybeConstantValue> {
    use MaybeConstantValue::*;

    match (lhs, rhs) {
        (U32(l), U32(r)) => Some(Bool(l <= r)),
        (I32(l), I32(r)) => Some(Bool(l <= r)),
        (F32(l), F32(r)) if is_safe_f32(l) && is_safe_f32(r) => Some(Bool(l <= r)),
        (U32(0), Variable(_)) => Some(Bool(true)),
        _ => None,
    }
}

fn try_reduce_gt(lhs: MaybeConstantValue, rhs: MaybeConstantValue) -> Option<MaybeConstantValue> {
    use MaybeConstantValue::*;

    match (lhs, rhs) {
        (U32(l), U32(r)) => Some(Bool(l > r)),
        (I32(l), I32(r)) => Some(Bool(l > r)),
        (F32(l), F32(r)) if is_safe_f32(l) && is_safe_f32(r) => Some(Bool(l > r)),
        (U32(0), Variable(_)) => Some(Bool(false)),
        _ => None,
    }
}

fn try_reduce_gt_eq(
    lhs: MaybeConstantValue,
    rhs: MaybeConstantValue,
) -> Option<MaybeConstantValue> {
    use MaybeConstantValue::*;

    match (lhs, rhs) {
        (U32(l), U32(r)) => Some(Bool(l >= r)),
        (I32(l), I32(r)) => Some(Bool(l >= r)),
        (F32(l), F32(r)) if is_safe_f32(l) && is_safe_f32(r) => Some(Bool(l >= r)),
        (Variable(_), U32(0)) => Some(Bool(true)),
        _ => None,
    }
}

fn try_reduce_eq(lhs: MaybeConstantValue, rhs: MaybeConstantValue) -> Option<MaybeConstantValue> {
    use MaybeConstantValue::*;

    match (lhs, rhs) {
        (U32(l), U32(r)) => Some(Bool(l == r)),
        (I32(l), I32(r)) => Some(Bool(l == r)),
        (F32(l), F32(r)) if is_safe_f32(l) && is_safe_f32(r) => Some(Bool(l == r)),
        (Bool(l), Bool(r)) => Some(Bool(l == r)),
        _ => None,
    }
}

fn try_reduce_not_eq(
    lhs: MaybeConstantValue,
    rhs: MaybeConstantValue,
) -> Option<MaybeConstantValue> {
    use MaybeConstantValue::*;

    match (lhs, rhs) {
        (U32(l), U32(r)) => Some(Bool(l != r)),
        (I32(l), I32(r)) => Some(Bool(l != r)),
        (F32(l), F32(r)) if is_safe_f32(l) && is_safe_f32(r) => Some(Bool(l != r)),
        (Bool(l), Bool(r)) => Some(Bool(l != r)),
        _ => None,
    }
}

fn try_reduce_and(lhs: MaybeConstantValue, rhs: MaybeConstantValue) -> Option<MaybeConstantValue> {
    use MaybeConstantValue::*;

    match (lhs, rhs) {
        (Bool(l), Bool(r)) => Some(Bool(l && r)),
        (Variable(v), Bool(true)) | (Bool(true), Variable(v)) => Some(Variable(v)),
        (Variable(_), Bool(false)) | (Bool(false), Variable(_)) => Some(Bool(false)),
        _ => None,
    }
}

fn try_reduce_or(lhs: MaybeConstantValue, rhs: MaybeConstantValue) -> Option<MaybeConstantValue> {
    use MaybeConstantValue::*;

    match (lhs, rhs) {
        (Bool(l), Bool(r)) => Some(Bool(l || r)),
        (Variable(_), Bool(true)) | (Bool(true), Variable(_)) => Some(Bool(true)),
        (Variable(v), Bool(false)) | (Bool(false), Variable(v)) => Some(Variable(v)),
        _ => None,
    }
}

fn try_reduce_convert_to_u32(val: MaybeConstantValue) -> Option<MaybeConstantValue> {
    use MaybeConstantValue::*;

    match val {
        U32(v) => Some(U32(v)),
        I32(v) => Some(U32(v as u32)),
        F32(v) if is_safe_f32(v) => Some(U32(v as u32)),
        Bool(v) => Some(U32(v as u32)),
        Variable(input) if input.ty == TY_U32 => Some(Variable(input)),
        _ => None,
    }
}

fn try_reduce_convert_to_i32(val: MaybeConstantValue) -> Option<MaybeConstantValue> {
    use MaybeConstantValue::*;

    match val {
        U32(v) => Some(I32(v as i32)),
        I32(v) => Some(I32(v)),
        F32(v) if is_safe_f32(v) => Some(I32(v as i32)),
        Bool(v) => Some(I32(v as i32)),
        Variable(input) if input.ty == TY_I32 => Some(Variable(input)),
        _ => None,
    }
}

fn try_reduce_convert_to_f32(val: MaybeConstantValue) -> Option<MaybeConstantValue> {
    use MaybeConstantValue::*;

    match val {
        U32(v) => Some(F32(v as f32)),
        I32(v) => Some(F32(v as f32)),
        F32(v) if is_safe_f32(v) => Some(F32(v)),
        Bool(v) => Some(F32(if v { 1.0 } else { 0.0 })),
        Variable(input) if input.ty == TY_F32 => Some(Variable(input)),
        _ => None,
    }
}

fn try_reduce_convert_to_bool(val: MaybeConstantValue) -> Option<MaybeConstantValue> {
    use MaybeConstantValue::*;

    match val {
        U32(v) => Some(Bool(v != 0)),
        I32(v) => Some(Bool(v != 0)),
        F32(v) if is_safe_f32(v) => Some(Bool(v != 0.0)),
        Bool(v) => Some(Bool(v)),
        Variable(input) if input.ty == TY_BOOL => Some(Variable(input)),
        _ => None,
    }
}

fn try_reduce_op_case_to_branch_selector(
    encoding: Int,
    cases: &[u128],
    val: MaybeConstantValue,
) -> Option<MaybeConstantValue> {
    use MaybeConstantValue::*;

    let val_bits = match val {
        U32(v) => v as u128,

        // For negative integers we'll need to truncate (for integer encodings with fewer bits) or
        // sign-extend (for integer encodings with more bits) to ensure would match the bit pattern
        // of a u128 case encoding.
        I32(v) => match encoding.size {
            IntSize::I8 => {
                // Truncate to 8 bits, zero-extend to 128 bits
                v as u8 as u128
            }
            IntSize::I16 => {
                // Truncate to 16 bits, zero-extend to 128 bits
                v as u16 as u128
            }
            IntSize::I32 => {
                // Zero-extend to 128 bits
                v as u32 as u128
            }
            IntSize::I64 => {
                // Sign-extend to 64 bits, zero-extend to 128 bits
                v as i64 as u64 as u128
            }
            IntSize::I128 => v as i128 as u128,
        },
        _ => panic!("op-case-to-branch-selector input value must be an integer"),
    };

    let branch_index = cases
        .iter()
        .position(|&case| case == val_bits)
        .unwrap_or(cases.len());

    Some(Predicate(branch_index as u32))
}

#[cfg(test)]
mod tests {
    use MaybeConstantValue::*;

    use super::*;
    use crate::rvsdg::ValueInput;
    use crate::ty::{TY_BOOL, TY_DUMMY, TY_F32, TY_I32, TY_U32};
    use crate::{FnSig, Function, Symbol};

    #[test]
    fn test_node_reduction_integration() {
        use BinaryOperator::*;
        let mut module = Module::new(Symbol::from_ref(""));
        let mut rvsdg = Rvsdg::new(module.ty.clone());

        let function = Function {
            name: Symbol::from_ref("f"),
            module: Symbol::from_ref("m"),
        };

        module.fn_sigs.register(
            function,
            FnSig {
                name: function.name,
                ty: TY_DUMMY,
                args: vec![],
                ret_ty: Some(TY_U32),
            },
        );

        let (_, region) = rvsdg.register_function(&module, function, []);

        // (1 + 2) + 3
        let c1 = rvsdg.add_const_u32(region, 1);
        let c2 = rvsdg.add_const_u32(region, 2);
        let c3 = rvsdg.add_const_u32(region, 3);

        let i1 = ValueInput::output(TY_U32, c1, 0);
        let i2 = ValueInput::output(TY_U32, c2, 0);
        let add1 = rvsdg.add_op_binary(region, Add, i1, i2);

        let i3 = ValueInput::output(TY_U32, add1, 0);
        let i4 = ValueInput::output(TY_U32, c3, 0);
        let add2 = rvsdg.add_op_binary(region, Add, i3, i4);

        rvsdg.reconnect_region_result(
            region,
            0,
            ValueOrigin::Output {
                producer: add2,
                output: 0,
            },
        );

        let mut reducer = NodeReducer::new();

        reducer.process_region(&mut rvsdg, region);

        let ValueOrigin::Output {
            producer,
            output: 0,
        } = rvsdg[region].value_results()[0].origin
        else {
            panic!("expected region result to connect to the first output of a node");
        };

        assert_eq!(rvsdg[producer].expect_const_u32().value(), 6);
    }

    #[test]
    fn test_try_reduce_binary_add() {
        use BinaryOperator::*;
        assert_eq!(try_reduce_binary(Add, U32(1), U32(2)), Some(U32(3)));
        assert_eq!(try_reduce_binary(Add, I32(1), I32(2)), Some(I32(3)));
        assert_eq!(try_reduce_binary(Add, F32(1.0), F32(2.0)), Some(F32(3.0)));

        let var = Variable(ValueInput::argument(TY_U32, 0));
        assert_eq!(try_reduce_binary(Add, var, U32(0)), Some(var));
        assert_eq!(try_reduce_binary(Add, U32(0), var), Some(var));
        let var = Variable(ValueInput::argument(TY_I32, 0));
        assert_eq!(try_reduce_binary(Add, var, I32(0)), Some(var));
        assert_eq!(try_reduce_binary(Add, I32(0), var), Some(var));

        // F32 safety
        assert_eq!(try_reduce_binary(Add, F32(1.0), F32(f32::NAN)), None);
        assert_eq!(try_reduce_binary(Add, F32(1.0), F32(f32::INFINITY)), None);
        assert_eq!(
            try_reduce_binary(Add, F32(1.0), F32(f32::MIN_POSITIVE / 2.0)),
            None
        );
        assert_eq!(try_reduce_binary(Add, F32(f32::MAX), F32(f32::MAX)), None);
        assert_eq!(try_reduce_binary(Add, F32(0.0), F32(1.0)), Some(F32(1.0)));
        assert_eq!(try_reduce_binary(Add, F32(-0.0), F32(0.0)), Some(F32(0.0)));
    }

    #[test]
    fn test_try_reduce_binary_sub() {
        use BinaryOperator::*;

        assert_eq!(try_reduce_binary(Sub, U32(5), U32(3)), Some(U32(2)));
        assert_eq!(try_reduce_binary(Sub, I32(5), I32(3)), Some(I32(2)));
        assert_eq!(try_reduce_binary(Sub, F32(5.0), F32(3.0)), Some(F32(2.0)));

        let var = Variable(ValueInput::argument(TY_U32, 0));
        assert_eq!(try_reduce_binary(Sub, var, U32(0)), Some(var));
        let var = Variable(ValueInput::argument(TY_I32, 0));
        assert_eq!(try_reduce_binary(Sub, var, I32(0)), Some(var));

        // F32 safety
        assert_eq!(try_reduce_binary(Sub, F32(1.0), F32(f32::NAN)), None);
        assert_eq!(
            try_reduce_binary(Sub, F32(f32::MIN_POSITIVE), F32(f32::MIN_POSITIVE)),
            Some(F32(0.0))
        );
    }

    #[test]
    fn test_try_reduce_binary_mul() {
        use BinaryOperator::*;

        assert_eq!(try_reduce_binary(Mul, U32(2), U32(3)), Some(U32(6)));
        assert_eq!(try_reduce_binary(Mul, I32(2), I32(3)), Some(I32(6)));
        assert_eq!(try_reduce_binary(Mul, F32(2.0), F32(3.0)), Some(F32(6.0)));

        let var = Variable(ValueInput::argument(TY_U32, 0));
        assert_eq!(try_reduce_binary(Mul, var, U32(1)), Some(var));
        assert_eq!(try_reduce_binary(Mul, U32(1), var), Some(var));
        let var = Variable(ValueInput::argument(TY_I32, 0));
        assert_eq!(try_reduce_binary(Mul, var, I32(1)), Some(var));
        assert_eq!(try_reduce_binary(Mul, I32(1), var), Some(var));
        let var = Variable(ValueInput::argument(TY_U32, 0));
        assert_eq!(try_reduce_binary(Mul, var, U32(0)), Some(U32(0)));
        assert_eq!(try_reduce_binary(Mul, U32(0), var), Some(U32(0)));
        let var = Variable(ValueInput::argument(TY_I32, 0));
        assert_eq!(try_reduce_binary(Mul, var, I32(0)), Some(I32(0)));
        assert_eq!(try_reduce_binary(Mul, I32(0), var), Some(I32(0)));

        // F32 safety
        assert_eq!(
            try_reduce_binary(Mul, F32(f32::MIN_POSITIVE), F32(0.1)),
            None
        );
    }

    #[test]
    fn test_try_reduce_binary_div() {
        use BinaryOperator::*;

        assert_eq!(try_reduce_binary(Div, U32(6), U32(2)), Some(U32(3)));
        assert_eq!(try_reduce_binary(Div, I32(6), I32(2)), Some(I32(3)));
        assert_eq!(try_reduce_binary(Div, F32(6.0), F32(2.0)), Some(F32(3.0)));

        let var = Variable(ValueInput::argument(TY_U32, 0));
        assert_eq!(try_reduce_binary(Div, var, U32(1)), Some(var));
        let var = Variable(ValueInput::argument(TY_I32, 0));
        assert_eq!(try_reduce_binary(Div, var, I32(1)), Some(var));

        assert_eq!(try_reduce_binary(Div, U32(1), U32(0)), None);
        assert_eq!(try_reduce_binary(Div, I32(1), I32(0)), None);
        assert_eq!(try_reduce_binary(Div, F32(1.0), F32(0.0)), None);

        // F32 safety
        assert_eq!(try_reduce_binary(Div, F32(1.0), F32(f32::INFINITY)), None);
        assert_eq!(
            try_reduce_binary(Div, F32(f32::MIN_POSITIVE), F32(2.0)),
            None
        );
    }

    #[test]
    fn test_try_reduce_binary_mod() {
        use BinaryOperator::*;

        assert_eq!(try_reduce_binary(Mod, U32(5), U32(3)), Some(U32(2)));
        assert_eq!(try_reduce_binary(Mod, I32(5), I32(3)), Some(I32(2)));
        assert_eq!(try_reduce_binary(Mod, F32(5.0), F32(3.0)), Some(F32(2.0)));

        assert_eq!(try_reduce_binary(Mod, U32(1), U32(0)), None);
        assert_eq!(try_reduce_binary(Mod, I32(1), I32(0)), None);

        // F32 safety
        assert_eq!(try_reduce_binary(Mod, F32(1.0), F32(0.0)), None);
        assert_eq!(try_reduce_binary(Mod, F32(f32::NAN), F32(1.0)), None);
    }

    #[test]
    fn test_try_reduce_binary_bit_and() {
        use BinaryOperator::*;

        assert_eq!(
            try_reduce_binary(BitAnd, U32(0b1100), U32(0b1010)),
            Some(U32(0b1000))
        );
        assert_eq!(
            try_reduce_binary(BitAnd, I32(0b1100), I32(0b1010)),
            Some(I32(0b1000))
        );

        let var = Variable(ValueInput::argument(TY_U32, 0));
        assert_eq!(try_reduce_binary(BitAnd, var, U32(0)), Some(U32(0)));
        assert_eq!(try_reduce_binary(BitAnd, U32(0), var), Some(U32(0)));
        let var = Variable(ValueInput::argument(TY_I32, 0));
        assert_eq!(try_reduce_binary(BitAnd, var, I32(0)), Some(I32(0)));
        assert_eq!(try_reduce_binary(BitAnd, I32(0), var), Some(I32(0)));
    }

    #[test]
    fn test_try_reduce_binary_bit_or() {
        use BinaryOperator::*;

        assert_eq!(
            try_reduce_binary(BitOr, U32(0b1100), U32(0b1010)),
            Some(U32(0b1110))
        );
        assert_eq!(
            try_reduce_binary(BitOr, I32(0b1100), I32(0b1010)),
            Some(I32(0b1110))
        );

        let var = Variable(ValueInput::argument(TY_U32, 0));
        assert_eq!(try_reduce_binary(BitOr, var, U32(0)), Some(var));
        assert_eq!(try_reduce_binary(BitOr, U32(0), var), Some(var));
        let var = Variable(ValueInput::argument(TY_I32, 0));
        assert_eq!(try_reduce_binary(BitOr, var, I32(0)), Some(var));
        assert_eq!(try_reduce_binary(BitOr, I32(0), var), Some(var));
    }

    #[test]
    fn test_try_reduce_binary_bit_xor() {
        use BinaryOperator::*;

        assert_eq!(
            try_reduce_binary(BitXor, U32(0b1100), U32(0b1010)),
            Some(U32(0b0110))
        );
        assert_eq!(
            try_reduce_binary(BitXor, I32(0b1100), I32(0b1010)),
            Some(I32(0b0110))
        );

        let var = Variable(ValueInput::argument(TY_U32, 0));
        assert_eq!(try_reduce_binary(BitXor, var, U32(0)), Some(var));
        assert_eq!(try_reduce_binary(BitXor, U32(0), var), Some(var));
        let var = Variable(ValueInput::argument(TY_I32, 0));
        assert_eq!(try_reduce_binary(BitXor, var, I32(0)), Some(var));
        assert_eq!(try_reduce_binary(BitXor, I32(0), var), Some(var));
    }

    #[test]
    fn test_try_reduce_binary_shl() {
        use BinaryOperator::*;

        assert_eq!(try_reduce_binary(Shl, U32(1), U32(2)), Some(U32(4)));
        assert_eq!(try_reduce_binary(Shl, I32(1), I32(2)), Some(I32(4)));

        let var = Variable(ValueInput::argument(TY_U32, 0));
        assert_eq!(try_reduce_binary(Shl, var, U32(0)), Some(var));
        let var = Variable(ValueInput::argument(TY_I32, 0));
        assert_eq!(try_reduce_binary(Shl, var, I32(0)), Some(var));
        let var = Variable(ValueInput::argument(TY_U32, 0));
        assert_eq!(try_reduce_binary(Shl, U32(0), var), Some(U32(0)));
        let var = Variable(ValueInput::argument(TY_I32, 0));
        assert_eq!(try_reduce_binary(Shl, I32(0), var), Some(I32(0)));
    }

    #[test]
    fn test_try_reduce_binary_shr() {
        use BinaryOperator::*;

        assert_eq!(try_reduce_binary(Shr, U32(4), U32(2)), Some(U32(1)));
        assert_eq!(try_reduce_binary(Shr, I32(4), I32(2)), Some(I32(1)));

        let var = Variable(ValueInput::argument(TY_U32, 0));
        assert_eq!(try_reduce_binary(Shr, var, U32(0)), Some(var));
        let var = Variable(ValueInput::argument(TY_I32, 0));
        assert_eq!(try_reduce_binary(Shr, var, I32(0)), Some(var));
        let var = Variable(ValueInput::argument(TY_U32, 0));
        assert_eq!(try_reduce_binary(Shr, U32(0), var), Some(U32(0)));
        let var = Variable(ValueInput::argument(TY_I32, 0));
        assert_eq!(try_reduce_binary(Shr, I32(0), var), Some(I32(0)));
    }

    #[test]
    fn test_try_reduce_binary_lt() {
        use BinaryOperator::*;

        assert_eq!(try_reduce_binary(Lt, U32(1), U32(2)), Some(Bool(true)));
        assert_eq!(try_reduce_binary(Lt, I32(1), I32(2)), Some(Bool(true)));
        assert_eq!(try_reduce_binary(Lt, F32(1.0), F32(2.0)), Some(Bool(true)));

        let var = Variable(ValueInput::argument(TY_U32, 0));
        assert_eq!(try_reduce_binary(Lt, var, U32(0)), Some(Bool(false)));

        // F32 safety
        assert_eq!(try_reduce_binary(Lt, F32(1.0), F32(f32::INFINITY)), None);
    }

    #[test]
    fn test_try_reduce_binary_lt_eq() {
        use BinaryOperator::*;

        assert_eq!(try_reduce_binary(LtEq, U32(1), U32(2)), Some(Bool(true)));
        assert_eq!(try_reduce_binary(LtEq, U32(2), U32(2)), Some(Bool(true)));

        let var = Variable(ValueInput::argument(TY_U32, 0));
        assert_eq!(try_reduce_binary(LtEq, U32(0), var), Some(Bool(true)));

        // F32 safety
        assert_eq!(try_reduce_binary(LtEq, F32(1.0), F32(f32::NAN)), None);
    }

    #[test]
    fn test_try_reduce_binary_gt() {
        use BinaryOperator::*;

        assert_eq!(try_reduce_binary(Gt, U32(2), U32(1)), Some(Bool(true)));

        let var = Variable(ValueInput::argument(TY_U32, 0));
        assert_eq!(try_reduce_binary(Gt, U32(0), var), Some(Bool(false)));

        // F32 safety
        assert_eq!(try_reduce_binary(Gt, F32(f32::INFINITY), F32(1.0)), None);
    }

    #[test]
    fn test_try_reduce_binary_gt_eq() {
        use BinaryOperator::*;

        assert_eq!(try_reduce_binary(GtEq, U32(2), U32(1)), Some(Bool(true)));
        assert_eq!(try_reduce_binary(GtEq, U32(2), U32(2)), Some(Bool(true)));

        let var = Variable(ValueInput::argument(TY_U32, 0));
        assert_eq!(try_reduce_binary(GtEq, var, U32(0)), Some(Bool(true)));

        // F32 safety
        assert_eq!(
            try_reduce_binary(GtEq, F32(1.0), F32(f32::MIN_POSITIVE / 2.0)),
            None
        );
    }

    #[test]
    fn test_try_reduce_binary_eq() {
        use BinaryOperator::*;

        assert_eq!(try_reduce_binary(Eq, U32(1), U32(1)), Some(Bool(true)));
        assert_eq!(
            try_reduce_binary(Eq, Bool(true), Bool(true)),
            Some(Bool(true))
        );

        // F32 safety
        assert_eq!(try_reduce_binary(Eq, F32(1.0), F32(f32::NAN)), None);
    }

    #[test]
    fn test_try_reduce_binary_not_eq() {
        use BinaryOperator::*;

        assert_eq!(try_reduce_binary(NotEq, U32(1), U32(2)), Some(Bool(true)));
        assert_eq!(
            try_reduce_binary(NotEq, Bool(true), Bool(false)),
            Some(Bool(true))
        );

        // F32 safety
        assert_eq!(try_reduce_binary(NotEq, F32(f32::NAN), F32(1.0)), None);
    }

    #[test]
    fn test_try_reduce_binary_and() {
        use BinaryOperator::*;

        assert_eq!(
            try_reduce_binary(And, Bool(true), Bool(false)),
            Some(Bool(false))
        );

        let var = Variable(ValueInput::argument(TY_BOOL, 0));
        assert_eq!(try_reduce_binary(And, var, Bool(true)), Some(var));
        assert_eq!(try_reduce_binary(And, Bool(true), var), Some(var));
        assert_eq!(try_reduce_binary(And, var, Bool(false)), Some(Bool(false)));
        assert_eq!(try_reduce_binary(And, Bool(false), var), Some(Bool(false)));
    }

    #[test]
    fn test_try_reduce_binary_or() {
        use BinaryOperator::*;

        assert_eq!(
            try_reduce_binary(Or, Bool(true), Bool(false)),
            Some(Bool(true))
        );

        let var = Variable(ValueInput::argument(TY_BOOL, 0));
        assert_eq!(try_reduce_binary(Or, var, Bool(true)), Some(Bool(true)));
        assert_eq!(try_reduce_binary(Or, Bool(true), var), Some(Bool(true)));
        assert_eq!(try_reduce_binary(Or, var, Bool(false)), Some(var));
        assert_eq!(try_reduce_binary(Or, Bool(false), var), Some(var));
    }

    #[test]
    fn test_try_reduce_unary() {
        use UnaryOperator::*;

        assert_eq!(try_reduce_unary(Not, Bool(true)), Some(Bool(false)));
        assert_eq!(try_reduce_unary(Neg, I32(5)), Some(I32(-5)));
        assert_eq!(try_reduce_unary(Neg, F32(1.0)), Some(F32(-1.0)));
        assert_eq!(try_reduce_unary(Neg, F32(f32::NAN)), None);
    }

    #[test]
    fn test_try_reduce_convert_to_u32() {
        assert_eq!(try_reduce_convert_to_u32(U32(1)), Some(U32(1)));
        assert_eq!(try_reduce_convert_to_u32(I32(1)), Some(U32(1)));
        assert_eq!(try_reduce_convert_to_u32(F32(1.0)), Some(U32(1)));
        assert_eq!(try_reduce_convert_to_u32(Bool(true)), Some(U32(1)));

        let var_u32 = Variable(ValueInput::argument(TY_U32, 0));
        let var_i32 = Variable(ValueInput::argument(TY_I32, 0));
        assert_eq!(try_reduce_convert_to_u32(var_u32), Some(var_u32));
        assert_eq!(try_reduce_convert_to_u32(var_i32), None);
    }

    #[test]
    fn test_try_reduce_convert_to_i32() {
        assert_eq!(try_reduce_convert_to_i32(U32(1)), Some(I32(1)));
        assert_eq!(try_reduce_convert_to_i32(I32(1)), Some(I32(1)));
        assert_eq!(try_reduce_convert_to_i32(F32(1.0)), Some(I32(1)));
        assert_eq!(try_reduce_convert_to_i32(Bool(true)), Some(I32(1)));

        let var_i32 = Variable(ValueInput::argument(TY_I32, 0));
        let var_u32 = Variable(ValueInput::argument(TY_U32, 0));
        assert_eq!(try_reduce_convert_to_i32(var_i32), Some(var_i32));
        assert_eq!(try_reduce_convert_to_i32(var_u32), None);
    }

    #[test]
    fn test_try_reduce_convert_to_f32() {
        assert_eq!(try_reduce_convert_to_f32(U32(1)), Some(F32(1.0)));
        assert_eq!(try_reduce_convert_to_f32(I32(1)), Some(F32(1.0)));
        assert_eq!(try_reduce_convert_to_f32(F32(1.0)), Some(F32(1.0)));
        assert_eq!(try_reduce_convert_to_f32(Bool(true)), Some(F32(1.0)));

        let var_f32 = Variable(ValueInput::argument(TY_F32, 0));
        let var_u32 = Variable(ValueInput::argument(TY_U32, 0));
        assert_eq!(try_reduce_convert_to_f32(var_f32), Some(var_f32));
        assert_eq!(try_reduce_convert_to_f32(var_u32), None);
    }

    #[test]
    fn test_try_reduce_convert_to_bool() {
        assert_eq!(try_reduce_convert_to_bool(U32(1)), Some(Bool(true)));
        assert_eq!(try_reduce_convert_to_bool(U32(0)), Some(Bool(false)));
        assert_eq!(try_reduce_convert_to_bool(I32(1)), Some(Bool(true)));
        assert_eq!(try_reduce_convert_to_bool(I32(0)), Some(Bool(false)));
        assert_eq!(try_reduce_convert_to_bool(F32(1.0)), Some(Bool(true)));
        assert_eq!(try_reduce_convert_to_bool(F32(0.0)), Some(Bool(false)));
        assert_eq!(try_reduce_convert_to_bool(Bool(true)), Some(Bool(true)));

        let var_bool = Variable(ValueInput::argument(TY_BOOL, 0));
        let var_u32 = Variable(ValueInput::argument(TY_U32, 0));
        assert_eq!(try_reduce_convert_to_bool(var_bool), Some(var_bool));
        assert_eq!(try_reduce_convert_to_bool(var_u32), None);
    }

    #[test]
    fn test_try_reduce_op_case_to_branch_selector() {
        use crate::ty::Int;

        let cases = [10u128, 20, 30];

        // Match case 0
        assert_eq!(
            try_reduce_op_case_to_branch_selector(Int::U32, &cases, U32(10)),
            Some(Predicate(0))
        );
        // Match case 1
        assert_eq!(
            try_reduce_op_case_to_branch_selector(Int::I32, &cases, I32(20)),
            Some(Predicate(1))
        );
        // No match -> default branch (cases.len())
        assert_eq!(
            try_reduce_op_case_to_branch_selector(Int::U32, &cases, U32(40)),
            Some(Predicate(3))
        );

        // I32
        let cases_neg = [(-1i32 as u32 as u128)];
        assert_eq!(
            try_reduce_op_case_to_branch_selector(Int::I32, &cases_neg, I32(-1)),
            Some(Predicate(0))
        );
    }
}
