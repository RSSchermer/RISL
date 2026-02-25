use std::ops::Deref;

use slotmap::{SecondaryMap, SlotMap};

use crate::scf::visit::TopDownVisitor;
use crate::scf::{
    Block, ExprBinding, ExpressionKind, If, IntrinsicOp, LocalBinding, LocalBindingData,
    Loop, LoopControl, Return, Scf, Statement, StatementKind, Switch,
    visit,
};

struct UseCounter {
    count: SecondaryMap<LocalBinding, u32>,
}

impl UseCounter {
    fn new(local_bindings: &SlotMap<LocalBinding, LocalBindingData>) -> Self {
        let mut count = SecondaryMap::with_capacity(local_bindings.capacity());

        for local_binding in local_bindings.keys() {
            count.insert(local_binding, 0);
        }

        UseCounter { count }
    }

    fn increment(&mut self, local_binding: LocalBinding) {
        self.count[local_binding] += 1;
    }

    fn count_if(&mut self, if_stmt: &If) {
        self.increment(if_stmt.condition());
    }

    fn count_switch(&mut self, switch_stmt: &Switch) {
        self.increment(switch_stmt.on());
    }

    fn count_loop(&mut self, loop_stmt: &Loop) {
        match loop_stmt.control() {
            LoopControl::Head(condition) => self.increment(condition),
            LoopControl::Tail(condition) => self.increment(condition),
            LoopControl::Infinite => {}
        }
    }

    fn count_return(&mut self, return_stmt: &Return) {
        if let Some(return_value) = return_stmt.value() {
            self.increment(return_value);
        }
    }

    fn count_intrinsic_op_binding<T>(&mut self, intrinsic_op: &IntrinsicOp<T>) {
        for arg in intrinsic_op.arguments() {
            self.increment(*arg);
        }
    }

    fn count_expr_binding(&mut self, _scf: &Scf, stmt: &ExprBinding) {
        match stmt.expression().kind() {
            ExpressionKind::FallbackValue
            | ExpressionKind::ConstU32(_)
            | ExpressionKind::ConstI32(_)
            | ExpressionKind::ConstF32(_)
            | ExpressionKind::ConstBool(_)
            | ExpressionKind::GlobalPtr(_) => {}
            ExpressionKind::OpUnary(op) => self.count_intrinsic_op_binding(op),
            ExpressionKind::OpBinary(op) => self.count_intrinsic_op_binding(op),
            ExpressionKind::OpVector(op) => self.count_intrinsic_op_binding(op),
            ExpressionKind::OpMatrix(op) => self.count_intrinsic_op_binding(op),
            ExpressionKind::OpConvertToU32(op) => self.count_intrinsic_op_binding(op),
            ExpressionKind::OpConvertToI32(op) => self.count_intrinsic_op_binding(op),
            ExpressionKind::OpConvertToF32(op) => self.count_intrinsic_op_binding(op),
            ExpressionKind::OpConvertToBool(op) => self.count_intrinsic_op_binding(op),
            ExpressionKind::OpFieldPtr(op) => self.count_intrinsic_op_binding(op),
            ExpressionKind::OpElementPtr(op) => self.count_intrinsic_op_binding(op),
            ExpressionKind::OpExtractField(op) => self.count_intrinsic_op_binding(op),
            ExpressionKind::OpExtractElement(op) => self.count_intrinsic_op_binding(op),
            ExpressionKind::OpLoad(op) => self.count_intrinsic_op_binding(op),
            ExpressionKind::OpArrayLength(op) => self.count_intrinsic_op_binding(op),
        }
    }
}

impl Deref for UseCounter {
    type Target = SecondaryMap<LocalBinding, u32>;

    fn deref(&self) -> &Self::Target {
        &self.count
    }
}

struct Visitor {
    counter: UseCounter,
}

impl TopDownVisitor for Visitor {
    fn visit_block(&mut self, scf: &Scf, block: Block) {
        for (_, value) in scf[block].control_flow_var_iter() {
            self.counter.increment(value)
        }

        visit::visit_block_top_down(self, scf, block);
    }

    fn visit_statement(&mut self, scf: &Scf, statement: Statement) {
        match scf[statement].kind() {
            StatementKind::If(stmt) => self.counter.count_if(stmt),
            StatementKind::Switch(stmt) => self.counter.count_switch(stmt),
            StatementKind::Loop(stmt) => self.counter.count_loop(stmt),
            StatementKind::Return(stmt) => self.counter.count_return(stmt),
            StatementKind::Alloca(_) => (),
            StatementKind::ExprBinding(stmt) => self.counter.count_expr_binding(scf, stmt),
            StatementKind::OpStore(stmt) => self.counter.count_intrinsic_op_binding(stmt),
        }

        visit::visit_statement_top_down(self, scf, statement);
    }
}

pub fn count_local_binding_use(scf: &Scf) -> SecondaryMap<LocalBinding, u32> {
    let counter = UseCounter::new(scf.local_bindings());
    let mut visitor = Visitor { counter };

    for function in scf.registered_functions() {
        let body = scf
            .get_function_body(function)
            .expect("function not registered");
        let body_block = body.block();

        visitor.visit_block(scf, body_block)
    }

    visitor.counter.count
}
