use std::ops::Deref;

use slotmap::{SecondaryMap, SlotMap};

use crate::scf::visit::TopDownVisitor;
use crate::scf::{
    Block, ExprBinding, ExpressionKind, If, LocalBinding, LocalBindingData, LocalBindingKind, Loop,
    LoopControl, OpCallBuiltin, Return, Scf, Statement, StatementKind, Store, Switch, visit,
};

pub struct Config {
    pub count_const_index_use: bool,
}

fn is_const_u32(scf: &Scf, local_binding: LocalBinding) -> bool {
    if let LocalBindingKind::ExprBinding(stmt) = scf[local_binding].kind()
        && let StatementKind::ExprBinding(stmt) = scf[*stmt].kind()
    {
        stmt.expression().kind().is_const_u32()
    } else {
        false
    }
}

struct UseCounter {
    count: SecondaryMap<LocalBinding, u32>,
    config: Config,
}

impl UseCounter {
    fn new(local_bindings: &SlotMap<LocalBinding, LocalBindingData>, config: Config) -> Self {
        let mut count = SecondaryMap::with_capacity(local_bindings.capacity());

        for local_binding in local_bindings.keys() {
            count.insert(local_binding, 0);
        }

        UseCounter { count, config }
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

    fn count_store(&mut self, store_stmt: &Store) {
        self.increment(store_stmt.pointer());
        self.increment(store_stmt.value());
    }

    fn count_expr_binding(&mut self, scf: &Scf, stmt: &ExprBinding) {
        match stmt.expression().kind() {
            ExpressionKind::FallbackValue
            | ExpressionKind::ConstU32(_)
            | ExpressionKind::ConstI32(_)
            | ExpressionKind::ConstF32(_)
            | ExpressionKind::ConstBool(_)
            | ExpressionKind::GlobalPtr(_) => {}
            ExpressionKind::OpUnary(op) => self.increment(op.operand()),
            ExpressionKind::OpBinary(op) => {
                self.increment(op.lhs());
                self.increment(op.rhs());
            }
            ExpressionKind::OpVector(op) => {
                for element in op.elements() {
                    self.increment(*element);
                }
            }
            ExpressionKind::OpMatrix(op) => {
                for column in op.columns() {
                    self.increment(*column);
                }
            }
            ExpressionKind::OpConvertToU32(op) => self.increment(op.value()),
            ExpressionKind::OpConvertToI32(op) => self.increment(op.value()),
            ExpressionKind::OpConvertToF32(op) => self.increment(op.value()),
            ExpressionKind::OpConvertToBool(op) => self.increment(op.value()),
            ExpressionKind::OpPtrElementPtr(op) => {
                self.increment(op.pointer());

                for index in op.indices() {
                    if self.config.count_const_index_use || !is_const_u32(scf, *index) {
                        self.increment(*index);
                    }
                }
            }
            ExpressionKind::OpExtractElement(op) => {
                self.increment(op.value());

                for index in op.indices() {
                    if self.config.count_const_index_use || !is_const_u32(scf, *index) {
                        self.increment(*index);
                    }
                }
            }
            ExpressionKind::OpLoad(ptr) => self.increment(*ptr),
            ExpressionKind::OpCallBuiltin(op) => self.count_call_builtin(op),
        }
    }

    fn count_call_builtin(&mut self, call_builtin: &OpCallBuiltin) {
        for arg in call_builtin.arguments() {
            self.increment(*arg);
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
            StatementKind::Store(stmt) => self.counter.count_store(stmt),
            StatementKind::CallBuiltin(stmt) => self.counter.count_call_builtin(stmt),
        }

        visit::visit_statement_top_down(self, scf, statement);
    }
}

pub fn count_local_binding_use(scf: &Scf, config: Config) -> SecondaryMap<LocalBinding, u32> {
    let counter = UseCounter::new(scf.local_bindings(), config);
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
