use crate::cfg::{BasicBlock, Cfg, Statement, Terminator};

pub trait Visitor: Sized {
    fn should_visit(&mut self, cfg: &Cfg, bb: BasicBlock) -> bool;

    fn visit_basic_block(&mut self, cfg: &Cfg, bb: BasicBlock) {
        visit_basic_block(self, cfg, bb);
    }

    #[allow(unused)]
    fn visit_statement(&mut self, cfg: &Cfg, statement: Statement) {}
}

pub fn visit_basic_block<V: Visitor>(visitor: &mut V, cfg: &Cfg, bb: BasicBlock) {
    if visitor.should_visit(cfg, bb) {
        for stmt in cfg[bb].statements() {
            visitor.visit_statement(cfg, *stmt);
        }

        match cfg[bb].terminator() {
            Terminator::Branch(b) => {
                for target in b.targets() {
                    visitor.visit_basic_block(cfg, *target);
                }
            }
            Terminator::Return(_) | Terminator::Unreachable => {}
        }
    }
}
