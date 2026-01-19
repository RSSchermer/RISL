use std::ops::{Deref, DerefMut};

use indexmap::IndexSet;
use rustc_hash::{FxBuildHasher, FxHashMap, FxHashSet};
use slotmap::SlotMap;

use crate::Function;
use crate::cfg::{
    BasicBlock, BasicBlockData, BlockPosition, Cfg, FunctionBody, LocalBinding, LocalBindingData,
    Statement, StatementData, Terminator,
};
use crate::cfg_to_rvsdg::control_flow_restructuring::exit_restructuring::restructure_exit;
use crate::ty::Type;

fn inverse_graph(cfg: &Cfg, function: Function) -> FxHashMap<BasicBlock, Vec<BasicBlock>> {
    let body = cfg
        .get_function_body(function)
        .expect("function not registered");
    let bb_count = body.basic_blocks().len();

    let mut inverse: FxHashMap<BasicBlock, Vec<BasicBlock>> =
        FxHashMap::with_capacity_and_hasher(bb_count, FxBuildHasher::default());

    for source in body.basic_blocks() {
        if let Terminator::Branch(b) = cfg[*source].terminator() {
            for dest in b.targets() {
                inverse.entry(*dest).or_default().push(*source);
            }
        }
    }

    inverse
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Edge {
    pub source: BasicBlock,
    pub dest: BasicBlock,
}

#[derive(Debug)]
pub struct Graph<'a> {
    cfg: &'a mut Cfg,
    function: Function,
    inverse_graph: FxHashMap<BasicBlock, Vec<BasicBlock>>,
    entry: BasicBlock,
    exit: BasicBlock,
}

impl<'a> Graph<'a> {
    pub fn init(cfg: &'a mut Cfg, function: Function) -> Self {
        let entry = cfg[function].entry_block();
        let exit = restructure_exit(cfg, function);

        let inverse_graph = inverse_graph(&cfg, function);

        Graph {
            cfg,
            function,
            inverse_graph,
            entry,
            exit,
        }
    }

    pub fn entry(&self) -> BasicBlock {
        self.entry
    }

    pub fn exit(&self) -> BasicBlock {
        self.exit
    }

    pub fn nodes(&self) -> impl Iterator<Item = BasicBlock> + use<'_> {
        self.cfg[self.function].basic_blocks().iter().copied()
    }

    pub fn append_block_branch_single(&mut self, target: BasicBlock) -> BasicBlock {
        let bb = self.cfg.add_basic_block(self.function);

        self.cfg
            .set_terminator(bb, Terminator::branch_single(target));
        self.inverse_graph.entry(target).or_default().push(bb);

        bb
    }

    pub fn append_block_branch_multiple(&mut self, predicate: LocalBinding) -> BasicBlock {
        let bb = self.cfg.add_basic_block(self.function);

        self.cfg
            .set_terminator(bb, Terminator::branch_multiple(predicate, []));

        bb
    }

    pub fn statements(&self, bb: BasicBlock) -> &IndexSet<Statement> {
        self.cfg[bb].statements()
    }

    pub fn add_value(&mut self, ty: Type) -> LocalBinding {
        self.cfg
            .add_stmt_uninitialized(self.entry, BlockPosition::Append, ty)
            .1
    }

    pub fn selector(&self, bb: BasicBlock) -> Option<LocalBinding> {
        self.cfg[bb].terminator().expect_branch().selector()
    }

    pub fn set_selector(&mut self, bb: BasicBlock, selector: LocalBinding) {
        self.cfg.set_branch_selector(bb, selector);
    }

    pub fn connect(&mut self, edge: Edge) {
        let Edge { source, dest } = edge;

        self.cfg.add_branch_target(source, dest);
        self.inverse_graph.entry(dest).or_default().push(source);
    }

    pub fn reconnect_dest(&mut self, edge: Edge, new_dest: BasicBlock) {
        let Edge { source, dest } = edge;

        // First point the out-edge from the source to the new destination
        let mut src_out_found = self.cfg.replace_branch_target(source, dest, new_dest);
        assert!(src_out_found, "edge source does not connect to edge dest");

        // Then disconnect the in-edge from the old destination
        let mut dest_in_found = false;
        let dst_in = self.inverse_graph.get_mut(&dest).unwrap();
        for i in 0..dst_in.len() {
            if dst_in[i] == source {
                dst_in.remove(i);
                dest_in_found = true;

                break;
            }
        }
        assert!(dest_in_found, "edge dest does not connect to edge source");

        // Finally, add an in-edge to the new destination
        self.inverse_graph.entry(new_dest).or_default().push(source);
    }

    pub fn divert(&mut self, edge: Edge, via: BasicBlock) {
        let Edge { source, dest } = edge;

        // First point the out-edge from the source to the "via" node

        let mut src_out_found = self.cfg.replace_branch_target(source, dest, via);

        assert!(src_out_found, "edge source does not connect to edge dest");

        // Then point the in-edge from the destination to the "via" node

        let mut dest_in_found = false;

        for bb in self.inverse_graph.entry(dest).or_default() {
            if *bb == source {
                *bb = via;
                dest_in_found = true;
            }
        }

        assert!(dest_in_found, "edge dest does not connect to edge source");

        // Finally, add an in-edge and out-edge to the "via" node for the source and destination
        // respectively
        self.inverse_graph.entry(via).or_default().push(source);
        self.cfg.add_branch_target(via, dest);
    }

    pub fn parents(&self, bb: BasicBlock) -> &[BasicBlock] {
        self.inverse_graph
            .get(&bb)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    pub fn children(&self, bb: BasicBlock) -> &[BasicBlock] {
        match self.cfg[bb].terminator() {
            Terminator::Branch(b) => b.targets(),
            Terminator::Return(_) => &[],
        }
    }

    pub fn branches_to_self(&self, bb: BasicBlock, edge_blacklist: &FxHashSet<Edge>) -> bool {
        self.children(bb).iter().any(|child| {
            *child == bb
                && !edge_blacklist.contains(&Edge {
                    source: bb,
                    dest: bb,
                })
        })
    }
}

impl Deref for Graph<'_> {
    type Target = Cfg;

    fn deref(&self) -> &Self::Target {
        &self.cfg
    }
}

impl DerefMut for Graph<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.cfg
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg::Terminator;
    use crate::ty::{TY_BOOL, TY_DUMMY};
    use crate::{FnArg, FnSig, Module, Symbol};

    #[test]
    fn test_connect() {
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
                    ty: TY_BOOL,
                    shader_io_binding: None,
                }],
                ret_ty: None,
            },
        );

        let mut cfg = Cfg::new(module.ty.clone());

        let body = cfg.register_function(&module, function);

        let arg_0 = body.argument_values()[0];

        let entry = cfg.add_basic_block(function);
        let bb0 = cfg.add_basic_block(function);
        let bb1 = cfg.add_basic_block(function);
        let bb2 = cfg.add_basic_block(function);
        let bb3 = cfg.add_basic_block(function);
        let exit = cfg.add_basic_block(function);

        let (_, selector) =
            cfg.add_stmt_op_bool_to_branch_selector(entry, BlockPosition::Append, arg_0.into());

        cfg.set_terminator(entry, Terminator::branch_multiple(selector, [bb0, bb1]));
        cfg.set_terminator(bb0, Terminator::branch_single(bb2));
        cfg.set_terminator(bb1, Terminator::branch_single(bb3));
        cfg.set_terminator(bb2, Terminator::branch_single(exit));
        cfg.set_terminator(bb3, Terminator::branch_single(exit));

        let mut graph = Graph::init(&mut cfg, function);

        graph.connect(Edge {
            source: bb0,
            dest: bb3,
        });

        assert_eq!(graph.children(bb0), &[bb2, bb3]);
        assert_eq!(graph.children(bb1), &[bb3]);
        assert_eq!(graph.children(bb2), &[exit]);
        assert_eq!(graph.children(bb3), &[exit]);

        assert_eq!(graph.parents(bb0), &[entry]);
        assert_eq!(graph.parents(bb1), &[entry]);
        assert_eq!(graph.parents(bb2), &[bb0]);
        assert_eq!(graph.parents(bb3), &[bb1, bb0]);
    }

    #[test]
    fn test_reconnect_dest() {
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
                    ty: TY_BOOL,
                    shader_io_binding: None,
                }],
                ret_ty: None,
            },
        );

        let mut cfg = Cfg::new(module.ty.clone());

        let body = cfg.register_function(&module, function);

        let arg_0 = body.argument_values()[0];

        let entry = cfg.add_basic_block(function);
        let bb0 = cfg.add_basic_block(function);
        let bb1 = cfg.add_basic_block(function);
        let bb2 = cfg.add_basic_block(function);
        let bb3 = cfg.add_basic_block(function);
        let exit = cfg.add_basic_block(function);

        let (_, selector) =
            cfg.add_stmt_op_bool_to_branch_selector(entry, BlockPosition::Append, arg_0.into());

        cfg.set_terminator(entry, Terminator::branch_multiple(selector, [bb0, bb1]));
        cfg.set_terminator(bb0, Terminator::branch_single(bb2));
        cfg.set_terminator(bb1, Terminator::branch_single(bb3));
        cfg.set_terminator(bb2, Terminator::branch_single(exit));
        cfg.set_terminator(bb3, Terminator::branch_single(exit));

        let mut graph = Graph::init(&mut cfg, function);

        graph.reconnect_dest(
            Edge {
                source: bb0,
                dest: bb2,
            },
            bb3,
        );

        assert_eq!(graph.children(bb0), &[bb3]);
        assert_eq!(graph.children(bb1), &[bb3]);
        assert_eq!(graph.children(bb2), &[exit]);
        assert_eq!(graph.children(bb3), &[exit]);

        assert_eq!(graph.parents(bb0), &[entry]);
        assert_eq!(graph.parents(bb1), &[entry]);
        assert_eq!(graph.parents(bb2), &[]);
        assert_eq!(graph.parents(bb3), &[bb1, bb0]);
    }
}
