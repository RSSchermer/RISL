use std::ops::Index;

use index_vec::IndexVec;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::cfg::{BasicBlock, LocalBinding};
use crate::cfg_to_rvsdg::control_flow_restructuring::{Edge, Graph};

index_vec::define_index_type! {
    pub struct ControlTreeNode = u32;
}

#[derive(Clone, PartialEq, Default, Debug)]
pub struct LinearNode {
    pub children: Vec<ControlTreeNode>,
}

#[derive(Clone, PartialEq, Debug)]
pub struct BranchingNode {
    pub selector: LocalBinding,
    pub branches: Vec<ControlTreeNode>,
}

#[derive(Clone, PartialEq, Debug)]
pub struct LoopNode {
    pub reentry_predicate: LocalBinding,
    pub inner: ControlTreeNode,
}

#[derive(Clone, PartialEq, Debug)]
pub enum ControlTreeNodeKind {
    BasicBlock(BasicBlock),
    Linear(LinearNode),
    Branching(BranchingNode),
    Loop(LoopNode),
}

impl ControlTreeNodeKind {
    pub fn expect_linear(&self) -> &LinearNode {
        if let ControlTreeNodeKind::Linear(n) = self {
            n
        } else {
            panic!("expected linear node")
        }
    }

    pub fn expect_linear_mut(&mut self) -> &mut LinearNode {
        if let ControlTreeNodeKind::Linear(n) = self {
            n
        } else {
            panic!("expected linear node")
        }
    }

    pub fn expect_basic_block(&self) -> BasicBlock {
        if let ControlTreeNodeKind::BasicBlock(bb) = self {
            *bb
        } else {
            panic!("expected basic block")
        }
    }

    pub fn expect_branching(&self) -> &BranchingNode {
        if let ControlTreeNodeKind::Branching(n) = self {
            n
        } else {
            panic!("expected branching node")
        }
    }

    pub fn expect_loop(&self) -> &LoopNode {
        if let ControlTreeNodeKind::Loop(n) = self {
            n
        } else {
            panic!("expected loop node")
        }
    }
}

/// Turns the set of re-entry edges produced by loop-restructuring a "loop info" map.
///
/// If a basic-block is the entry block for a looping region, it will have an entry in this map
/// where the value is the loop's tail block (the basic-block that will branch either out of the
/// loop, or back to the loop's entry block). Note that the entry block and the tail block may be
/// the same block.
fn reentry_edges_to_loop_info(
    reentry_edges: &FxHashSet<Edge>,
) -> FxHashMap<BasicBlock, BasicBlock> {
    let mut loop_info = FxHashMap::default();

    for edge in reentry_edges {
        loop_info.insert(edge.dest, edge.source);
    }

    loop_info
}

struct ControlTreeGenerator<'a> {
    graph: &'a Graph<'a>,
    branch_info: &'a FxHashMap<BasicBlock, BasicBlock>,
    loop_info: FxHashMap<BasicBlock, BasicBlock>,
    nodes: IndexVec<ControlTreeNode, ControlTreeNodeKind>,
    root: ControlTreeNode,
}

impl<'a> ControlTreeGenerator<'a> {
    fn new(
        graph: &'a Graph<'a>,
        branch_info: &'a FxHashMap<BasicBlock, BasicBlock>,
        reentry_edges: &FxHashSet<Edge>,
    ) -> Self {
        let loop_info = reentry_edges_to_loop_info(reentry_edges);
        let mut nodes = IndexVec::new();
        let root = nodes.push(ControlTreeNodeKind::Linear(Default::default()));

        ControlTreeGenerator {
            graph,
            branch_info,
            loop_info,
            nodes,
            root,
        }
    }

    fn visit(&mut self, bb: BasicBlock, parent: Option<ControlTreeNode>, end: Option<BasicBlock>) {
        let parent = parent.unwrap_or(self.root);

        if Some(bb) == end {
            return;
        }

        if let Some(loop_tail) = self.loop_info.remove(&bb) {
            let reentry_predicate = self
                .graph
                .selector(loop_tail)
                .expect("loop tail should have a reentry predicate selector");

            let child_node = self
                .nodes
                .push(ControlTreeNodeKind::Linear(Default::default()));
            let loop_node = self.nodes.push(ControlTreeNodeKind::Loop(LoopNode {
                reentry_predicate,
                inner: child_node,
            }));

            self.nodes[parent]
                .expect_linear_mut()
                .children
                .push(loop_node);

            // Note that above, we `remove` the `loop_info` entry rather than just `get` it. We want to
            // further classify the loop's entry block (it may be e.g. a branching block also); if it
            // had not been removed from the `loop_info`, it would end up being classified as a loop
            // entry block again, and we would trigger infinite recursion.
            self.visit(bb, Some(child_node), Some(loop_tail));

            // In the above `visit` call, we passed the `loop_tail` as the visit's `end`, which
            // means that the `loop_tail` itself will not have been added yet (even if this is a
            // single node loop where `bb == loop_tail`), so we add it now.
            let tail_node = self.nodes.push(ControlTreeNodeKind::BasicBlock(loop_tail));

            self.nodes[child_node]
                .expect_linear_mut()
                .children
                .push(tail_node);

            let continuation = self.graph.children(loop_tail)[0];

            self.visit(continuation, Some(parent), end);
        } else if let Some(branch_exit) = self.branch_info.get(&bb).copied() {
            let selector = self
                .graph
                .selector(bb)
                .expect("branching node must have a selector");
            let mut branches = Vec::with_capacity(self.graph.children(bb).len());

            for child in self.graph.children(bb) {
                let branch_node = self
                    .nodes
                    .push(ControlTreeNodeKind::Linear(Default::default()));

                branches.push(branch_node);

                self.visit(*child, Some(branch_node), Some(branch_exit));
            }

            // We insert 2 nodes for every branching block: a basic-block node that contains the
            // statements and the actual branching terminator, and the actual branching node that only
            // holds the branches. That means that to resolve the branching predicate, a branching node
            // must look to its previous sibling.

            let bb_node = self.nodes.push(ControlTreeNodeKind::BasicBlock(bb));
            let branching_node = self
                .nodes
                .push(ControlTreeNodeKind::Branching(BranchingNode {
                    selector,
                    branches,
                }));

            let p = self.nodes[parent].expect_linear_mut();

            p.children.push(bb_node);
            p.children.push(branching_node);

            self.visit(branch_exit, Some(parent), end);
        } else {
            let node = self.nodes.push(ControlTreeNodeKind::BasicBlock(bb));

            self.nodes[parent].expect_linear_mut().children.push(node);

            let children = self.graph.children(bb);

            assert!(children.len() <= 1);

            for child in children {
                self.visit(*child, Some(parent), end);
            }
        }
    }

    fn into_control_tree(self) -> ControlTree {
        ControlTree {
            nodes: self.nodes,
            root: self.root,
        }
    }
}

#[derive(Clone, PartialEq, Debug)]
pub struct ControlTree {
    nodes: IndexVec<ControlTreeNode, ControlTreeNodeKind>,
    root: ControlTreeNode,
}

impl ControlTree {
    pub fn generate(
        graph: &Graph,
        reentry_edges: &FxHashSet<Edge>,
        branch_info: &FxHashMap<BasicBlock, BasicBlock>,
    ) -> Self {
        let mut generator = ControlTreeGenerator::new(graph, branch_info, reentry_edges);

        generator.visit(graph.entry(), None, None);
        generator.into_control_tree()
    }

    pub fn root(&self) -> ControlTreeNode {
        self.root
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
}

impl Index<ControlTreeNode> for ControlTree {
    type Output = ControlTreeNodeKind;

    fn index(&self, node: ControlTreeNode) -> &Self::Output {
        &self.nodes[node]
    }
}

#[cfg(test)]
mod tests {
    use smallvec::smallvec;

    use super::*;
    use crate::cfg::{
        Assign, BlockPosition, Branch, Cfg, FunctionBody, InlineConst, LocalBindingData, OpBinary,
        StatementData, Terminator, Value,
    };
    use crate::cfg_to_rvsdg::control_flow_restructuring::{
        restructure_branches, restructure_loops,
    };
    use crate::ty::{TY_BOOL, TY_DUMMY, TY_U32};
    use crate::{BinaryOperator, FnArg, FnSig, Function, Module, Symbol};

    #[test]
    fn test_control_tree_generate() {
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

        let mut cfg = Cfg::new(module.ty.clone());

        let body = cfg.register_function(&module, function);

        let x = body.argument_values()[0];
        let y = body.argument_values()[1];

        let enter = body.entry_block();
        let bb0 = cfg.add_basic_block(function);
        let bb1 = cfg.add_basic_block(function);
        let bb2 = cfg.add_basic_block(function);
        let bb3 = cfg.add_basic_block(function);
        let bb4 = cfg.add_basic_block(function);
        let exit = cfg.add_basic_block(function);

        let (_, r) = cfg.add_stmt_uninitialized(enter, BlockPosition::Append, TY_U32);
        cfg.set_terminator(enter, Terminator::branch_single(bb0));

        let (_, c) = cfg.add_stmt_op_binary(
            bb0,
            BlockPosition::Append,
            BinaryOperator::NotEq,
            y.into(),
            0u32.into(),
        );
        cfg.set_terminator(bb0, Terminator::branch_multiple(c, [bb1, bb2]));

        cfg.add_stmt_assign(bb1, BlockPosition::Append, r, 0u32.into());
        cfg.set_terminator(bb1, Terminator::branch_single(bb4));

        let (_, t) = cfg.add_stmt_bind(bb2, BlockPosition::Append, y.into());
        let (_, mod_result) = cfg.add_stmt_op_binary(
            bb2,
            BlockPosition::Append,
            BinaryOperator::Mod,
            x.into(),
            y.into(),
        );
        cfg.add_stmt_assign(bb2, BlockPosition::Append, y, mod_result.into());
        cfg.add_stmt_assign(bb2, BlockPosition::Append, x, t.into());
        cfg.set_terminator(bb2, Terminator::branch_single(bb3));

        cfg.add_stmt_assign(bb3, BlockPosition::Append, r, 1u32.into());
        cfg.set_terminator(bb3, Terminator::branch_single(bb4));

        cfg.set_terminator(bb4, Terminator::branch_multiple(r, [exit, bb0]));

        cfg.set_terminator(exit, Terminator::Return(Some(x.into())));

        let mut graph = Graph::init(&mut cfg, function);
        let reentry_edges = restructure_loops(&mut graph);
        let branch_info = restructure_branches(&mut graph, &reentry_edges);

        let control_tree = ControlTree::generate(&graph, &reentry_edges, &branch_info);

        // Expected control-tree layout:
        //
        //                      Linear (A)
        //                          |
        //              -------------------------
        //              |           |           |
        //          BB:enter (B)   Loop (C)   BB:exit (D)
        //                          |
        //                      Linear (E)
        //                          |
        //              -----------------------------
        //              |           |               |
        //         BB:bb0 (F)    Branching (G)   BB:bb4 (H)
        //                          |
        //              ----------------------------
        //              |                          |
        //          Linear (I)                 Linear (J)
        //              |                          |
        //           BB:bb1 (K)             ----------------
        //                                  |              |
        //                               BB:bb2 (L)     BB:bb3 (M)
        //

        #[allow(non_snake_case)]
        {
            let A = control_tree[control_tree.root()].expect_linear();
            assert_eq!(A.children.len(), 3);

            let B = control_tree[A.children[0]].expect_basic_block();
            assert_eq!(B, enter);

            let C = control_tree[A.children[1]].expect_loop();

            let D = control_tree[A.children[2]].expect_basic_block();
            assert_eq!(D, exit);

            let E = control_tree[C.inner].expect_linear();
            assert_eq!(E.children.len(), 3);

            let F = control_tree[E.children[0]].expect_basic_block();
            assert_eq!(F, bb0);

            let G = control_tree[E.children[1]].expect_branching();
            assert_eq!(G.branches.len(), 2);

            let H = control_tree[E.children[2]].expect_basic_block();
            assert_eq!(H, bb4);

            let I = control_tree[G.branches[0]].expect_linear();
            assert_eq!(I.children.len(), 1);

            let J = control_tree[I.children[0]].expect_basic_block();
            assert_eq!(J, bb1);

            let K = control_tree[G.branches[1]].expect_linear();
            assert_eq!(K.children.len(), 2);

            let L = control_tree[K.children[0]].expect_basic_block();
            assert_eq!(L, bb2);

            let M = control_tree[K.children[1]].expect_basic_block();
            assert_eq!(M, bb3);
        }
    }

    #[test]
    fn test_control_tree_generate_single_node_loop() {
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
                    ty: TY_U32,
                    shader_io_binding: None,
                }],
                ret_ty: Some(TY_U32),
            },
        );

        let mut cfg = Cfg::new(module.ty.clone());

        let body = cfg.register_function(&module, function);

        let a0 = body.argument_values()[0];

        let bb0 = body.entry_block();
        let bb1 = cfg.add_basic_block(function);

        let (_, add_result) = cfg.add_stmt_op_binary(
            bb0,
            BlockPosition::Append,
            BinaryOperator::Add,
            a0.into(),
            1u32.into(),
        );
        let (_, test_result) = cfg.add_stmt_op_binary(
            bb0,
            BlockPosition::Append,
            BinaryOperator::Gt,
            add_result.into(),
            10u32.into(),
        );
        let (_, predicate) =
            cfg.add_stmt_op_bool_to_branch_selector(bb0, BlockPosition::Append, test_result.into());
        cfg.set_terminator(
            bb0,
            Terminator::branch_multiple(predicate.into(), [bb1, bb0]),
        );

        cfg.set_terminator(bb1, Terminator::return_value(add_result.into()));

        let mut graph = Graph::init(&mut cfg, function);
        let reentry_edges = restructure_loops(&mut graph);
        let branch_info = restructure_branches(&mut graph, &reentry_edges);

        let control_tree = ControlTree::generate(&graph, &reentry_edges, &branch_info);

        // Expected control-tree layout:
        //
        //           Linear (A)
        //               |
        //         -------------
        //         |           |
        //        Loop (B)   BB:bb1 (C)
        //         |
        //     Linear (D)
        //         |
        //     BB:bb0 (F)
        //

        #[allow(non_snake_case)]
        {
            let A = control_tree[control_tree.root()].expect_linear();
            assert_eq!(A.children.len(), 2);

            let B = control_tree[A.children[0]].expect_loop();

            let C = control_tree[A.children[1]].expect_basic_block();
            assert_eq!(C, bb1);

            let D = control_tree[B.inner].expect_linear();
            assert_eq!(D.children.len(), 1);

            let E = control_tree[D.children[0]].expect_basic_block();
            assert_eq!(E, bb0);
        }
    }
}
