use indexmap::IndexSet;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::cfg::{BasicBlock, BlockPosition, BranchSelector, LocalBinding};
use crate::cfg_to_rvsdg::control_flow_restructuring::strongly_connected_components::{
    SccStructure, strongly_connected_components,
};
use crate::cfg_to_rvsdg::control_flow_restructuring::{Edge, Graph};
use crate::ty::{TY_BOOL, TY_U32};
use crate::{BinaryOperator, UnaryOperator};

/// Restructures the loops in the graph.
///
/// Returns the set of re-entry edges (edges that connect the tail block of a loop to the entry
/// block of the loop) after restructuring.
pub fn restructure_loops(graph: &mut Graph) -> FxHashSet<Edge> {
    let mut reentry_edges = FxHashSet::default();

    restructure_loops_internal(graph, &mut reentry_edges);

    reentry_edges
}

fn restructure_loops_internal(graph: &mut Graph, reentry_edges: &mut FxHashSet<Edge>) {
    let mut components = strongly_connected_components(graph, reentry_edges);

    while components.len() > 0 {
        for component in &components {
            restructure_loop(graph, reentry_edges, component);
        }

        components = strongly_connected_components(graph, reentry_edges);
    }
}

fn restructure_loop(
    graph: &mut Graph,
    reentry_edges: &mut FxHashSet<Edge>,
    scc: &IndexSet<BasicBlock>,
) {
    let scc_structure = SccStructure::analyse(graph, scc);

    let reentry_edge = if scc_structure.is_tail_controlled_loop() {
        // The control-flow graph is already in the desired structure.

        let reentry_edge = scc_structure.repetition_edges[0];

        normalize_terminator(graph, reentry_edge);

        reentry_edge
    } else {
        let branch_selector = graph.add_value(TY_U32);

        let (entry, branch_mapping) =
            restructure_loop_entry(graph, &scc_structure, branch_selector);

        restructure_loop_tail(
            graph,
            &scc_structure,
            branch_selector,
            entry,
            branch_mapping,
        )
    };

    reentry_edges.insert(reentry_edge);
}

/// Normalize the terminator of the tail block of a loop that is already in the desired structure.
///
/// Though the graph structure is already in the desired form, we may still need to normalize the
/// tail block's terminator. We want the branch-selector to be a boolean value, we want the first
/// target to be the loop's entry block, and we want the second target to be the continuation block.
fn normalize_terminator(graph: &mut Graph, reentry_edge: Edge) {
    let entry = reentry_edge.dest;
    let tail = reentry_edge.source;

    let needs_reversal = graph.children(tail)[0] != entry;

    match graph.selector(tail) {
        BranchSelector::Bool(condition) => {
            let condition = *condition;

            if needs_reversal {
                // We need to invert both the condition and the branch target order.

                let (_, new_condition) = graph.add_stmt_op_unary(
                    tail,
                    BlockPosition::Append,
                    UnaryOperator::Not,
                    condition.into(),
                );

                graph.set_selector(tail, BranchSelector::Bool(new_condition));
                graph.reverse_child_order(tail);
            }
        }
        BranchSelector::Case { value, cases } => {
            // We'll need to convert the `value` to a boolean value. We'll do this by comparing
            // it to the first case. If we need to invert the branch target order, we'll also
            // need to invert the new boolean value. We can combine this into a single operation
            // by simply selecting the inverse comparison operator.

            let cmp = if needs_reversal {
                BinaryOperator::NotEq
            } else {
                BinaryOperator::Eq
            };

            let value = *value;
            let case = cases[0];

            let (_, condition) = graph.add_stmt_op_binary(
                tail,
                BlockPosition::Append,
                cmp,
                value.into(),
                case.into(),
            );

            graph.set_selector(tail, BranchSelector::Bool(condition));

            if needs_reversal {
                graph.reverse_child_order(tail);
            }
        }
        BranchSelector::U32(value) => {
            // We'll need to convert the `value` to a boolean value. We'll do this by comparing
            // it to `0`. If we need to invert the branch target order, we'll also need to
            // invert the new boolean value. We can combine this into a single operation by
            // simply selecting the inverse comparison operator.

            let cmp = if needs_reversal {
                BinaryOperator::NotEq
            } else {
                BinaryOperator::Eq
            };

            let value = *value;

            let (_, condition) = graph.add_stmt_op_binary(
                tail,
                BlockPosition::Append,
                cmp,
                value.into(),
                0u32.into(),
            );

            graph.set_selector(tail, BranchSelector::Bool(condition));

            if needs_reversal {
                graph.reverse_child_order(tail);
            }
        }
        BranchSelector::Single => {
            unreachable!("a tail-controlled loop has exactly two targets")
        }
    }
}

fn restructure_loop_entry(
    graph: &mut Graph,
    structure: &SccStructure,
    branch_selector: LocalBinding,
) -> (BasicBlock, FxHashMap<BasicBlock, u32>) {
    let entry = graph.append_block_branch_u32(branch_selector);
    let mut branch_mapping = FxHashMap::default();

    for (i, edge) in structure.entry_edges.iter().enumerate() {
        let intermediate = graph.append_block_branch_single(entry);

        graph.add_stmt_assign(
            intermediate,
            BlockPosition::Append,
            branch_selector,
            (i as u32).into(),
        );

        graph.reconnect_dest(*edge, intermediate);
        graph.connect(Edge {
            source: entry,
            dest: edge.dest,
        });

        branch_mapping.insert(edge.dest, i as u32);
    }

    (entry, branch_mapping)
}

fn restructure_loop_tail(
    graph: &mut Graph,
    structure: &SccStructure,
    branch_selector: LocalBinding,
    entry: BasicBlock,
    entry_branch_mapping: FxHashMap<BasicBlock, u32>,
) -> Edge {
    let reentry_condition = graph.add_value(TY_BOOL);

    let exit = graph.append_block_branch_u32(branch_selector);
    let tail = graph.append_block_branch_bool(reentry_condition, entry, exit);

    for edge in &structure.repetition_edges {
        let intermediate = graph.append_block_branch_single(tail);
        let branch_index = *entry_branch_mapping
            .get(&edge.dest)
            .expect("no branch for repetition edge");

        graph.add_stmt_assign(
            intermediate,
            BlockPosition::Append,
            branch_selector,
            branch_index.into(),
        );

        // Set the `reentry_condition` to `true` to indicate that we will be repeating the loop.
        graph.add_stmt_assign(
            intermediate,
            BlockPosition::Append,
            reentry_condition,
            true.into(),
        );

        graph.reconnect_dest(*edge, intermediate);
    }

    for (i, edge) in structure.exit_edges.iter().enumerate() {
        let intermediate = graph.append_block_branch_single(tail);

        graph.add_stmt_assign(
            intermediate,
            BlockPosition::Append,
            branch_selector,
            (i as u32).into(),
        );

        // Set the `reentry_selector` to `false` to indicate that we will be exiting the loop.
        graph.add_stmt_assign(
            intermediate,
            BlockPosition::Append,
            reentry_condition,
            false.into(),
        );

        graph.reconnect_dest(*edge, intermediate);
        graph.connect(Edge {
            source: exit,
            dest: edge.dest,
        })
    }

    Edge {
        source: tail,
        dest: entry,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg::{Cfg, Terminator};
    use crate::ty::TY_DUMMY;
    use crate::{FnArg, FnSig, Function, Module, Symbol};

    #[test]
    fn test_loop_restructuring() {
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
                    FnArg {
                        ty: TY_U32,
                        shader_io_binding: None,
                    },
                ],
                ret_ty: None,
            },
        );

        let mut cfg = Cfg::new(module.ty.clone());

        let body = cfg.register_function(&module, function);

        let a0 = body.argument_values()[0];
        let a1 = body.argument_values()[1];
        let a2 = body.argument_values()[2];

        //
        //       bb0
        //       /  \
        //      /    \
        //     v      v
        //    bb1    bb2
        //    |  ^  ^  |
        //    |   \/   |
        //    |   /\   |
        //    v  /  \  v
        //   bb3      bb4
        //    |        |
        //    v        v
        //   bb5      bb6
        //     \      /
        //      \    /
        //       v  v
        //       exit
        //

        let bb0 = body.entry_block();
        let bb1 = cfg.add_basic_block(function);
        let bb2 = cfg.add_basic_block(function);
        let bb3 = cfg.add_basic_block(function);
        let bb4 = cfg.add_basic_block(function);
        let bb5 = cfg.add_basic_block(function);
        let bb6 = cfg.add_basic_block(function);
        let bb7 = cfg.add_basic_block(function);

        cfg.set_terminator(bb0, Terminator::branch_u32(a0, [bb1, bb2]));
        cfg.set_terminator(bb1, Terminator::branch_single(bb3));
        cfg.set_terminator(bb2, Terminator::branch_single(bb4));
        cfg.set_terminator(bb3, Terminator::branch_u32(a1, [bb5, bb2]));
        cfg.set_terminator(bb4, Terminator::branch_u32(a2, [bb6, bb1]));
        cfg.set_terminator(bb5, Terminator::branch_single(bb7));
        cfg.set_terminator(bb6, Terminator::branch_single(bb7));

        let mut graph = Graph::init(&mut cfg, function);

        restructure_loops(&mut graph);

        // Restructured:
        //
        //            bb0
        //           /  \
        //          /    \
        //         v      v
        //        bb9    bb10
        //          \    /
        //           v  v
        //            bb8 <--------|
        //           /  \          |
        //          v    v         |
        //        bb1    bb2       |
        //        |        |       |
        //        v        v       |
        //       bb3      bb4      |
        //       / \      / \      |
        //      v   v    v   v     |
        //    bb14 bb16 bb13 bb15  |
        //       \   |  |   /      |
        //        v  v  v  v       |
        //           bb11 ---------|
        //            |
        //            v
        //           bb12
        //          /   \
        //         v     v
        //       bb5      bb6
        //         \      /
        //          \    /
        //           v  v
        //           bb7
        //

        assert_eq!(graph.children(bb0).len(), 2);

        let bb9 = graph.children(bb0)[0];
        let bb10 = graph.children(bb0)[1];

        assert_eq!(graph.children(bb9).len(), 1);

        let bb8 = graph.children(bb9)[0];

        assert_eq!(graph.children(bb10), &[bb8]);

        assert_eq!(graph.children(bb8), &[bb2, bb1]);

        assert_eq!(graph.children(bb1), &[bb3]);

        assert_eq!(graph.children(bb2), &[bb4]);

        assert_eq!(graph.children(bb3).len(), 2);

        let bb14 = graph.children(bb3)[0];
        let bb16 = graph.children(bb3)[1];

        assert_eq!(graph.children(bb4).len(), 2);

        let bb13 = graph.children(bb4)[0];
        let bb15 = graph.children(bb4)[1];

        assert_eq!(graph.children(bb13).len(), 1);

        let bb11 = graph.children(bb13)[0];

        assert_eq!(graph.children(bb14), &[bb11]);
        assert_eq!(graph.children(bb15), &[bb11]);
        assert_eq!(graph.children(bb16), &[bb11]);

        assert_eq!(graph.children(bb11).len(), 2);
        assert_eq!(graph.children(bb11)[0], bb8);

        let bb12 = graph.children(bb11)[1];

        assert_eq!(graph.children(bb12), &[bb6, bb5]);
    }

    #[test]
    fn test_normalize_terminator_already_normalized() {
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

        let cond = body.argument_values()[0];

        //   entry <---|
        //     |       |
        //     v       |
        //    tail ----|
        //     |
        //     v
        //   exit

        let entry = body.entry_block();
        let tail = cfg.add_basic_block(function);
        let exit = cfg.add_basic_block(function);

        cfg.set_terminator(entry, Terminator::branch_single(tail));
        cfg.set_terminator(tail, Terminator::branch_bool(cond, entry, exit));

        let mut graph = Graph::init(&mut cfg, function);

        let reentry_edge = Edge {
            source: tail,
            dest: entry,
        };

        normalize_terminator(&mut graph, reentry_edge);

        assert_eq!(graph.children(tail), &[entry, exit]);
        let BranchSelector::Bool(c) = graph.selector(tail) else {
            panic!("Expected Bool selector")
        };
        assert_eq!(*c, cond);
    }

    #[test]
    fn test_normalize_terminator_bool_selector_reversed() {
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

        let cond = body.argument_values()[0];

        //   entry <---|
        //     |       |
        //     v       |
        //    tail ----|
        //     |
        //     v
        //   exit
        //
        // Reversed: tail -> [exit, entry]

        let entry = body.entry_block();
        let tail = cfg.add_basic_block(function);
        let exit = cfg.add_basic_block(function);

        cfg.set_terminator(entry, Terminator::branch_single(tail));
        // We set the targets in reversed order: [exit, entry]
        cfg.set_terminator(tail, Terminator::branch_bool(cond, exit, entry));

        let mut graph = Graph::init(&mut cfg, function);

        let reentry_edge = Edge {
            source: tail,
            dest: entry,
        };

        normalize_terminator(&mut graph, reentry_edge);

        // Verify that the targets are now correctly ordered: [entry, exit]
        assert_eq!(graph.children(tail), &[entry, exit]);

        // Verify that the selector is a Bool and is the inverse of the original condition.
        let BranchSelector::Bool(new_cond) = graph.selector(tail) else {
            panic!("Expected Bool selector")
        };

        let statements = graph.statements(tail);
        assert_eq!(statements.len(), 1);

        let stmt = statements[0];
        let op_unary = graph[stmt].expect_op_unary();
        assert_eq!(op_unary.operator(), UnaryOperator::Not);
        assert_eq!(op_unary.value(), cond.into());
        assert_eq!(op_unary.result(), *new_cond);

        assert_ne!(*new_cond, cond);
    }

    #[test]
    fn test_normalize_terminator_case_selector() {
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
                ret_ty: None,
            },
        );

        let mut cfg = Cfg::new(module.ty.clone());

        let body = cfg.register_function(&module, function);

        let value = body.argument_values()[0];

        //   entry <---|
        //     |       |
        //     v       |
        //    tail ----|
        //     |
        //     v
        //   exit
        //
        // Case: tail -> [entry, exit] (entry is case 42, exit is default)

        let entry = body.entry_block();
        let tail = cfg.add_basic_block(function);
        let exit = cfg.add_basic_block(function);

        cfg.set_terminator(entry, Terminator::branch_single(tail));
        cfg.set_terminator(
            tail,
            Terminator::branch_case(value, vec![42], vec![entry, exit]),
        );

        let mut graph = Graph::init(&mut cfg, function);

        let reentry_edge = Edge {
            source: tail,
            dest: entry,
        };

        normalize_terminator(&mut graph, reentry_edge);

        // Verify that the targets are correctly ordered: [entry, exit]
        assert_eq!(graph.children(tail), &[entry, exit]);

        // Verify that the selector is now a Bool.
        let BranchSelector::Bool(condition) = graph.selector(tail) else {
            panic!("Expected Bool selector")
        };

        // Verify that the comparison operation was added to the tail block.
        let statements = graph.statements(tail);
        assert_eq!(statements.len(), 1);

        let stmt = statements[0];
        let op_binary = graph[stmt].expect_op_binary();
        assert_eq!(op_binary.operator(), BinaryOperator::Eq);
        assert_eq!(op_binary.lhs(), value.into());
        assert_eq!(op_binary.rhs(), 42u32.into());
        assert_eq!(op_binary.result(), *condition);
    }

    #[test]
    fn test_normalize_terminator_case_selector_reversed() {
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
                ret_ty: None,
            },
        );

        let mut cfg = Cfg::new(module.ty.clone());

        let body = cfg.register_function(&module, function);

        let value = body.argument_values()[0];

        //   entry <---|
        //     |       |
        //     v       |
        //    tail ----|
        //     |
        //     v
        //   exit
        //
        // Case: tail -> [exit, entry] (exit is case 42, entry is default)
        // Reversed: entry is NOT at index 0.

        let entry = body.entry_block();
        let tail = cfg.add_basic_block(function);
        let exit = cfg.add_basic_block(function);

        cfg.set_terminator(entry, Terminator::branch_single(tail));
        // We set the targets in reversed order: [exit, entry]
        // Entry is at index 1, so needs_reversal will be true.
        cfg.set_terminator(
            tail,
            Terminator::branch_case(value, vec![42], vec![exit, entry]),
        );

        let mut graph = Graph::init(&mut cfg, function);

        let reentry_edge = Edge {
            source: tail,
            dest: entry,
        };

        normalize_terminator(&mut graph, reentry_edge);

        // Verify that the targets are correctly ordered: [entry, exit]
        assert_eq!(graph.children(tail), &[entry, exit]);

        // Verify that the selector is now a Bool.
        let BranchSelector::Bool(condition) = graph.selector(tail) else {
            panic!("Expected Bool selector")
        };

        // Verify that the comparison operation was added to the tail block.
        // Since it needed reversal, it should be NotEq.
        let statements = graph.statements(tail);
        assert_eq!(statements.len(), 1);

        let stmt = statements[0];
        let op_binary = graph[stmt].expect_op_binary();
        assert_eq!(op_binary.operator(), BinaryOperator::NotEq);
        assert_eq!(op_binary.lhs(), value.into());
        assert_eq!(op_binary.rhs(), 42u32.into());
        assert_eq!(op_binary.result(), *condition);
    }

    #[test]
    fn test_normalize_terminator_u32_selector() {
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
                ret_ty: None,
            },
        );

        let mut cfg = Cfg::new(module.ty.clone());

        let body = cfg.register_function(&module, function);

        let value = body.argument_values()[0];

        //   entry <---|
        //     |       |
        //     v       |
        //    tail ----|
        //     |
        //     v
        //   exit
        //
        // U32: tail -> [entry, exit] (entry is index 0, exit is index 1)

        let entry = body.entry_block();
        let tail = cfg.add_basic_block(function);
        let exit = cfg.add_basic_block(function);

        cfg.set_terminator(entry, Terminator::branch_single(tail));
        cfg.set_terminator(tail, Terminator::branch_u32(value, [entry, exit]));

        let mut graph = Graph::init(&mut cfg, function);

        let reentry_edge = Edge {
            source: tail,
            dest: entry,
        };

        normalize_terminator(&mut graph, reentry_edge);

        // Verify that the targets are correctly ordered: [entry, exit]
        assert_eq!(graph.children(tail), &[entry, exit]);

        // Verify that the selector is now a Bool.
        let BranchSelector::Bool(condition) = graph.selector(tail) else {
            panic!("Expected Bool selector")
        };

        // Verify that the comparison operation was added to the tail block.
        let statements = graph.statements(tail);
        assert_eq!(statements.len(), 1);

        let stmt = statements[0];
        let op_binary = graph[stmt].expect_op_binary();
        assert_eq!(op_binary.operator(), BinaryOperator::Eq);
        assert_eq!(op_binary.lhs(), value.into());
        assert_eq!(op_binary.rhs(), 0u32.into());
        assert_eq!(op_binary.result(), *condition);
    }

    #[test]
    fn test_normalize_terminator_u32_selector_reversed() {
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
                ret_ty: None,
            },
        );

        let mut cfg = Cfg::new(module.ty.clone());

        let body = cfg.register_function(&module, function);

        let value = body.argument_values()[0];

        //   entry <---|
        //     |       |
        //     v       |
        //    tail ----|
        //     |
        //     v
        //   exit
        //
        // U32: tail -> [exit, entry] (exit is index 0, entry is index 1)
        // Reversed: entry is NOT at index 0.

        let entry = body.entry_block();
        let tail = cfg.add_basic_block(function);
        let exit = cfg.add_basic_block(function);

        cfg.set_terminator(entry, Terminator::branch_single(tail));
        // We set the targets in reversed order: [exit, entry]
        cfg.set_terminator(tail, Terminator::branch_u32(value, [exit, entry]));

        let mut graph = Graph::init(&mut cfg, function);

        let reentry_edge = Edge {
            source: tail,
            dest: entry,
        };

        normalize_terminator(&mut graph, reentry_edge);

        // Verify that the targets are correctly ordered: [entry, exit]
        assert_eq!(graph.children(tail), &[entry, exit]);

        // Verify that the selector is now a Bool.
        let BranchSelector::Bool(condition) = graph.selector(tail) else {
            panic!("Expected Bool selector")
        };

        // Verify that the comparison operation was added to the tail block.
        // Since it needed reversal, it should be NotEq.
        let statements = graph.statements(tail);
        assert_eq!(statements.len(), 1);

        let stmt = statements[0];
        let op_binary = graph[stmt].expect_op_binary();
        assert_eq!(op_binary.operator(), BinaryOperator::NotEq);
        assert_eq!(op_binary.lhs(), value.into());
        assert_eq!(op_binary.rhs(), 0u32.into());
        assert_eq!(op_binary.result(), *condition);
    }
}
