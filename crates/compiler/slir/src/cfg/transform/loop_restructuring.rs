use indexmap::IndexSet;
use rustc_hash::FxHashSet;

use crate::cfg::analyze::strongly_connected_components::{
    SccStructure, StronglyConnectedComponent, strongly_connected_components,
};
use crate::cfg::{BasicBlock, BlockPosition, BranchSelector, Cfg, Edge, LocalBinding, Terminator};
use crate::ty::{TY_BOOL, TY_U32};
use crate::{BinaryOperator, Function, UnaryOperator};

/// Restructures the loops in the graph.
///
/// Returns the set of re-entry edges (edges that connect the tail block of a loop to the entry
/// block of the loop) after restructuring.
pub fn restructure_loops(cfg: &mut Cfg, function: Function) -> FxHashSet<Edge> {
    let mut reentry_edges = FxHashSet::default();
    let mut components = strongly_connected_components(cfg, function, &reentry_edges);

    while components.len() > 0 {
        for component in &components {
            restructure_loop(cfg, &mut reentry_edges, component);
        }

        components = strongly_connected_components(cfg, function, &reentry_edges);
    }

    reentry_edges
}

fn restructure_loop(
    cfg: &mut Cfg,
    reentry_edges: &mut FxHashSet<Edge>,
    scc: &StronglyConnectedComponent,
) {
    let scc_structure = SccStructure::analyse(cfg, scc);

    let reentry_edge = if scc_structure.is_tail_controlled_loop() {
        // The control-flow graph is already in the desired structure.

        let reentry_edge = scc_structure.repetition_edges[0];

        normalize_terminator(cfg, reentry_edge);

        reentry_edge
    } else {
        let entry = cfg[scc.function].entry_block();
        let (_, branch_selector) = cfg.add_stmt_uninitialized(entry, BlockPosition::Append, TY_U32);

        let restructured_entry = restructure_loop_entry(cfg, &scc_structure, branch_selector);

        restructure_loop_tail(cfg, &scc_structure, branch_selector, restructured_entry)
    };

    reentry_edges.insert(reentry_edge);
}

/// Normalize the terminator of the tail block of a loop that is already in the desired structure.
///
/// Though the graph structure is already in the desired form, we may still need to normalize the
/// tail block's terminator. We want the branch-selector to be a boolean value, we want the first
/// target to be the loop's entry block, and we want the second target to be the continuation block.
fn normalize_terminator(cfg: &mut Cfg, reentry_edge: Edge) {
    let entry = reentry_edge.dest;
    let tail = reentry_edge.source;

    let needs_reversal = cfg.successors(tail)[0] != entry;

    match cfg[tail].terminator().expect_branch().selector() {
        BranchSelector::Bool(condition) => {
            let condition = *condition;

            if needs_reversal {
                // We need to invert both the condition and the branch target order.

                let (_, new_condition) = cfg.add_stmt_op_unary(
                    tail,
                    BlockPosition::Append,
                    UnaryOperator::Not,
                    condition.into(),
                );

                cfg.set_branch_selector(tail, BranchSelector::Bool(new_condition));
                cfg.reverse_branch_targets(tail);
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

            let (_, condition) =
                cfg.add_stmt_op_binary(tail, BlockPosition::Append, cmp, value.into(), case.into());

            cfg.set_branch_selector(tail, BranchSelector::Bool(condition));

            if needs_reversal {
                cfg.reverse_branch_targets(tail);
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

            let (_, condition) =
                cfg.add_stmt_op_binary(tail, BlockPosition::Append, cmp, value.into(), 0u32.into());

            cfg.set_branch_selector(tail, BranchSelector::Bool(condition));

            if needs_reversal {
                cfg.reverse_branch_targets(tail);
            }
        }
        BranchSelector::Single => {
            unreachable!("a tail-controlled loop has exactly two targets")
        }
    }
}

/// Information about the restructured loop entry.
struct RestructuredLoopEntry {
    /// The new unified entry block.
    basic_block: BasicBlock,
    /// An index set that maps each original entry-edge destination and each original
    /// repitition-edge destination to the index of a target branch in the new entry block's
    /// terminator.
    target_index_mapping: IndexSet<BasicBlock>,
}

fn restructure_loop_entry(
    cfg: &mut Cfg,
    structure: &SccStructure,
    branch_selector: LocalBinding,
) -> RestructuredLoopEntry {
    // Add a new entry block to the CFG. This will be the target of the restructured loop's reentry
    // edge. We'll also route all original entry edges through this new entry block, though not
    // directly; we'll create a new "intermediate" block for each entry edge. For the `i`th
    // original `entry_edge_i`, we'll reconnect `entry_edge_i.source` to the `i`th intermediate
    // block. In the `i`th intermediate block, we'll add an "assign" statement that will set the
    // `branch_selector` to `i`. The `i` intermediate block will then unconditionally branch to the
    // new `entry` block. The entry block then branches on the `branch_selector` to its `i`th
    // target, which we set to the original `entry_edge_i.dest`.

    let entry = cfg.add_basic_block(structure.function);

    cfg.set_branch_selector(entry, BranchSelector::U32(branch_selector));

    let mut target_index_mapping = IndexSet::default();

    for edge in structure.entry_edges.iter() {
        let (index, inserted) = target_index_mapping.insert_full(edge.dest);

        if inserted {
            cfg.add_branch_target(entry, edge.dest);
        }

        let intermediate = cfg.add_basic_block(structure.function);

        cfg.add_stmt_assign(
            intermediate,
            BlockPosition::Append,
            branch_selector,
            (index as u32).into(),
        );
        cfg.set_terminator(intermediate, Terminator::branch_single(entry));

        cfg.replace_branch_target(edge.source, edge.dest, intermediate);
    }

    for edge in structure.repetition_edges.iter() {
        if target_index_mapping.insert(edge.dest) {
            cfg.add_branch_target(entry, edge.dest);
        }
    }

    RestructuredLoopEntry {
        basic_block: entry,
        target_index_mapping,
    }
}

fn restructure_loop_tail(
    cfg: &mut Cfg,
    structure: &SccStructure,
    branch_selector: LocalBinding,
    entry: RestructuredLoopEntry,
) -> Edge {
    let function_entry = cfg[structure.function].entry_block();
    let (_, reentry_condition) =
        cfg.add_stmt_uninitialized(function_entry, BlockPosition::Append, TY_BOOL);

    // The new unified exit block of the restructured loop. If the `reentry_condition` is `false`,
    // then the new `tail` block (below) will flow to here. This new `exit` block will then
    // redirect flow to one of the prior exit blocks.
    let exit = cfg.add_basic_block(structure.function);

    cfg.set_terminator(exit, Terminator::branch_u32(branch_selector, []));

    // The new tail block of the restructured loop. If the `reentry_condition` is `true`, then this
    // block will flow to the new unified `entry` block (created by `restruct_loop_entry`),
    // otherwise this new tail block will flow to the new unified `exit` block created above.
    let tail = cfg.add_basic_block(structure.function);

    cfg.set_terminator(
        tail,
        Terminator::branch_bool(reentry_condition, entry.basic_block, exit),
    );

    for edge in &structure.repetition_edges {
        let intermediate = cfg.add_basic_block(structure.function);

        let target_index = entry.target_index_mapping.get_index_of(&edge.dest).expect(
            "all repetition edge destinations should have been added as part of \
            `restructure_loop_entry`",
        );

        cfg.add_stmt_assign(
            intermediate,
            BlockPosition::Append,
            branch_selector,
            (target_index as u32).into(),
        );
        // Set the `reentry_condition` to `true` to indicate that we will be repeating the loop.
        cfg.add_stmt_assign(
            intermediate,
            BlockPosition::Append,
            reentry_condition,
            true.into(),
        );
        cfg.set_terminator(intermediate, Terminator::branch_single(tail));

        cfg.replace_branch_target(edge.source, edge.dest, intermediate);
    }

    let mut exit_index_mapping: IndexSet<BasicBlock> = IndexSet::default();

    for edge in structure.exit_edges.iter() {
        let (index, inserted) = exit_index_mapping.insert_full(edge.dest);

        if inserted {
            cfg.add_branch_target(exit, edge.dest);
        }

        let intermediate = cfg.add_basic_block(structure.function);

        cfg.add_stmt_assign(
            intermediate,
            BlockPosition::Append,
            branch_selector,
            (index as u32).into(),
        );
        // Set the `reentry_selector` to `false` to indicate that we will be exiting the loop.
        cfg.add_stmt_assign(
            intermediate,
            BlockPosition::Append,
            reentry_condition,
            false.into(),
        );
        cfg.set_terminator(intermediate, Terminator::branch_single(tail));

        cfg.replace_branch_target(edge.source, edge.dest, intermediate);
    }

    Edge {
        source: tail,
        dest: entry.basic_block,
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

        restructure_loops(&mut cfg, function);

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

        assert_eq!(cfg.successors(bb0).len(), 2);

        let bb9 = cfg.successors(bb0)[0];
        let bb10 = cfg.successors(bb0)[1];

        assert_eq!(cfg.successors(bb9).len(), 1);

        let bb8 = cfg.successors(bb9)[0];

        assert_eq!(cfg.successors(bb10), &[bb8]);

        assert_eq!(cfg.successors(bb8), &[bb2, bb1]);

        assert_eq!(cfg.successors(bb1), &[bb3]);

        assert_eq!(cfg.successors(bb2), &[bb4]);

        assert_eq!(cfg.successors(bb3).len(), 2);

        let bb14 = cfg.successors(bb3)[0];
        let bb16 = cfg.successors(bb3)[1];

        assert_eq!(cfg.successors(bb4).len(), 2);

        let bb13 = cfg.successors(bb4)[0];
        let bb15 = cfg.successors(bb4)[1];

        assert_eq!(cfg.successors(bb13).len(), 1);

        let bb11 = cfg.successors(bb13)[0];

        assert_eq!(cfg.successors(bb14), &[bb11]);
        assert_eq!(cfg.successors(bb15), &[bb11]);
        assert_eq!(cfg.successors(bb16), &[bb11]);

        assert_eq!(cfg.successors(bb11).len(), 2);
        assert_eq!(cfg.successors(bb11)[0], bb8);

        let bb12 = cfg.successors(bb11)[1];

        assert_eq!(cfg.successors(bb12), &[bb6, bb5]);
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

        let reentry_edge = Edge {
            source: tail,
            dest: entry,
        };

        normalize_terminator(&mut cfg, reentry_edge);

        assert_eq!(cfg.successors(tail), &[entry, exit]);
        let BranchSelector::Bool(c) = cfg[tail].terminator().expect_branch().selector() else {
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

        let reentry_edge = Edge {
            source: tail,
            dest: entry,
        };

        normalize_terminator(&mut cfg, reentry_edge);

        // Verify that the targets are now correctly ordered: [entry, exit]
        assert_eq!(cfg.successors(tail), &[entry, exit]);

        // Verify that the selector is a Bool and is the inverse of the original condition.
        let BranchSelector::Bool(new_cond) = cfg[tail].terminator().expect_branch().selector()
        else {
            panic!("Expected Bool selector")
        };

        let statements = cfg[tail].statements();
        assert_eq!(statements.len(), 1);

        let stmt = statements[0];
        let op_unary = cfg[stmt].expect_op_unary();
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

        let reentry_edge = Edge {
            source: tail,
            dest: entry,
        };

        normalize_terminator(&mut cfg, reentry_edge);

        // Verify that the targets are correctly ordered: [entry, exit]
        assert_eq!(cfg.successors(tail), &[entry, exit]);

        // Verify that the selector is now a Bool.
        let BranchSelector::Bool(condition) = cfg[tail].terminator().expect_branch().selector()
        else {
            panic!("Expected Bool selector")
        };

        // Verify that the comparison operation was added to the tail block.
        let statements = cfg[tail].statements();
        assert_eq!(statements.len(), 1);

        let stmt = statements[0];
        let op_binary = cfg[stmt].expect_op_binary();
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

        let reentry_edge = Edge {
            source: tail,
            dest: entry,
        };

        normalize_terminator(&mut cfg, reentry_edge);

        // Verify that the targets are correctly ordered: [entry, exit]
        assert_eq!(cfg.successors(tail), &[entry, exit]);

        // Verify that the selector is now a Bool.
        let BranchSelector::Bool(condition) = cfg[tail].terminator().expect_branch().selector()
        else {
            panic!("Expected Bool selector")
        };

        // Verify that the comparison operation was added to the tail block.
        // Since it needed reversal, it should be NotEq.
        let statements = cfg[tail].statements();
        assert_eq!(statements.len(), 1);

        let stmt = statements[0];
        let op_binary = cfg[stmt].expect_op_binary();
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

        let reentry_edge = Edge {
            source: tail,
            dest: entry,
        };

        normalize_terminator(&mut cfg, reentry_edge);

        // Verify that the targets are correctly ordered: [entry, exit]
        assert_eq!(cfg.successors(tail), &[entry, exit]);

        // Verify that the selector is now a Bool.
        let BranchSelector::Bool(condition) = cfg[tail].terminator().expect_branch().selector()
        else {
            panic!("Expected Bool selector")
        };

        // Verify that the comparison operation was added to the tail block.
        let statements = cfg[tail].statements();
        assert_eq!(statements.len(), 1);

        let stmt = statements[0];
        let op_binary = cfg[stmt].expect_op_binary();
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

        let reentry_edge = Edge {
            source: tail,
            dest: entry,
        };

        normalize_terminator(&mut cfg, reentry_edge);

        // Verify that the targets are correctly ordered: [entry, exit]
        assert_eq!(cfg.successors(tail), &[entry, exit]);

        // Verify that the selector is now a Bool.
        let BranchSelector::Bool(condition) = cfg[tail].terminator().expect_branch().selector()
        else {
            panic!("Expected Bool selector")
        };

        // Verify that the comparison operation was added to the tail block.
        // Since it needed reversal, it should be NotEq.
        let statements = cfg[tail].statements();
        assert_eq!(statements.len(), 1);

        let stmt = statements[0];
        let op_binary = cfg[stmt].expect_op_binary();
        assert_eq!(op_binary.operator(), BinaryOperator::NotEq);
        assert_eq!(op_binary.lhs(), value.into());
        assert_eq!(op_binary.rhs(), 0u32.into());
        assert_eq!(op_binary.result(), *condition);
    }
}
