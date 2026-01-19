use indexmap::IndexSet;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::cfg::{Assign, BasicBlock, BlockPosition, LocalBinding};
use crate::cfg_to_rvsdg::control_flow_restructuring::strongly_connected_components::{
    SccStructure, strongly_connected_components,
};
use crate::cfg_to_rvsdg::control_flow_restructuring::{Edge, Graph};
use crate::ty::TY_PREDICATE;

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
        // Already in the desired structure

        scc_structure.repetition_edges[0]
    } else {
        let branch_selector = graph.add_value(TY_PREDICATE);

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

fn restructure_loop_entry(
    graph: &mut Graph,
    structure: &SccStructure,
    branch_selector: LocalBinding,
) -> (BasicBlock, FxHashMap<BasicBlock, u32>) {
    let entry = graph.append_block_branch_multiple(branch_selector);
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
    let reentry_selector = graph.add_value(TY_PREDICATE);

    let tail = graph.append_block_branch_multiple(reentry_selector);
    let exit = graph.append_block_branch_multiple(branch_selector);

    graph.connect(Edge {
        source: tail,
        dest: exit,
    });
    graph.connect(Edge {
        source: tail,
        dest: entry,
    });

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

        // Set the `reentry_selector` selector to `1` to indicate that we will be repeating the
        // loop.
        graph.add_stmt_assign(
            intermediate,
            BlockPosition::Append,
            reentry_selector,
            1u32.into(),
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

        // Set the `reentry_selector` selector to `0` to indicate that we will be exiting the
        // loop.
        graph.add_stmt_assign(
            intermediate,
            BlockPosition::Append,
            reentry_selector,
            0u32.into(),
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
                        ty: TY_PREDICATE,
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

        cfg.set_terminator(bb0, Terminator::branch_multiple(a0, [bb1, bb2]));
        cfg.set_terminator(bb1, Terminator::branch_single(bb3));
        cfg.set_terminator(bb2, Terminator::branch_single(bb4));
        cfg.set_terminator(bb3, Terminator::branch_multiple(a1, [bb5, bb2]));
        cfg.set_terminator(bb4, Terminator::branch_multiple(a2, [bb6, bb1]));
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
        assert_eq!(graph.children(bb11)[1], bb8);

        let bb12 = graph.children(bb11)[0];

        assert_eq!(graph.children(bb12), &[bb6, bb5]);
    }
}
