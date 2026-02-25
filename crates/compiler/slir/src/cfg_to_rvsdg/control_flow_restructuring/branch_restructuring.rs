use indexmap::IndexSet;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::cfg::{BasicBlock, BlockPosition};
use crate::cfg_to_rvsdg::control_flow_restructuring::{Edge, Graph};
use crate::ty::TY_PREDICATE;

pub struct Reachable {
    kind: ReachableKind,
    most_recent: Edge,
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum ReachableKind {
    One,
    Multiple,
}

struct Reachability {
    node_state: FxHashMap<BasicBlock, Reachable>,
}

impl Reachability {
    fn new() -> Self {
        Reachability {
            node_state: Default::default(),
        }
    }

    fn update(&mut self, node: BasicBlock, reached_from: Edge) -> bool {
        let mut keep_going = true;

        self.node_state
            .entry(node)
            .and_modify(|state| {
                if state.most_recent == reached_from {
                    keep_going = false;
                } else {
                    if matches!(state.kind, ReachableKind::One) {
                        state.kind = ReachableKind::Multiple
                    }
                }
            })
            .or_insert(Reachable {
                kind: ReachableKind::One,
                most_recent: reached_from,
            });

        keep_going
    }
}

fn dominated_sets(
    graph: &Graph,
    reentry_edges: &FxHashSet<Edge>,
    head_end: BasicBlock,
    end: BasicBlock,
) -> FxHashMap<Edge, FxHashSet<BasicBlock>> {
    if head_end == end {
        return Default::default();
    }

    let mut reachability = Reachability::new();

    for child in graph.children(head_end) {
        let edge = Edge {
            source: head_end,
            dest: *child,
        };

        // We ignore loop re-entry edges during branch restructuring
        if !reentry_edges.contains(&edge) {
            reachability_search(graph, reentry_edges, edge, *child, end, &mut reachability);
        }
    }

    let mut dominated_sets: FxHashMap<Edge, FxHashSet<BasicBlock>> = FxHashMap::default();

    for (bb, reachable) in reachability.node_state {
        if matches!(reachable.kind, ReachableKind::One) {
            dominated_sets
                .entry(reachable.most_recent)
                .or_default()
                .insert(bb);
        }
    }

    dominated_sets
}

fn reachability_search(
    graph: &Graph,
    reentry_edges: &FxHashSet<Edge>,
    edge: Edge,
    current: BasicBlock,
    end: BasicBlock,
    reachability: &mut Reachability,
) {
    let keep_going = reachability.update(current, edge);

    if current != end && keep_going {
        for child in graph.children(current) {
            // We ignore loop re-entry edges during branch restructuring
            if !reentry_edges.contains(&Edge {
                source: current,
                dest: *child,
            }) {
                reachability_search(graph, reentry_edges, edge, *child, end, reachability);
            }
        }
    }
}

#[derive(Default)]
struct TailStructure {
    continuation_nodes: IndexSet<BasicBlock>,
    branch_continuation_edges: FxHashMap<Edge, Vec<Edge>>,
}

impl TailStructure {
    fn analyse(
        graph: &Graph,
        reentry_edges: &FxHashSet<Edge>,
        head_end: BasicBlock,
        end: BasicBlock,
    ) -> Self {
        let dominated_sets = dominated_sets(graph, reentry_edges, head_end, end);

        let mut continuation_nodes = IndexSet::new();
        let mut branch_continuation_edges = FxHashMap::default();

        for branch_head in graph.children(head_end) {
            let branch_edge = Edge {
                source: head_end,
                dest: *branch_head,
            };
            let mut continuation_edges = Vec::new();

            if let Some(dominated) = dominated_sets.get(&branch_edge) {
                for bb in dominated {
                    for child in graph.children(*bb) {
                        if !dominated.contains(child) {
                            // Note that we now know not just that the `child` is not dominated by the
                            // current branch edge, but also that it cannot be dominated by any of the
                            // other branch edges that originate from the `branch_head` node: any node
                            // reachable via more than 1 branch edge is by definition not dominated by
                            // any of those branch edges. Therefor, the `child` must be part of the tail.

                            continuation_nodes.insert(*child);
                            continuation_edges.push(Edge {
                                source: *bb,
                                dest: *child,
                            })
                        }
                    }
                }
            } else {
                continuation_nodes.insert(branch_edge.dest);
                continuation_edges.push(branch_edge);
            }

            branch_continuation_edges.insert(branch_edge, continuation_edges);
        }

        TailStructure {
            continuation_nodes,
            branch_continuation_edges,
        }
    }
}

fn find_head_end(
    graph: &Graph,
    reentry_edges: &FxHashSet<Edge>,
    root: BasicBlock,
    end: BasicBlock,
) -> BasicBlock {
    let mut current = root;

    while current != end {
        let mut non_reentry_branch_count = 0;
        let mut non_reentry_edge_index = 0;

        for (index, child) in graph.children(current).iter().enumerate() {
            if !reentry_edges.contains(&Edge {
                source: current,
                dest: *child,
            }) {
                non_reentry_branch_count += 1;
                non_reentry_edge_index = index;
            }
        }

        if non_reentry_branch_count != 1 {
            break;
        }

        current = graph.children(current)[non_reentry_edge_index];
    }

    current
}

fn restructure_branches_internal(
    graph: &mut Graph,
    reentry_edges: &FxHashSet<Edge>,
    branch_info: &mut FxHashMap<BasicBlock, BasicBlock>,
    root: BasicBlock,
    end: BasicBlock,
) {
    let head_end = find_head_end(graph, reentry_edges, root, end);

    if head_end == end {
        // Nothing to restructure
        return;
    }

    let tail_structure = TailStructure::analyse(graph, reentry_edges, head_end, end);

    let tail_start = if tail_structure.continuation_nodes.len() == 1 {
        // Continuation to tail already has required structure, but we may still need to restructure
        // the branch sub-graphs and the tail sub-structure

        let continuation_node = *tail_structure.continuation_nodes.iter().next().unwrap();

        for i in 0..graph.children(head_end).len() {
            let branch = graph.children(head_end)[i];

            restructure_branches_internal(
                graph,
                reentry_edges,
                branch_info,
                branch,
                continuation_node,
            );
        }

        restructure_branches_internal(graph, reentry_edges, branch_info, continuation_node, end);

        continuation_node
    } else {
        let predicate = graph.add_value(TY_PREDICATE);
        let continuation_node = graph.append_block_branch_multiple(predicate);

        let mut indices = FxHashMap::default();

        for (index, node) in tail_structure.continuation_nodes.iter().enumerate() {
            graph.connect(Edge {
                source: continuation_node,
                dest: *node,
            });
            indices.insert(*node, index);
        }

        for i in 0..graph.children(head_end).len() {
            let branch = graph.children(head_end)[i];
            let continuation_edges = tail_structure
                .branch_continuation_edges
                .get(&Edge {
                    source: head_end,
                    dest: branch,
                })
                .unwrap();

            let (branch_start, branch_end) = if continuation_edges.len() == 1 {
                let edge = continuation_edges[0];

                // If this branch is empty, make the continuation node both the start and end; this
                // should terminate immediately on the next recursion.
                let branch_start = if branch == edge.dest {
                    continuation_node
                } else {
                    branch
                };

                (branch_start, continuation_node)
            } else {
                // Consolidate all continuation edges from this branch via a single node.
                let new_end = graph.append_block_branch_single(continuation_node);

                (branch, new_end)
            };

            for edge in continuation_edges {
                let intermediate = graph.append_block_branch_single(branch_end);
                let index = *indices
                    .get(&edge.dest)
                    .expect("no continuation node for continuation edge");

                graph.add_stmt_assign(
                    intermediate,
                    BlockPosition::Append,
                    predicate,
                    (index as u32).into(),
                );
                graph.reconnect_dest(*edge, intermediate);
            }

            restructure_branches_internal(
                graph,
                reentry_edges,
                branch_info,
                branch_start,
                branch_end,
            );
        }

        continuation_node
    };

    restructure_branches_internal(graph, reentry_edges, branch_info, tail_start, end);
    branch_info.insert(head_end, tail_start);
}

/// Restructures the branches in the graph.
///
/// Also requires the set of re-entry edges that returned from loop restructuring. Re-entry edges
/// are ignored by branch restructuring.
///
/// Returns a "branch info" map, where if a basic-block starts a branching region, it will have an
/// entry in this map where the value is the consolidation block (the basic-block into which the
/// branches reconsolidate).
pub fn restructure_branches(
    graph: &mut Graph,
    reentry_edges: &FxHashSet<Edge>,
) -> FxHashMap<BasicBlock, BasicBlock> {
    let mut branch_info = FxHashMap::default();

    restructure_branches_internal(
        graph,
        reentry_edges,
        &mut branch_info,
        graph.entry(),
        graph.exit(),
    );

    branch_info
}

#[cfg(test)]
mod tests {
    use rustc_hash::FxHashSet;

    use super::*;
    use crate::cfg::{Cfg, Terminator};
    use crate::ty::TY_DUMMY;
    use crate::{FnArg, FnSig, Function, Module, Symbol};

    #[test]
    fn test_dominated_sets() {
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
                ],
                ret_ty: None,
            },
        );

        let mut cfg = Cfg::new(module.ty.clone());

        let body = cfg.register_function(&module, function);

        let a0 = body.argument_values()[0];
        let a1 = body.argument_values()[1];

        //
        //            bb0
        //             | \
        //             |  \
        //             v   \
        //            bb1   \
        //            /  \   \
        //           /    \   \
        //          v      v   \
        //         bb2    bb3   |
        //          |      |   /
        //          |      |  /
        //          |      v v
        //          |      bb4
        //          |      /
        //           \    /
        //            v  v
        //            bb5
        //

        let bb0 = body.entry_block();
        let bb1 = cfg.add_basic_block(function);
        let bb2 = cfg.add_basic_block(function);
        let bb3 = cfg.add_basic_block(function);
        let bb4 = cfg.add_basic_block(function);
        let bb5 = cfg.add_basic_block(function);

        cfg.set_terminator(bb0, Terminator::branch_multiple(a0, [bb1, bb4]));
        cfg.set_terminator(bb1, Terminator::branch_multiple(a1, [bb2, bb3]));
        cfg.set_terminator(bb2, Terminator::branch_single(bb5));
        cfg.set_terminator(bb3, Terminator::branch_single(bb4));
        cfg.set_terminator(bb4, Terminator::branch_single(bb5));

        let graph = Graph::init(&mut cfg, function);

        let dominated_sets =
            dominated_sets(&graph, &FxHashSet::default(), graph.entry(), graph.exit());

        assert_eq!(dominated_sets.len(), 1);
        assert_eq!(
            dominated_sets
                .get(&Edge {
                    source: bb0,
                    dest: bb1,
                })
                .unwrap(),
            &FxHashSet::from_iter([bb1, bb2, bb3])
        );
    }

    #[test]
    fn test_tail_structure_analyse() {
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
                ],
                ret_ty: None,
            },
        );

        let mut cfg = Cfg::new(module.ty.clone());

        let body = cfg.register_function(&module, function);

        let a0 = body.argument_values()[0];
        let a1 = body.argument_values()[1];

        //
        //            bb0
        //             | \
        //             |  \
        //             v   \
        //            bb1   \
        //            /  \   \
        //           /    \   \
        //          v      v   \
        //         bb2    bb3   |
        //          |      |   /
        //          |      |  /
        //          |      v v
        //          |      bb4
        //          |      /
        //           \    /
        //            v  v
        //            bb5
        //

        let bb0 = body.entry_block();
        let bb1 = cfg.add_basic_block(function);
        let bb2 = cfg.add_basic_block(function);
        let bb3 = cfg.add_basic_block(function);
        let bb4 = cfg.add_basic_block(function);
        let bb5 = cfg.add_basic_block(function);

        cfg.set_terminator(bb0, Terminator::branch_multiple(a0, [bb1, bb4]));
        cfg.set_terminator(bb1, Terminator::branch_multiple(a1, [bb2, bb3]));
        cfg.set_terminator(bb2, Terminator::branch_single(bb5));
        cfg.set_terminator(bb3, Terminator::branch_single(bb4));
        cfg.set_terminator(bb4, Terminator::branch_single(bb5));

        let graph = Graph::init(&mut cfg, function);

        let head_end = find_head_end(&graph, &FxHashSet::default(), graph.entry(), graph.exit());
        let tail_structure =
            TailStructure::analyse(&graph, &FxHashSet::default(), head_end, graph.exit());

        assert_eq!(
            &tail_structure.continuation_nodes,
            &IndexSet::from([bb4, bb5])
        );
        assert_eq!(tail_structure.branch_continuation_edges.len(), 2);
        assert_eq!(
            FxHashSet::from_iter(
                tail_structure
                    .branch_continuation_edges
                    .get(&Edge {
                        source: bb0,
                        dest: bb1,
                    })
                    .unwrap()
                    .iter()
                    .copied()
            ),
            FxHashSet::from_iter([
                Edge {
                    source: bb2,
                    dest: bb5,
                },
                Edge {
                    source: bb3,
                    dest: bb4,
                }
            ])
        );
        assert_eq!(
            FxHashSet::from_iter(
                tail_structure
                    .branch_continuation_edges
                    .get(&Edge {
                        source: bb0,
                        dest: bb4,
                    })
                    .unwrap()
                    .iter()
                    .copied()
            ),
            FxHashSet::from_iter([Edge {
                source: bb0,
                dest: bb4,
            }])
        );
    }

    #[test]
    fn test_restructure_branches() {
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
                ],
                ret_ty: None,
            },
        );

        let mut cfg = Cfg::new(module.ty.clone());

        let body = cfg.register_function(&module, function);

        let a0 = body.argument_values()[0];
        let a1 = body.argument_values()[1];

        //
        //            bb0
        //             | \
        //             |  \
        //             v   \
        //            bb1   \
        //            /  \   \
        //           /    \   \
        //          v      v   \
        //         bb2    bb3   |
        //          |      |   /
        //          |      |  /
        //          |      v v
        //          |      bb4
        //          |      /
        //           \    /
        //            v  v
        //            bb5
        //

        let bb0 = body.entry_block();
        let bb1 = cfg.add_basic_block(function);
        let bb2 = cfg.add_basic_block(function);
        let bb3 = cfg.add_basic_block(function);
        let bb4 = cfg.add_basic_block(function);
        let bb5 = cfg.add_basic_block(function);

        cfg.set_terminator(bb0, Terminator::branch_multiple(a0, [bb1, bb4]));
        cfg.set_terminator(bb1, Terminator::branch_multiple(a1, [bb2, bb3]));
        cfg.set_terminator(bb2, Terminator::branch_single(bb5));
        cfg.set_terminator(bb3, Terminator::branch_single(bb4));
        cfg.set_terminator(bb4, Terminator::branch_single(bb5));

        let mut graph = Graph::init(&mut cfg, function);

        let branch_info = restructure_branches(&mut graph, &FxHashSet::default());

        // Restructured:
        //
        //            bb0
        //             | \
        //             |  \
        //             v   \
        //            bb1   \
        //            /  \   \
        //           /    \   \
        //          v      v   \
        //         bb2    bb3   |
        //          |      |    |
        //          |      |    v
        //         bb8    bb9   bb10
        //          \      /   /
        //           \    /   /
        //            v  v   /
        //            bb7   /
        //             |   /
        //             |  /
        //             v v
        //             bb6
        //              | \
        //              |  v
        //              |  bb4
        //              |  /
        //              v v
        //              bb5
        //

        assert_eq!(graph.children(bb0).len(), 2);
        assert_eq!(graph.children(bb0)[0], bb1);

        let bb10 = graph.children(bb0)[1];

        assert_eq!(graph.children(bb1), &[bb2, bb3]);

        assert_eq!(graph.children(bb2).len(), 1);

        let bb9 = graph.children(bb2)[0];

        assert_eq!(graph.children(bb3).len(), 1);

        let bb8 = graph.children(bb3)[0];

        assert_eq!(graph.children(bb4), &[bb5]);

        assert!(graph.children(bb5).is_empty());

        assert_eq!(graph.children(bb8).len(), 1);

        let bb7 = graph.children(bb8)[0];

        assert_eq!(graph.children(bb7).len(), 1);

        let bb6 = graph.children(bb7)[0];

        assert_eq!(graph.children(bb6), &[bb4, bb5]);

        assert_eq!(graph.children(bb9), &[bb7]);

        assert_eq!(graph.children(bb10), &[bb6]);

        assert_eq!(branch_info.len(), 3);
        assert_eq!(branch_info.get(&bb0), Some(&bb6));
        assert_eq!(branch_info.get(&bb1), Some(&bb7));
        assert_eq!(branch_info.get(&bb6), Some(&bb5));
    }
}
