//! Provides a data structure for efficiently querying the predecessors of basic blocks in a
//! [Cfg].
//!
//! The [Cfg] naturally encodes a basic-block's successors (see [Cfg::successors]), that is, the
//! other basic-blocks the control-flow can flow to when a basic-block terminates. This module
//! computes a basic-block's predecessors: the basic-blocks that can flow into that basic-block.
//!
//! Note that a basic-block can be its own predecessor (just like it can be its own successor; a
//! single basic-block loop).
//!
//! To compute the predecessors for each of the basic-blocks in a particular [Function]'s body, call
//! the [predecessors] function. This returns a [Predecessors] data structure that can be indexed by
//! [BasicBlock] keys to query that basic-block's predecessors.
//!
//! Note that the [Predecessors] data structure represents a snapshot of the [Cfg] of the moment
//! that the [predecessors] function was called. If the [Cfg] is later modified in a way that
//! affects control-flow, then the predecessor information may be outdated and [predecessors]
//! should be called again to compute a fresh instance.
//!
//! The successor list of a basic-block is a [Cfg] is expected to not contain duplicate [BasicBlock]
//! keys, otherwise the [Cfg] is not in a valid state. If the successor lists (as returned by
//! [Cfg::successors]) indeed do not contain duplicates, then predecessor lists can also be expected
//! not to contain duplicates. However, if successor lists do contain duplicates, then predecessor
//! lists may also contain duplicates.

use std::ops::{Index, Range};

use slotmap::SecondaryMap;

use crate::Function;
use crate::cfg::{BasicBlock, Cfg};

/// A data structure for efficiently querying the predecessors of basic blocks in a [Cfg].
///
/// See module-level documentation for details.
pub struct Predecessors {
    data: Vec<BasicBlock>,
    ranges: SecondaryMap<BasicBlock, Range<usize>>,
}

impl Index<BasicBlock> for Predecessors {
    type Output = [BasicBlock];

    fn index(&self, bb: BasicBlock) -> &Self::Output {
        let range = self.ranges[bb].clone();

        &self.data[range]
    }
}

/// Constructs a data structure that allows you to efficiently query to the predecessors of
/// basic-blocks in the body of the given function.
pub fn predecessors(cfg: &Cfg, function: Function) -> Predecessors {
    let body = cfg
        .get_function_body(function)
        .expect("function not registered");
    let bb_count = body.basic_blocks().len();

    let mut counts: SecondaryMap<BasicBlock, usize> = SecondaryMap::with_capacity(bb_count);

    for bb in body.basic_blocks() {
        for successor in cfg.successors(*bb) {
            let entry = counts.entry(*successor).unwrap().or_default();

            *entry += 1;
        }
    }

    let mut offset = 0;
    let mut ranges: SecondaryMap<BasicBlock, Range<usize>> = SecondaryMap::with_capacity(bb_count);

    for bb in body.basic_blocks() {
        let count = counts.get(*bb).copied().unwrap_or_default();

        // We initialize with an empty range. We'll use this in the next step to decide where to
        // insert into the `data` list.
        ranges.insert(*bb, offset..offset);

        offset += count;
    }

    // Initialize the `data` list with null keys.
    let mut data = vec![BasicBlock::default(); offset];

    for bb in body.basic_blocks() {
        for successor in cfg.successors(*bb) {
            // We just inserted a range for every basic-block in the function body in the loop
            // above, so we can expect a range to exist.
            let range = ranges
                .get_mut(*successor)
                .expect("there should be a range for every basic-block in the function body");

            // We use the end to keep track of the next insertion point for each of the predecessor
            // lists. We initialize each range end to be identical to its start, then increment
            // after each predecessor insertion. Because the range starts were initialized as a
            // prefix-sum over the predecessor counts, in the final result none of the ranges will
            // overlap.
            data[range.end] = *bb;
            range.end += 1;
        }
    }

    Predecessors { data, ranges }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg::Terminator;
    use crate::ty::{TY_DUMMY, TY_U32};
    use crate::{FnArg, FnSig, Module, Symbol};

    #[test]
    fn test_predecessors() {
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

        cfg.set_terminator(bb0, Terminator::branch_u32(a0, [bb1, bb4]));
        cfg.set_terminator(bb1, Terminator::branch_u32(a1, [bb2, bb3]));
        cfg.set_terminator(bb2, Terminator::branch_single(bb5));
        cfg.set_terminator(bb3, Terminator::branch_single(bb4));
        cfg.set_terminator(bb4, Terminator::branch_single(bb5));

        let preds = predecessors(&cfg, function);

        assert_eq!(&preds[bb0], &[]);
        assert_eq!(&preds[bb1], &[bb0]);
        assert_eq!(&preds[bb2], &[bb1]);
        assert_eq!(&preds[bb3], &[bb1]);

        let mut bb4_preds = preds[bb4].to_vec();
        bb4_preds.sort();
        let mut expected_bb4_preds = vec![bb0, bb3];
        expected_bb4_preds.sort();
        assert_eq!(bb4_preds, expected_bb4_preds);

        let mut bb5_preds = preds[bb5].to_vec();
        bb5_preds.sort();
        let mut expected_bb5_preds = vec![bb2, bb4];
        expected_bb5_preds.sort();
        assert_eq!(bb5_preds, expected_bb5_preds);
    }
}
