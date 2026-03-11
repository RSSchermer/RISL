//! Transforms away terminators of kind [Terminator::Unreachable].
//!
//! To accommodate straight-forward construction of SLIR-CFG from Rust's Mid-level Intermediate
//! Representation (MIR) we support basic-blocks that terminate with "unreachable". The Rust
//! compiler likes to generate such basic-blocks for exhaustive `match` expressions without a
//! "catch-all" pattern; here it represents the "default" branch with an empty basic-block with an
//! "unreachable" terminator to indicate that the default branch will go unused.
//!
//! We have to transform away basic-blocks that end in an "unreachable" terminator, as other
//! transforms we apply ([loop_restructuring][0] and [branch_restructuring][1]) assume control-flow
//! graphs with exactly one "exit node": a basic-block that has no successors. The
//! [exit_restructuring][2] transform will ensure that there is exactly one basic-block that ends in
//! a [Terminator::Return] terminator. Since that already fills our quota, all
//! [Terminator::Unreachable] terminators will have to be eliminated.
//!
//! Fortunately, reaching a basic-block with an "unreachable" terminator is explicitly Undefined
//! Behavior. We are therefore allowed to assume that such terminators will indeed never be reached.
//! As reaching any part (statement/terminator) of a basic-block implies reaching all parts of a
//! basic-block, the terminator being unreachable implies that all the basic-block's statements are
//! also unreachable. Consequently, we can eliminate an "unreachable" terminator by removing the
//! entire basic-block, without affecting the semantics of the program, assuming the program is
//! valid (contains no Undefined Behavior).
//!
//! However, this may leave the terminators of the predecessors of the removed basic-block in an
//! invalid state, with a branch target that references a basic-block that no longer exists. The
//! meat of this transform is therefore concerned with adjusting the terminators of the unreachable
//! block's predecessors.
//!
//! All predecessors can be assumed to end in [Branch][3] terminators (as only basic-blocks that end
//! in [Branch][3] terminators can have successors by definition). The exact adjustment that is made
//! depends on the [BranchSelector] kind of the terminator's [Branch::selector][4] and the number of
//! [Branch::targets][5]:
//!
//! - There is only a single branch target: this implies that reaching the predecessors terminator
//!   would mean reaching the unreachable block. Since the unreachable block cannot be reached, this
//!   implies that the predecessor's terminator was in actuality also "unreachable". We set the
//!   predecessor's terminator to [Terminator::Unreachable] and recurse.
//! - There are exactly two branch targets: since one branch target is an unreachable block, this
//!   implies the terminator will always select the other branch. We replace the predecessor's
//!   branch-selector with a [BranchSelector::Single] and remove the "unreachable" branch target.
//! - [BranchSelector::Case] with more than 2 branch targets: a valid "case" selector will have one
//!   less case in its `cases` list than there are branch targets. If the unreachable block is the
//!   last target is the `targets` list, then we remove the last target from the `targets` list, and
//!   we remove the last case from the `cases` list. Otherwise, if the unreachable block is the
//!   `i`th target, then we remove the `i`th target from the `targets` list, and we remove the `i`th
//!   case from the `cases` list.
//! - [BranchSelector::U32] with more than 2 branch targets: if the unreachable block is the last
//!   target in the `targets` list, then we simply remove the last target. Otherwise, we leverage
//!   the fact that a "u32" selector for a branching terminator with `N` targets is equivalent to a
//!   "case" selector with `N-1` cases where the cases are consecutive integers starting at `0`. For
//!   example, a "u32" selector for a branching terminator with 4 targets is equivalent to a "case"
//!   selector with cases list `[0, 1, 2]`. We therefore convert to the equivalent "case" selector
//!   and modify the `cases` list as described for [BranchSelector::Case] above.
//!
//! [0]: crate::cfg::transform::loop_restructuring
//! [1]: crate::cfg::transform::branch_restructuring
//! [2]: crate::cfg::transform::exit_restructuring
//! [3]: crate::cfg::Branch
//! [4]: crate::cfg::Branch::selector
//! [5]: crate::cfg::Branch::targets

use crate::Function;
use crate::cfg::analyze::predecessors::predecessors;
use crate::cfg::{BranchSelector, Cfg, Terminator};

pub fn eliminate_unreachable_terminators(cfg: &mut Cfg, function: Function) {
    loop {
        let mut changed = false;
        let mut to_remove = Vec::new();

        let body = cfg[function].basic_blocks().clone();

        for &bb in &body {
            if matches!(cfg[bb].terminator(), Terminator::Unreachable) {
                to_remove.push(bb);
            }
        }

        if to_remove.is_empty() {
            break;
        }

        let predecessors = predecessors(cfg, function);

        for bb in to_remove {
            for predecessor in predecessors[bb].iter().copied() {
                let branch = cfg[predecessor].terminator().expect_branch().clone();
                let targets = branch.targets().to_vec();

                if targets.len() == 1 {
                    assert_eq!(targets[0], bb);

                    cfg.set_terminator(predecessor, Terminator::Unreachable);
                } else if targets.len() == 2 {
                    let other_target = if targets[0] == bb {
                        targets[1]
                    } else {
                        targets[0]
                    };

                    cfg.set_terminator(predecessor, Terminator::branch_single(other_target));
                } else {
                    let target_index = targets.iter().position(|&t| t == bb).unwrap();

                    let mut new_targets = targets.clone();
                    new_targets.remove(target_index);

                    match branch.selector().clone() {
                        BranchSelector::Case { value, mut cases } => {
                            if target_index < cases.len() {
                                cases.remove(target_index);
                            } else {
                                // It's the last target, remove the last case.
                                cases.pop();
                            }

                            cfg.set_terminator(
                                predecessor,
                                Terminator::branch_case(value, cases, new_targets),
                            );
                        }
                        BranchSelector::U32(value) => {
                            if target_index == new_targets.len() {
                                // Last target, just remove it.
                                cfg.set_terminator(
                                    predecessor,
                                    Terminator::branch_u32(value, new_targets),
                                );
                            } else {
                                // The unreachable block is not the last target, convert to a Case
                                // selector.

                                let mut cases: Vec<u32> = (0..targets.len() as u32 - 1).collect();
                                cases.remove(target_index);

                                cfg.set_terminator(
                                    predecessor,
                                    Terminator::branch_case(value, cases, new_targets),
                                );
                            }
                        }
                        _ => unreachable!("expected Case or U32 selector for >2 targets"),
                    }
                }

                changed = true;
            }

            cfg.remove_basic_block(bb);
        }

        if !changed {
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ty::{TY_BOOL, TY_U32};
    use crate::{FnArg, FnSig, Module, Symbol};

    #[test]
    fn test_predecessor_single_branch_target() {
        let mut module = Module::new(Symbol::from_ref("test_module"));
        let function = Function {
            name: Symbol::from_ref("test_func"),
            module: module.name,
        };

        module.fn_sigs.register(
            function,
            FnSig {
                name: Default::default(),
                ty: TY_BOOL,
                args: vec![FnArg {
                    ty: TY_BOOL,
                    shader_io_binding: None,
                }],
                ret_ty: None,
            },
        );

        let mut cfg = Cfg::new(module.ty.clone());

        let body = cfg.register_function(&module, function);

        let bb_entry = body.entry_block();
        let arg0 = body.argument_values()[0];

        let bb_ret = cfg.add_basic_block(function);
        let bb_mid = cfg.add_basic_block(function);
        let bb_unreachable = cfg.add_basic_block(function);

        cfg.set_terminator(bb_entry, Terminator::branch_bool(arg0, bb_ret, bb_mid));
        cfg.set_terminator(bb_ret, Terminator::return_void());

        cfg.set_terminator(bb_mid, Terminator::branch_single(bb_unreachable));
        cfg.set_terminator(bb_unreachable, Terminator::Unreachable);

        eliminate_unreachable_terminators(&mut cfg, function);

        assert!(!cfg[function].basic_blocks().contains(&bb_unreachable));
        assert!(!cfg[function].basic_blocks().contains(&bb_mid));

        let branch = cfg[bb_entry].terminator().expect_branch();

        assert!(matches!(branch.selector(), BranchSelector::Single));
        assert_eq!(branch.targets(), &[bb_ret]);
    }

    #[test]
    fn test_predecessor_two_branch_targets() {
        let mut module = Module::new(Symbol::from_ref("test_module"));
        let function = Function {
            name: Symbol::from_ref("test_func"),
            module: module.name,
        };

        module.fn_sigs.register(
            function,
            FnSig {
                name: Default::default(),
                ty: TY_BOOL,
                args: vec![FnArg {
                    ty: TY_BOOL,
                    shader_io_binding: None,
                }],
                ret_ty: None,
            },
        );

        let mut cfg = Cfg::new(module.ty.clone());

        let body = cfg.register_function(&module, function);

        let bb_entry = body.entry_block();
        let arg0 = body.argument_values()[0];

        let bb_ret = cfg.add_basic_block(function);
        let bb_unreachable = cfg.add_basic_block(function);

        cfg.set_terminator(
            bb_entry,
            Terminator::branch_bool(arg0, bb_ret, bb_unreachable),
        );
        cfg.set_terminator(bb_ret, Terminator::return_void());

        cfg.set_terminator(bb_unreachable, Terminator::Unreachable);

        eliminate_unreachable_terminators(&mut cfg, function);

        assert!(!cfg[function].basic_blocks().contains(&bb_unreachable));

        let branch = cfg[bb_entry].terminator().expect_branch();

        assert!(matches!(branch.selector(), BranchSelector::Single));
        assert_eq!(branch.targets(), &[bb_ret]);
    }

    #[test]
    fn test_predecessor_case_selector_last_target() {
        let mut module = Module::new(Symbol::from_ref("test_module"));
        let function = Function {
            name: Symbol::from_ref("test_func"),
            module: module.name,
        };

        module.fn_sigs.register(
            function,
            FnSig {
                name: Default::default(),
                ty: TY_BOOL,
                args: vec![FnArg {
                    ty: TY_U32,
                    shader_io_binding: None,
                }],
                ret_ty: None,
            },
        );

        let mut cfg = Cfg::new(module.ty.clone());

        let body = cfg.register_function(&module, function);

        let bb_entry = body.entry_block();
        let arg0 = body.argument_values()[0];

        let bb0 = cfg.add_basic_block(function);
        let bb1 = cfg.add_basic_block(function);
        let bb_unreachable = cfg.add_basic_block(function);

        cfg.set_terminator(
            bb_entry,
            Terminator::branch_case(arg0, [0, 1], [bb0, bb1, bb_unreachable]),
        );
        cfg.set_terminator(bb0, Terminator::return_void());
        cfg.set_terminator(bb1, Terminator::return_void());
        cfg.set_terminator(bb_unreachable, Terminator::Unreachable);

        eliminate_unreachable_terminators(&mut cfg, function);

        assert!(!cfg[function].basic_blocks().contains(&bb_unreachable));

        let branch = cfg[bb_entry].terminator().expect_branch();

        if let BranchSelector::Case { cases, .. } = branch.selector() {
            assert_eq!(cases.as_slice(), &[0]);
            assert_eq!(branch.targets(), &[bb0, bb1]);
        } else {
            panic!("expected case selector");
        }
    }

    #[test]
    fn test_predecessor_case_selector_middle_target() {
        let mut module = Module::new(Symbol::from_ref("test_module"));
        let function = Function {
            name: Symbol::from_ref("test_func"),
            module: module.name,
        };

        module.fn_sigs.register(
            function,
            FnSig {
                name: Default::default(),
                ty: TY_BOOL,
                args: vec![FnArg {
                    ty: TY_U32,
                    shader_io_binding: None,
                }],
                ret_ty: None,
            },
        );

        let mut cfg = Cfg::new(module.ty.clone());

        let body = cfg.register_function(&module, function);

        let bb_entry = body.entry_block();
        let arg0 = body.argument_values()[0];

        let bb0 = cfg.add_basic_block(function);
        let bb_unreachable = cfg.add_basic_block(function);
        let bb2 = cfg.add_basic_block(function);
        let bb3 = cfg.add_basic_block(function);

        cfg.set_terminator(
            bb_entry,
            Terminator::branch_case(arg0, [0, 1, 2], [bb0, bb_unreachable, bb2, bb3]),
        );
        cfg.set_terminator(bb0, Terminator::return_void());
        cfg.set_terminator(bb_unreachable, Terminator::Unreachable);
        cfg.set_terminator(bb2, Terminator::return_void());
        cfg.set_terminator(bb3, Terminator::return_void());

        eliminate_unreachable_terminators(&mut cfg, function);

        assert!(!cfg[function].basic_blocks().contains(&bb_unreachable));

        let branch = cfg[bb_entry].terminator().expect_branch();

        if let BranchSelector::Case { cases, .. } = branch.selector() {
            assert_eq!(cases.as_slice(), &[0, 2]);
            assert_eq!(branch.targets(), &[bb0, bb2, bb3]);
        } else {
            panic!("expected case selector");
        }
    }

    #[test]
    fn test_predecessor_u32_selector_last_target() {
        let mut module = Module::new(Symbol::from_ref("test_module"));
        let function = Function {
            name: Symbol::from_ref("test_func"),
            module: module.name,
        };

        module.fn_sigs.register(
            function,
            FnSig {
                name: Default::default(),
                ty: TY_BOOL,
                args: vec![FnArg {
                    ty: TY_U32,
                    shader_io_binding: None,
                }],
                ret_ty: None,
            },
        );

        let mut cfg = Cfg::new(module.ty.clone());

        let body = cfg.register_function(&module, function);

        let bb_entry = body.entry_block();
        let arg0 = body.argument_values()[0];

        let bb0 = cfg.add_basic_block(function);
        let bb1 = cfg.add_basic_block(function);
        let bb_unreachable = cfg.add_basic_block(function);

        cfg.set_terminator(
            bb_entry,
            Terminator::branch_u32(arg0, [bb0, bb1, bb_unreachable]),
        );
        cfg.set_terminator(bb0, Terminator::return_void());
        cfg.set_terminator(bb1, Terminator::return_void());
        cfg.set_terminator(bb_unreachable, Terminator::Unreachable);

        eliminate_unreachable_terminators(&mut cfg, function);

        assert!(!cfg[function].basic_blocks().contains(&bb_unreachable));

        let branch = cfg[bb_entry].terminator().expect_branch();

        assert!(matches!(branch.selector(), BranchSelector::U32(_)));
        assert_eq!(branch.targets(), &[bb0, bb1]);
    }

    #[test]
    fn test_predecessor_u32_selector_middle_target() {
        let mut module = Module::new(Symbol::from_ref("test_module"));
        let function = Function {
            name: Symbol::from_ref("test_func"),
            module: module.name,
        };

        module.fn_sigs.register(
            function,
            FnSig {
                name: Default::default(),
                ty: TY_BOOL,
                args: vec![FnArg {
                    ty: TY_U32,
                    shader_io_binding: None,
                }],
                ret_ty: None,
            },
        );

        let mut cfg = Cfg::new(module.ty.clone());

        let body = cfg.register_function(&module, function);

        let bb_entry = body.entry_block();
        let arg0 = body.argument_values()[0];

        let bb0 = cfg.add_basic_block(function);
        let bb_unreachable = cfg.add_basic_block(function);
        let bb2 = cfg.add_basic_block(function);
        let bb3 = cfg.add_basic_block(function);

        cfg.set_terminator(
            bb_entry,
            Terminator::branch_u32(arg0, [bb0, bb_unreachable, bb2, bb3]),
        );
        cfg.set_terminator(bb0, Terminator::return_void());
        cfg.set_terminator(bb_unreachable, Terminator::Unreachable);
        cfg.set_terminator(bb2, Terminator::return_void());
        cfg.set_terminator(bb3, Terminator::return_void());

        eliminate_unreachable_terminators(&mut cfg, function);

        assert!(!cfg[function].basic_blocks().contains(&bb_unreachable));

        let branch = cfg[bb_entry].terminator().expect_branch();

        if let BranchSelector::Case { cases, .. } = branch.selector() {
            assert_eq!(cases.as_slice(), &[0, 2]);
            assert_eq!(branch.targets(), &[bb0, bb2, bb3]);
        } else {
            panic!("expected case selector");
        }
    }
}
