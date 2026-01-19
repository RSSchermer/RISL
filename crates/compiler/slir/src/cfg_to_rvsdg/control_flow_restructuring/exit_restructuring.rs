use crate::Function;
use crate::cfg::{BasicBlock, BlockPosition, Branch, Cfg, Terminator};

pub fn restructure_exit(cfg: &mut Cfg, function: Function) -> BasicBlock {
    let body = cfg
        .get_function_body(function)
        .expect("function body should exist");
    let entry_block = body.entry_block();
    let return_ty = body.return_ty();

    let mut exit_nodes = Vec::new();

    for bb in body.basic_blocks() {
        if cfg[*bb].terminator().is_return() {
            exit_nodes.push(*bb);
        }
    }

    if exit_nodes.len() > 1 {
        // If the function returns a value, then create a new uninitialized binding at the start of
        // the function to which we can assign the return values of the various exits. We will
        // return this binding's value from the new single exit.
        let return_binding = return_ty.map(|ty| {
            cfg.add_stmt_uninitialized(entry_block, BlockPosition::Prepend, ty)
                .1
        });

        let new_exit = cfg.add_basic_block(function);

        for old_exit in exit_nodes {
            if let Some(value) = cfg[old_exit].terminator().expect_return() {
                let binding = return_binding
                    .expect("branch shouldn't return a value if the function returns void");

                cfg.add_stmt_assign(old_exit, BlockPosition::Append, binding, *value);
            }

            cfg.set_terminator(old_exit, Terminator::branch_single(new_exit));
        }

        if let Some(return_binding) = return_binding {
            cfg.set_terminator(new_exit, Terminator::return_value(return_binding.into()));
        }

        new_exit
    } else if exit_nodes.len() == 1 {
        exit_nodes[0]
    } else {
        // An empty block that returns `void`
        cfg.add_basic_block(function)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ty::{TY_BOOL, TY_DUMMY};
    use crate::{FnArg, FnSig, Module, Symbol};

    #[test]
    fn test_restructure_exit() {
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

        // Before:
        //
        //       bb0
        //       /  \
        //      /    \
        //     v      v
        //    bb1    bb2
        //

        let bb0 = body.entry_block();
        let bb1 = cfg.add_basic_block(function);
        let bb2 = cfg.add_basic_block(function);

        let (_, predicate) =
            cfg.add_stmt_op_bool_to_branch_selector(bb0, BlockPosition::Append, arg_0.into());

        cfg.set_terminator(bb0, Terminator::branch_multiple(predicate, [bb1, bb2]));

        restructure_exit(&mut cfg, function);

        let body = cfg
            .get_function_body(function)
            .expect("function body should exist");

        assert_eq!(body.basic_blocks().len(), 4);

        let new_exit = body.basic_blocks()[3];

        let bb1_term = cfg[bb1].terminator().expect_branch();

        assert_eq!(bb1_term.targets(), &[new_exit]);

        let bb2_term = cfg[bb2].terminator().expect_branch();

        assert_eq!(bb2_term.targets(), &[new_exit]);

        let new_exit_term = cfg[new_exit].terminator().expect_return();

        assert!(new_exit_term.is_none());
    }

    #[test]
    fn test_restructure_exit_infinite_loop() {
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
                args: vec![],
                ret_ty: None,
            },
        );

        let mut cfg = Cfg::new(module.ty.clone());

        let body = cfg.register_function(&module, function);

        // Before:
        //
        //       bb0
        //       /  ^
        //      /    \
        //     v      \
        //    bb1--->bb2
        //

        let bb0 = body.entry_block();
        let bb1 = cfg.add_basic_block(function);
        let bb2 = cfg.add_basic_block(function);

        cfg.set_terminator(bb0, Terminator::branch_single(bb1));
        cfg.set_terminator(bb1, Terminator::branch_single(bb2));
        cfg.set_terminator(bb2, Terminator::branch_single(bb0));

        restructure_exit(&mut cfg, function);

        let body = cfg
            .get_function_body(function)
            .expect("function body should exist");

        assert_eq!(body.basic_blocks().len(), 4);

        let exit = body.basic_blocks()[3];

        assert_eq!(
            cfg[bb0].terminator(),
            &Terminator::Branch(Branch::single(bb1))
        );
        assert_eq!(
            cfg[bb1].terminator(),
            &Terminator::Branch(Branch::single(bb2))
        );
        assert_eq!(
            cfg[bb2].terminator(),
            &Terminator::Branch(Branch::single(bb0))
        );

        assert_eq!(cfg[exit].terminator(), &Terminator::Return(None));
    }
}
