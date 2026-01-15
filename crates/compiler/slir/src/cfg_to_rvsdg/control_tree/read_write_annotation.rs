use indexmap::IndexSet;

use crate::cfg::{Assign, BasicBlock, Bind, Cfg, IntrinsicOp, LocalBinding, OpCall, StatementData, Terminator, Uninitialized, Value};
use crate::cfg_to_rvsdg::control_tree::control_tree::{
    BranchingNode, ControlTree, ControlTreeNode, ControlTreeNodeKind, LinearNode, LoopNode,
};
use crate::cfg_to_rvsdg::control_tree::slice_annotation::SliceAnnotation;

pub trait WithReadValues {
    fn with_read_values<F>(&self, f: F)
    where
        F: FnMut(&Value);
}

impl WithReadValues for Assign {
    fn with_read_values<F>(&self, mut f: F)
    where
        F: FnMut(&Value),
    {
        f(&self.value());
    }
}

impl WithReadValues for Bind {
    fn with_read_values<F>(&self, mut f: F)
    where
        F: FnMut(&Value),
    {
        f(&self.value());
    }
}

impl WithReadValues for Uninitialized {
    fn with_read_values<F>(&self, mut f: F)
    where
        F: FnMut(&Value),
    {
    }
}

impl<T> WithReadValues for IntrinsicOp<T> {
    fn with_read_values<F>(&self, mut f: F)
    where
        F: FnMut(&Value),
    {
        for arg in self.arguments() {
            f(arg)
        }
    }
}

impl WithReadValues for OpCall {
    fn with_read_values<F>(&self, f: F)
    where
        F: FnMut(&Value),
    {
        self.arguments().iter().for_each(f);
    }
}

macro_rules! impl_with_read_values_statement {
    ($($op:ident,)*) => {
        impl WithReadValues for StatementData {
            fn with_read_values<F>(&self, f: F)
            where
                F: FnMut(&Value),
            {
                match self {
                    $(StatementData::$op(op) => op.with_read_values(f),)*
                }
            }
        }
    };
}

impl_with_read_values_statement! {
    Bind,
    Uninitialized,
    Assign,
    OpAlloca,
    OpLoad,
    OpStore,
    OpExtractField,
    OpExtractElement,
    OpFieldPtr,
    OpElementPtr,
    OpVariantPtr,
    OpGetDiscriminant,
    OpSetDiscriminant,
    OpOffsetSlice,
    OpUnary,
    OpBinary,
    OpCall,
    OpCaseToBranchSelector,
    OpBoolToBranchSelector,
    OpConvertToU32,
    OpConvertToI32,
    OpConvertToF32,
    OpConvertToBool,
}

pub trait WithWrittenValues {
    fn with_written_values<F>(&self, f: F)
    where
        F: FnMut(&LocalBinding);
}

impl WithWrittenValues for Assign {
    fn with_written_values<F>(&self, mut f: F)
    where
        F: FnMut(&LocalBinding),
    {
        f(&self.local_binding().into())
    }
}

impl WithWrittenValues for Bind {
    fn with_written_values<F>(&self, mut f: F)
    where
        F: FnMut(&LocalBinding),
    {
        f(&self.local_binding().into())
    }
}

impl WithWrittenValues for Uninitialized {
    fn with_written_values<F>(&self, _: F)
    where
        F: FnMut(&LocalBinding),
    {
    }
}

impl<T> WithWrittenValues for IntrinsicOp<T> {
    fn with_written_values<F>(&self, f: F)
    where
        F: FnMut(&LocalBinding)
    {
        self.maybe_result().as_ref().map(f);
    }
}

impl WithWrittenValues for OpCall {
    fn with_written_values<F>(&self, mut f: F)
    where
        F: FnMut(&LocalBinding),
    {
        self.maybe_result().as_ref().map(f);
    }
}

macro_rules! impl_with_written_values_statement {
    ($($op:ident,)*) => {
        impl WithWrittenValues for StatementData {
            fn with_written_values<F>(&self, f: F)
            where
                F: FnMut(&LocalBinding),
            {
                match self {
                    $(StatementData::$op(op) => op.with_written_values(f),)*
                }
            }
        }
    };
}

impl_with_written_values_statement! {
    Bind,
    Uninitialized,
    Assign,
    OpAlloca,
    OpLoad,
    OpStore,
    OpExtractField,
    OpExtractElement,
    OpFieldPtr,
    OpElementPtr,
    OpVariantPtr,
    OpGetDiscriminant,
    OpSetDiscriminant,
    OpOffsetSlice,
    OpUnary,
    OpBinary,
    OpCall,
    OpCaseToBranchSelector,
    OpBoolToBranchSelector,
    OpConvertToU32,
    OpConvertToI32,
    OpConvertToF32,
    OpConvertToBool,
}

pub struct ReadWriteAnnotationVisitor<'a> {
    control_tree: &'a ControlTree,
    cfg: &'a Cfg,
    read_state: SliceAnnotation<LocalBinding>,
    write_state: SliceAnnotation<LocalBinding>,
    read_accum: IndexSet<LocalBinding>,
    write_accum: IndexSet<LocalBinding>,
}

impl<'a> ReadWriteAnnotationVisitor<'a> {
    fn new(control_tree: &'a ControlTree, cfg: &'a Cfg) -> Self {
        ReadWriteAnnotationVisitor {
            control_tree,
            cfg,
            read_state: SliceAnnotation::new(control_tree.node_count()),
            write_state: SliceAnnotation::new(control_tree.node_count()),
            read_accum: Default::default(),
            write_accum: Default::default(),
        }
    }

    fn visit(&mut self, node: ControlTreeNode) {
        match &self.control_tree[node] {
            ControlTreeNodeKind::BasicBlock(bb) => self.visit_basic_block((node, *bb)),
            ControlTreeNodeKind::Linear(data) => self.visit_linear_node((node, data)),
            ControlTreeNodeKind::Branching(data) => self.visit_branching_node((node, data)),
            ControlTreeNodeKind::Loop(data) => self.visit_loop_node((node, data)),
        }
    }

    fn visit_basic_block(&mut self, (node, bb): (ControlTreeNode, BasicBlock)) {
        self.read_accum.clear();
        self.write_accum.clear();

        match self.cfg[bb].terminator() {
            Terminator::Branch(b) => {
                if let Some(selector) = b.selector() {
                    self.read_accum.insert(selector);
                }
            }
            Terminator::Return(value) => {
                if let Some(Value::Local(value)) = value {
                    self.read_accum.insert(*value);
                }
            }
        }

        for statement in self.cfg[bb].statements().iter().rev() {
            let statement = &self.cfg[*statement];

            statement.with_written_values(|v| {
                self.read_accum.shift_remove(v);
                self.write_accum.insert(*v);
            });

            statement.with_read_values(|value| {
                if let Value::Local(value) = value {
                    self.read_accum.insert(*value);
                }
            });
        }

        self.read_state.set(node, self.read_accum.iter().copied());
        self.write_state.set(node, self.write_accum.iter().copied());
    }

    fn visit_linear_node(&mut self, (node, data): (ControlTreeNode, &LinearNode)) {
        for child in &data.children {
            self.visit(*child);
        }

        self.read_accum.clear();
        self.write_accum.clear();

        for child in data.children.iter().rev() {
            for value in self.write_state.get(*child) {
                self.read_accum.shift_remove(value);
                self.write_accum.insert(*value);
            }

            self.read_accum.extend(self.read_state.get(*child));
        }

        self.read_state.set(node, self.read_accum.iter().copied());
        self.write_state.set(node, self.write_accum.iter().copied());
    }

    fn visit_branching_node(&mut self, (node, data): (ControlTreeNode, &BranchingNode)) {
        for branch in &data.branches {
            self.visit(*branch);
        }

        self.write_accum.clear();

        for (i, branch) in data.branches.iter().enumerate() {
            if i == 0 {
                self.write_accum.extend(self.write_state.get(*branch));
            } else {
                // Retain only the intersection

                // Temporarily re-use the read accumulation set to help construct the write set
                // intersection.
                self.read_accum.clear();
                self.read_accum.extend(self.write_state.get(*branch));

                self.write_accum.retain(|v| self.read_accum.contains(v));
            }
        }

        self.read_accum.clear();

        for branch in &data.branches {
            self.read_accum.extend(self.read_state.get(*branch));
        }

        self.read_state.set(node, self.read_accum.iter().copied());
        self.write_state.set(node, self.write_accum.iter().copied());
    }

    fn visit_loop_node(&mut self, (node, data): (ControlTreeNode, &LoopNode)) {
        self.visit(data.inner);

        self.read_state.copy(data.inner, node);
        self.write_state.copy(data.inner, node);
    }

    fn into_annotation_state(
        self,
    ) -> (SliceAnnotation<LocalBinding>, SliceAnnotation<LocalBinding>) {
        (self.read_state, self.write_state)
    }
}

pub fn annotate_read_write(
    control_tree: &ControlTree,
    cfg: &Cfg,
) -> (SliceAnnotation<LocalBinding>, SliceAnnotation<LocalBinding>) {
    let mut visitor = ReadWriteAnnotationVisitor::new(control_tree, cfg);

    visitor.visit(control_tree.root());
    visitor.into_annotation_state()
}

#[cfg(test)]
mod tests {
    use smallvec::smallvec;

    use super::*;
    use crate::cfg::{
        Assign, BlockPosition, Branch, InlineConst, LocalBindingData, OpBinary, Terminator,
    };
    use crate::cfg_to_rvsdg::control_flow_restructuring::{
        Graph, restructure_branches, restructure_loops,
    };
    use crate::ty::{TY_BOOL, TY_DUMMY, TY_U32};
    use crate::{BinaryOperator, FnArg, FnSig, Function, Module, Symbol};

    #[test]
    fn annotate_read_write_bb_test() {
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

        let x = body.argument_values()[0];

        let bb0 = body.entry_block();

        let (_, c) = cfg.add_stmt_op_binary(
            bb0,
            BlockPosition::Append,
            BinaryOperator::Add,
            x.into(),
            1u32.into(),
        );

        let mut graph = Graph::init(&mut cfg, function);
        let reentry_edges = restructure_loops(&mut graph);
        let branch_info = restructure_branches(&mut graph, &reentry_edges);

        let control_tree = ControlTree::generate(&graph, &reentry_edges, &branch_info);

        let (read, write) = annotate_read_write(&control_tree, &cfg);

        #[allow(non_snake_case)]
        {
            let A = control_tree.root();
            let A_children = &control_tree[A].expect_linear().children;
            let B = A_children[0];

            assert_eq!(read.get(A), &[x]);
            assert_eq!(read.get(B), &[x]);

            assert_eq!(write.get(A), &[c]);
            assert_eq!(write.get(B), &[c]);
        }
    }

    #[test]
    fn annotate_read_write_test() {
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
        let (_, s) = cfg.add_stmt_op_binary(
            bb2,
            BlockPosition::Append,
            BinaryOperator::Mod,
            x.into(),
            y.into(),
        );
        cfg.add_stmt_assign(bb2, BlockPosition::Append, y, s.into());
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

        // Control-tree layout:
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
        //
        // Expected read and write annotations:
        //
        //     | Node | Read       | Write         |
        //     |------|------------|---------------|
        //     | A    | x, y       | r, c          |
        //     | B    |            |               |
        //     | C    | x, y       | r, c          |
        //     | D    | x          |               |
        //     | E    | x, y       | r, c          |
        //     | F    | y          | c             |
        //     | G    | x, y       | r             |
        //     | H    | r          |               |
        //     | I    |            | r             |
        //     | J    | x, y       | r, x, y, s, t |
        //     | K    |            | r             |
        //     | L    | x, y       | x, y, s, t    |
        //     | M    |            | r             |
        //

        let (read, write) = annotate_read_write(&control_tree, &cfg);

        #[allow(non_snake_case)]
        {
            let A = control_tree.root();
            let A_children = &control_tree[A].expect_linear().children;
            let B = A_children[0];
            let C = A_children[1];
            let D = A_children[2];
            let E = control_tree[C].expect_loop().inner;
            let E_children = &control_tree[E].expect_linear().children;
            let F = E_children[0];
            let G = E_children[1];
            let G_branches = &control_tree[G].expect_branching().branches;
            let H = E_children[2];
            let I = G_branches[0];
            let I_children = &control_tree[I].expect_linear().children;
            let J = G_branches[1];
            let J_children = &control_tree[J].expect_linear().children;
            let K = I_children[0];
            let L = J_children[0];
            let M = J_children[1];

            assert_eq!(read.get(A), &[x, y]);
            assert_eq!(read.get(B), &[]);
            assert_eq!(read.get(C), &[x, y]);
            assert_eq!(read.get(D), &[x]);
            assert_eq!(read.get(E), &[x, y]);
            assert_eq!(read.get(F), &[y]);
            assert_eq!(read.get(G), &[x, y]);
            assert_eq!(read.get(H), &[r]);
            assert_eq!(read.get(I), &[]);
            assert_eq!(read.get(J), &[x, y]);
            assert_eq!(read.get(K), &[]);
            assert_eq!(read.get(L), &[x, y]);
            assert_eq!(read.get(M), &[]);

            assert_eq!(write.get(A), &[r, c]);
            assert_eq!(write.get(B), &[]);
            assert_eq!(write.get(C), &[r, c]);
            assert_eq!(write.get(D), &[]);
            assert_eq!(write.get(E), &[r, c]);
            assert_eq!(write.get(F), &[c]);
            assert_eq!(write.get(G), &[r]);
            assert_eq!(write.get(H), &[]);
            assert_eq!(write.get(I), &[r]);
            assert_eq!(write.get(J), &[r, x, y, s, t]);
            assert_eq!(write.get(K), &[r]);
            assert_eq!(write.get(L), &[x, y, s, t]);
            assert_eq!(write.get(M), &[r]);
        }
    }
}
