use rustc_hash::FxHashMap;
use slotmap::{Key, KeyData};

use crate::rvsdg::{
    Connectivity, Node, NodeKind, Region, Rvsdg, SimpleNode, StateOrigin, ValueInput, ValueOrigin,
};

fn canonical_cache_key<K: Key>(k0: K, k1: K) -> (KeyData, KeyData) {
    let d0 = k0.data();
    let d1 = k1.data();

    if d0 < d1 { (d0, d1) } else { (d1, d0) }
}

/// A tool for determining if two RVSDG regions are structurally identical.
///
/// It performs a deep comparison of the regions by tracing their value and state results back to
/// their arguments.
pub struct RegionIdentityChecker {
    node_cache: FxHashMap<(KeyData, KeyData), bool>,
}

impl RegionIdentityChecker {
    /// Creates a new [RegionIdentityChecker].
    pub fn new() -> Self {
        Self {
            node_cache: FxHashMap::default(),
        }
    }

    /// Returns `true` if the two regions are structurally identical.
    pub fn compare_regions(&mut self, rvsdg: &Rvsdg, r0: Region, r1: Region) -> bool {
        self.node_cache.clear();

        self.compare_regions_internal(rvsdg, r0, r1)
    }

    fn compare_regions_internal(&mut self, rvsdg: &Rvsdg, r0: Region, r1: Region) -> bool {
        if r0 == r1 {
            return true;
        }

        let d0 = &rvsdg[r0];
        let d1 = &rvsdg[r1];

        // Compare arguments count and types
        if d0.value_arguments().len() != d1.value_arguments().len() {
            return false;
        }

        for (a0, a1) in d0.value_arguments().iter().zip(d1.value_arguments()) {
            if a0.ty != a1.ty {
                return false;
            }
        }

        // Compare results
        if d0.value_results().len() != d1.value_results().len() {
            return false;
        }

        for (res0, res1) in d0.value_results().iter().zip(d1.value_results()) {
            // This also checks type equivalence, so we don't compare the result types separately
            // here.
            if !self.compare_value_inputs(rvsdg, res0, res1) {
                return false;
            }
        }

        // Compare state result
        if !self.compare_state_origins(rvsdg, d0.state_result(), d1.state_result()) {
            return false;
        }

        true
    }

    fn compare_nodes(&mut self, rvsdg: &Rvsdg, n0: Node, n1: Node) -> bool {
        if n0 == n1 {
            return true;
        }

        let key = canonical_cache_key(n0, n1);

        if let Some(&identical) = self.node_cache.get(&key) {
            return identical;
        }

        // Conservatively insert false to prevent infinitely recursing through potential cycles
        // within a region's value/state flow (though these are not expected in a valid RVSDG).
        self.node_cache.insert(key, false);

        let identical = self.compare_nodes_internal(rvsdg, n0, n1);

        self.node_cache.insert(key, identical);

        identical
    }

    fn compare_nodes_internal(&mut self, rvsdg: &Rvsdg, n0: Node, n1: Node) -> bool {
        let d0 = &rvsdg[n0];
        let d1 = &rvsdg[n1];

        // 1. Compare kind shallowly
        if !self.compare_kinds_shallow(d0.kind(), d1.kind()) {
            return false;
        }

        // 2. Compare inputs
        let in0 = d0.value_inputs();
        let in1 = d1.value_inputs();

        if in0.len() != in1.len() {
            return false;
        }

        for (i0, i1) in in0.iter().zip(in1) {
            if !self.compare_value_inputs(rvsdg, i0, i1) {
                return false;
            }
        }

        // 3. Compare state input
        let s0 = d0.state();
        let s1 = d1.state();

        match (s0, s1) {
            (Some(s0), Some(s1)) => {
                if !self.compare_state_origins(rvsdg, &s0.origin, &s1.origin) {
                    return false;
                }
            }
            (None, None) => {}
            _ => return false,
        }

        // 4. Compare outputs (only types and count)
        let out0 = d0.value_outputs();
        let out1 = d1.value_outputs();

        if out0.len() != out1.len() {
            return false;
        }

        for (o0, o1) in out0.iter().zip(out1) {
            if o0.ty != o1.ty {
                return false;
            }
        }

        // 5. Compare internal regions
        match (d0.kind(), d1.kind()) {
            (NodeKind::Switch(s0), NodeKind::Switch(s1)) => {
                // Number of branches already checked in compare_kinds_shallow
                for (b0, b1) in s0.branches().iter().zip(s1.branches()) {
                    if !self.compare_regions_internal(rvsdg, *b0, *b1) {
                        return false;
                    }
                }
            }
            (NodeKind::Loop(l0), NodeKind::Loop(l1)) => {
                if !self.compare_regions_internal(rvsdg, l0.loop_region(), l1.loop_region()) {
                    return false;
                }
            }
            _ => {}
        }

        true
    }

    fn compare_value_inputs(&mut self, rvsdg: &Rvsdg, i0: &ValueInput, i1: &ValueInput) -> bool {
        if i0.ty != i1.ty {
            return false;
        }

        self.compare_value_origins(rvsdg, &i0.origin, &i1.origin)
    }

    fn compare_value_origins(&mut self, rvsdg: &Rvsdg, o0: &ValueOrigin, o1: &ValueOrigin) -> bool {
        match (o0, o1) {
            (ValueOrigin::Argument(idx0), ValueOrigin::Argument(idx1)) => idx0 == idx1,
            (
                ValueOrigin::Output {
                    producer: p0,
                    output: out0,
                },
                ValueOrigin::Output {
                    producer: p1,
                    output: out1,
                },
            ) => *out0 == *out1 && self.compare_nodes(rvsdg, *p0, *p1),
            _ => false,
        }
    }

    fn compare_state_origins(&mut self, rvsdg: &Rvsdg, o0: &StateOrigin, o1: &StateOrigin) -> bool {
        match (o0, o1) {
            (StateOrigin::Argument, StateOrigin::Argument) => true,
            (StateOrigin::Node(p0), StateOrigin::Node(p1)) => self.compare_nodes(rvsdg, *p0, *p1),
            _ => false,
        }
    }

    fn compare_kinds_shallow(&self, k0: &NodeKind, k1: &NodeKind) -> bool {
        match (k0, k1) {
            (NodeKind::Simple(s0), NodeKind::Simple(s1)) => {
                self.compare_simple_nodes_shallow(s0, s1)
            }
            (NodeKind::Switch(s0), NodeKind::Switch(s1)) => {
                s0.branches().len() == s1.branches().len()
            }
            (NodeKind::Loop(_), NodeKind::Loop(_)) => true,
            // We don't handle equivalences in the global region, as there is only one global region
            // anyway.
            _ => false,
        }
    }

    fn compare_simple_nodes_shallow(&self, s0: &SimpleNode, s1: &SimpleNode) -> bool {
        use SimpleNode::*;

        match (s0, s1) {
            (ConstU32(n0), ConstU32(n1)) => n0.value() == n1.value(),
            (ConstI32(n0), ConstI32(n1)) => n0.value() == n1.value(),
            (ConstF32(n0), ConstF32(n1)) => {
                // We conservatively use bit-wise equality for now.
                n0.value().to_bits() == n1.value().to_bits()
            }
            (ConstBool(n0), ConstBool(n1)) => n0.value() == n1.value(),
            (ConstPredicate(n0), ConstPredicate(n1)) => n0.value() == n1.value(),
            (ConstFallback(_), ConstFallback(_)) => true,
            (OpAlloca(n0), OpAlloca(n1)) => n0.intrinsic() == n1.intrinsic(),
            (OpLoad(n0), OpLoad(n1)) => n0.intrinsic() == n1.intrinsic(),
            (OpStore(n0), OpStore(n1)) => n0.intrinsic() == n1.intrinsic(),
            (OpExtractField(n0), OpExtractField(n1)) => n0.intrinsic() == n1.intrinsic(),
            (OpExtractElement(n0), OpExtractElement(n1)) => n0.intrinsic() == n1.intrinsic(),
            (OpFieldPtr(n0), OpFieldPtr(n1)) => n0.intrinsic() == n1.intrinsic(),
            (OpElementPtr(n0), OpElementPtr(n1)) => n0.intrinsic() == n1.intrinsic(),
            (OpDiscriminantPtr(n0), OpDiscriminantPtr(n1)) => n0.intrinsic() == n1.intrinsic(),
            (OpVariantPtr(n0), OpVariantPtr(n1)) => n0.intrinsic() == n1.intrinsic(),
            (OpGetDiscriminant(n0), OpGetDiscriminant(n1)) => n0.intrinsic() == n1.intrinsic(),
            (OpSetDiscriminant(n0), OpSetDiscriminant(n1)) => n0.intrinsic() == n1.intrinsic(),
            (OpOffsetSlice(n0), OpOffsetSlice(n1)) => n0.intrinsic() == n1.intrinsic(),
            (OpGetSliceOffset(n0), OpGetSliceOffset(n1)) => n0.intrinsic() == n1.intrinsic(),
            (OpUnary(n0), OpUnary(n1)) => n0.intrinsic() == n1.intrinsic(),
            (OpBinary(n0), OpBinary(n1)) => n0.intrinsic() == n1.intrinsic(),
            (OpMax(n0), OpMax(n1)) => n0.intrinsic() == n1.intrinsic(),
            (OpMin(n0), OpMin(n1)) => n0.intrinsic() == n1.intrinsic(),
            (OpRoundToEven(n0), OpRoundToEven(n1)) => n0.intrinsic() == n1.intrinsic(),
            (OpSaturate(n0), OpSaturate(n1)) => n0.intrinsic() == n1.intrinsic(),
            (OpFloor(n0), OpFloor(n1)) => n0.intrinsic() == n1.intrinsic(),
            (OpCeil(n0), OpCeil(n1)) => n0.intrinsic() == n1.intrinsic(),
            (OpClamp(n0), OpClamp(n1)) => n0.intrinsic() == n1.intrinsic(),
            (OpFract(n0), OpFract(n1)) => n0.intrinsic() == n1.intrinsic(),
            (OpFusedMulAdd(n0), OpFusedMulAdd(n1)) => n0.intrinsic() == n1.intrinsic(),
            (OpTrunc(n0), OpTrunc(n1)) => n0.intrinsic() == n1.intrinsic(),
            (OpSqrt(n0), OpSqrt(n1)) => n0.intrinsic() == n1.intrinsic(),
            (OpInverseSqrt(n0), OpInverseSqrt(n1)) => n0.intrinsic() == n1.intrinsic(),
            (OpExp(n0), OpExp(n1)) => n0.intrinsic() == n1.intrinsic(),
            (OpExp2(n0), OpExp2(n1)) => n0.intrinsic() == n1.intrinsic(),
            (OpLog(n0), OpLog(n1)) => n0.intrinsic() == n1.intrinsic(),
            (OpLog2(n0), OpLog2(n1)) => n0.intrinsic() == n1.intrinsic(),
            (OpPowf(n0), OpPowf(n1)) => n0.intrinsic() == n1.intrinsic(),
            (OpStep(n0), OpStep(n1)) => n0.intrinsic() == n1.intrinsic(),
            (OpSmoothStep(n0), OpSmoothStep(n1)) => n0.intrinsic() == n1.intrinsic(),
            (OpCos(n0), OpCos(n1)) => n0.intrinsic() == n1.intrinsic(),
            (OpAcos(n0), OpAcos(n1)) => n0.intrinsic() == n1.intrinsic(),
            (OpCosh(n0), OpCosh(n1)) => n0.intrinsic() == n1.intrinsic(),
            (OpAcosh(n0), OpAcosh(n1)) => n0.intrinsic() == n1.intrinsic(),
            (OpSin(n0), OpSin(n1)) => n0.intrinsic() == n1.intrinsic(),
            (OpAsin(n0), OpAsin(n1)) => n0.intrinsic() == n1.intrinsic(),
            (OpSinh(n0), OpSinh(n1)) => n0.intrinsic() == n1.intrinsic(),
            (OpAsinh(n0), OpAsinh(n1)) => n0.intrinsic() == n1.intrinsic(),
            (OpTan(n0), OpTan(n1)) => n0.intrinsic() == n1.intrinsic(),
            (OpAtan(n0), OpAtan(n1)) => n0.intrinsic() == n1.intrinsic(),
            (OpTanh(n0), OpTanh(n1)) => n0.intrinsic() == n1.intrinsic(),
            (OpAtanh(n0), OpAtanh(n1)) => n0.intrinsic() == n1.intrinsic(),
            (OpToRadians(n0), OpToRadians(n1)) => n0.intrinsic() == n1.intrinsic(),
            (OpToDegrees(n0), OpToDegrees(n1)) => n0.intrinsic() == n1.intrinsic(),
            (OpVector(n0), OpVector(n1)) => n0.intrinsic() == n1.intrinsic(),
            (OpMatrix(n0), OpMatrix(n1)) => n0.intrinsic() == n1.intrinsic(),
            (OpCaseToBranchSelector(n0), OpCaseToBranchSelector(n1)) => {
                n0.intrinsic() == n1.intrinsic()
            }
            (OpBoolToBranchSelector(n0), OpBoolToBranchSelector(n1)) => {
                n0.intrinsic() == n1.intrinsic()
            }
            (OpBranchSelectorToCase(n0), OpBranchSelectorToCase(n1)) => {
                n0.intrinsic() == n1.intrinsic()
            }
            (OpConvertToU32(n0), OpConvertToU32(n1)) => n0.intrinsic() == n1.intrinsic(),
            (OpConvertToI32(n0), OpConvertToI32(n1)) => n0.intrinsic() == n1.intrinsic(),
            (OpConvertToF32(n0), OpConvertToF32(n1)) => n0.intrinsic() == n1.intrinsic(),
            (OpConvertToBool(n0), OpConvertToBool(n1)) => n0.intrinsic() == n1.intrinsic(),
            (OpArrayLength(n0), OpArrayLength(n1)) => n0.intrinsic() == n1.intrinsic(),
            // OpCall nodes are entirely defined by their inputs (the first input is the function
            // being called)
            (OpCall(_), OpCall(_)) => true,
            (ValueProxy(_), ValueProxy(_)) => true,
            (Reaggregation(_), Reaggregation(_)) => true,
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rvsdg::{StateOrigin, ValueInput, ValueOrigin, ValueOutput};
    use crate::ty::{TY_DUMMY, TY_PTR_U32, TY_U32};
    use crate::{BinaryOperator, FnArg, FnSig, Function, Module, Symbol};

    #[test]
    fn test_identical_simple_regions() {
        let module = Module::new(Symbol::from_ref(""));
        let mut rvsdg = Rvsdg::new(module.ty.clone());
        let mut module = module;

        let function = Function {
            name: Symbol::from_ref("f"),
            module: Symbol::from_ref("m"),
        };

        module.fn_sigs.register(
            function,
            FnSig {
                name: function.name,
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

        let (_, r0) = rvsdg.register_function(&module, function, Vec::new());
        let (_, r1) = rvsdg.register_function(&module, function, Vec::new());

        for r in [r0, r1] {
            let arg0 = ValueInput::argument(TY_U32, 0);
            let arg1 = ValueInput::argument(TY_U32, 1);
            let node = rvsdg.add_op_binary(r, BinaryOperator::Add, arg0, arg1);

            rvsdg.reconnect_region_result(
                r,
                0,
                ValueOrigin::Output {
                    producer: node,
                    output: 0,
                },
            );
        }

        let mut checker = RegionIdentityChecker::new();

        assert!(checker.compare_regions(&rvsdg, r0, r1));
    }

    #[test]
    fn test_different_ops() {
        let module = Module::new(Symbol::from_ref(""));
        let mut rvsdg = Rvsdg::new(module.ty.clone());
        let mut module = module;

        let function = Function {
            name: Symbol::from_ref("f"),
            module: Symbol::from_ref("m"),
        };

        module.fn_sigs.register(
            function,
            FnSig {
                name: function.name,
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

        let (_, r0) = rvsdg.register_function(&module, function, Vec::new());
        let (_, r1) = rvsdg.register_function(&module, function, Vec::new());

        {
            let arg0 = ValueInput::argument(TY_U32, 0);
            let arg1 = ValueInput::argument(TY_U32, 1);
            let node = rvsdg.add_op_binary(r0, BinaryOperator::Add, arg0, arg1);

            rvsdg.reconnect_region_result(
                r0,
                0,
                ValueOrigin::Output {
                    producer: node,
                    output: 0,
                },
            );
        }
        {
            let arg0 = ValueInput::argument(TY_U32, 0);
            let arg1 = ValueInput::argument(TY_U32, 1);
            let node = rvsdg.add_op_binary(r1, BinaryOperator::Sub, arg0, arg1);
            rvsdg.reconnect_region_result(
                r1,
                0,
                ValueOrigin::Output {
                    producer: node,
                    output: 0,
                },
            );
        }

        let mut checker = RegionIdentityChecker::new();

        assert!(!checker.compare_regions(&rvsdg, r0, r1));
    }

    #[test]
    fn test_different_inputs() {
        let module = Module::new(Symbol::from_ref(""));
        let mut rvsdg = Rvsdg::new(module.ty.clone());
        let mut module = module;

        let function = Function {
            name: Symbol::from_ref("f"),
            module: Symbol::from_ref("m"),
        };

        module.fn_sigs.register(
            function,
            FnSig {
                name: function.name,
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

        let (_, r0) = rvsdg.register_function(&module, function, Vec::new());
        let (_, r1) = rvsdg.register_function(&module, function, Vec::new());

        {
            let arg0 = ValueInput::argument(TY_U32, 0);
            let arg1 = ValueInput::argument(TY_U32, 1);
            let node = rvsdg.add_op_binary(r0, BinaryOperator::Add, arg0, arg1);

            rvsdg.reconnect_region_result(
                r0,
                0,
                ValueOrigin::Output {
                    producer: node,
                    output: 0,
                },
            );
        }
        {
            let arg0 = ValueInput::argument(TY_U32, 0);
            let arg1 = ValueInput::argument(TY_U32, 0); // Different input
            let node = rvsdg.add_op_binary(r1, BinaryOperator::Add, arg0, arg1);

            rvsdg.reconnect_region_result(
                r1,
                0,
                ValueOrigin::Output {
                    producer: node,
                    output: 0,
                },
            );
        }

        let mut checker = RegionIdentityChecker::new();

        assert!(!checker.compare_regions(&rvsdg, r0, r1));
    }

    #[test]
    fn test_multiple_simple_nodes() {
        let module = Module::new(Symbol::from_ref(""));
        let mut rvsdg = Rvsdg::new(module.ty.clone());
        let mut module = module;

        let function = Function {
            name: Symbol::from_ref("f"),
            module: Symbol::from_ref("m"),
        };

        module.fn_sigs.register(
            function,
            FnSig {
                name: function.name,
                ty: TY_DUMMY,
                args: vec![FnArg {
                    ty: TY_U32,
                    shader_io_binding: None,
                }],
                ret_ty: Some(TY_U32),
            },
        );

        let (_, r0) = rvsdg.register_function(&module, function, Vec::new());
        let (_, r1) = rvsdg.register_function(&module, function, Vec::new());

        for r in [r0, r1] {
            let arg0 = ValueInput::argument(TY_U32, 0);
            let node_a = rvsdg.add_op_binary(r, BinaryOperator::Add, arg0.clone(), arg0.clone());
            let node_b = rvsdg.add_op_binary(
                r,
                BinaryOperator::Mul,
                ValueInput::output(TY_U32, node_a, 0),
                ValueInput::output(TY_U32, node_a, 0),
            );

            rvsdg.reconnect_region_result(
                r,
                0,
                ValueOrigin::Output {
                    producer: node_b,
                    output: 0,
                },
            );
        }

        let mut checker = RegionIdentityChecker::new();

        assert!(checker.compare_regions(&rvsdg, r0, r1));
    }

    #[test]
    fn test_state_flow() {
        let module = Module::new(Symbol::from_ref(""));
        let mut rvsdg = Rvsdg::new(module.ty.clone());
        let mut module = module;

        let ptr_ty = TY_PTR_U32;
        let function = Function {
            name: Symbol::from_ref("f"),
            module: Symbol::from_ref("m"),
        };

        module.fn_sigs.register(
            function,
            FnSig {
                name: function.name,
                ty: TY_DUMMY,
                args: vec![
                    FnArg {
                        ty: ptr_ty,
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

        let (_, r0) = rvsdg.register_function(&module, function, Vec::new());
        let (_, r1) = rvsdg.register_function(&module, function, Vec::new());

        for r in [r0, r1] {
            let ptr = ValueInput::argument(ptr_ty, 0);
            let val = ValueInput::argument(TY_U32, 1);

            let load = rvsdg.add_op_load(r, ptr.clone(), StateOrigin::Argument);
            let _store = rvsdg.add_op_store(r, ptr, val, StateOrigin::Node(load));

            rvsdg.reconnect_region_result(
                r,
                0,
                ValueOrigin::Output {
                    producer: load,
                    output: 0,
                },
            );
        }

        let mut checker = RegionIdentityChecker::new();

        assert!(checker.compare_regions(&rvsdg, r0, r1));
    }

    #[test]
    fn test_different_state_flow() {
        let module = Module::new(Symbol::from_ref(""));
        let mut rvsdg = Rvsdg::new(module.ty.clone());
        let mut module = module;

        let ptr_ty = TY_PTR_U32;
        let function = Function {
            name: Symbol::from_ref("f"),
            module: Symbol::from_ref("m"),
        };

        module.fn_sigs.register(
            function,
            FnSig {
                name: function.name,
                ty: TY_DUMMY,
                args: vec![
                    FnArg {
                        ty: ptr_ty,
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

        let (_, r0) = rvsdg.register_function(&module, function, Vec::new());
        let (_, r1) = rvsdg.register_function(&module, function, Vec::new());

        // Region 0: Load then Store
        {
            let ptr = ValueInput::argument(ptr_ty, 0);
            let val = ValueInput::argument(TY_U32, 1);

            let load = rvsdg.add_op_load(r0, ptr.clone(), StateOrigin::Argument);
            let _store = rvsdg.add_op_store(r0, ptr, val, StateOrigin::Node(load));

            rvsdg.reconnect_region_result(
                r0,
                0,
                ValueOrigin::Output {
                    producer: load,
                    output: 0,
                },
            );
        }

        // Region 1: Store then Load
        {
            let ptr = ValueInput::argument(ptr_ty, 0);
            let val = ValueInput::argument(TY_U32, 1);

            let store = rvsdg.add_op_store(r1, ptr.clone(), val, StateOrigin::Argument);
            let load = rvsdg.add_op_load(r1, ptr, StateOrigin::Node(store));

            rvsdg.reconnect_region_result(
                r1,
                0,
                ValueOrigin::Output {
                    producer: load,
                    output: 0,
                },
            );
        }

        let mut checker = RegionIdentityChecker::new();

        assert!(!checker.compare_regions(&rvsdg, r0, r1));
    }

    #[test]
    fn test_identical_nested_switches() {
        let module = Module::new(Symbol::from_ref(""));
        let mut rvsdg = Rvsdg::new(module.ty.clone());
        let mut module = module;

        let function = Function {
            name: Symbol::from_ref("f"),
            module: Symbol::from_ref("m"),
        };

        module.fn_sigs.register(
            function,
            FnSig {
                name: function.name,
                ty: TY_DUMMY,
                args: vec![FnArg {
                    ty: crate::ty::TY_PREDICATE,
                    shader_io_binding: None,
                }],
                ret_ty: Some(TY_U32),
            },
        );

        let (_, r0) = rvsdg.register_function(&module, function, Vec::new());
        let (_, r1) = rvsdg.register_function(&module, function, Vec::new());

        for r in [r0, r1] {
            let selector = ValueInput::argument(crate::ty::TY_PREDICATE, 0);

            let switch_node =
                rvsdg.add_switch(r, vec![selector], vec![ValueOutput::new(TY_U32)], None);

            // Both branches are identical: they return constant 42
            for _ in 0..2 {
                let branch = rvsdg.add_switch_branch(switch_node);
                let c = rvsdg.add_const_u32(branch, 42);

                rvsdg.reconnect_region_result(
                    branch,
                    0,
                    ValueOrigin::Output {
                        producer: c,
                        output: 0,
                    },
                );
            }

            rvsdg.reconnect_region_result(
                r,
                0,
                ValueOrigin::Output {
                    producer: switch_node,
                    output: 0,
                },
            );
        }

        let mut checker = RegionIdentityChecker::new();

        assert!(checker.compare_regions(&rvsdg, r0, r1));
    }

    #[test]
    fn test_different_nested_switches() {
        let module = Module::new(Symbol::from_ref(""));
        let mut rvsdg = Rvsdg::new(module.ty.clone());
        let mut module = module;

        let function = Function {
            name: Symbol::from_ref("f"),
            module: Symbol::from_ref("m"),
        };

        module.fn_sigs.register(
            function,
            FnSig {
                name: function.name,
                ty: TY_DUMMY,
                args: vec![FnArg {
                    ty: crate::ty::TY_PREDICATE,
                    shader_io_binding: None,
                }],
                ret_ty: Some(TY_U32),
            },
        );

        let (_, r0) = rvsdg.register_function(&module, function, Vec::new());
        let (_, r1) = rvsdg.register_function(&module, function, Vec::new());

        // R0: branch returns 42
        {
            let selector = ValueInput::argument(crate::ty::TY_PREDICATE, 0);
            let switch_node =
                rvsdg.add_switch(r0, vec![selector], vec![ValueOutput::new(TY_U32)], None);
            let branch = rvsdg.add_switch_branch(switch_node);
            let c = rvsdg.add_const_u32(branch, 42);

            rvsdg.reconnect_region_result(
                branch,
                0,
                ValueOrigin::Output {
                    producer: c,
                    output: 0,
                },
            );

            rvsdg.reconnect_region_result(
                r0,
                0,
                ValueOrigin::Output {
                    producer: switch_node,
                    output: 0,
                },
            );
        }

        // R1: branch returns 43
        {
            let selector = ValueInput::argument(crate::ty::TY_PREDICATE, 0);
            let switch_node =
                rvsdg.add_switch(r1, vec![selector], vec![ValueOutput::new(TY_U32)], None);
            let branch = rvsdg.add_switch_branch(switch_node);
            let c = rvsdg.add_const_u32(branch, 43);

            rvsdg.reconnect_region_result(
                branch,
                0,
                ValueOrigin::Output {
                    producer: c,
                    output: 0,
                },
            );

            rvsdg.reconnect_region_result(
                r1,
                0,
                ValueOrigin::Output {
                    producer: switch_node,
                    output: 0,
                },
            );
        }

        let mut checker = RegionIdentityChecker::new();

        assert!(!checker.compare_regions(&rvsdg, r0, r1));
    }

    #[test]
    fn test_identical_loops() {
        let module = Module::new(Symbol::from_ref(""));
        let mut rvsdg = Rvsdg::new(module.ty.clone());
        let mut module = module;

        let function = Function {
            name: Symbol::from_ref("f"),
            module: Symbol::from_ref("m"),
        };

        module.fn_sigs.register(
            function,
            FnSig {
                name: function.name,
                ty: TY_DUMMY,
                args: vec![FnArg {
                    ty: TY_U32,
                    shader_io_binding: None,
                }],
                ret_ty: Some(TY_U32),
            },
        );

        let (_, r0) = rvsdg.register_function(&module, function, Vec::new());
        let (_, r1) = rvsdg.register_function(&module, function, Vec::new());

        for r in [r0, r1] {
            let arg = ValueInput::argument(TY_U32, 0);
            let (loop_node, loop_region) = rvsdg.add_loop(r, vec![arg], None);

            // Loop body:
            // Result 0: re-entry predicate (constant false to exit immediately)
            // Result 1: next value (return arg + 1)
            let c_false = rvsdg.add_const_bool(loop_region, false);
            let loop_arg = ValueInput::argument(TY_U32, 0);
            let c_1 = rvsdg.add_const_u32(loop_region, 1);
            let add = rvsdg.add_op_binary(
                loop_region,
                BinaryOperator::Add,
                loop_arg,
                ValueInput::output(TY_U32, c_1, 0),
            );

            rvsdg.reconnect_region_result(
                loop_region,
                0,
                ValueOrigin::Output {
                    producer: c_false,
                    output: 0,
                },
            );
            rvsdg.reconnect_region_result(
                loop_region,
                1,
                ValueOrigin::Output {
                    producer: add,
                    output: 0,
                },
            );

            rvsdg.reconnect_region_result(
                r,
                0,
                ValueOrigin::Output {
                    producer: loop_node,
                    output: 0,
                },
            );
        }

        let mut checker = RegionIdentityChecker::new();

        assert!(checker.compare_regions(&rvsdg, r0, r1));
    }

    #[test]
    fn test_different_loops() {
        let module = Module::new(Symbol::from_ref(""));
        let mut rvsdg = Rvsdg::new(module.ty.clone());
        let mut module = module;

        let function = Function {
            name: Symbol::from_ref("f"),
            module: Symbol::from_ref("m"),
        };

        module.fn_sigs.register(
            function,
            FnSig {
                name: function.name,
                ty: TY_DUMMY,
                args: vec![FnArg {
                    ty: TY_U32,
                    shader_io_binding: None,
                }],
                ret_ty: Some(TY_U32),
            },
        );

        let (_, r0) = rvsdg.register_function(&module, function, Vec::new());
        let (_, r1) = rvsdg.register_function(&module, function, Vec::new());

        // R0: adds 1
        {
            let arg = ValueInput::argument(TY_U32, 0);
            let (loop_node, loop_region) = rvsdg.add_loop(r0, vec![arg], None);
            let c_false = rvsdg.add_const_bool(loop_region, false);
            let loop_arg = ValueInput::argument(TY_U32, 0);
            let c_1 = rvsdg.add_const_u32(loop_region, 1);
            let add = rvsdg.add_op_binary(
                loop_region,
                BinaryOperator::Add,
                loop_arg,
                ValueInput::output(TY_U32, c_1, 0),
            );
            rvsdg.reconnect_region_result(
                loop_region,
                0,
                ValueOrigin::Output {
                    producer: c_false,
                    output: 0,
                },
            );
            rvsdg.reconnect_region_result(
                loop_region,
                1,
                ValueOrigin::Output {
                    producer: add,
                    output: 0,
                },
            );
            rvsdg.reconnect_region_result(
                r0,
                0,
                ValueOrigin::Output {
                    producer: loop_node,
                    output: 0,
                },
            );
        }

        // R1: adds 2
        {
            let arg = ValueInput::argument(TY_U32, 0);
            let (loop_node, loop_region) = rvsdg.add_loop(r1, vec![arg], None);
            let c_false = rvsdg.add_const_bool(loop_region, false);
            let loop_arg = ValueInput::argument(TY_U32, 0);
            let c_2 = rvsdg.add_const_u32(loop_region, 2);
            let add = rvsdg.add_op_binary(
                loop_region,
                BinaryOperator::Add,
                loop_arg,
                ValueInput::output(TY_U32, c_2, 0),
            );
            rvsdg.reconnect_region_result(
                loop_region,
                0,
                ValueOrigin::Output {
                    producer: c_false,
                    output: 0,
                },
            );
            rvsdg.reconnect_region_result(
                loop_region,
                1,
                ValueOrigin::Output {
                    producer: add,
                    output: 0,
                },
            );
            rvsdg.reconnect_region_result(
                r1,
                0,
                ValueOrigin::Output {
                    producer: loop_node,
                    output: 0,
                },
            );
        }

        let mut checker = RegionIdentityChecker::new();

        assert!(!checker.compare_regions(&rvsdg, r0, r1));
    }
}
