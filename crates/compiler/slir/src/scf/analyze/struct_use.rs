//! Identifies the set of all struct types that are reachable from the live parts of the module.
//!
//! There are 3 sources of struct data:
//!
//! 1. The struct data was present from the start of the program as a global value.
//! 2. The struct data was allocated by an alloca statement.
//! 3. The struct data was part of parent struct data as a nested field.
//!
//! Therefore, to find all struct types used by the program, we check all used global values (as
//! identified by a `global_use` analysis pass) for struct types, and we visit all alloca statements
//! reachable from the module's entry-points and check for struct types. Then we recursively check
//! for nested struct types in all struct types we found thus far.

use std::collections::VecDeque;

use rustc_hash::FxHashSet;

use crate::scf::analyze::global_use::GlobalUse;
use crate::scf::visit::TopDownVisitor;
use crate::scf::{Scf, Statement, StatementKind, visit};
use crate::ty::Type;
use crate::{Constant, Module, WorkgroupBinding};

struct StructUseAnalyzer {
    used_structs: FxHashSet<Type>,
}

impl StructUseAnalyzer {
    fn new() -> Self {
        StructUseAnalyzer {
            used_structs: Default::default(),
        }
    }

    fn collect_from_uniform_bindings(&mut self, module: &Module) {
        for binding in module.uniform_bindings.values() {
            if module.ty.kind(binding.ty).is_struct() {
                self.used_structs.insert(binding.ty);
            }
        }
    }

    fn collect_from_storage_bindings(&mut self, module: &Module) {
        for binding in module.storage_bindings.values() {
            if module.ty.kind(binding.ty).is_struct() {
                self.used_structs.insert(binding.ty);
            }
        }
    }

    fn collect_from_workgroup_bindings(
        &mut self,
        module: &Module,
        used_bindings: impl Iterator<Item = WorkgroupBinding>,
    ) {
        for binding in used_bindings {
            let data = &module.workgroup_bindings[binding];

            if module.ty.kind(data.ty).is_struct() {
                self.used_structs.insert(data.ty);
            }
        }
    }

    fn collect_from_constants(
        &mut self,
        module: &Module,
        used_constants: impl Iterator<Item = Constant>,
    ) {
        for constant in used_constants {
            let data = &module.constants[constant];

            if module.ty.kind(data.ty()).is_struct() {
                self.used_structs.insert(data.ty());
            }
        }
    }

    fn collect_from_alloca_statements(&mut self, module: &Module, scf: &Scf) {
        for (entry_point, _) in module.entry_points.iter() {
            if let Some(body) = scf.get_function_body(entry_point) {
                self.visit_block(scf, body.block());
            }
        }
    }

    fn collect_nested_structs(&mut self, module: &Module) {
        let mut queue = VecDeque::from_iter(self.used_structs.iter().copied());

        while let Some(ty) = queue.pop_front() {
            let kind = module.ty.kind(ty);
            let struct_data = kind.expect_struct();

            for field in &struct_data.fields {
                if module.ty.kind(field.ty).is_struct() {
                    if self.used_structs.insert(field.ty) {
                        queue.push_back(field.ty);
                    }
                }
            }
        }
    }
}

impl TopDownVisitor for StructUseAnalyzer {
    fn visit_statement(&mut self, scf: &Scf, statement: Statement) {
        if let StatementKind::Alloca(alloca) = scf[statement].kind()
            && scf.ty().kind(alloca.ty()).is_struct()
        {
            self.used_structs.insert(alloca.ty());
        }

        visit::visit_statement_top_down(self, scf, statement);
    }
}

pub fn collect_used_structs(module: &Module, scf: &Scf, global_use: &GlobalUse) -> FxHashSet<Type> {
    let mut analyzer = StructUseAnalyzer::new();

    // Note that we always consider all uniform bindings and storage bindings as "used", since
    // they are part of a module's public interface. For workgroup bindings and constants, we
    // only collect structs when they are actually used, based on the results of a `global_use`
    // analysis pass.

    analyzer.collect_from_uniform_bindings(module);
    analyzer.collect_from_storage_bindings(module);
    analyzer.collect_from_workgroup_bindings(module, global_use.workgroup_bindings.iter().copied());
    analyzer.collect_from_constants(module, global_use.constants.iter().copied());
    analyzer.collect_from_alloca_statements(module, scf);
    analyzer.collect_nested_structs(module);

    analyzer.used_structs
}
