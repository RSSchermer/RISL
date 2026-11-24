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

use rustc_hash::FxHashSet;

use crate::scf::analyze::global_use::GlobalUse;
use crate::scf::visit::TopDownVisitor;
use crate::scf::{Scf, Statement, StatementKind, visit};
use crate::ty::{Type, TypeKind};
use crate::{Constant, Module, WorkgroupBinding};

struct StructUseAnalyzer {
    used_structs: FxHashSet<Type>,
    visited: FxHashSet<Type>,
    queue: Vec<Type>,
}

impl StructUseAnalyzer {
    fn new() -> Self {
        StructUseAnalyzer {
            used_structs: Default::default(),
            visited: Default::default(),
            queue: Vec::new(),
        }
    }

    fn collect_from_uniform_bindings(&mut self, module: &Module) {
        for binding in module.uniform_bindings.values() {
            self.queue.push(binding.ty);
        }
    }

    fn collect_from_storage_bindings(&mut self, module: &Module) {
        for binding in module.storage_bindings.values() {
            self.queue.push(binding.ty);
        }
    }

    fn collect_from_workgroup_bindings(
        &mut self,
        module: &Module,
        used_bindings: impl Iterator<Item = WorkgroupBinding>,
    ) {
        for binding in used_bindings {
            let data = &module.workgroup_bindings[binding];

            self.queue.push(data.ty);
        }
    }

    fn collect_from_constants(
        &mut self,
        module: &Module,
        used_constants: impl Iterator<Item = Constant>,
    ) {
        for constant in used_constants {
            let data = &module.constants[constant];

            self.queue.push(data.ty());
        }
    }

    fn collect_from_alloca_statements(&mut self, module: &Module, scf: &Scf) {
        for (entry_point, _) in module.entry_points.iter() {
            let sig = &module.fn_sigs[entry_point];

            self.queue.extend(sig.args.iter().map(|arg| arg.ty));
            self.queue.extend(sig.ret_ty);

            if let Some(body) = scf.get_function_body(entry_point) {
                self.visit_block(scf, body.block());
            }
        }
    }

    fn collect_nested_structs(&mut self, module: &Module) {
        while let Some(ty) = self.queue.pop() {
            if !self.visited.insert(ty) {
                continue;
            }

            match &*module.ty.kind(ty) {
                TypeKind::Struct(data) => {
                    self.used_structs.insert(ty);
                    self.queue.extend(data.fields.iter().map(|field| field.ty));
                }
                TypeKind::Array { element_ty, .. } | TypeKind::Slice { element_ty, .. } => {
                    self.queue.push(*element_ty);
                }
                TypeKind::Ptr(pointee) => self.queue.push(*pointee),
                _ => {}
            }
        }
    }
}

impl TopDownVisitor for StructUseAnalyzer {
    fn visit_statement(&mut self, scf: &Scf, statement: Statement) {
        match scf[statement].kind() {
            StatementKind::Alloca(alloca) => self.queue.push(alloca.ty()),
            StatementKind::ExprBinding(binding) => self.queue.push(binding.expression().ty()),
            _ => {}
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
