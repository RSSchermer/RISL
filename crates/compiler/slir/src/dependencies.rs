use std::borrow::Borrow;

use indexmap::{IndexMap, IndexSet};

use crate::cfg::{Cfg, StatementData};
use crate::import::FunctionImporter;
use crate::{Function, Module, Symbol};

pub trait DependencyLoader {
    type ModuleData<'a>: Borrow<Module> + 'a
    where
        Self: 'a;
    type CfgData<'a>: Borrow<Cfg> + 'a
    where
        Self: 'a;

    fn load<'a>(&'a mut self, module: Symbol) -> (Self::ModuleData<'a>, Self::CfgData<'a>);
}

struct ModuleDependencies {
    function: IndexSet<Function>,
    // TODO: constants
}

impl ModuleDependencies {
    fn new() -> Self {
        Self {
            function: IndexSet::new(),
        }
    }
}

struct DependencyImporter {
    queue: IndexMap<Symbol, ModuleDependencies>,
    function_importer: FunctionImporter,
}

impl DependencyImporter {
    fn new() -> Self {
        Self {
            queue: IndexMap::default(),
            function_importer: FunctionImporter::new(),
        }
    }

    fn import_dependencies(
        &mut self,
        dst_module: &mut Module,
        dst_cfg: &mut Cfg,
        loader: &mut impl DependencyLoader,
    ) {
        self.gather_dst_dependencies(dst_cfg);

        while let Some(module) = self.queue.first().map(|e| *e.0) {
            let (src_module, src_cfg) = loader.load(module);

            while let Some(function) = self.queue.get_mut(&module).unwrap().function.pop() {
                self.gather_function_dependencies(src_cfg.borrow(), dst_cfg, function);

                self.function_importer.import_function(
                    (src_module.borrow(), src_cfg.borrow()),
                    (dst_module, dst_cfg),
                    function,
                );
            }

            self.queue.shift_remove(&module);
        }
    }

    fn gather_dst_dependencies(&mut self, dst_cfg: &Cfg) {
        for function in dst_cfg.registered_functions() {
            self.gather_function_dependencies(dst_cfg, dst_cfg, function);
        }
    }

    fn gather_function_dependencies(&mut self, src_cfg: &Cfg, dst_cfg: &Cfg, function: Function) {
        if let Some(body) = src_cfg.get_function_body(function) {
            for bb in body.basic_blocks() {
                for statement in src_cfg[*bb].statements() {
                    if let StatementData::OpCall(op_call) = &src_cfg[*statement] {
                        // Check if the dependency was already imported into the destination CFG.
                        if dst_cfg.get_function_body(op_call.callee()).is_none() {
                            self.enqueue_function_dependency(op_call.callee());
                        }
                    }
                }
            }
        }
    }

    fn enqueue_function_dependency(&mut self, function: Function) {
        let module_deps = self
            .queue
            .entry(function.module)
            .or_insert_with(|| ModuleDependencies::new());

        module_deps.function.insert(function);
    }
}

pub fn import_dependencies(module: &mut Module, cfg: &mut Cfg, loader: &mut impl DependencyLoader) {
    let mut importer = DependencyImporter::new();

    importer.import_dependencies(module, cfg, loader);
}
