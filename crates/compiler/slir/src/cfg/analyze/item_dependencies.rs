use indexmap::IndexSet;
use rustc_hash::FxHashMap;

use crate::cfg::{
    Assign, Bind, Cfg, InlineConst, IntrinsicOp, OpCall, RootIdentifier, StatementData,
    Uninitialized, Value,
};
use crate::ty::Type;
use crate::{Constant, Function, Module, StorageBinding, UniformBinding, WorkgroupBinding};

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Item {
    Function(Function),
    UniformBinding(UniformBinding),
    StorageBinding(StorageBinding),
    WorkgroupBinding(WorkgroupBinding),
    Constant(Constant),
}

impl Item {
    pub fn ty(&self, module: &Module) -> Type {
        match *self {
            Item::Function(f) => module.fn_sigs[f].ty,
            Item::UniformBinding(b) => module.uniform_bindings[b].ty,
            Item::StorageBinding(b) => module.storage_bindings[b].ty,
            Item::WorkgroupBinding(b) => module.workgroup_bindings[b].ty,
            Item::Constant(c) => module.constants[c].ty(),
        }
    }
}

pub trait WithItemDependencies {
    fn with_item_dependencies<F>(&self, f: F)
    where
        F: FnMut(Item);
}

impl WithItemDependencies for Value {
    fn with_item_dependencies<F>(&self, mut f: F)
    where
        F: FnMut(Item),
    {
        match self {
            Value::InlineConst(InlineConst::Ptr(ptr)) => match ptr.root_identifier() {
                RootIdentifier::Uniform(b) => {
                    f(Item::UniformBinding(b));
                }
                RootIdentifier::Storage(b) => {
                    f(Item::StorageBinding(b));
                }
                RootIdentifier::Workgroup(b) => {
                    f(Item::WorkgroupBinding(b));
                }
                RootIdentifier::Constant(c) => f(Item::Constant(c)),
                _ => {}
            },
            _ => {}
        }
    }
}

impl WithItemDependencies for Bind {
    fn with_item_dependencies<F>(&self, mut f: F)
    where
        F: FnMut(Item),
    {
        self.value().with_item_dependencies(&mut f);
    }
}

impl WithItemDependencies for Uninitialized {
    fn with_item_dependencies<F>(&self, _f: F)
    where
        F: FnMut(Item),
    {
    }
}

impl WithItemDependencies for Assign {
    fn with_item_dependencies<F>(&self, mut f: F)
    where
        F: FnMut(Item),
    {
        self.value().with_item_dependencies(&mut f);
    }
}

impl<T> WithItemDependencies for IntrinsicOp<T> {
    fn with_item_dependencies<F>(&self, mut f: F)
    where
        F: FnMut(Item),
    {
        for arg in self.arguments() {
            arg.with_item_dependencies(&mut f);
        }
    }
}

impl WithItemDependencies for OpCall {
    fn with_item_dependencies<F>(&self, mut f: F)
    where
        F: FnMut(Item),
    {
        for arg in self.arguments() {
            arg.with_item_dependencies(&mut f);
        }

        f(Item::Function(self.callee()));
    }
}

macro_rules! impl_collect_dependencies_statement {
    ($($op:ident,)*) => {
        impl WithItemDependencies for StatementData {
            fn with_item_dependencies<F>(&self, mut f: F) where F: FnMut(Item) {
                match self {
                    $(StatementData::$op(s) => s.with_item_dependencies(&mut f),)*
                }
            }
        }
    };
}

impl_collect_dependencies_statement! {
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
    OpArrayLength,
}

fn collect_body_dependencies(cfg: &Cfg, function: Function) -> IndexSet<Item> {
    let mut dependencies = IndexSet::new();

    let body = &cfg[function];

    for bb in body.basic_blocks() {
        for statement in cfg[*bb].statements() {
            cfg[*statement].with_item_dependencies(|item| {
                dependencies.insert(item);
            })
        }
    }

    dependencies
}

pub type ItemDependencies = FxHashMap<Item, IndexSet<Item>>;

pub fn item_dependencies(cfg: &Cfg) -> ItemDependencies {
    let mut dep_map = FxHashMap::default();

    for function in cfg.registered_functions() {
        let dependencies = collect_body_dependencies(cfg, function);

        dep_map.insert(Item::Function(function), dependencies);
    }

    dep_map
}
