use indexmap::{IndexMap, IndexSet};
use risl_smi as smi;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::cfg::analyze::item_dependencies::{Item, item_dependencies};
use crate::cfg::visit::Visitor;
use crate::cfg::{BasicBlock, Cfg, Statement, StatementData};
use crate::smi::buffer_layout::BufferLayoutProvider;
use crate::ty::{ScalarKind, Type, TypeKind, TypeRegistry, VectorSize};
use crate::{
    Constant, ConstantKind, EntryPointKind, FnSig, Function, InterpolationSampling,
    InterpolationType, Module, OverridableConstantKind, ShaderIOBinding, StorageBinding,
    UniformBinding,
};

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Resource {
    Uniform(UniformBinding),
    Storage(StorageBinding),
    // TODO: texture and sampler bindings
}

#[derive(Clone, Debug)]
struct FunctionDependencies {
    resources: FxHashSet<Resource>,
    overridable_constants: FxHashSet<Constant>,
}

impl FunctionDependencies {
    fn extend(&mut self, other: &FunctionDependencies) {
        self.resources.extend(other.resources.iter().copied());
        self.overridable_constants
            .extend(other.overridable_constants.iter().copied());
    }
}

struct DependencyCollector {
    item_dependencies: FxHashMap<Item, IndexSet<Item>>,
    function_dependencies: FxHashMap<Function, FunctionDependencies>,
    visited_bbs: FxHashSet<BasicBlock>,
}

impl DependencyCollector {
    fn new(cfg: &Cfg) -> Self {
        Self {
            item_dependencies: item_dependencies(cfg),
            function_dependencies: Default::default(),
            visited_bbs: Default::default(),
        }
    }

    fn collect(&mut self, module: &Module, cfg: &Cfg, function: Function) -> &FunctionDependencies {
        // Check the cache first.
        if self.function_dependencies.contains_key(&function) {
            return &self.function_dependencies[&function];
        }

        let mut resources = FxHashSet::default();
        let mut overridable_constants = FxHashSet::default();

        if let Some(deps) = self.item_dependencies.get(&Item::Function(function)) {
            for item in deps {
                match item {
                    Item::UniformBinding(b) => {
                        resources.insert(Resource::Uniform(*b));
                    }
                    Item::StorageBinding(b) => {
                        resources.insert(Resource::Storage(*b));
                    }
                    Item::Constant(c) => match module.constants[*c].kind() {
                        ConstantKind::ByteData(_) => {}
                        ConstantKind::Overridable(_) => {
                            overridable_constants.insert(*c);
                        }
                        ConstantKind::Expression => {
                            todo!()
                        }
                    },
                    _ => {}
                }
            }
        }

        let mut deps = FunctionDependencies {
            resources,
            overridable_constants,
        };

        // We don't clear the visited_bbs set for each function, since we recursively visit nested
        // functions and we may not have finished visiting the calling function yet. BasicBlock
        // tokens are globally unique within a CFG, so we can mix them in a single set without
        // issue. Because we cache collection results, we also don't need to worry about visiting
        // the same function twice.

        let entry_bb = cfg
            .get_function_body(function)
            .expect("function not registered")
            .entry_block();

        BodyVisitor {
            module,
            collector: self,
            deps: &mut deps,
        }
        .visit_basic_block(cfg, entry_bb);

        // Cache the result.
        self.function_dependencies.insert(function, deps);

        &self.function_dependencies[&function]
    }
}

struct BodyVisitor<'a> {
    module: &'a Module,
    collector: &'a mut DependencyCollector,
    deps: &'a mut FunctionDependencies,
}

impl Visitor for BodyVisitor<'_> {
    fn should_visit(&mut self, _cfg: &Cfg, bb: BasicBlock) -> bool {
        self.collector.visited_bbs.insert(bb)
    }

    fn visit_statement(&mut self, cfg: &Cfg, statement: Statement) {
        if let StatementData::OpCall(op) = &cfg[statement] {
            let callee_deps = self.collector.collect(self.module, cfg, op.callee());

            self.deps.extend(callee_deps);
        }
    }
}

fn build_resource_binding_map(
    module: &Module,
    resources: FxHashSet<Resource>,
) -> IndexMap<Resource, smi::ResourceBinding> {
    let mut buffer_layout_provider = BufferLayoutProvider::new();
    let mut resource_binding_map = IndexMap::default();

    for resource in resources {
        let mapped = match resource {
            Resource::Uniform(b) => {
                let data = &module.uniform_bindings[b];
                let layout = buffer_layout_provider.layout(&module.ty, data.ty).clone();

                assert!(
                    layout.tail.is_none(),
                    "uniform buffer resources must be sized"
                );

                smi::ResourceBinding {
                    group: data.resource_binding.group,
                    binding: data.resource_binding.binding,
                    resource_type: smi::ResourceType::Uniform(smi::SizedBufferLayout {
                        memory_units: layout.head.into(),
                    }),
                }
            }
            Resource::Storage(b) => {
                let data = &module.storage_bindings[b];
                let layout = buffer_layout_provider.layout(&module.ty, data.ty).clone();
                let layout = smi::UnsizedBufferLayout {
                    sized_head: layout.head.into(),
                    unsized_tail: layout.tail,
                };

                let resource_type = if data.writable {
                    smi::ResourceType::StorageReadWrite(layout)
                } else {
                    smi::ResourceType::StorageRead(layout)
                };

                smi::ResourceBinding {
                    group: data.resource_binding.group,
                    binding: data.resource_binding.binding,
                    resource_type,
                }
            }
        };

        resource_binding_map.insert(resource, mapped);
    }

    resource_binding_map.sort_by(|_, a, _, b| a.cmp(b));

    resource_binding_map
}

fn build_overridable_constant_map(
    module: &Module,
    constants: FxHashSet<Constant>,
) -> IndexMap<Constant, smi::OverridableConstant> {
    let mut overridable_constant_map = IndexMap::default();

    for constant in constants {
        let data = module.constants[constant].kind().expect_overridable();

        let (constant_type, required) = match data.kind() {
            OverridableConstantKind::Float(Some(_)) => (smi::OverridableConstantType::Float, false),
            OverridableConstantKind::Float(None) => (smi::OverridableConstantType::Float, true),
            OverridableConstantKind::Bool(Some(_)) => (smi::OverridableConstantType::Bool, false),
            OverridableConstantKind::Bool(None) => (smi::OverridableConstantType::Bool, true),
            OverridableConstantKind::SignedInteger(Some(_)) => {
                (smi::OverridableConstantType::SignedInteger, false)
            }
            OverridableConstantKind::SignedInteger(None) => {
                (smi::OverridableConstantType::SignedInteger, true)
            }
            OverridableConstantKind::UnsignedInteger(Some(_)) => {
                (smi::OverridableConstantType::UnsignedInteger, false)
            }
            OverridableConstantKind::UnsignedInteger(None) => {
                (smi::OverridableConstantType::UnsignedInteger, true)
            }
        };

        overridable_constant_map.insert(
            constant,
            smi::OverridableConstant {
                id: data.id() as u16,
                constant_type,
                required,
            },
        );
    }

    overridable_constant_map.sort_by(|_, a, _, b| a.cmp(b));

    overridable_constant_map
}

fn shader_stage(entry_point_kind: &EntryPointKind) -> smi::ShaderStage {
    match entry_point_kind {
        EntryPointKind::Vertex => smi::ShaderStage::Vertex,
        EntryPointKind::Fragment => smi::ShaderStage::Fragment,
        EntryPointKind::Compute(_, _, _) => smi::ShaderStage::Compute,
    }
}

fn interpolation_type_to_smi(interpolation_type: InterpolationType) -> smi::InterpolationType {
    match interpolation_type {
        InterpolationType::Perspective => smi::InterpolationType::Perspective,
        InterpolationType::Linear => smi::InterpolationType::Linear,
        InterpolationType::Flat => smi::InterpolationType::Flat,
    }
}

fn interpolation_sampling_to_smi(sampling: InterpolationSampling) -> smi::Sampling {
    match sampling {
        InterpolationSampling::Centroid => smi::Sampling::Centroid,
        InterpolationSampling::Sample => smi::Sampling::Sample,
        InterpolationSampling::Center => smi::Sampling::Center,
        InterpolationSampling::First => smi::Sampling::First,
        InterpolationSampling::Either => smi::Sampling::Either,
    }
}

fn collect_io_binding(
    collection: &mut Vec<smi::IoBinding>,
    binding: &ShaderIOBinding,
    ty_registry: &TypeRegistry,
    ty: Type,
) {
    use ScalarKind::*;
    use TypeKind::*;
    use VectorSize::*;

    if let ShaderIOBinding::Location {
        location,
        interpolation,
        ..
    } = binding
    {
        let binding_type = match &*ty_registry.kind(ty) {
            Scalar(I32) => smi::IoBindingType::SignedInteger,
            Scalar(U32) => smi::IoBindingType::UnsignedInteger,
            Scalar(F32) => smi::IoBindingType::Float,
            Vector(v) => match (v.scalar, v.size) {
                (I32, Two) => smi::IoBindingType::SignedIntegerVector2,
                (I32, Three) => smi::IoBindingType::SignedIntegerVector3,
                (I32, Four) => smi::IoBindingType::SignedIntegerVector4,
                (U32, Two) => smi::IoBindingType::UnsignedIntegerVector2,
                (U32, Three) => smi::IoBindingType::UnsignedIntegerVector3,
                (U32, Four) => smi::IoBindingType::UnsignedIntegerVector4,
                (F32, Two) => smi::IoBindingType::FloatVector2,
                (F32, Three) => smi::IoBindingType::FloatVector3,
                (F32, Four) => smi::IoBindingType::FloatVector4,
                _ => panic!(
                    "location attributes may only be applied to numeric scalar or vector types"
                ),
            },
            _ => {
                panic!("location attributes may only be applied to numeric scalar or vector types")
            }
        };

        let interpolate = interpolation.map(|i| smi::Interpolate {
            interpolation_type: interpolation_type_to_smi(i.tpe),
            sampling: i.sampling.map(interpolation_sampling_to_smi),
        });

        collection.push(smi::IoBinding {
            location: *location,
            binding_type,
            interpolate,
        });
    }
}

fn collect_struct_io_bindings(
    collection: &mut Vec<smi::IoBinding>,
    ty_registry: &TypeRegistry,
    ty: Type,
) {
    for field in ty_registry.kind(ty).expect_struct().fields.iter() {
        field
            .io_binding
            .as_ref()
            .map(|b| collect_io_binding(collection, b, ty_registry, field.ty));
    }
}

fn collect_input_bindings(ty_registry: &TypeRegistry, sig: &FnSig) -> Vec<smi::IoBinding> {
    let mut collection = Vec::new();

    for arg in &sig.args {
        if let Some(io_binding) = arg.shader_io_binding.as_ref() {
            collect_io_binding(&mut collection, io_binding, ty_registry, arg.ty);
        } else if ty_registry.kind(arg.ty).is_struct() {
            collect_struct_io_bindings(&mut collection, ty_registry, arg.ty);
        }
    }

    collection.sort();

    collection
}

fn collect_output_bindings(ty_registry: &TypeRegistry, sig: &FnSig) -> Vec<smi::IoBinding> {
    let mut collection = Vec::new();

    if let Some(ret_ty) = sig.ret_ty
        && ty_registry.kind(ret_ty).is_struct()
    {
        collect_struct_io_bindings(&mut collection, ty_registry, ret_ty);
        collection.sort();
    }

    collection
}

pub fn build_smi(module: &Module, cfg: &Cfg) -> smi::ShaderModuleInterface {
    let mut dependency_collector = DependencyCollector::new(cfg);

    let mut resources = FxHashSet::default();
    let mut overridable_constants = FxHashSet::default();

    for (entry_point, _) in module.entry_points.iter() {
        let deps = dependency_collector.collect(module, cfg, entry_point);

        resources.extend(deps.resources.iter().copied());
        overridable_constants.extend(deps.overridable_constants.iter().copied());
    }

    let resource_binding_map = build_resource_binding_map(module, resources);
    let overridable_constant_map = build_overridable_constant_map(module, overridable_constants);

    let mut entry_points = Vec::new();

    for (function, entry_point) in module.entry_points.iter() {
        let deps = dependency_collector.collect(module, cfg, function);
        let sig = &module.fn_sigs[function];

        let mut overridable_constants = deps
            .overridable_constants
            .iter()
            .map(|c| overridable_constant_map.get_index_of(c).unwrap())
            .collect::<Vec<_>>();
        let mut resource_bindings = deps
            .resources
            .iter()
            .map(|r| resource_binding_map.get_index_of(r).unwrap())
            .collect::<Vec<_>>();

        overridable_constants.sort();
        resource_bindings.sort();

        entry_points.push(smi::EntryPoint {
            name: entry_point.name.to_string().into(),
            stage: shader_stage(&entry_point.kind),
            input_bindings: collect_input_bindings(&module.ty, sig).into(),
            output_bindings: collect_output_bindings(&module.ty, sig).into(),
            overridable_constants: overridable_constants.into(),
            resource_bindings: resource_bindings.into(),
        })
    }

    let overridable_constants = overridable_constant_map.into_values().collect::<Vec<_>>();
    let resource_bindings = resource_binding_map.into_values().collect::<Vec<_>>();

    entry_points.sort_by(|a, b| a.name.cmp(&b.name));

    smi::ShaderModuleInterface {
        overridable_constants: overridable_constants.into(),
        resource_bindings: resource_bindings.into(),
        entry_points: entry_points.into(),
    }
}

#[cfg(test)]
mod tests {
    use risl_smi::{
        IoBinding, MemoryUnit, MemoryUnitLayout, OverridableConstantType, ResourceType,
        SizedBufferLayout, UnsizedBufferLayout,
    };

    use super::*;
    use crate::cfg::{BlockPosition, ConstPtr, RootIdentifier, Terminator};
    use crate::ty::{Struct, StructField, TY_DUMMY, TY_F32, TY_U32, TY_VEC4_F32};
    use crate::{
        EntryPoint, FnArg, Interpolation, ResourceBinding, StorageBindingData, Symbol,
        UniformBindingData,
    };

    #[test]
    fn test_build_smi() {
        let mut module = Module::new(Symbol::from_ref(""));

        let entry_point_0_output_ty = module.ty.register(TypeKind::Struct(Struct {
            fields: vec![
                StructField {
                    offset: 0,
                    ty: TY_VEC4_F32,
                    io_binding: Some(ShaderIOBinding::Position { invariant: false }),
                },
                StructField {
                    offset: 16,
                    ty: TY_F32,
                    io_binding: Some(ShaderIOBinding::Location {
                        location: 0,
                        blend_src: None,
                        interpolation: Some(Interpolation {
                            tpe: InterpolationType::Linear,
                            sampling: Some(InterpolationSampling::Sample),
                        }),
                    }),
                },
                StructField {
                    offset: 20,
                    ty: TY_F32,
                    io_binding: Some(ShaderIOBinding::Location {
                        location: 2,
                        blend_src: None,
                        interpolation: None,
                    }),
                },
            ],
        }));

        let entry_point_0 = Function {
            name: Symbol::from_ref("entry_point_0"),
            module: Symbol::from_ref(""),
        };
        module.fn_sigs.register(
            entry_point_0,
            FnSig {
                name: Symbol::from_ref("entry_point_0"),
                ty: TY_DUMMY,
                args: vec![
                    FnArg {
                        ty: TY_F32,
                        shader_io_binding: Some(ShaderIOBinding::Location {
                            location: 0,
                            blend_src: None,
                            interpolation: None,
                        }),
                    },
                    FnArg {
                        ty: TY_F32,
                        shader_io_binding: Some(ShaderIOBinding::Location {
                            location: 1,
                            blend_src: None,
                            interpolation: None,
                        }),
                    },
                ],
                ret_ty: Some(entry_point_0_output_ty),
            },
        );

        let entry_point_1 = Function {
            name: Symbol::from_ref("entry_point_1"),
            module: Symbol::from_ref(""),
        };
        module.fn_sigs.register(
            entry_point_1,
            FnSig {
                name: Symbol::from_ref("entry_point_1"),
                ty: TY_DUMMY,
                args: vec![FnArg {
                    ty: TY_U32,
                    shader_io_binding: Some(ShaderIOBinding::LocalInvocationIndex),
                }],
                ret_ty: None,
            },
        );

        let called_fn = Function {
            name: Symbol::from_ref("called_fn"),
            module: Symbol::from_ref(""),
        };
        module.fn_sigs.register(
            called_fn,
            FnSig {
                name: Symbol::from_ref("called_fn"),
                ty: TY_DUMMY,
                args: vec![],
                ret_ty: None,
            },
        );

        module.entry_points.register(
            entry_point_0,
            EntryPoint {
                name: Symbol::from_ref("vertex_main"),
                kind: EntryPointKind::Vertex,
            },
        );
        module.entry_points.register(
            entry_point_1,
            EntryPoint {
                name: Symbol::from_ref("compute_main"),
                kind: EntryPointKind::Compute(1, 1, 1),
            },
        );

        // Used by both entry_point_0 and entry_point_1.
        let overridable_0 = Constant {
            name: Symbol::from_ref("overridable_0"),
            module: Symbol::from_ref(""),
        };
        module.constants.register_overridable(
            overridable_0,
            0,
            OverridableConstantKind::UnsignedInteger(Some(0)),
        );
        // Used by called_fn.
        let overridable_1 = Constant {
            name: Symbol::from_ref("overridable_1"),
            module: Symbol::from_ref(""),
        };
        module.constants.register_overridable(
            overridable_1,
            10,
            OverridableConstantKind::UnsignedInteger(None),
        );
        // Not used by any function.
        let overridable_2 = Constant {
            name: Symbol::from_ref("overridable_2"),
            module: Symbol::from_ref(""),
        };
        module.constants.register_overridable(
            overridable_2,
            20,
            OverridableConstantKind::UnsignedInteger(None),
        );

        // Used by both entry_point_0 and entry_point_1.
        let uniform_0 = module.uniform_bindings.register(UniformBindingData {
            ty: TY_F32,
            resource_binding: ResourceBinding {
                group: 0,
                binding: 0,
            },
        });
        // Used by the called_fn.
        let uniform_1 = module.uniform_bindings.register(UniformBindingData {
            ty: TY_F32,
            resource_binding: ResourceBinding {
                group: 0,
                binding: 1,
            },
        });
        // Used by entry_point_0 in an unreachable basic-block (dead code).
        let storage_0 = module.storage_bindings.register(StorageBindingData {
            ty: TY_F32,
            resource_binding: ResourceBinding {
                group: 1,
                binding: 0,
            },
            writable: false,
        });
        // Not used by any function.
        let storage_1 = module.storage_bindings.register(StorageBindingData {
            ty: TY_F32,
            resource_binding: ResourceBinding {
                group: 1,
                binding: 3,
            },
            writable: false,
        });
        // Used by entry_point_1.
        let storage_read_write_0 = module.storage_bindings.register(StorageBindingData {
            ty: TY_F32,
            resource_binding: ResourceBinding {
                group: 4,
                binding: 0,
            },
            writable: true,
        });

        let mut cfg = Cfg::new(module.ty.clone());

        // Build the CFG for entry_point_0

        let entry_point_0_body = cfg.register_function(&module, entry_point_0);

        let entry_point_0_bb0 = entry_point_0_body.entry_block();
        let entry_point_0_bb1 = cfg.add_basic_block(entry_point_0);
        let entry_point_0_bb2 = cfg.add_basic_block(entry_point_0);

        cfg.add_stmt_op_load(
            entry_point_0_bb0,
            BlockPosition::Append,
            ConstPtr::new(&module, &cfg, RootIdentifier::Uniform(uniform_0)).into(),
        );
        cfg.add_stmt_op_load(
            entry_point_0_bb0,
            BlockPosition::Append,
            ConstPtr::new(&module, &cfg, RootIdentifier::Constant(overridable_0)).into(),
        );
        cfg.add_stmt_op_call(
            entry_point_0_bb0,
            BlockPosition::Append,
            called_fn,
            None,
            [],
        );
        let (_, entry_point_0_bb0_predicate) = cfg.add_stmt_op_bool_to_branch_selector(
            entry_point_0_bb0,
            BlockPosition::Append,
            false.into(),
        );
        cfg.set_terminator(
            entry_point_0_bb0,
            Terminator::branch_multiple(
                entry_point_0_bb0_predicate,
                [entry_point_0_bb1, entry_point_0_bb2],
            ),
        );

        cfg.add_stmt_op_load(
            entry_point_0_bb1,
            BlockPosition::Append,
            ConstPtr::new(&module, &cfg, RootIdentifier::Storage(storage_0)).into(),
        );
        cfg.set_terminator(
            entry_point_0_bb1,
            Terminator::branch_single(entry_point_0_bb2),
        );

        let (_, entry_point_0_ret_alloca) = cfg.add_stmt_op_alloca(
            entry_point_0_bb2,
            BlockPosition::Append,
            entry_point_0_output_ty,
        );
        let (_, entry_point_0_ret_val) = cfg.add_stmt_op_load(
            entry_point_0_bb2,
            BlockPosition::Append,
            entry_point_0_ret_alloca.into(),
        );
        cfg.set_terminator(
            entry_point_0_bb2,
            Terminator::return_value(entry_point_0_ret_val.into()),
        );

        // Build the CFG for entry_point_1
        let entry_point_1_body = cfg.register_function(&module, entry_point_1);

        let entry_point_1_bb0 = entry_point_1_body.entry_block();

        cfg.add_stmt_op_load(
            entry_point_1_bb0,
            BlockPosition::Append,
            ConstPtr::new(&module, &cfg, RootIdentifier::Uniform(uniform_0)).into(),
        );
        cfg.add_stmt_op_load(
            entry_point_1_bb0,
            BlockPosition::Append,
            ConstPtr::new(&module, &cfg, RootIdentifier::Constant(overridable_0)).into(),
        );
        cfg.add_stmt_op_load(
            entry_point_1_bb0,
            BlockPosition::Append,
            ConstPtr::new(&module, &cfg, RootIdentifier::Storage(storage_read_write_0)).into(),
        );
        cfg.set_terminator(entry_point_1_bb0, Terminator::return_void());

        // Build the CFG for called_fn
        let called_fn_body = cfg.register_function(&module, called_fn);

        let called_fn_bb0 = called_fn_body.entry_block();

        cfg.add_stmt_op_load(
            called_fn_bb0,
            BlockPosition::Append,
            ConstPtr::new(&module, &cfg, RootIdentifier::Uniform(uniform_1)).into(),
        );
        cfg.add_stmt_op_load(
            called_fn_bb0,
            BlockPosition::Append,
            ConstPtr::new(&module, &cfg, RootIdentifier::Constant(overridable_1)).into(),
        );
        cfg.set_terminator(called_fn_bb0, Terminator::return_void());

        let smi = build_smi(&module, &cfg);

        assert_eq!(
            smi.overridable_constants,
            vec![
                smi::OverridableConstant {
                    id: 0,
                    constant_type: OverridableConstantType::UnsignedInteger,
                    required: false,
                },
                smi::OverridableConstant {
                    id: 10,
                    constant_type: OverridableConstantType::UnsignedInteger,
                    required: true,
                }
            ]
        );

        assert_eq!(
            smi.resource_bindings,
            vec![
                smi::ResourceBinding {
                    group: 0,
                    binding: 0,
                    resource_type: ResourceType::Uniform(SizedBufferLayout {
                        memory_units: vec![MemoryUnit {
                            offset: 0,
                            layout: MemoryUnitLayout::Float,
                        }]
                        .into(),
                    }),
                },
                smi::ResourceBinding {
                    group: 0,
                    binding: 1,
                    resource_type: ResourceType::Uniform(SizedBufferLayout {
                        memory_units: vec![MemoryUnit {
                            offset: 0,
                            layout: MemoryUnitLayout::Float,
                        }]
                        .into(),
                    }),
                },
                smi::ResourceBinding {
                    group: 1,
                    binding: 0,
                    resource_type: ResourceType::StorageRead(UnsizedBufferLayout {
                        sized_head: vec![MemoryUnit {
                            offset: 0,
                            layout: MemoryUnitLayout::Float,
                        }]
                        .into(),
                        unsized_tail: None,
                    }),
                },
                smi::ResourceBinding {
                    group: 4,
                    binding: 0,
                    resource_type: ResourceType::StorageReadWrite(UnsizedBufferLayout {
                        sized_head: vec![MemoryUnit {
                            offset: 0,
                            layout: MemoryUnitLayout::Float,
                        }]
                        .into(),
                        unsized_tail: None,
                    }),
                }
            ]
        );

        assert_eq!(smi.entry_points.len(), 2);

        assert_eq!(&smi.entry_points[0].name, "compute_main");
        assert_eq!(smi.entry_points[0].stage, smi::ShaderStage::Compute);
        assert_eq!(&smi.entry_points[0].input_bindings, &[]);
        assert_eq!(&smi.entry_points[0].output_bindings, &[]);
        assert_eq!(&smi.entry_points[0].overridable_constants, &[0]);
        assert_eq!(&smi.entry_points[0].resource_bindings, &[0, 3]);

        assert_eq!(smi.entry_points[1].name, "vertex_main");
        assert_eq!(smi.entry_points[1].stage, smi::ShaderStage::Vertex);
        assert_eq!(
            &smi.entry_points[1].input_bindings,
            &[
                IoBinding {
                    location: 0,
                    binding_type: smi::IoBindingType::Float,
                    interpolate: None
                },
                IoBinding {
                    location: 1,
                    binding_type: smi::IoBindingType::Float,
                    interpolate: None
                },
            ]
        );
        assert_eq!(
            &smi.entry_points[1].output_bindings,
            &[
                IoBinding {
                    location: 0,
                    binding_type: smi::IoBindingType::Float,
                    interpolate: Some(smi::Interpolate {
                        interpolation_type: smi::InterpolationType::Linear,
                        sampling: Some(smi::Sampling::Sample),
                    })
                },
                IoBinding {
                    location: 2,
                    binding_type: smi::IoBindingType::Float,
                    interpolate: None
                },
            ]
        );
        assert_eq!(&smi.entry_points[1].overridable_constants, &[0, 1]);
        assert_eq!(&smi.entry_points[1].resource_bindings, &[0, 1, 2]);
    }
}
