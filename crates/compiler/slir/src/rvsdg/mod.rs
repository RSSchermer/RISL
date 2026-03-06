//! The Regionalized Value-State Dependence Graph (RVSDG) intermediate representation.
//!
//! This module contains data structures for representing a shader-module as an RVSDG. It also
//! contains tooling for analyzing and transforming such RVSDGs in the [visit], [analyze] and
//! [transform] submodules. The RVSDG is SLIR's "workhorse" representation: most of the code
//! transforms in this crate operate on the RVSDG representation. The RVSDG representation is not
//! meant to be constructed directly from a source language; the typical work-flow would be to first
//! translate the source language representation (e.g., Rust MIR) to a SLIR Control-Flow-Graph (see
//! [crate::cfg]) and to then convert it to an RVSDG (see [crate::cfg_to_rvsdg]).
//!
//! The [Rvsdg] data structure in this module is based on the RVSDG data structure described in
//! [Reissmann et al., 2020][0], and familiarizing yourself with this publication is strongly
//! recommended before continuing to read here. There are some notable differences between the
//! RVSDG data structure in this module and the one described in the paper:
//!
//! - Reissmann et al. use the term "Lambda Node" to describe nodes that model functions and
//!   procedures. We use the term "Function Node" instead, see [FunctionNode] and
//!   [NodeKind::Function].
//! - Reissmann et al. use the term "Delta Node" to describe nodes that model global values. We
//!   split this into four different node kinds:
//!
//!   - Uniform Binding Nodes represent a shader module's uniform resource bindings; see
//!     [UniformBindingNode] and [NodeKind::UniformBinding].
//!   - Storage Binding Nodes represent a shader module's storage resource bindings; see
//!     [StorageBindingNode] and [NodeKind::StorageBinding].
//!   - Workgroup Binding Nodes represent a shader module's workgroup shared memory bindings; see
//!     [WorkgroupBindingNode] and [NodeKind::WorkgroupBinding].
//!   - Constant Nodes represent a shader module's constant values; see [ConstantNode] and
//!     [NodeKind::Constant].
//!
//! - Reissmann et al. use the term "Phi Node" to describe nodes that model recursion. We do not
//!   support recursion at all, so this node kind has no equivalent in our implementation.
//! - Reismann et al. use the term "Gramma Node" to describe nodes that model symmetric control-flow
//!   splits and joins. We use the term "Switch Node" instead, see [SwitchNode] and
//!   [NodeKind::Switch].
//! - Reissmann et al. use the term "Theta Node" to describe nodes that model tail-controlled loops.
//!   We use the term "Loop Node" instead, see [LoopNode] and [NodeKind::Loop].
//! - Reissmann et al. use the term "Omega Node" to describe nodes that model a translation unit
//!   (like a shader module). The [Rvsdg] data structure itself could be considered to map to the
//!   Omega node concept, though it does not implement the common interfaces all other node kinds
//!   implement. A [Rvsdg] has an associated "global region" ([Rvsdg::global_region]) that maps to
//!   the omega node's region. Reissmann et al. use this region's argument to model external
//!   dependencies (e.g., a function from a different translation unit); we don't allow external
//!   dependencies and instead always import all dependencies, so in our implementation the global
//!   region always has zero arguments.
//!
//! # Regions and Region Data
//!
//! We refer to regions with [Region] values. These are simple keys/identifiers that do not
//! themselves hold any data. The [Rvsdg] implement `Index<Region>`, which resolves a [Region]
//! to a [RegionData] reference:
//!
//! ```
//! # fn f(region: Region) {
//! let region_data = &rvsdg[region];
//! # }
//! ```
//!
//! The [RegionData] holds the actual information about the region, including its owner node, and
//! the nodes it contains. For a more complete description of the information it contains, refer to
//! the documentation for the [RegionData] struct.
//!
//! Region's are always associated with an "owner node" (see [RegionData::owner]), with the
//! exception of the global region ([Rvsdg::global_region]. Node kinds that may own regions are:
//!
//! - Function nodes (see [FunctionNode] and [NodeKind::Function]), which own a "body" region (see
//!   [FunctionNode::body_region]).
//! - Loop nodes (see [LoopNode] and [NodeKind::Loop]), which own a "loop" region (see
//!   [LoopNode::loop_region]).
//! - Switch nodes (see [SwitchNode] and [NodeKind::Switch]), which own one or more "branch" regions
//!   (see [SwitchNode::branches]).
//!
//! # Nodes and Node Data
//!
//! We refer to nodes with [Node] values. These are simple keys/identifiers that do not themselves
//! hold any data. The [Rvsdg] implement `Index<Node>`, which resolves a [Node] to a [NodeData]
//! reference:
//!
//! ```
//! # fn f(node: Node) {
//! let node_data = &rvsdg[node];
//! # }
//! ```
//!
//! The [NodeData] holds the actual information about the node, such as the region it belongs to,
//! and information specific to the node's kind. Refer to the documentation for the [NodeKind] enum
//! for more information on the various node kinds available in this module.
//!
//! # Value-Flow
//!
//! Values may flow from region arguments and node outputs, to region results and node inputs. We
//! model the producing side of such value-flow edges (region arguments and node outputs) with the
//! [ValueOutput] type, and we model the consuming side (region results and node inputs) with the
//! [ValueInput] type. Refer to the documentation for these respective types for more information.
//!
//! The [visit] submodule provides tools for tracing value-flow through the RVSDG.
//!
//! # The State Chain
//!
//! Every region has an associated "state chain" that starts at the body region's "state argument"
//! and ends at the body region's "state result". Various node kinds need to be "linked" into this
//! state chain to maintain the correct instruction order (see [Reissmann et al., 2020][0] for an
//! in-depth explanation). Such node kinds will have an associated [State] value that describes the
//! node's position in the state chain, see the documentation for the [State] struct for details.
//!
//! If a region's state argument connects directly to that region's state result, then that region's
//! state chain is considered "unused"; otherwise (if the state chain passes through one or more
//! nodes), the region's state chain is considered "used".
//!
//! A [LoopNode] is considered to use the state chain if it's loop region (see
//! [LoopNode::loop_region]) uses the state chain. A [SwitchNode] is considered to use the state
//! chain if any of its branch regions (see [SwitchNode::branches]) use the state chain. If a
//! [LoopNode] or [SwitchNode] uses the state chain, then it *must* be linked into its region's
//! state chain. If a [LoopNode] or [SwitchNode] does not use the state chain, then it *should not*
//! be linked into its regions state chain; though it is not considered invalid for a [LoopNode] or
//! [SwitchNode] that does not use the state chain to be linked into the state chain, this imposes
//! unnecessary ordering constraints that may hinder certain optimizing transformations.
//!
//! # Modifying the RVSDG
//!
//! All modifications to the RVSDG are done via methods on the [Rvsdg] data structure; there is
//! currently no way to directly modify [RegionData] or [NodeData]. The [Rvsdg] defines an "add"
//! method for each of the node kinds. [FunctionNode]s, [UniformBindingNode]s, [StorageBindingNode]s,
//! [WorkgroupBindingNode]s, and [ConstantNode]s are always added to the [Rvsdg]'s global region.
//! For all other node kinds, the first argument to the add method is always the [Region] in which
//! the node should be added.
//!
//! Certain node kinds can be modified after they have been added to the [Rvsdg]. For example,
//! [Rvsdg::add_switch_branch] can be used to add a new (empty) branch region to an existing switch
//! node.
//!
//! If a modification operation results in the addition of a new node or region, the operation will
//! typically return a [Node] or [Region] respectively (or both; for example,
//! [Rvsdg::register_function] returns a `(Node, Region)` pair that represents the newly added
//! [FunctionNode] and its "body" region).
//!
//! All methods that modify the RVSDG strive to maintain certain invariants:
//!
//! - Value-flow edges must never cross region boundaries. For example, when adding a new node to
//!   the RVSDG, the "add" method will verify that the node's inputs originate from the same region
//!   as the region to which the node is being added.
//! - The state chain must never cross region boundaries. For example, when adding a node kind that
//!   needs to be linked into the state chain, the state edge into which the node is being added
//!   must be in the same region as the region to which the node is being added.
//! - The type of the [ValueOutput] to which a [ValueInput]s origin resolves, must be compatible
//!   (as defined by [TypeRegistry::can_coerce]) with the type declared for the [ValueInput].
//!
//! Specific operations may enforce additional invariants. For example, [Rvsdg::add_op_load] will
//! verify that the value provided as its `ptr_input` (which represents the pointer to load from)
//! does in fact have a pointer type.
//!
//! If a modification operation is invoked with arguments that violate any of its invariants, this
//! will result in a panic.
//!
//! # Debugging and RVSDG Dumps
//!
//! The graph-like nature of an RVSDG makes it difficult to understand the structure of the graph
//! from plain text. While [Rvsdg] does implement [Debug], this is not the recommended way of
//! debugging the RVSDG. Instead, use the [Rvsdg::dump_to_file] method to create a "dump file". The
//! `slir` crate has two companion crates that can then be used to explore an RVSDG dump:
//!
//! - The `slir-explorer` crate: this is a web-based tool that can render the RVSDG dump as an
//!   interactive SVG image. This is the recommended way to explore the RVSDG for humans.
//! - The `rvsdg-dump-cli` crate: this is a command-line tool that can be used to query the dump
//!   for information and render textual representations of parts of the RVSDG to the terminal.
//!   This is the recommended way to explore the RVSDG for LLM agents.
//!
//! Refer to the README documents for these crates for more details on their use.
//!
//! [0]: https://arxiv.org/abs/1912.05036

pub mod analyse;
pub mod transform;
pub mod visit;

mod rvsdg;

pub use self::rvsdg::*;
use crate::ty::TypeRegistry;
