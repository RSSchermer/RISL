# rvsdg-dump-cli

`rvsdg-dump-cli` is a command-line tool designed to explore and analyze RVSDG (Regional
Value-State Dependence Graph) binary dumps produced by the `slir` compiler.

Note that the primary audience for this tool is LLM agents. If you are a human, you might prefer
the [slir-explorer](../slir-explorer) web application instead, which can render an RVSDG dump as
an interactive SVG diagram.

## Overview

The RVSDG is a hierarchical graph representation where nodes are contained within regions, and
regions are owned by nodes (e.g., `Switch`, `Loop`, or `Function`). This tool provides a textual,
semantic view of this graph, making it easier to trace data and state flow, identify structural
issues, and debug compiler transformations. For more information on the RVSDG, refer to the 
documentation for the `slir::rvsdg` module in the `slir` crate.

## Usage

The RISL project defines the Cargo alias `rvsdg-dump` to ensure this tool runs with an up-to-date 
build. The recommended way to invoke this tool is through this Cargo alias:

```bash
cargo rvsdg-dump <path-to-dump> [mode-flag] [options]
```

### Exploration Modes

- **Discovery (`--list`)**: Provides a high-level inventory of the module, listing all registered
  functions, global bindings (Uniform, Storage, Workgroup), and global constants.
- **Function Analysis (`--function <name|ID>`)**: Analyzes a specific function. It shows the
  function's dependencies in the global region and renders its body region.
- **Region Inspection (`--region <ID>`)**: Zooms in on a specific region (e.g., a branch of a
  `Switch` or the body of a `Loop`).
- **Node Inspection (`--node <ID>`)**: Inspects a single node in isolation, including any inlined
  child regions.
- **Value Tracing (`--trace-value <ValueID>`)**: Traces the flow of a specific value within its
  region.
    - **Backward Trace**: If you provide a node input or region result, it shows the origin of the 
      value.
    - **Forward Trace**: If you provide a node output or region argument, it lists all users of the 
      value.
- **State Tracing (`--trace-state <RegionID>`)**: Visualizes the linear chain of side-effecting
  operations (the state chain) within a region, from the region's state argument to the region's 
  state result.
- **Type Inspection (`--type <TypeID>`)**: Displays the detailed structure of a registered type
  (fields for structs, variants for enums).

Note that because node IDs, region IDs, and value IDs contain parentheses, they need to be wrapped in
quotes. For example:

```bash
cargo rvsdg-dump rvsdg.dump --node "Node(1v1)"
```

### Options

- `--inline-max-node-count <N>`: Maximum nodes a region can have to be rendered inline (default: 5).
- `--inline-max-nesting_level <N>`: Maximum depth of region inlining (default: 1).
- `--no-color`: Disables ANSI color output (recommended for LLM processing).

## Interpreting the Output

### Identifiers

- **Nodes**: `Node(ID)` (e.g., `Node(1v1)`).
- **Regions**: `Region(ID)` (e.g., `Region(2v1)`).
- **Types**: `struct(ID)` or `enum(ID)`.
- **Value IDs**:
    - `Node(ID)iN`: Node input $N$ (e.g., `Node(1v1)i0`).
    - `Node(ID)eN`: Node output $N$ (e.g., `Node(1v1)e0`).
    - `Region(ID)aN`: Region argument $N$ (e.g., `Region(1v1)a0`).
    - `Region(ID)rN`: Region result $N$ (e.g., `Region(1v1)r0`).
- **State Origins**:
    - `Node(ID)s`: Node state output (e.g., `Node(1v1)s`).
    - `Region(ID)s`: Region state argument (e.g., `Region(1v1)s`).

### Node Rendering

A typical node is rendered as:
`[Node(ID)] OpName{opconfig}(inputs) (state: origin) -> outputs`

Example:
`[Node(4v1)] OpBinary{operator: +}(Region(2v1)a2, Region(2v1)a3) -> Node(4v1)e0 : u32`

- **OpName**: Names like `OpLoad`, `OpStore`, `OpBinary`.
- **opconfig**: Static parameters in curly braces that configure the nodes behavior (e.g., 
  `{operator: +}`, `{field_index: 1}`).
- **inputs**: A list of value IDs that describe the node's inputs. These can be region
  arguments (e.g., `Region(2v1)a2`), or node-outputs (e.g., `Node(4v1)e2`).
- **state**: If the node is side-effecting, its state dependency is shown as `(state: origin)`,
  where origin is either `Arg` (region state argument) or a preceding `Node(ID)`.
- **outputs**: If the node producer output-values, a list of the value `valueID : type` pairs.
  (e.g., `Node(4v1)e0 : u32`).

### Region Rendering

A region (e.g., a function body, a loop body, or a switch branch) is rendered as a an argument
list, followed by a list of nodes, followed by a result list:

```text
Body Region (Region(2v1)):
  Arguments: [Region(2v1)a0: u32, Region(2v1)s: State]
  [Node(4v1)] OpBinary{operator: +}(Region(2v1)a0, Region(2v1)a0) -> Node(4v1)e0 : u32
  Results: [Node(4v1)e0, Region(2v1)s]
```

Example of a nested region (Switch branch):
```text
[Node(5v1)] Switch(Region(2v1)a0) -> Node(5v1)e0 : u32
  Branch 0 (Region(3v1)):
    Arguments: [Region(3v1)a0: u32, Region(3v1)s: State]
    [Node(6v1)] Constant -> Node(6v1)e0 : u32
    Results: [Node(6v1)e0, Region(3v1)s]
```

- **Header**: Identifies the region's role and its ID (e.g., `Body Region (Region(2v1)):`, 
  `Branch 0 (Region(3v1)):`, or `Loop Region (Region(4v1)):`).
- **Arguments**: A list of value arguments (`Region(ID)aN`) and their types, plus the implicit 
  state argument (`Region(ID)s: State`).
- **Nodes**: A list of the nodes in the region. The nodes are ordered such that their inputs are
  always satisfied by earlier nodes.
- **Results**: A list of the region's value results (mapped to the producing node output or 
  region argument) and the region's state result (`Region(ID)s`).

### Hierarchical View (Smart Inlining)

The tool uses a hybrid approach to manage large graphs:
- **Inlined**: Small regions are expanded directly under their owner node.
- **Summarized**: Large regions are collapsed: `Branch 1 (Region(4v1)): 8 child nodes`.
  Use `--region Region(4v1)` to inspect them.

## LLM Agent Workflow

1.  **Start with `--list`**: Get an overview of the available functions and global resources.
2.  **Analyze Entry Points**: Use `--function <name>` to look at the top-level logic of functions of
    interest.
3.  **Investigate Control Flow**: If a `Switch` or `Loop` has collapsed regions, use `--region <ID>`
    to zoom in, if the region is of interest.
4.  **Trace Data Flow**: If you see an unexpected value, use `--trace-value` on its ID to find where
    it comes from (for node inputs and region results) or where it goes (for node outputs and region
    arguments).
5.  **Verify State Order**: Use `--trace-state <RegionID>` to get a concise overview of the region's 
    state chain, if the state chain is of interest.
6.  **Check Types**: If you see `struct(ID)` or `enum(ID)`, use `--type <TypeID>` to understand the data 
    layout, if the type is of interest.
