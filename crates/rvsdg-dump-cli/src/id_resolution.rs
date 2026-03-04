use std::sync::OnceLock;

use anyhow::{Context, Result, anyhow};
use regex::Regex;
use slir::rvsdg::{Node, Region};
use slotmap::KeyData;

static NODE_REGION_REGEX: OnceLock<Regex> = OnceLock::new();
static VALUE_ID_REGEX: OnceLock<Regex> = OnceLock::new();

fn get_node_region_regex() -> &'static Regex {
    NODE_REGION_REGEX.get_or_init(|| Regex::new(r"^(Node|Region)\((\d+)v(\d+)\)$").unwrap())
}

fn get_value_id_regex() -> &'static Regex {
    VALUE_ID_REGEX
        .get_or_init(|| Regex::new(r"^(Node|Region)\((\d+)v(\d+)\)([eiar])(\d+)$").unwrap())
}

fn construct_key<T: From<KeyData>>(index: u32, version: u32) -> T {
    let ffi = (version as u64) << 32 | (index as u64);
    KeyData::from_ffi(ffi).into()
}

pub fn parse_node_id(id: &str) -> Result<Node> {
    let caps = get_node_region_regex()
        .captures(id)
        .ok_or_else(|| anyhow!("Invalid ID format: {}. Expected Node(indexvversion)", id))?;

    if &caps[1] != "Node" {
        return Err(anyhow!("Expected Node ID, found {}", &caps[1]));
    }

    let index: u32 = caps[2].parse().context("Failed to parse node index")?;
    let version: u32 = caps[3].parse().context("Failed to parse node version")?;

    Ok(construct_key(index, version))
}

pub fn parse_region_id(id: &str) -> Result<Region> {
    let caps = get_node_region_regex()
        .captures(id)
        .ok_or_else(|| anyhow!("Invalid ID format: {}. Expected Region(indexvversion)", id))?;

    if &caps[1] != "Region" {
        return Err(anyhow!("Expected Region ID, found {}", &caps[1]));
    }

    let index: u32 = caps[2].parse().context("Failed to parse region index")?;
    let version: u32 = caps[3].parse().context("Failed to parse region version")?;

    Ok(construct_key(index, version))
}

#[derive(Debug, Clone, Copy)]
pub enum ParsedValueId {
    NodeOutput(Node, u32),
    NodeInput(Node, u32),
    RegionArgument(Region, u32),
    RegionResult(Region, u32),
}

pub fn parse_value_id(id: &str) -> Result<ParsedValueId> {
    let caps = get_value_id_regex().captures(id)
        .ok_or_else(|| anyhow!("Invalid Value ID format: {}. Expected Node(indexvversion)[ei]index or Region(indexvversion)[ar]index", id))?;

    let kind = &caps[1];
    let index: u32 = caps[2].parse().context("Failed to parse index")?;
    let version: u32 = caps[3].parse().context("Failed to parse version")?;
    let type_char = &caps[4];
    let sub_index: u32 = caps[5].parse().context("Failed to parse sub-index")?;

    match (kind, type_char) {
        ("Node", "e") => Ok(ParsedValueId::NodeOutput(
            construct_key(index, version),
            sub_index,
        )),
        ("Node", "i") => Ok(ParsedValueId::NodeInput(
            construct_key(index, version),
            sub_index,
        )),
        ("Region", "a") => Ok(ParsedValueId::RegionArgument(
            construct_key(index, version),
            sub_index,
        )),
        ("Region", "r") => Ok(ParsedValueId::RegionResult(
            construct_key(index, version),
            sub_index,
        )),
        _ => Err(anyhow!(
            "Invalid combination of kind and type: {}{}",
            kind,
            type_char
        )),
    }
}
