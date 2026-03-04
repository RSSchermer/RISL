use anyhow::{Result, anyhow};
use nom::bytes::complete::tag;
use nom::character::complete::{char, digit1, one_of};
use nom::combinator::map_res;
use nom::sequence::delimited;
use nom::{IResult, Parser};
use slir::rvsdg::{Node, Region};
use slotmap::KeyData;

fn parse_u32(input: &str) -> IResult<&str, u32> {
    map_res(digit1, |s: &str| s.parse::<u32>()).parse(input)
}

fn parse_id_inner(input: &str) -> IResult<&str, (u32, u32)> {
    (parse_u32, char('v'), parse_u32)
        .map(|(index, _, version)| (index, version))
        .parse(input)
}

fn construct_key<T: From<KeyData>>(index: u32, version: u32) -> T {
    let ffi = (version as u64) << 32 | (index as u64);
    KeyData::from_ffi(ffi).into()
}

pub fn parse_node_id(id: &str) -> Result<Node> {
    let (_, (index, version)) = delimited(tag("Node("), parse_id_inner, char(')'))
        .parse(id)
        .map_err(|e| {
            anyhow!(
                "Invalid Node ID format: {}. Expected Node(<index>v<version>)",
                e
            )
        })?;

    Ok(construct_key(index, version))
}

pub fn parse_region_id(id: &str) -> Result<Region> {
    let (_, (index, version)) = delimited(tag("Region("), parse_id_inner, char(')'))
        .parse(id)
        .map_err(|e| {
            anyhow!(
                "Invalid Region ID format: {}. Expected Region(<index>v<version>)",
                e
            )
        })?;

    Ok(construct_key(index, version))
}

pub fn parse_type_id(id: &str) -> Result<u32> {
    // Matches "struct(0)", "enum(0)" or "0"
    let mut parser = nom::branch::alt((
        delimited(tag("struct("), parse_u32, char(')')),
        delimited(tag("enum("), parse_u32, char(')')),
        parse_u32,
    ));

    let (_, ty_id) = parser.parse(id).map_err(|e| {
        anyhow!(
            "Invalid Type ID format: {}. Expected struct(<id>), enum(<id>), or <id>",
            e
        )
    })?;

    Ok(ty_id)
}

#[derive(Debug, Clone, Copy)]
pub enum ParsedValueId {
    NodeOutput(Node, u32),
    NodeInput(Node, u32),
    RegionArgument(Region, u32),
    RegionResult(Region, u32),
}

pub fn parse_value_id(id: &str) -> Result<ParsedValueId> {
    let (_, (kind, (index, version), type_char, sub_index)) = (
        nom::branch::alt((tag("Node"), tag("Region"))),
        delimited(char('('), parse_id_inner, char(')')),
        one_of("eiar"),
        parse_u32,
    )
        .parse(id)
        .map_err(|e| {
            anyhow!(
                "Invalid Value ID format: {}. Expected Node(<index>v<version>)[ei]<index> or \
                Region(<index>v<version>)[ar]<index>",
                e
            )
        })?;

    match (kind, type_char) {
        ("Node", 'e') => Ok(ParsedValueId::NodeOutput(
            construct_key(index, version),
            sub_index,
        )),
        ("Node", 'i') => Ok(ParsedValueId::NodeInput(
            construct_key(index, version),
            sub_index,
        )),
        ("Region", 'a') => Ok(ParsedValueId::RegionArgument(
            construct_key(index, version),
            sub_index,
        )),
        ("Region", 'r') => Ok(ParsedValueId::RegionResult(
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
