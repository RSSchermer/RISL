use crate::rvsdg::{Node, NodeKind, Region, Rvsdg, SimpleNode};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ScalarConstant {
    U32(u32),
    I32(i32),
    F32(u32),
    Bool(bool),
    Predicate(u32),
}

impl ScalarConstant {
    pub fn from_node(rvsdg: &Rvsdg, node: Node, output: u32) -> Option<Self> {
        if output != 0 {
            return None;
        }

        match rvsdg[node].kind() {
            NodeKind::Simple(SimpleNode::ConstU32(value)) => Some(Self::U32(value.value())),
            NodeKind::Simple(SimpleNode::ConstI32(value)) => Some(Self::I32(value.value())),
            NodeKind::Simple(SimpleNode::ConstF32(value)) => {
                Some(Self::F32(value.value().to_bits()))
            }
            NodeKind::Simple(SimpleNode::ConstBool(value)) => Some(Self::Bool(value.value())),
            NodeKind::Simple(SimpleNode::ConstPredicate(value)) => {
                Some(Self::Predicate(value.value()))
            }
            _ => None,
        }
    }

    pub fn add_to_region(self, rvsdg: &mut Rvsdg, region: Region) -> Node {
        match self {
            Self::U32(value) => rvsdg.add_const_u32(region, value),
            Self::I32(value) => rvsdg.add_const_i32(region, value),
            Self::F32(value) => rvsdg.add_const_f32(region, f32::from_bits(value)),
            Self::Bool(value) => rvsdg.add_const_bool(region, value),
            Self::Predicate(value) => rvsdg.add_const_predicate(region, value),
        }
    }

    pub fn integer_encoding(self) -> Option<u128> {
        match self {
            Self::U32(value) => Some(value as u128),
            Self::I32(value) => Some(value as u32 as u128),
            _ => None,
        }
    }
}
