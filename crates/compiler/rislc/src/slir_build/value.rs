#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Value {
    Value(slir::cfg::Value),
    FnAddr(slir::Function),
    Constant(slir::Constant),
    Void,
}

impl Value {
    pub fn expect_value(&self) -> slir::cfg::Value {
        if let Value::Value(v) = self {
            *v
        } else {
            panic!("expected a value")
        }
    }

    pub fn expect_fn_addr(&self) -> slir::Function {
        if let Value::FnAddr(f) = self {
            *f
        } else {
            panic!("expected a function address")
        }
    }

    pub fn expect_constant(&self) -> slir::Constant {
        if let Value::Constant(c) = self {
            *c
        } else {
            panic!("expected a constant")
        }
    }
}

impl From<slir::cfg::Value> for Value {
    fn from(value: slir::cfg::Value) -> Self {
        Value::Value(value)
    }
}

impl From<slir::cfg::LocalBinding> for Value {
    fn from(value: slir::cfg::LocalBinding) -> Self {
        Value::Value(slir::cfg::Value::Local(value))
    }
}

impl From<slir::cfg::InlineConst> for Value {
    fn from(value: slir::cfg::InlineConst) -> Self {
        Value::Value(slir::cfg::Value::InlineConst(value))
    }
}

impl From<slir::Function> for Value {
    fn from(value: slir::Function) -> Self {
        Value::FnAddr(value)
    }
}
