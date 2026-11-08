mod fold;
mod is_empty;
mod len;
mod next;
mod next_back;
mod rfold;
mod size_hint;
mod try_fold;
mod try_rfold;

use empa::abi;
use risl::gpu;

// Empa requires Copy inputs; this host type shares `Value`'s storage layout.
#[derive(Copy, Clone, abi::Sized)]
struct InputValue {
    value: u32,
}

// Non-`Copy`, with an observable `Clone` to distinguish cloning from copying the source.
#[derive(abi::Sized)]
#[gpu]
struct Value {
    value: u32,
}

#[gpu]
impl Clone for Value {
    fn clone(&self) -> Self {
        Value {
            value: self.value + 1,
        }
    }
}
