mod advance_back_by;
mod advance_by;
mod count;
mod fold;
mod is_empty;
mod last;
mod len;
mod next;
mod next_back;
mod nth;
mod rfold;
mod size_hint;
mod try_fold;
mod try_rfold;

use empa::abi;
use risl::gpu;

// Host inputs share `Value`'s storage layout without its custom `Clone`.
#[derive(Copy, Clone, abi::Sized)]
struct InputValue {
    value: u32,
}

// An observable `Clone` distinguishes copying from accidentally cloning the source.
#[derive(Copy, abi::Sized)]
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
