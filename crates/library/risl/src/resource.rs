pub use risl_macros::resource;

pub(crate) mod seal {
    pub trait Sealed {}
}

pub trait Resource: seal::Sealed {}
