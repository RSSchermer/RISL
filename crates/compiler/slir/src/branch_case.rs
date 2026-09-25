use std::fmt;
use std::num::TryFromIntError;

use serde::{Deserialize, Serialize};

use crate::ty::IntSize;

/// A branch case encoded as the zero-extended bit pattern of an integer.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Debug)]
pub struct BranchCase(u128);

impl BranchCase {
    /// Attempt to re-encode a case as an `i32` case.
    pub fn try_cast_i32(self, source_size: IntSize) -> Result<Self, TryFromIntError> {
        let value = match source_size {
            IntSize::I8 => i32::from(i8::try_from(self)?),
            IntSize::I16 => i32::from(i16::try_from(self)?),
            IntSize::I32 => i32::try_from(self)?,
            IntSize::I64 => i32::try_from(i64::try_from(self)?)?,
            IntSize::I128 => i32::try_from(i128::from(self))?,
        };

        Ok(Self::from(value))
    }

    /// Attempt to re-encode a case as an `u32` case.
    pub fn try_cast_u32(self) -> Result<Self, TryFromIntError> {
        u32::try_from(self).map(Self::from)
    }
}

impl fmt::Display for BranchCase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}

macro_rules! impl_narrow_integer_conversions {
    ($signed:ty, $unsigned:ty) => {
        impl From<$signed> for BranchCase {
            fn from(value: $signed) -> Self {
                Self(value as $unsigned as u128)
            }
        }

        impl From<$unsigned> for BranchCase {
            fn from(value: $unsigned) -> Self {
                Self(value as u128)
            }
        }

        impl TryFrom<BranchCase> for $signed {
            type Error = TryFromIntError;

            fn try_from(value: BranchCase) -> Result<Self, Self::Error> {
                Ok(<$unsigned>::try_from(value.0)? as $signed)
            }
        }

        impl TryFrom<BranchCase> for $unsigned {
            type Error = TryFromIntError;

            fn try_from(value: BranchCase) -> Result<Self, Self::Error> {
                <$unsigned>::try_from(value.0)
            }
        }
    };
}

impl_narrow_integer_conversions!(i8, u8);
impl_narrow_integer_conversions!(i16, u16);
impl_narrow_integer_conversions!(i32, u32);
impl_narrow_integer_conversions!(i64, u64);

impl From<i128> for BranchCase {
    fn from(value: i128) -> Self {
        Self(value as u128)
    }
}

impl From<u128> for BranchCase {
    fn from(value: u128) -> Self {
        Self(value)
    }
}

impl From<BranchCase> for i128 {
    fn from(value: BranchCase) -> Self {
        value.0 as i128
    }
}

impl From<BranchCase> for u128 {
    fn from(value: BranchCase) -> Self {
        value.0
    }
}

#[cfg(test)]
mod tests {
    use super::BranchCase;
    use crate::ty::IntSize;

    #[test]
    fn display_case_as_unsigned_decimal() {
        assert_eq!(BranchCase::from(-1i8).to_string(), "255");
        assert_eq!(
            BranchCase::from(u128::MAX).to_string(),
            u128::MAX.to_string()
        );
    }

    #[test]
    fn serialized_case_matches_u128() {
        let config = bincode::config::standard();

        for value in [0u128, 255, u128::MAX] {
            let case = BranchCase::from(value);
            let serialized = bincode::serde::encode_to_vec(case, config).unwrap();

            assert_eq!(
                serialized,
                bincode::serde::encode_to_vec(value, config).unwrap()
            );
            assert_eq!(
                bincode::serde::decode_from_slice::<BranchCase, _>(&serialized, config)
                    .unwrap()
                    .0,
                case
            );
        }
    }

    #[test]
    fn cast_signed_cases_to_i32_encoding() {
        assert_eq!(
            BranchCase::from(-1i8).try_cast_i32(IntSize::I8),
            Ok(BranchCase::from(-1i32))
        );
        assert_eq!(
            BranchCase::from(-1i16).try_cast_i32(IntSize::I16),
            Ok(BranchCase::from(-1i32))
        );
        assert_eq!(
            BranchCase::from(-1i32).try_cast_i32(IntSize::I32),
            Ok(BranchCase::from(-1i32))
        );
        assert_eq!(
            BranchCase::from(-1i64).try_cast_i32(IntSize::I64),
            Ok(BranchCase::from(-1i32))
        );
        assert_eq!(
            BranchCase::from(-1i128).try_cast_i32(IntSize::I128),
            Ok(BranchCase::from(-1i32))
        );
        assert_eq!(
            BranchCase::from(i16::MIN).try_cast_i32(IntSize::I16),
            Ok(BranchCase::from(i16::MIN as i32))
        );
        assert_eq!(
            BranchCase::from(i32::MIN as i64).try_cast_i32(IntSize::I64),
            Ok(BranchCase::from(i32::MIN))
        );
        assert_eq!(
            BranchCase::from(i32::MAX as i128).try_cast_i32(IntSize::I128),
            Ok(BranchCase::from(i32::MAX))
        );
        assert!(
            BranchCase::from(i32::MIN as i128 - 1)
                .try_cast_i32(IntSize::I128)
                .is_err()
        );
    }

    #[test]
    fn cast_unsigned_cases_to_u32_encoding() {
        assert_eq!(
            BranchCase::from(0u32).try_cast_u32(),
            Ok(BranchCase::from(0u32))
        );
        assert_eq!(
            BranchCase::from(1u32).try_cast_u32(),
            Ok(BranchCase::from(1u32))
        );
        assert_eq!(
            BranchCase::from(u32::MAX).try_cast_u32(),
            Ok(BranchCase::from(u32::MAX))
        );
        assert!(
            BranchCase::from(u32::MAX as u128 + 1)
                .try_cast_u32()
                .is_err()
        );
    }
}
