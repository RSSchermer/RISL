use std::marker::PhantomData;
use std::ops::{Deref, RangeInclusive};
use std::sync::{Arc, PoisonError, RwLock};
use std::{fmt, mem};

use indexmap::IndexSet;
use serde::{Deserialize, Serialize};

use crate::{Function, ShaderIOBinding};

#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Debug)]
pub struct Type(TypeInner);

impl Type {
    pub fn registration_id(&self) -> Option<usize> {
        if let TypeInner::Registered(id) = &self.0 {
            Some(*id)
        } else {
            None
        }
    }

    pub fn from_registration_id(id: usize) -> Self {
        Type(TypeInner::Registered(id))
    }

    pub fn to_string(&self, ty_registry: &TypeRegistry) -> String {
        ty_registry.kind(*self).to_string(ty_registry)
    }

    pub fn is_scalar(&self) -> bool {
        matches!(*self, TY_U32 | TY_I32 | TY_F32 | TY_BOOL)
    }

    pub fn is_numeric_scalar(&self) -> bool {
        matches!(*self, TY_U32 | TY_I32 | TY_F32)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Debug)]
enum TypeInner {
    U32,
    I32,
    F32,
    Bool,
    Vec2U32,
    Vec2I32,
    Vec2F32,
    Vec2Bool,
    Vec3U32,
    Vec3I32,
    Vec3F32,
    Vec3Bool,
    Vec4U32,
    Vec4I32,
    Vec4F32,
    Vec4Bool,
    Mat2x2,
    Mat2x3,
    Mat2x4,
    Mat3x2,
    Mat3x3,
    Mat3x4,
    Mat4x2,
    Mat4x3,
    Mat4x4,
    AtomicU32,
    AtomicI32,
    AtomicF32,
    AtomicBool,
    Predicate,
    PtrU32,
    Dummy,
    Registered(usize),
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Debug)]
pub struct Vector {
    pub scalar: ScalarKind,
    pub size: VectorSize,
}

impl Vector {
    pub fn vec2_f32() -> Vector {
        Vector {
            scalar: ScalarKind::F32,
            size: VectorSize::Two,
        }
    }

    pub fn vec2_u32() -> Vector {
        Vector {
            scalar: ScalarKind::U32,
            size: VectorSize::Two,
        }
    }

    pub fn vec2_i32() -> Vector {
        Vector {
            scalar: ScalarKind::I32,
            size: VectorSize::Two,
        }
    }

    pub fn vec2_bool() -> Vector {
        Vector {
            scalar: ScalarKind::Bool,
            size: VectorSize::Two,
        }
    }

    pub fn vec3_f32() -> Vector {
        Vector {
            scalar: ScalarKind::F32,
            size: VectorSize::Three,
        }
    }

    pub fn vec3_u32() -> Vector {
        Vector {
            scalar: ScalarKind::U32,
            size: VectorSize::Three,
        }
    }

    pub fn vec3_i32() -> Vector {
        Vector {
            scalar: ScalarKind::I32,
            size: VectorSize::Three,
        }
    }

    pub fn vec3_bool() -> Vector {
        Vector {
            scalar: ScalarKind::Bool,
            size: VectorSize::Three,
        }
    }

    pub fn vec4_f32() -> Vector {
        Vector {
            scalar: ScalarKind::F32,
            size: VectorSize::Four,
        }
    }

    pub fn vec4_u32() -> Vector {
        Vector {
            scalar: ScalarKind::U32,
            size: VectorSize::Four,
        }
    }

    pub fn vec4_i32() -> Vector {
        Vector {
            scalar: ScalarKind::I32,
            size: VectorSize::Four,
        }
    }

    pub fn vec4_bool() -> Vector {
        Vector {
            scalar: ScalarKind::Bool,
            size: VectorSize::Four,
        }
    }
}

impl fmt::Display for Vector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.size {
            VectorSize::Two => write!(f, "vec2<{}>", self.scalar),
            VectorSize::Three => write!(f, "vec3<{}>", self.scalar),
            VectorSize::Four => write!(f, "vec4<{}>", self.scalar),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Debug)]
pub struct Matrix {
    pub rows: VectorSize,
    pub columns: VectorSize,
    pub scalar: ScalarKind,
}

impl Matrix {
    pub fn column_vector(&self) -> Vector {
        Vector {
            scalar: self.scalar,
            size: self.rows,
        }
    }

    pub fn column_ty(&self) -> Type {
        match (self.scalar, self.rows) {
            (ScalarKind::I32, VectorSize::Two) => TY_VEC2_I32,
            (ScalarKind::I32, VectorSize::Three) => TY_VEC3_I32,
            (ScalarKind::I32, VectorSize::Four) => TY_VEC4_I32,
            (ScalarKind::U32, VectorSize::Two) => TY_VEC2_U32,
            (ScalarKind::U32, VectorSize::Three) => TY_VEC3_U32,
            (ScalarKind::U32, VectorSize::Four) => TY_VEC4_U32,
            (ScalarKind::F32, VectorSize::Two) => TY_VEC2_F32,
            (ScalarKind::F32, VectorSize::Three) => TY_VEC3_F32,
            (ScalarKind::F32, VectorSize::Four) => TY_VEC4_F32,
            (ScalarKind::Bool, VectorSize::Two) => TY_VEC2_BOOL,
            (ScalarKind::Bool, VectorSize::Three) => TY_VEC3_BOOL,
            (ScalarKind::Bool, VectorSize::Four) => TY_VEC4_BOOL,
        }
    }
}

impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match (self.columns, self.rows) {
            (VectorSize::Two, VectorSize::Two) => write!(f, "mat2x2<{}>", self.scalar),
            (VectorSize::Two, VectorSize::Three) => write!(f, "mat2x3<{}>", self.scalar),
            (VectorSize::Two, VectorSize::Four) => write!(f, "mat2x4<{}>", self.scalar),
            (VectorSize::Three, VectorSize::Two) => write!(f, "mat3x2<{}>", self.scalar),
            (VectorSize::Three, VectorSize::Three) => write!(f, "mat3x3<{}>", self.scalar),
            (VectorSize::Three, VectorSize::Four) => write!(f, "mat3x4<{}>", self.scalar),
            (VectorSize::Four, VectorSize::Two) => write!(f, "mat4x2<{}>", self.scalar),
            (VectorSize::Four, VectorSize::Three) => write!(f, "mat4x3<{}>", self.scalar),
            (VectorSize::Four, VectorSize::Four) => write!(f, "mat4x4<{}>", self.scalar),
        }
    }
}

#[derive(Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Debug)]
pub struct Struct {
    pub fields: Vec<StructField>,
}

#[derive(Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Debug)]
pub struct Enum {
    /// The ordered list of variants in the enum.
    ///
    /// Note that each variant must have a unique [EnumVariant::discriminant] value, otherwise the
    /// enum is invalid (UB).
    pub variants: Vec<EnumVariant>,

    /// Describes the type of the discriminant for this enum.
    ///
    /// Is used to decode the [EnumVariant::discriminant] values in this enum's [variants] list.
    pub discriminant_ty: Int,

    /// Describes the type of tag-values for this enum.
    pub tag_ty: EnumTagTy,

    /// Describes how the tag-value encodes the enum's discriminant.
    pub tag_encoding: EnumTagEncoding,

    /// The offset in bytes of the tag-value within the enum data.
    pub tag_offset: u64,
}

impl Enum {
    pub fn read_discriminant(&self, enum_data: &[u8]) -> EnumDiscriminant {
        let tag_value = self.read_tag_value(enum_data);

        match &self.tag_encoding {
            EnumTagEncoding::Direct => {
                let tag_ty = self.tag_ty.expect_int();

                let mut v = tag_value;

                if tag_ty.size != self.discriminant_ty.size {
                    // The discriminant-size may be different from the tag-size (if the user
                    // declared an explicit discriminant type for the enum). To convert the tag to
                    // the discriminant size:
                    //
                    // 1. For signed types we first sign-extend the tag to 128 bits.
                    // 2. For both signed and unsigned types, we truncate back down to the
                    //    discriminant size.

                    if tag_ty.signed {
                        // Sign-extend

                        let size = self.tag_ty.size();
                        let shift = 128 - (size * 8);

                        v = ((v << shift) as i128 >> shift) as u128;
                    }

                    // Truncate
                    v &= self.discriminant_ty.size.mask();
                }

                EnumDiscriminant {
                    value: v,
                    ty: self.discriminant_ty,
                }
            }
            EnumTagEncoding::Niche {
                valid_range,
                untagged_variant,
                niche_variants,
                niche_start,
            } => {
                let variant_index = if valid_range.contains(tag_value) {
                    *untagged_variant
                } else {
                    let niche_index = tag_value.wrapping_sub(*niche_start) & self.tag_ty.mask();
                    let count = (niche_variants.end() - niche_variants.start() + 1) as u128;

                    if niche_index < count {
                        niche_variants.start() + niche_index as usize
                    } else {
                        panic!("invalid tag value for niche-encoded enum");
                    }
                };

                EnumDiscriminant {
                    value: self.variants[variant_index].discriminant,
                    ty: self.discriminant_ty,
                }
            }
        }
    }

    pub fn variant_index_for(&self, discriminant: EnumDiscriminant) -> Option<usize> {
        self.variants
            .iter()
            .position(|v| v.discriminant == discriminant.value)
    }

    pub fn resolve_variant_index(&self, enum_data: &[u8]) -> usize {
        let discriminant = self.read_discriminant(enum_data);

        self.variant_index_for(discriminant)
            .expect("enum tag-value should encode a valid discriminant")
    }

    fn read_tag_value(&self, enum_data: &[u8]) -> u128 {
        let tag_offset = self.tag_offset as usize;
        let tag_data = &enum_data[tag_offset..];

        match self.tag_ty {
            EnumTagTy::Int(int) => match int.size {
                IntSize::I8 => enum_data[0] as u128,
                IntSize::I16 => u16::from_le_bytes(tag_data[..2].try_into().unwrap()) as u128,
                IntSize::I32 => u32::from_le_bytes(tag_data[..4].try_into().unwrap()) as u128,
                IntSize::I64 => u64::from_le_bytes(tag_data[..8].try_into().unwrap()) as u128,
                IntSize::I128 => u128::from_le_bytes(tag_data[..16].try_into().unwrap()),
            },
            EnumTagTy::Pointer => u64::from_le_bytes(enum_data[..8].try_into().unwrap()) as u128,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Debug)]
pub struct EnumVariant {
    /// The [Type] of the variant.
    ///
    /// Must resolve to a struct type.
    pub ty: Type,

    /// The value of the discriminant for this variant.
    ///
    /// The value stored depends on the [Enum::discriminant_ty] for this enum. For a discriminant
    /// with an [Int] type that has a size of `N` bits, the `N` least significant bits store the
    /// discriminant value; the remaining bits are all `0`.
    pub discriminant: u128,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Debug)]
pub struct WrappingRange {
    pub start: u128,
    pub end: u128,
}

impl WrappingRange {
    pub fn contains(&self, v: u128) -> bool {
        if self.start <= self.end {
            v >= self.start && v <= self.end
        } else {
            v >= self.start || v <= self.end
        }
    }
}

#[derive(Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Debug)]
pub enum EnumTagEncoding {
    /// The tag value directly represents the discriminant of the active variant.
    ///
    /// This is the "standard" enum representation where a dedicated field (the tag) stores an
    /// integer value identifying the variant. While this value often matches the variant index, it
    /// may be explicitly assigned by the user (e.g., `enum E { A = 10 }`).
    ///
    /// To resolve the active variant:
    ///
    /// 1. Read the tag value `v` from the [Enum::tag_offset] as raw bits of [Enum::tag_primitive]
    ///    size, zero-extended to a `u128`.
    /// 2. If [Enum::tag_primitive] is signed, sign-extend `v` from [Enum::tag_primitive]'s size
    ///    to 128 bits.
    /// 3. Truncate `v` to the size of the [Enum::discriminant_ty].
    /// 4. Look up the variant in the [Enum::variants] vector for which the
    ///    [EnumVariant::discriminant] is equal to `v`.
    Direct,

    /// The tag is a field of one of the variants (the "niche" field), and its value is used to
    /// distinguish between the "untagged" variant and one or more "niche" variants.
    ///
    /// This encoding is a common Rust optimization used when a field in one variant (the
    /// `untagged_variant`) has "invalid" bit patterns (niches) that are not used by the field's
    /// type. These invalid patterns are repurposed to represent other variants (the
    /// `niche_variants`).
    ///
    /// Example: `Option<&T>`
    /// - `&T` is a non-null reference, so the bit pattern `0` (null) is "invalid".
    /// - `Some(&T)` is the `untagged_variant` (e.g., index 1). It owns the memory.
    /// - `None` is a niche variant (e.g., index 0).
    /// - If the value at `tag_offset` is non-zero, it's `Some`. If it's zero, it's `None`.
    ///
    /// To resolve the active variant index:
    ///
    /// 1. Read the tag value `v` from the [Enum::tag_offset] as raw bits of [Enum::tag_primitive]
    ///    size, zero-extended to a `u128`.
    /// 2. If `v` is within `valid_range`, the active variant is `untagged_variant`.
    /// 3. Otherwise, calculate the niche index:
    ///    `niche_index = v.wrapping_sub(niche_start) & Enum::tag_primitive.mask()`.
    /// 4. If `niche_index` is less than the count of variants in `niche_variants`:
    ///    The active variant is `niche_variants.start() + (niche_index as usize)`.
    ///    If this results in the `untagged_variant` index, the tag is invalid (UB).
    /// 5. Otherwise, the result is undefined (UB).
    ///
    /// Note that `niche_variants` is a contiguous range of indices. It is possible for
    /// `untagged_variant` index to fall within this range; if it does, the niche value that would
    /// correspond to it is simply never used by the compiler.
    Niche {
        /// The range of bit patterns that are considered valid for the niche field of the
        /// `untagged_variant`.
        ///
        /// Any value `V` that falls within this range indicates that the `untagged_variant` is
        /// active. Values outside this range are niches and represent one of the `niche_variants`.
        valid_range: WrappingRange,

        /// The index of the variant that is active when the tag value represents a valid
        /// value for the niche field.
        untagged_variant: usize,

        /// The range of variant indices that are encoded as niches.
        ///
        /// These indices are represented by bit patterns that are outside the valid range of the
        /// niche field in the `untagged_variant`.
        niche_variants: RangeInclusive<usize>,

        /// The bit pattern value that corresponds to the first variant in `niche_variants`.
        ///
        /// Niche variants are mapped linearly from this starting value. For a tag value `v`,
        /// the corresponding variant index is
        /// `v.wrapping_sub(niche_start) + niche_variants.start()`, provided the result is within
        /// the `niche_variants` range.
        niche_start: u128,
    },
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Debug)]
pub enum EnumTagTy {
    Int(Int),
    Pointer,
}

impl EnumTagTy {
    pub fn expect_int(&self) -> &Int {
        if let EnumTagTy::Int(int) = self {
            int
        } else {
            panic!("not an integer tag primitive");
        }
    }

    fn size(&self) -> usize {
        match self {
            EnumTagTy::Int(int) => int.size.bytes(),
            EnumTagTy::Pointer => 8, // Assuming 64-bit target for now
        }
    }

    fn mask(&self) -> u128 {
        match self {
            EnumTagTy::Int(int) => int.size.mask(),
            EnumTagTy::Pointer => 0xFFFFFFFF_FFFFFFFF,
        }
    }
}

impl From<Int> for EnumTagTy {
    fn from(int: Int) -> Self {
        EnumTagTy::Int(int)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Debug)]
pub struct Int {
    pub size: IntSize,
    pub signed: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Debug)]
pub enum IntSize {
    I8,
    I16,
    I32,
    I64,
    I128,
}

impl IntSize {
    pub fn bytes(&self) -> usize {
        match self {
            IntSize::I8 => 1,
            IntSize::I16 => 2,
            IntSize::I32 => 4,
            IntSize::I64 => 8,
            IntSize::I128 => 16,
        }
    }

    fn mask(&self) -> u128 {
        match self {
            IntSize::I8 => 0xFF,
            IntSize::I16 => 0xFFFF,
            IntSize::I32 => 0xFFFFFFFF,
            IntSize::I64 => 0xFFFFFFFF_FFFFFFFF,
            IntSize::I128 => 0xFFFFFFFF_FFFFFFFF_FFFFFFFF_FFFFFFFF,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct EnumDiscriminant {
    pub value: u128,
    pub ty: Int,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Debug)]
pub struct StructField {
    pub offset: u64,
    pub ty: Type,
    pub io_binding: Option<ShaderIOBinding>,
}

#[derive(Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Debug)]
pub enum TypeKind {
    Scalar(ScalarKind),
    Atomic(ScalarKind),
    Vector(Vector),
    Matrix(Matrix),
    Array {
        /// The type of the elements in the array.
        element_ty: Type,
        /// The number of elements in the array.
        count: u64,
        /// The stride between elements in bytes.
        stride: u64,
    },
    Slice {
        element_ty: Type,
        stride: u64,
    },
    Struct(Struct),
    Enum(Enum),
    Ptr(Type),
    Function(Function),
    Predicate,
    Dummy,
}

impl TypeKind {
    pub fn is_scalar(&self) -> bool {
        matches!(self, TypeKind::Scalar(_))
    }

    pub fn expect_scalar(&self) -> &ScalarKind {
        if let TypeKind::Scalar(scalar) = self {
            scalar
        } else {
            panic!("not a scalar type");
        }
    }

    pub fn is_vector(&self) -> bool {
        matches!(self, TypeKind::Vector(_))
    }

    pub fn expect_vector(&self) -> &Vector {
        if let TypeKind::Vector(v) = self {
            v
        } else {
            panic!("not a vector type");
        }
    }

    pub fn is_matrix(&self) -> bool {
        matches!(self, TypeKind::Matrix(_))
    }

    pub fn expect_matrix(&self) -> &Matrix {
        if let TypeKind::Matrix(m) = self {
            m
        } else {
            panic!("not a matrix type");
        }
    }

    pub fn expect_fn(&self) -> &Function {
        if let TypeKind::Function(function) = self {
            function
        } else {
            panic!("not a function type");
        }
    }

    pub fn is_struct(&self) -> bool {
        matches!(self, TypeKind::Struct(_))
    }

    pub fn expect_struct(&self) -> &Struct {
        if let TypeKind::Struct(struct_data) = self {
            struct_data
        } else {
            panic!("not an struct type");
        }
    }

    pub fn is_enum(&self) -> bool {
        matches!(self, TypeKind::Enum(_))
    }

    pub fn expect_enum(&self) -> &Enum {
        if let TypeKind::Enum(enum_data) = self {
            enum_data
        } else {
            panic!("not an enum type");
        }
    }

    pub fn is_ptr(&self) -> bool {
        matches!(self, TypeKind::Ptr(_))
    }

    pub fn expect_ptr(&self) -> Type {
        if let TypeKind::Ptr(ptr) = self {
            *ptr
        } else {
            panic!("not a pointer type");
        }
    }

    pub fn is_aggregate(&self) -> bool {
        matches!(
            self,
            TypeKind::Struct(_) | TypeKind::Enum(_) | TypeKind::Array { .. }
        )
    }

    pub fn is_slice(&self) -> bool {
        matches!(self, TypeKind::Slice { .. })
    }
}

impl TypeKind {
    fn to_string(&self, ty_registry: &TypeRegistry) -> String {
        match self {
            TypeKind::Scalar(scalar) => scalar.to_string(),
            TypeKind::Atomic(scalar) => format!("atomic<{}>", scalar),
            TypeKind::Vector(v) => v.to_string(),
            TypeKind::Matrix(m) => m.to_string(),
            TypeKind::Array {
                element_ty, count, ..
            } => {
                format!("array<{}, {}>", element_ty.to_string(ty_registry), count)
            }
            TypeKind::Slice { element_ty, .. } => {
                format!("array<{}>", element_ty.to_string(ty_registry))
            }
            TypeKind::Struct(_) => "struct".to_string(),
            TypeKind::Enum(_) => "enum".to_string(),
            TypeKind::Ptr(pointee_ty) => format!("ptr<{}>", pointee_ty.to_string(ty_registry)),
            TypeKind::Function(f) => format!("Function_{}_{}", f.module, f.name),
            TypeKind::Predicate => format!("predicate"),
            TypeKind::Dummy => "dummy".to_string(),
        }
    }
}

impl From<Vector> for TypeKind {
    fn from(value: Vector) -> Self {
        TypeKind::Vector(value)
    }
}

impl From<Matrix> for TypeKind {
    fn from(value: Matrix) -> Self {
        TypeKind::Matrix(value)
    }
}

impl From<Struct> for TypeKind {
    fn from(value: Struct) -> Self {
        TypeKind::Struct(value)
    }
}

impl From<Enum> for TypeKind {
    fn from(value: Enum) -> Self {
        TypeKind::Enum(value)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Debug)]
pub enum ScalarKind {
    I32,
    U32,
    F32,
    Bool,
}

impl ScalarKind {
    pub fn is_numeric(&self) -> bool {
        match self {
            ScalarKind::I32 | ScalarKind::U32 | ScalarKind::F32 => true,
            ScalarKind::Bool => false,
        }
    }

    pub fn is_integer(&self) -> bool {
        match self {
            ScalarKind::I32 | ScalarKind::U32 => true,
            ScalarKind::F32 | ScalarKind::Bool => false,
        }
    }

    pub fn ty(&self) -> Type {
        match self {
            ScalarKind::I32 => TY_I32,
            ScalarKind::U32 => TY_U32,
            ScalarKind::F32 => TY_F32,
            ScalarKind::Bool => TY_BOOL,
        }
    }
}

impl fmt::Display for ScalarKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ScalarKind::I32 => write!(f, "i32"),
            ScalarKind::U32 => write!(f, "u32"),
            ScalarKind::F32 => write!(f, "f32"),
            ScalarKind::Bool => write!(f, "bool"),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Debug)]
pub enum VectorSize {
    Two,
    Three,
    Four,
}

impl VectorSize {
    pub fn to_u32(&self) -> u32 {
        match self {
            VectorSize::Two => 2,
            VectorSize::Three => 3,
            VectorSize::Four => 4,
        }
    }

    pub fn to_usize(&self) -> usize {
        match self {
            VectorSize::Two => 2,
            VectorSize::Three => 3,
            VectorSize::Four => 4,
        }
    }
}

pub const TY_KIND_U32: TypeKind = TypeKind::Scalar(ScalarKind::U32);
pub const TY_KIND_I32: TypeKind = TypeKind::Scalar(ScalarKind::I32);
pub const TY_KIND_F32: TypeKind = TypeKind::Scalar(ScalarKind::F32);
pub const TY_KIND_BOOL: TypeKind = TypeKind::Scalar(ScalarKind::Bool);
pub const TY_KIND_VEC2_U32: TypeKind = TypeKind::Vector(Vector {
    scalar: ScalarKind::U32,
    size: VectorSize::Two,
});
pub const TY_KIND_VEC2_I32: TypeKind = TypeKind::Vector(Vector {
    scalar: ScalarKind::I32,
    size: VectorSize::Two,
});
pub const TY_KIND_VEC2_F32: TypeKind = TypeKind::Vector(Vector {
    scalar: ScalarKind::F32,
    size: VectorSize::Two,
});
pub const TY_KIND_VEC2_BOOL: TypeKind = TypeKind::Vector(Vector {
    scalar: ScalarKind::Bool,
    size: VectorSize::Two,
});
pub const TY_KIND_VEC3_U32: TypeKind = TypeKind::Vector(Vector {
    scalar: ScalarKind::U32,
    size: VectorSize::Three,
});
pub const TY_KIND_VEC3_I32: TypeKind = TypeKind::Vector(Vector {
    scalar: ScalarKind::I32,
    size: VectorSize::Three,
});
pub const TY_KIND_VEC3_F32: TypeKind = TypeKind::Vector(Vector {
    scalar: ScalarKind::F32,
    size: VectorSize::Three,
});
pub const TY_KIND_VEC3_BOOL: TypeKind = TypeKind::Vector(Vector {
    scalar: ScalarKind::Bool,
    size: VectorSize::Three,
});
pub const TY_KIND_VEC4_U32: TypeKind = TypeKind::Vector(Vector {
    scalar: ScalarKind::U32,
    size: VectorSize::Four,
});
pub const TY_KIND_VEC4_I32: TypeKind = TypeKind::Vector(Vector {
    scalar: ScalarKind::I32,
    size: VectorSize::Four,
});
pub const TY_KIND_VEC4_F32: TypeKind = TypeKind::Vector(Vector {
    scalar: ScalarKind::F32,
    size: VectorSize::Four,
});
pub const TY_KIND_VEC4_BOOL: TypeKind = TypeKind::Vector(Vector {
    scalar: ScalarKind::Bool,
    size: VectorSize::Four,
});
pub const TY_KIND_MAT2X2: TypeKind = TypeKind::Matrix(Matrix {
    rows: VectorSize::Two,
    columns: VectorSize::Two,
    scalar: ScalarKind::F32,
});
pub const TY_KIND_MAT2X3: TypeKind = TypeKind::Matrix(Matrix {
    rows: VectorSize::Three,
    columns: VectorSize::Two,
    scalar: ScalarKind::F32,
});
pub const TY_KIND_MAT2X4: TypeKind = TypeKind::Matrix(Matrix {
    rows: VectorSize::Four,
    columns: VectorSize::Two,
    scalar: ScalarKind::F32,
});
pub const TY_KIND_MAT3X2: TypeKind = TypeKind::Matrix(Matrix {
    rows: VectorSize::Two,
    columns: VectorSize::Three,
    scalar: ScalarKind::F32,
});
pub const TY_KIND_MAT3X3: TypeKind = TypeKind::Matrix(Matrix {
    rows: VectorSize::Three,
    columns: VectorSize::Three,
    scalar: ScalarKind::F32,
});
pub const TY_KIND_MAT3X4: TypeKind = TypeKind::Matrix(Matrix {
    rows: VectorSize::Four,
    columns: VectorSize::Three,
    scalar: ScalarKind::F32,
});
pub const TY_KIND_MAT4X2: TypeKind = TypeKind::Matrix(Matrix {
    rows: VectorSize::Two,
    columns: VectorSize::Four,
    scalar: ScalarKind::F32,
});
pub const TY_KIND_MAT4X3: TypeKind = TypeKind::Matrix(Matrix {
    rows: VectorSize::Three,
    columns: VectorSize::Four,
    scalar: ScalarKind::F32,
});
pub const TY_KIND_MAT4X4: TypeKind = TypeKind::Matrix(Matrix {
    rows: VectorSize::Four,
    columns: VectorSize::Four,
    scalar: ScalarKind::F32,
});
pub const TY_KIND_ATOMIC_U32: TypeKind = TypeKind::Atomic(ScalarKind::U32);
pub const TY_KIND_ATOMIC_I32: TypeKind = TypeKind::Atomic(ScalarKind::I32);
pub const TY_KIND_ATOMIC_F32: TypeKind = TypeKind::Atomic(ScalarKind::F32);
pub const TY_KIND_ATOMIC_BOOL: TypeKind = TypeKind::Atomic(ScalarKind::Bool);
pub const TY_KIND_PREDICATE: TypeKind = TypeKind::Predicate;
pub const TY_KIND_PTR_U32: TypeKind = TypeKind::Ptr(TY_U32);
pub const TY_KIND_DUMMY: TypeKind = TypeKind::Dummy;

pub const TY_U32: Type = Type(TypeInner::U32);
pub const TY_I32: Type = Type(TypeInner::I32);
pub const TY_F32: Type = Type(TypeInner::F32);
pub const TY_BOOL: Type = Type(TypeInner::Bool);
pub const TY_VEC2_U32: Type = Type(TypeInner::Vec2U32);
pub const TY_VEC2_I32: Type = Type(TypeInner::Vec2I32);
pub const TY_VEC2_F32: Type = Type(TypeInner::Vec2F32);
pub const TY_VEC2_BOOL: Type = Type(TypeInner::Vec2Bool);
pub const TY_VEC3_U32: Type = Type(TypeInner::Vec3U32);
pub const TY_VEC3_I32: Type = Type(TypeInner::Vec3I32);
pub const TY_VEC3_F32: Type = Type(TypeInner::Vec3F32);
pub const TY_VEC3_BOOL: Type = Type(TypeInner::Vec3Bool);
pub const TY_VEC4_U32: Type = Type(TypeInner::Vec4U32);
pub const TY_VEC4_I32: Type = Type(TypeInner::Vec4I32);
pub const TY_VEC4_F32: Type = Type(TypeInner::Vec4F32);
pub const TY_VEC4_BOOL: Type = Type(TypeInner::Vec4Bool);
pub const TY_MAT2X2: Type = Type(TypeInner::Mat2x2);
pub const TY_MAT2X3: Type = Type(TypeInner::Mat2x3);
pub const TY_MAT2X4: Type = Type(TypeInner::Mat2x4);
pub const TY_MAT3X2: Type = Type(TypeInner::Mat3x2);
pub const TY_MAT3X3: Type = Type(TypeInner::Mat3x3);
pub const TY_MAT3X4: Type = Type(TypeInner::Mat3x4);
pub const TY_MAT4X2: Type = Type(TypeInner::Mat4x2);
pub const TY_MAT4X3: Type = Type(TypeInner::Mat4x3);
pub const TY_MAT4X4: Type = Type(TypeInner::Mat4x4);
pub const TY_ATOMIC_U32: Type = Type(TypeInner::AtomicU32);
pub const TY_ATOMIC_I32: Type = Type(TypeInner::AtomicI32);
pub const TY_ATOMIC_F32: Type = Type(TypeInner::AtomicF32);
pub const TY_ATOMIC_BOOL: Type = Type(TypeInner::AtomicBool);
pub const TY_PREDICATE: Type = Type(TypeInner::Predicate);
pub const TY_PTR_U32: Type = Type(TypeInner::PtrU32);
pub const TY_DUMMY: Type = Type(TypeInner::Dummy);

#[derive(Clone, Default, Serialize, Deserialize, Debug)]
pub struct TypeRegistry {
    #[serde(with = "crate::serde::arc_rwlock")]
    store: Arc<RwLock<IndexSet<Box<TypeKind>>>>,
}

impl TypeRegistry {
    pub fn register(&self, ty_kind: TypeKind) -> Type {
        match &ty_kind {
            TypeKind::Scalar(ScalarKind::I32) => return TY_I32,
            TypeKind::Scalar(ScalarKind::U32) => return TY_U32,
            TypeKind::Scalar(ScalarKind::F32) => return TY_F32,
            TypeKind::Scalar(ScalarKind::Bool) => return TY_BOOL,
            TypeKind::Atomic(ScalarKind::U32) => return TY_ATOMIC_U32,
            TypeKind::Atomic(ScalarKind::I32) => return TY_ATOMIC_I32,
            TypeKind::Atomic(ScalarKind::F32) => return TY_ATOMIC_F32,
            TypeKind::Atomic(ScalarKind::Bool) => return TY_ATOMIC_BOOL,
            TypeKind::Vector(Vector {
                scalar: ScalarKind::U32,
                size: VectorSize::Two,
            }) => return TY_VEC2_U32,
            TypeKind::Vector(Vector {
                scalar: ScalarKind::I32,
                size: VectorSize::Two,
            }) => return TY_VEC2_I32,
            TypeKind::Vector(Vector {
                scalar: ScalarKind::F32,
                size: VectorSize::Two,
            }) => return TY_VEC2_F32,
            TypeKind::Vector(Vector {
                scalar: ScalarKind::Bool,
                size: VectorSize::Two,
            }) => return TY_VEC2_BOOL,
            TypeKind::Vector(Vector {
                scalar: ScalarKind::U32,
                size: VectorSize::Three,
            }) => return TY_VEC3_U32,
            TypeKind::Vector(Vector {
                scalar: ScalarKind::I32,
                size: VectorSize::Three,
            }) => return TY_VEC3_I32,
            TypeKind::Vector(Vector {
                scalar: ScalarKind::F32,
                size: VectorSize::Three,
            }) => return TY_VEC3_F32,
            TypeKind::Vector(Vector {
                scalar: ScalarKind::Bool,
                size: VectorSize::Three,
            }) => return TY_VEC3_BOOL,
            TypeKind::Vector(Vector {
                scalar: ScalarKind::U32,
                size: VectorSize::Four,
            }) => return TY_VEC4_U32,
            TypeKind::Vector(Vector {
                scalar: ScalarKind::I32,
                size: VectorSize::Four,
            }) => return TY_VEC4_I32,
            TypeKind::Vector(Vector {
                scalar: ScalarKind::F32,
                size: VectorSize::Four,
            }) => return TY_VEC4_F32,
            TypeKind::Vector(Vector {
                scalar: ScalarKind::Bool,
                size: VectorSize::Four,
            }) => return TY_VEC4_BOOL,
            TypeKind::Matrix(Matrix {
                rows: VectorSize::Two,
                columns: VectorSize::Two,
                scalar: ScalarKind::F32,
            }) => return TY_MAT2X2,
            TypeKind::Matrix(Matrix {
                rows: VectorSize::Three,
                columns: VectorSize::Two,
                scalar: ScalarKind::F32,
            }) => return TY_MAT2X3,
            TypeKind::Matrix(Matrix {
                rows: VectorSize::Four,
                columns: VectorSize::Two,
                scalar: ScalarKind::F32,
            }) => return TY_MAT2X4,
            TypeKind::Matrix(Matrix {
                rows: VectorSize::Two,
                columns: VectorSize::Three,
                scalar: ScalarKind::F32,
            }) => return TY_MAT3X2,
            TypeKind::Matrix(Matrix {
                rows: VectorSize::Three,
                columns: VectorSize::Three,
                scalar: ScalarKind::F32,
            }) => return TY_MAT3X3,
            TypeKind::Matrix(Matrix {
                rows: VectorSize::Four,
                columns: VectorSize::Three,
                scalar: ScalarKind::F32,
            }) => return TY_MAT3X4,
            TypeKind::Matrix(Matrix {
                rows: VectorSize::Two,
                columns: VectorSize::Four,
                scalar: ScalarKind::F32,
            }) => return TY_MAT4X2,
            TypeKind::Matrix(Matrix {
                rows: VectorSize::Three,
                columns: VectorSize::Four,
                scalar: ScalarKind::F32,
            }) => return TY_MAT4X3,
            TypeKind::Matrix(Matrix {
                rows: VectorSize::Four,
                columns: VectorSize::Four,
                scalar: ScalarKind::F32,
            }) => return TY_MAT4X4,
            TypeKind::Predicate => return TY_PREDICATE,
            TypeKind::Ptr(TY_U32) => return TY_PTR_U32,
            #[cfg(test)]
            TypeKind::Dummy => return TY_DUMMY,
            _ => (),
        }

        // We check if the type kind was already registered via a read lock first. Only if it wasn't
        // already registered, do we request a write lock.
        let read_lock = self.store.read().unwrap_or_else(PoisonError::into_inner);

        let index = if let Some(index) = read_lock.get_index_of(&ty_kind) {
            index
        } else {
            // Make sure we drop the read lock before the try to get the write lock, otherwise this
            // will immediately deadlock.
            mem::drop(read_lock);

            self.store
                .write()
                .unwrap_or_else(PoisonError::into_inner)
                .insert_full(Box::new(ty_kind))
                .0
        };

        Type(TypeInner::Registered(index))
    }

    pub fn kind(&self, ty: Type) -> KindRef<'_> {
        match ty.0 {
            TypeInner::U32 => KindRef::from_static(&TY_KIND_U32),
            TypeInner::I32 => KindRef::from_static(&TY_KIND_I32),
            TypeInner::F32 => KindRef::from_static(&TY_KIND_F32),
            TypeInner::Bool => KindRef::from_static(&TY_KIND_BOOL),
            TypeInner::Vec2U32 => KindRef::from_static(&TY_KIND_VEC2_U32),
            TypeInner::Vec2I32 => KindRef::from_static(&TY_KIND_VEC2_I32),
            TypeInner::Vec2F32 => KindRef::from_static(&TY_KIND_VEC2_F32),
            TypeInner::Vec2Bool => KindRef::from_static(&TY_KIND_VEC2_BOOL),
            TypeInner::Vec3U32 => KindRef::from_static(&TY_KIND_VEC3_U32),
            TypeInner::Vec3I32 => KindRef::from_static(&TY_KIND_VEC3_I32),
            TypeInner::Vec3F32 => KindRef::from_static(&TY_KIND_VEC3_F32),
            TypeInner::Vec3Bool => KindRef::from_static(&TY_KIND_VEC3_BOOL),
            TypeInner::Vec4U32 => KindRef::from_static(&TY_KIND_VEC4_U32),
            TypeInner::Vec4I32 => KindRef::from_static(&TY_KIND_VEC4_I32),
            TypeInner::Vec4F32 => KindRef::from_static(&TY_KIND_VEC4_F32),
            TypeInner::Vec4Bool => KindRef::from_static(&TY_KIND_VEC4_BOOL),
            TypeInner::Mat2x2 => KindRef::from_static(&TY_KIND_MAT2X2),
            TypeInner::Mat2x3 => KindRef::from_static(&TY_KIND_MAT2X3),
            TypeInner::Mat2x4 => KindRef::from_static(&TY_KIND_MAT2X4),
            TypeInner::Mat3x2 => KindRef::from_static(&TY_KIND_MAT3X2),
            TypeInner::Mat3x3 => KindRef::from_static(&TY_KIND_MAT3X3),
            TypeInner::Mat3x4 => KindRef::from_static(&TY_KIND_MAT3X4),
            TypeInner::Mat4x2 => KindRef::from_static(&TY_KIND_MAT4X2),
            TypeInner::Mat4x3 => KindRef::from_static(&TY_KIND_MAT4X3),
            TypeInner::Mat4x4 => KindRef::from_static(&TY_KIND_MAT4X4),
            TypeInner::AtomicU32 => KindRef::from_static(&TY_KIND_ATOMIC_U32),
            TypeInner::AtomicI32 => KindRef::from_static(&TY_KIND_ATOMIC_I32),
            TypeInner::AtomicF32 => KindRef::from_static(&TY_KIND_ATOMIC_F32),
            TypeInner::AtomicBool => KindRef::from_static(&TY_KIND_ATOMIC_BOOL),
            TypeInner::Predicate => KindRef::from_static(&TY_KIND_PREDICATE),
            TypeInner::PtrU32 => KindRef::from_static(&TY_KIND_PTR_U32),
            TypeInner::Dummy => KindRef::from_static(&TY_KIND_DUMMY),
            TypeInner::Registered(index) => {
                let store = self.store.read().unwrap_or_else(PoisonError::into_inner);
                let boxed = store.get_index(index).expect("unregistered type");
                let ptr = boxed.as_ref() as *const TypeKind;

                KindRef {
                    ptr,
                    _marker: Default::default(),
                }
            }
        }
    }

    pub fn import(&self, other_reg: &TypeRegistry, ty: Type) -> Type {
        let mut ty_kind = other_reg.kind(ty).deref().clone();

        match &mut ty_kind {
            TypeKind::Array {
                element_ty: base, ..
            } => {
                *base = self.import(other_reg, *base);
            }
            TypeKind::Struct(struct_data) => {
                for field in &mut struct_data.fields {
                    field.ty = self.import(other_reg, field.ty);
                }
            }
            TypeKind::Enum(enum_data) => {
                for variant in &mut enum_data.variants {
                    variant.ty = self.import(other_reg, variant.ty);
                }
            }
            _ => (),
        }

        self.register(ty_kind)
    }

    /// Whether the `src_ty` can be "coerced" into the `dst_ty`.
    ///
    /// A type can always be coerced into itself, so when the `src_ty` and `dst_ty` are the same
    /// type, this always returns `true`.
    ///
    /// In addition, we currently allow a pointer to a sized array to coerce into a pointer to an
    /// unsized array (a slice) with the same element type and the same stride. For example,
    /// `ptr<array<u32, 4>>` can coerce into `ptr<array<u32>>` (note: not the other way around,
    /// `ptr<array<u32>>` cannot be coerced into `ptr<array<u32, 4>>`).
    ///
    /// Otherwise, returns `false`.
    pub fn can_coerce(&self, src_ty: Type, dst_ty: Type) -> bool {
        if src_ty == dst_ty {
            return true;
        }

        if let TypeKind::Ptr(src_pointee_ty) = *self.kind(src_ty)
            && let TypeKind::Array {
                element_ty: src_element_ty,
                stride: src_stride,
                ..
            } = *self.kind(src_pointee_ty)
        {
            if let TypeKind::Ptr(dst_pointee_ty) = *self.kind(dst_ty)
                && let TypeKind::Slice {
                    element_ty: dst_element_ty,
                    stride: dst_stride,
                } = *self.kind(dst_pointee_ty)
            {
                return src_element_ty == dst_element_ty && src_stride == dst_stride;
            }
        }

        false
    }
}

pub struct KindRef<'a> {
    ptr: *const TypeKind,
    _marker: PhantomData<&'a TypeKind>,
}

impl KindRef<'static> {
    fn from_static(kind: &'static TypeKind) -> Self {
        KindRef {
            ptr: kind as *const TypeKind,
            _marker: Default::default(),
        }
    }
}

impl Clone for KindRef<'_> {
    fn clone(&self) -> Self {
        KindRef {
            ptr: self.ptr,
            _marker: Default::default(),
        }
    }
}

impl Copy for KindRef<'_> {}

impl AsRef<TypeKind> for KindRef<'_> {
    fn as_ref(&self) -> &TypeKind {
        // SAFETY
        //
        // There are 2 ways a `KindRef` can be created:
        //
        // 1. From a static reference to a `TypeKind`. In this case the `TypeKind` will never drop
        //    and the pointer will always remain valid.
        // 2. By registering with a TypeRegistry. In this case the TyRef's lifetime ensures that the
        //    `TypeRegistry` store with which it is associated cannot have dropped. As the interface
        //    does not expose a mechanism for removing registered types from the store, that
        //    implies that the `Box` the `TyRef`'s pointer points to will also not have dropped.
        //    Therefore, the pointer must still be valid.
        unsafe { &*self.ptr }
    }
}

impl Deref for KindRef<'_> {
    type Target = TypeKind;

    fn deref(&self) -> &Self::Target {
        self.as_ref()
    }
}
