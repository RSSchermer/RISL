use std::marker::PhantomData;
use std::ops::Deref;
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
    pub variants: Vec<Type>,
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
                    *variant = self.import(other_reg, *variant);
                }
            }
            _ => (),
        }

        self.register(ty_kind)
    }

    pub fn is_compatible(&self, t0: Type, t1: Type) -> bool {
        if t0 == t1 {
            // A type is always compatible with itself
            return true;
        }

        return true;
        //
        // use ScalarKind::*;
        // use TypeKind::*;
        //
        // let k0 = self.kind(t0);
        // let k1 = self.kind(t1);
        //
        // match (&*k0, &*k1) {
        //     // TODO: the predicate compatibilities are a bit of a stop-gap. I think the predicate
        //     // type itself needs to be rethought.
        //     (Predicate, Scalar(U32))
        //     | (Scalar(U32), Predicate)
        //     | (Predicate, Scalar(Bool))
        //     | (Scalar(Bool), Predicate) => return true,
        //     _ => {}
        // }
        //
        // false
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
