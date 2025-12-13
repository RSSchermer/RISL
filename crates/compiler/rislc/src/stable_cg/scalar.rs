use std::num::NonZero;

use rustc_middle::bug;
use rustc_public::abi::{FloatLength, IntegerLength, Primitive, ValueAbi, WrappingRange};
use rustc_public::mir::alloc::AllocId;
use rustc_public::target::{MachineInfo, MachineSize};
use rustc_public::ty::{Allocation, Prov, RigidTy, Size, TyKind};
use rustc_public::{Error, abi};

use crate::stable_cg::TyAndLayout;
use crate::stable_cg::layout::ScalarExt;

#[derive(Clone, PartialEq, Debug)]
pub enum Scalar {
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    Pointer(Pointer),
}

#[derive(Clone, PartialEq, Debug)]
pub struct Pointer {
    pub alloc_id: AllocId,
    pub offset: u64,
    pub pointee_layout: TyAndLayout,
}

impl Scalar {
    pub fn read_from_alloc(
        allocation: &Allocation,
        offset: usize,
        abi: abi::Scalar,
        layout: &TyAndLayout,
    ) -> Scalar {
        let size = abi.size(&MachineInfo::target());
        let end = offset + size.bytes();

        let data = allocation.read_partial_uint(offset..end).expect(
            "not enough data in allocation to satisfy the size requirements of the scalar ABI",
        );

        if matches!(
            abi,
            abi::Scalar::Initialized {
                value: Primitive::Int {
                    length: IntegerLength::I8,
                    signed: false
                },
                valid_range: WrappingRange { start: 0, end: 1 }
            }
        ) {
            return Scalar::Bool(data == 1);
        }

        match abi.primitive() {
            Primitive::Int { signed: true, .. } => {
                Scalar::I32(i32::from_ne_bytes((data as u32).to_ne_bytes()))
            }
            Primitive::Int { signed: false, .. } => Scalar::U32(data as u32),
            Primitive::Float {
                length: FloatLength::F32,
            } => Scalar::F32(f32::from_ne_bytes((data as u32).to_ne_bytes())),
            Primitive::Pointer(_) => {
                use RigidTy::*;

                let pointee_ty = match layout.ty.kind() {
                    TyKind::RigidTy(Ref(_, ty, _) | RawPtr(ty, _)) => ty,
                    _ => bug!("pointer primitive should have a pointer-like type"),
                };

                let prov_index = allocation
                    .provenance
                    .ptrs
                    .binary_search_by(|(offset, _)| offset.cmp(offset))
                    .expect("provenance not found for pointer");
                let (_, Prov(alloc_id)) = allocation.provenance.ptrs[prov_index];

                Scalar::Pointer(Pointer {
                    alloc_id,
                    offset: data as u64,
                    pointee_layout: TyAndLayout::expect_from_ty(pointee_ty),
                })
            }
            _ => bug!("primitive type not supported by RISL"),
        }
    }
}
