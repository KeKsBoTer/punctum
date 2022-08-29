use std::ops::AddAssign;

use bytemuck::{Pod, Zeroable};
use nalgebra::{Point3, RealField, Scalar, Vector3};
use num_traits::ToPrimitive;
use serde::{Deserialize, Serialize};
use vulkano::pipeline::graphics::vertex_input::{Vertex as VulkanoVertex, VertexMemberInfo};
use vulkano::pipeline::graphics::vertex_input::{VertexMember, VertexMemberTy};

use crate::{ply::PlyType, sh::SH_0};
use serde_big_array::BigArray;

pub trait BaseFloat:
    Scalar + ToPrimitive + Default + RealField + PlyType + Copy + Zeroable + AddAssign
{
}
impl<T: Scalar + ToPrimitive + Default + RealField + PlyType + Copy + Zeroable + AddAssign>
    BaseFloat for T
{
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, Zeroable)]
#[repr(C)]
pub struct Vertex<F: BaseFloat> {
    pub position: Point3<F>,
    pub color: Vector3<u8>,
}

unsafe impl VulkanoVertex for Vertex<f32> {
    fn member(name: &str) -> Option<VertexMemberInfo> {
        match name {
            "position" => Some(VertexMemberInfo {
                offset: 0,
                ty: VertexMemberTy::F32,
                array_size: 3,
            }),
            "color" => Some(VertexMemberInfo {
                offset: 4 * 3,
                ty: VertexMemberTy::U32,
                array_size: 1,
            }),
            _ => None,
        }
    }
}

// unsafe impl Zeroable for Vertex<f32> {}
unsafe impl Pod for Vertex<f32> {}

impl From<Vertex<f64>> for Vertex<f32> {
    fn from(item: Vertex<f64>) -> Self {
        Vertex {
            position: item.position.cast(),
            color: item.color,
        }
    }
}

pub const NUM_COEFS: usize = (4 + 1) * (4 + 1);

#[derive(Clone, Copy, Debug, Serialize, Deserialize, Zeroable)]
#[repr(C)]
pub struct SHCoefficients<const T: usize = NUM_COEFS>(
    #[serde(with = "BigArray")] pub [Vector3<f32>; T],
);

unsafe impl<const T: usize> VertexMember for SHCoefficients<T> {
    #[inline]
    fn format() -> (VertexMemberTy, usize) {
        let (ty, sz) = Vector3::<f32>::format();
        (ty, sz * T)
    }
}

impl<const T: usize> SHCoefficients<T> {
    pub fn new_from_color(color: Vector3<f32>) -> Self {
        let mut coefs = [Vector3::zeros(); T];
        coefs[0].x = color.x / SH_0;
        coefs[0].y = color.y / SH_0;
        coefs[0].z = color.z / SH_0;
        SHCoefficients(coefs)
    }

    pub fn l_max() -> u64 {
        (T as f32).sqrt() as u64 - 1
    }
}

impl<const T: usize> Into<SHCoefficients<T>> for [Vector3<f32>; T] {
    fn into(self) -> SHCoefficients<T> {
        SHCoefficients(self)
    }
}

unsafe impl Pod for SHCoefficients {}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, Zeroable)]
#[repr(C)]
pub struct SHVertex<F: BaseFloat> {
    pub position: Point3<F>,
    pub radius: F,
    pub coefficients: SHCoefficients,
}

impl Into<SHVertex<f32>> for SHVertex<f64> {
    fn into(self) -> SHVertex<f32> {
        SHVertex {
            position: self.position.cast(),
            radius: self.radius as f32,
            coefficients: self.coefficients.clone(),
        }
    }
}

impl<F: BaseFloat> SHVertex<F> {
    pub fn new(position: Point3<F>, radius: F, coefficients: SHCoefficients) -> Self {
        Self {
            position,
            radius,
            coefficients,
        }
    }

    pub fn new_with_color(position: Point3<F>, radius: F, color: Vector3<f32>) -> Self {
        Self {
            position,
            radius,
            coefficients: SHCoefficients::new_from_color(color),
        }
    }
}

impl<F: BaseFloat> Default for SHVertex<F> {
    fn default() -> Self {
        Self {
            position: Point3::origin(),
            radius: F::from_subset(&1.),
            coefficients: SHCoefficients::zeroed(),
        }
    }
}

unsafe impl Pod for SHVertex<f32> {}
vulkano::impl_vertex!(SHVertex<f32>, position, radius, coefficients);
