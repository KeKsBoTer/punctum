use std::ops::AddAssign;

use bytemuck::{Pod, Zeroable};
use nalgebra::{Point3, RealField, Scalar, Vector4};
use num_traits::{ToPrimitive, Zero};
use serde::{Deserialize, Serialize};
use vulkano::pipeline::graphics::vertex_input::{VertexMember, VertexMemberTy};

use crate::ply::{Color, PlyType};
use serde_big_array::BigArray;

pub trait BaseFloat:
    Scalar + ToPrimitive + Default + RealField + PlyType + Copy + Zeroable + AddAssign
{
}
impl<T: Scalar + ToPrimitive + Default + RealField + PlyType + Copy + Zeroable + AddAssign>
    BaseFloat for T
{
}

pub trait BaseColor:
    Scalar + Default + Color + Zero + PlyType + Copy + Zeroable + AddAssign
{
}
impl<T: Scalar + Zero + Color + Default + PlyType + Copy + Zeroable + AddAssign> BaseColor for T {}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, Zeroable)]
#[repr(C)]
pub struct Vertex<F: BaseFloat, C: BaseColor> {
    pub position: Point3<F>,
    // pub normal: Vector3<f32>,
    pub color: Vector4<C>,
}

// unsafe impl Zeroable for Vertex<f32, f32> {}
unsafe impl Pod for Vertex<f32, f32> {}

vulkano::impl_vertex!(Vertex<f32,f32>, position, color);

impl From<Vertex<f64, u8>> for Vertex<f32, f32> {
    fn from(item: Vertex<f64, u8>) -> Self {
        Vertex {
            position: item.position.cast(),
            color: item.color.cast() / 255.,
        }
    }
}

impl From<Vertex<f64, u8>> for Vertex<f32, u8> {
    fn from(item: Vertex<f64, u8>) -> Self {
        Vertex {
            position: item.position.cast(),
            color: item.color,
        }
    }
}

impl From<Vertex<f32, f32>> for Vertex<f32, u8> {
    fn from(item: Vertex<f32, f32>) -> Self {
        Vertex {
            position: item.position.clone(),
            color: Vector4::new(
                (item.color.x * 255.) as u8,
                (item.color.y * 255.) as u8,
                (item.color.z * 255.) as u8,
                (item.color.w * 255.) as u8,
            ),
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, Zeroable)]
#[repr(C)]
pub struct SHCoefficients<const T: usize>(#[serde(with = "BigArray")] [Vector4<f32>; T]);

unsafe impl Pod for SHCoefficients<121> {}

impl<const T: usize> Into<SHCoefficients<T>> for [Vector4<f32>; T] {
    fn into(self) -> SHCoefficients<T> {
        SHCoefficients(self)
    }
}

unsafe impl<const T: usize> VertexMember for SHCoefficients<T> {
    #[inline]
    fn format() -> (VertexMemberTy, usize) {
        let (ty, sz) = Vector4::<f32>::format();
        (ty, sz * T)
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, Zeroable)]
#[repr(C)]
pub struct SHVertex<F: BaseFloat, const T: usize> {
    pub position: Point3<F>,
    // we need to add padding manually here since we only use the coefficients in a uniform buffer
    _pad: F,
    pub coefficients: SHCoefficients<T>,
}

impl<const T: usize> Into<SHVertex<f32, T>> for SHVertex<f64, T> {
    fn into(self) -> SHVertex<f32, T> {
        SHVertex {
            position: self.position.cast(),
            _pad: 0.,
            coefficients: self.coefficients.clone(),
        }
    }
}

impl<F: BaseFloat, const T: usize> SHVertex<F, T> {
    pub fn new(position: Point3<F>, coefficients: SHCoefficients<T>) -> Self {
        Self {
            position,
            _pad: F::from_subset(&0.),
            coefficients,
        }
    }
}

impl<F: BaseFloat, const T: usize> Default for SHVertex<F, T> {
    fn default() -> Self {
        Self {
            position: Point3::origin(),
            _pad: F::from_subset(&0.),
            coefficients: SHCoefficients::zeroed(),
        }
    }
}

// unsafe impl Zeroable for Vertex<f32, f32> {}
unsafe impl Pod for SHVertex<f32, 121> {}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, Zeroable, Pod, Default)]
#[repr(C)]
pub struct IndexVertex {
    pub index: u32,
}

vulkano::impl_vertex!(IndexVertex, index);
