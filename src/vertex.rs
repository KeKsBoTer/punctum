use std::ops::AddAssign;

use bytemuck::{Pod, Zeroable};
use nalgebra::{Point3, RealField, Scalar, Vector3, Vector4};
use num_traits::{ToPrimitive, Zero};
use serde::{Deserialize, Serialize};

use crate::{
    ply::{Color, PlyType},
    sh::sh_at_point,
};
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
    pub color: Vector3<C>,
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
            color: Vector3::new(
                (item.color.x * 255.) as u8,
                (item.color.y * 255.) as u8,
                (item.color.z * 255.) as u8,
            ),
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, Zeroable)]
#[repr(C)]
pub struct SHCoefficients<const T: usize = 25>(#[serde(with = "BigArray")] [Vector4<f32>; T]);

impl<const T: usize> SHCoefficients<T> {
    pub fn new_from_color(color: Vector3<f32>) -> Self {
        let mut coefs = [Vector4::zeros(); T];
        let sh_factor = sh_at_point(0, 0, 0., 0.);
        coefs[0].x = color.x / sh_factor;
        coefs[0].y = color.y / sh_factor;
        coefs[0].z = color.z / sh_factor;
        coefs[0].w = 1. / sh_factor;
        SHCoefficients(coefs)
    }
}

impl<const T: usize> Into<SHCoefficients<T>> for [Vector4<f32>; T] {
    fn into(self) -> SHCoefficients<T> {
        SHCoefficients(self)
    }
}

unsafe impl Pod for SHCoefficients {}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, Zeroable)]
#[repr(C)]
pub struct SHVertex<F: BaseFloat> {
    pub position: Point3<F>,
    pub size: F,
    pub coefficients: SHCoefficients,
}

impl Into<SHVertex<f32>> for SHVertex<f64> {
    fn into(self) -> SHVertex<f32> {
        SHVertex {
            position: self.position.cast(),
            size: self.size as f32,
            coefficients: self.coefficients.clone(),
        }
    }
}

impl<F: BaseFloat> SHVertex<F> {
    pub fn new(position: Point3<F>, size: F, coefficients: SHCoefficients) -> Self {
        Self {
            position,
            size,
            coefficients,
        }
    }
}

impl<F: BaseFloat> Default for SHVertex<F> {
    fn default() -> Self {
        Self {
            position: Point3::origin(),
            size: F::from_subset(&1.),
            coefficients: SHCoefficients::zeroed(),
        }
    }
}

unsafe impl Pod for SHVertex<f32> {}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, Zeroable, Pod, Default)]
#[repr(C)]
pub struct IndexVertex {
    pub index: u32,
}

vulkano::impl_vertex!(IndexVertex, index);
