use std::ops::AddAssign;

use bytemuck::{Pod, Zeroable};
use nalgebra::{Point3, RealField, Scalar, Vector3, Vector4};
use num_traits::{ToPrimitive, Zero};
use serde::{Deserialize, Serialize};
use vulkano::pipeline::graphics::vertex_input::{VertexMember, VertexMemberTy};

use crate::{
    ply::{Color, PlyType},
    sh::SH_0,
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

pub trait NormColor {
    fn to_norm(&self) -> f32;
}

impl NormColor for u8 {
    fn to_norm(&self) -> f32 {
        (*self as f32) / 255.
    }
}
impl NormColor for f32 {
    fn to_norm(&self) -> f32 {
        *self
    }
}

pub trait BaseColor:
    Scalar + Default + Color + Zero + PlyType + Copy + Zeroable + AddAssign + NormColor
{
}
impl<T: Scalar + Zero + Color + Default + PlyType + Copy + Zeroable + AddAssign + NormColor>
    BaseColor for T
{
}

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

unsafe impl<const T: usize> VertexMember for SHCoefficients<T> {
    #[inline]
    fn format() -> (VertexMemberTy, usize) {
        let (ty, sz) = Vector4::<f32>::format();
        (ty, sz * T)
    }
}

impl<const T: usize> SHCoefficients<T> {
    pub fn new_from_color(color: Vector3<f32>) -> Self {
        let mut coefs = [Vector4::zeros(); T];
        coefs[0].x = color.x / SH_0;
        coefs[0].y = color.y / SH_0;
        coefs[0].z = color.z / SH_0;
        coefs[0].w = 1. / SH_0;
        SHCoefficients(coefs)
    }

    pub fn update_color(&mut self, color: Vector3<f32>, num_before: usize) {
        let n_new = 1. / ((num_before + 1) as f32);
        let n_old = num_before as f32 * n_new;
        self.0[0].x = self.0[0].x * n_old + color.x / SH_0 * n_new;
        self.0[0].x = self.0[0].y * n_old + color.y / SH_0 * n_new;
        self.0[0].x = self.0[0].z * n_old + color.z / SH_0 * n_new;
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

    pub fn new_with_color(position: Point3<F>, size: F, color: Vector3<f32>) -> Self {
        Self {
            position,
            size,
            coefficients: SHCoefficients::new_from_color(color),
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
vulkano::impl_vertex!(SHVertex<f32>, position, size, coefficients);
