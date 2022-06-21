use std::ops::AddAssign;

use bytemuck::{Pod, Zeroable};
use nalgebra::{Point3, RealField, Scalar, Vector4};
use num_traits::{ToPrimitive, Zero};
use ply_rs::ply::{self, Addable, ElementDef, PropertyDef, PropertyType, ScalarType};
use serde::{Deserialize, Serialize};

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

// unsafe impl Zeroable for Vertex<f32, u8> {}
unsafe impl Pod for Vertex<f32, u8> {}
// unsafe impl Zeroable for Vertex<f64, u8> {}
unsafe impl Pod for Vertex<f64, u8> {}

vulkano::impl_vertex!(Vertex<f32,f32>, position, color);

impl<F: BaseFloat, C: BaseColor> ply::PropertyAccess for Vertex<F, C> {
    fn new() -> Self {
        Vertex {
            position: Point3::origin(),
            // normal: Vector3::zeros(),
            color: Vector4::zeros(),
        }
    }

    fn set_property(&mut self, key: String, property: ply::Property) {
        match (key.as_ref(), property) {
            ("x", ply::Property::Float(v)) => self.position[0] = F::from_f32(v).unwrap(),
            ("y", ply::Property::Float(v)) => self.position[1] = F::from_f32(v).unwrap(),
            ("z", ply::Property::Float(v)) => self.position[2] = F::from_f32(v).unwrap(),
            ("x", ply::Property::Double(v)) => self.position[0] = F::from_f64(v).unwrap(),
            ("y", ply::Property::Double(v)) => self.position[1] = F::from_f64(v).unwrap(),
            ("z", ply::Property::Double(v)) => self.position[2] = F::from_f64(v).unwrap(),
            ("nx", ply::Property::Float(_)) => {}
            ("ny", ply::Property::Float(_)) => {}
            ("nz", ply::Property::Float(_)) => {}
            ("red", ply::Property::UChar(v)) => self.color[0] = C::from_u8(v),
            ("green", ply::Property::UChar(v)) => self.color[1] = C::from_u8(v),
            ("blue", ply::Property::UChar(v)) => self.color[2] = C::from_u8(v),
            ("alpha", ply::Property::UChar(v)) => self.color[3] = C::from_u8(v),
            ("vertex_indices", _) => {} // ignore
            (k, _) => panic!("Vertex: Unexpected key/value combination: key: {}", k),
        };
    }

    #[inline]
    fn get_float(&self, _property_name: &String) -> Option<f32> {
        match _property_name.as_str() {
            "x" => Some(self.position[0].to_f32().unwrap()),
            "y" => Some(self.position[1].to_f32().unwrap()),
            "z" => Some(self.position[2].to_f32().unwrap()),
            "red" => Some(<C as Color>::to_f32(self.color[0])),
            "green" => Some(<C as Color>::to_f32(self.color[1])),
            "blue" => Some(<C as Color>::to_f32(self.color[2])),
            "alpha" => Some(<C as Color>::to_f32(self.color[3])),
            _ => None,
        }
    }
    #[inline]
    fn get_double(&self, _property_name: &String) -> Option<f64> {
        match _property_name.as_str() {
            "x" => Some(self.position[0].to_f64().unwrap()),
            "y" => Some(self.position[1].to_f64().unwrap()),
            "z" => Some(self.position[2].to_f64().unwrap()),
            _ => None,
        }
    }

    #[inline]
    fn get_uchar(&self, _property_name: &String) -> Option<u8> {
        match _property_name.as_str() {
            "red" => Some(<C as Color>::to_u8(self.color[0])),
            "green" => Some(<C as Color>::to_u8(self.color[1])),
            "blue" => Some(<C as Color>::to_u8(self.color[2])),
            "alpha" => Some(<C as Color>::to_u8(self.color[3])),
            _ => None,
        }
    }
}

impl<F: BaseFloat, C: BaseColor> Vertex<F, C> {
    pub fn element_def(name: String) -> ElementDef {
        let pos_type = PropertyType::Scalar(F::ply_type());
        let color_type = PropertyType::Scalar(C::ply_type());

        let mut point_element = ElementDef::new(name);
        let p = PropertyDef::new("x".to_string(), pos_type.clone());
        point_element.properties.add(p);
        let p = PropertyDef::new("y".to_string(), pos_type.clone());
        point_element.properties.add(p);
        let p = PropertyDef::new("z".to_string(), pos_type);
        point_element.properties.add(p);
        let p = PropertyDef::new("red".to_string(), color_type.clone());
        point_element.properties.add(p);
        let p = PropertyDef::new("green".to_string(), color_type.clone());
        point_element.properties.add(p);
        let p = PropertyDef::new("blue".to_string(), color_type.clone());
        point_element.properties.add(p);
        let p = PropertyDef::new("alpha".to_string(), color_type);
        point_element.properties.add(p);
        return point_element;
    }
}

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

pub trait PlyType {
    fn ply_type() -> ScalarType;
    fn matches(prop: ply::Property) -> bool;
}

impl PlyType for f32 {
    fn ply_type() -> ScalarType {
        ScalarType::Float
    }
    fn matches(prop: ply::Property) -> bool {
        match prop {
            ply::Property::Float(_) => true,
            _ => false,
        }
    }
}

impl PlyType for f64 {
    fn ply_type() -> ScalarType {
        ScalarType::Double
    }
    fn matches(prop: ply::Property) -> bool {
        match prop {
            ply::Property::Double(_) => true,
            _ => false,
        }
    }
}
impl PlyType for u8 {
    fn ply_type() -> ScalarType {
        ScalarType::UChar
    }
    fn matches(prop: ply::Property) -> bool {
        match prop {
            ply::Property::UChar(_) => true,
            _ => false,
        }
    }
}

pub trait Color {
    fn from_u8(v: u8) -> Self;
    fn from_f32(v: f32) -> Self;

    fn to_u8(v: Self) -> u8;
    fn to_f32(v: Self) -> f32;
}

impl Color for f32 {
    fn from_u8(v: u8) -> Self {
        (v as f32) / 255.
    }

    fn from_f32(v: f32) -> Self {
        v
    }

    fn to_u8(v: Self) -> u8 {
        (v * 255.) as u8
    }

    fn to_f32(v: Self) -> f32 {
        v
    }
}

impl Color for u8 {
    fn from_u8(v: u8) -> Self {
        v
    }

    fn from_f32(v: f32) -> Self {
        (v * 255.) as u8
    }

    fn to_u8(v: u8) -> u8 {
        v
    }

    fn to_f32(v: u8) -> f32 {
        v as f32 / 255.
    }
}
