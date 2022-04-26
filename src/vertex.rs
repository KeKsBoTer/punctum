use bytemuck::{Pod, Zeroable};
use nalgebra::{Point3, RealField, Scalar, Vector4};
use num_traits::Zero;
use ply_rs::ply::{self, Addable, ElementDef, PropertyDef, PropertyType, ScalarType};
use serde::{Deserialize, Serialize};

pub trait BaseFloat: Scalar + RealField + Copy {}
impl<T: Scalar + RealField + Copy> BaseFloat for T {}

pub trait BaseColor: Scalar + Zero + Copy {}
impl<T: Scalar + Zero + Copy> BaseColor for T {}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct Vertex<F: BaseFloat, C: BaseColor> {
    pub position: Point3<F>,
    // pub normal: Vector3<f32>,
    pub color: Vector4<C>,
}

unsafe impl Zeroable for Vertex<f32, f32> {}
unsafe impl Pod for Vertex<f32, f32> {}

vulkano::impl_vertex!(Vertex<f32,f32>, position, color);

impl ply::PropertyAccess for Vertex<f32, u8> {
    fn new() -> Self {
        Vertex {
            position: Point3::origin(),
            // normal: Vector3::zeros(),
            color: Vector4::zeros(),
        }
    }

    fn set_property(&mut self, key: String, property: ply::Property) {
        match (key.as_ref(), property) {
            ("x", ply::Property::Float(v)) => self.position[0] = v,
            ("y", ply::Property::Float(v)) => self.position[1] = v,
            ("z", ply::Property::Float(v)) => self.position[2] = v,
            ("nx", ply::Property::Float(_)) => {}
            ("ny", ply::Property::Float(_)) => {}
            ("nz", ply::Property::Float(_)) => {}
            ("red", ply::Property::UChar(v)) => self.color[0] = v,
            ("green", ply::Property::UChar(v)) => self.color[1] = v,
            ("blue", ply::Property::UChar(v)) => self.color[2] = v,
            ("alpha", ply::Property::UChar(v)) => self.color[3] = v,
            ("vertex_indices", _) => {} // ignore
            (k, _) => panic!("Vertex: Unexpected key/value combination: key: {}", k),
        }
    }

    fn get_uchar(&self, _property_name: &String) -> Option<u8> {
        match _property_name.as_str() {
            "red" => Some(self.color[0]),
            "green" => Some(self.color[1]),
            "blue" => Some(self.color[2]),
            "alpha" => Some(self.color[3]),
            _ => None,
        }
    }
    fn get_float(&self, _property_name: &String) -> Option<f32> {
        match _property_name.as_str() {
            "x" => Some(self.position[0]),
            "y" => Some(self.position[1]),
            "z" => Some(self.position[2]),
            _ => None,
        }
    }
}

impl ply::PropertyAccess for Vertex<f32, f32> {
    fn new() -> Self {
        Vertex {
            position: Point3::origin(),
            // normal: Vector3::zeros(),
            color: Vector4::zeros(),
        }
    }

    fn set_property(&mut self, key: String, property: ply::Property) {
        match (key.as_ref(), property) {
            ("x", ply::Property::Float(v)) => self.position[0] = v,
            ("y", ply::Property::Float(v)) => self.position[1] = v,
            ("z", ply::Property::Float(v)) => self.position[2] = v,
            ("nx", ply::Property::Float(_)) => {}
            ("ny", ply::Property::Float(_)) => {}
            ("nz", ply::Property::Float(_)) => {}
            ("red", ply::Property::UChar(v)) => self.color[0] = (v as f32) / 255.,
            ("green", ply::Property::UChar(v)) => self.color[1] = (v as f32) / 255.,
            ("blue", ply::Property::UChar(v)) => self.color[2] = (v as f32) / 255.,
            ("alpha", ply::Property::UChar(v)) => self.color[3] = (v as f32) / 255.,
            ("vertex_indices", _) => {} // ignore
            (k, _) => panic!("Vertex: Unexpected key/value combination: key: {}", k),
        }
    }
}

impl Vertex<f32, f32> {
    pub fn element_def(name: String) -> ElementDef {
        let mut point_element = ElementDef::new(name);
        let p = PropertyDef::new("x".to_string(), PropertyType::Scalar(ScalarType::Float));
        point_element.properties.add(p);
        let p = PropertyDef::new("y".to_string(), PropertyType::Scalar(ScalarType::Float));
        point_element.properties.add(p);
        let p = PropertyDef::new("z".to_string(), PropertyType::Scalar(ScalarType::Float));
        point_element.properties.add(p);
        let p = PropertyDef::new("red".to_string(), PropertyType::Scalar(ScalarType::UChar));
        point_element.properties.add(p);
        let p = PropertyDef::new("green".to_string(), PropertyType::Scalar(ScalarType::UChar));
        point_element.properties.add(p);
        let p = PropertyDef::new("blue".to_string(), PropertyType::Scalar(ScalarType::UChar));
        point_element.properties.add(p);
        let p = PropertyDef::new("alpha".to_string(), PropertyType::Scalar(ScalarType::UChar));
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
