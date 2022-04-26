use bytemuck::{Pod, Zeroable};
use nalgebra::{Point3, RealField, Scalar, Vector4};
use ply_rs::ply::{self, Addable, ElementDef, PropertyDef, PropertyType, ScalarType};
use serde::{Deserialize, Serialize};

pub trait BaseFloat: Scalar + RealField + Copy {}
impl<T: Scalar + RealField + Copy> BaseFloat for T {}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct Vertex<F: BaseFloat> {
    pub position: Point3<F>,
    // pub normal: Vector3<f32>,
    #[serde(skip_serializing)]
    pub color: Vector4<f32>,
}

unsafe impl Zeroable for Vertex<f32> {}
unsafe impl Pod for Vertex<f32> {}

pub trait PointPosition<F>
where
    F: BaseFloat,
{
    fn position(&self) -> &Point3<F>;
}

impl<F: BaseFloat> PointPosition<F> for Vertex<F> {
    #[inline]
    fn position(&self) -> &Point3<F> {
        &self.position
    }
}

vulkano::impl_vertex!(Vertex<f32>, position, color);

impl<F: BaseFloat> ply::PropertyAccess for Vertex<F> {
    fn new() -> Self {
        Vertex {
            position: Point3::<F>::origin(),
            // normal: Vector3::zeros(),
            color: Vector4::zeros(),
        }
    }

    fn set_property(&mut self, key: String, property: ply::Property) {
        match (key.as_ref(), property) {
            ("x", ply::Property::Float(v)) => self.position[0] = F::from_f32(v).unwrap(),
            ("y", ply::Property::Float(v)) => self.position[1] = F::from_f32(v).unwrap(),
            ("z", ply::Property::Float(v)) => self.position[2] = F::from_f32(v).unwrap(),
            // ("nx", ply::Property::Float(v)) => self.normal[0] = v,
            // ("ny", ply::Property::Float(v)) => self.normal[1] = v,
            // ("nz", ply::Property::Float(v)) => self.normal[2] = v,
            ("red", ply::Property::UChar(v)) => self.color[0] = (v as f32) / 255.,
            ("green", ply::Property::UChar(v)) => self.color[1] = (v as f32) / 255.,
            ("blue", ply::Property::UChar(v)) => self.color[2] = (v as f32) / 255.,
            ("alpha", ply::Property::UChar(v)) => self.color[3] = (v as f32) / 255.,
            ("vertex_indices", _) => {} // ignore
            (k, _) => panic!("Vertex: Unexpected key/value combination: key: {}", k),
        }
    }

    fn get_uchar(&self, _property_name: &String) -> Option<u8> {
        match _property_name.as_str() {
            "red" => Some((self.color[0] * 255.) as u8),
            "green" => Some((self.color[1] * 255.) as u8),
            "blue" => Some((self.color[2] * 255.) as u8),
            "alpha" => Some((self.color[3] * 255.) as u8),
            _ => None,
        }
    }
    fn get_float(&self, _property_name: &String) -> Option<f32> {
        match _property_name.as_str() {
            "x" => self.position[0].to_subset().and_then(|v| Some(v as f32)),
            "y" => self.position[0].to_subset().and_then(|v| Some(v as f32)),
            "z" => self.position[0].to_subset().and_then(|v| Some(v as f32)),
            _ => None,
        }
    }
}

impl<F: BaseFloat> Vertex<F> {
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
