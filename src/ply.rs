use nalgebra::{Point3, Vector3};
use ply_rs::ply::{self, Addable, ElementDef, PropertyDef, PropertyType, ScalarType};

use crate::{BaseFloat, Vertex};

impl<F: BaseFloat> ply::PropertyAccess for Vertex<F> {
    fn new() -> Self {
        Vertex {
            position: Point3::origin(),
            color: Vector3::zeros(),
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
            ("red", ply::Property::UChar(v)) => self.color[0] = v,
            ("green", ply::Property::UChar(v)) => self.color[1] = v,
            ("blue", ply::Property::UChar(v)) => self.color[2] = v,
            ("vertex_indices", _) => {} // ignore
            (_, _) => {}
        };
    }

    #[inline]
    fn get_float(&self, _property_name: &String) -> Option<f32> {
        match _property_name.as_str() {
            "x" => Some(self.position[0].to_f32().unwrap()),
            "y" => Some(self.position[1].to_f32().unwrap()),
            "z" => Some(self.position[2].to_f32().unwrap()),
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
            "red" => Some(self.color.x),
            "green" => Some(self.color.y),
            "blue" => Some(self.color.z),
            _ => None,
        }
    }
}

impl<F: BaseFloat> Vertex<F> {
    pub fn element_def(name: String) -> ElementDef {
        let pos_type = PropertyType::Scalar(F::ply_type());
        let color_type = PropertyType::Scalar(ScalarType::UChar);

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
        return point_element;
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
