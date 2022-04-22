use bytemuck::{Pod, Zeroable};
use ply_rs::ply::{self, Addable, ElementDef, PropertyDef, PropertyType, ScalarType};

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub color: [f32; 4],
}

vulkano::impl_vertex!(Vertex, position, normal, color);

impl ply::PropertyAccess for Vertex {
    fn new() -> Self {
        Vertex::zeroed()
    }

    fn set_property(&mut self, key: String, property: ply::Property) {
        match (key.as_ref(), property) {
            ("x", ply::Property::Float(v)) => self.position[0] = v,
            ("y", ply::Property::Float(v)) => self.position[1] = v,
            ("z", ply::Property::Float(v)) => self.position[2] = v,
            ("nx", ply::Property::Float(v)) => self.normal[0] = v,
            ("ny", ply::Property::Float(v)) => self.normal[1] = v,
            ("nz", ply::Property::Float(v)) => self.normal[2] = v,
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
            "x" => Some(self.position[0]),
            "y" => Some(self.position[1]),
            "z" => Some(self.position[2]),
            _ => None,
        }
    }
}

impl Vertex {
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
