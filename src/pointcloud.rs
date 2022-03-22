use std::sync::Arc;

use bytemuck::Zeroable;
use ply_rs::ply;
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer},
    device::Device,
};

use crate::vertex::Vertex;

pub struct PointCloud {
    pub buffer: Arc<CpuAccessibleBuffer<[Vertex]>>,
}

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
            ("alpha", ply::Property::UChar(_)) => {} // ignore alpha
            (k, _) => panic!("Vertex: Unexpected key/value combination: key: {}", k),
        }
    }
}

impl PointCloud {
    pub fn from_ply_file(device: Arc<Device>, filename: &str) -> Self {
        let mut f = std::fs::File::open(filename).unwrap();

        // create a parser
        let p = ply_rs::parser::Parser::<Vertex>::new();

        // use the parser: read the entire file
        let ply = p.read_ply(&mut f);

        // make sure it did work
        assert!(ply.is_ok());
        let ply = ply.unwrap();

        let points = ply.payload.get("vertex").unwrap().clone();

        let vertex_buffer = CpuAccessibleBuffer::from_iter(
            device,
            BufferUsage::vertex_buffer(),
            false,
            points.into_iter(),
        )
        .unwrap();

        PointCloud {
            buffer: vertex_buffer,
        }
    }
}
