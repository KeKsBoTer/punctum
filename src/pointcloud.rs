use std::sync::Arc;

use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer},
    device::Device,
};

use crate::vertex::Vertex;

pub struct PointCloud {
    pub buffer: Arc<CpuAccessibleBuffer<[Vertex]>>,
}

impl PointCloud {
    pub fn new(device: Arc<Device>) -> Self {
        let vertex1 = Vertex {
            position: [-0.5, -0.5, 1.],
        };
        let vertex2 = Vertex {
            position: [0.0, 0.5, 1.],
        };
        let vertex3 = Vertex {
            position: [0.5, -0.25, 1.],
        };

        let vertex_buffer = CpuAccessibleBuffer::from_iter(
            device,
            BufferUsage::vertex_buffer(),
            false,
            vec![vertex1, vertex2, vertex3].into_iter(),
        )
        .unwrap();

        PointCloud {
            buffer: vertex_buffer,
        }
    }
}
