use std::sync::Arc;

use cgmath::{Bounded, EuclideanSpace, MetricSpace, Point3, Vector3};
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer},
    device::Device,
};

use crate::vertex::Vertex;

pub struct PointCloud {
    data: Vec<Vertex>,
    bbox: BoundingBox,
}

impl PointCloud {
    pub fn from_ply_file(filename: &str) -> Self {
        let mut f = std::fs::File::open(filename).unwrap();

        // create a parser
        let p = ply_rs::parser::Parser::<Vertex>::new();

        // use the parser: read the entire file
        let ply = p.read_ply(&mut f);

        // make sure it did work
        assert!(ply.is_ok());
        let ply = ply.unwrap();

        let points = ply.payload.get("vertex").unwrap().clone();
        let bbox = PointCloud::calc_bbox(&points);

        PointCloud {
            data: points,
            bbox: bbox,
        }
    }

    // scales all points to fix into a sphere with radius 1 and center at 0.0;
    pub fn scale_to_unit_sphere(&mut self) {
        let center = self.bbox.center();
        let max_size = self
            .data
            .iter()
            .map(|p| Point3::from(p.position).distance2(center))
            .reduce(|acum, item| acum.max(item))
            .unwrap()
            .sqrt();
        self.data.iter_mut().for_each(|p| {
            p.position[0] = (p.position[0] - center.x) / max_size;
            p.position[1] = (p.position[1] - center.y) / max_size;
            p.position[2] = (p.position[2] - center.z) / max_size;
        });
        self.bbox = PointCloud::calc_bbox(&self.data);
    }

    fn calc_bbox(points: &Vec<Vertex>) -> BoundingBox {
        let mut min_corner = Point3::max_value();
        let mut max_corner = Point3::min_value();
        for v in points.iter() {
            min_corner = min_corner.zip(v.position.into(), |x: f32, y| x.min(y));
            max_corner = max_corner.zip(v.position.into(), |x: f32, y| x.max(y));
        }
        BoundingBox {
            min: min_corner,
            max: max_corner,
        }
    }

    pub fn bounding_box(&self) -> &BoundingBox {
        &self.bbox
    }

    pub fn points(&self) -> &Vec<Vertex> {
        &self.data
    }
}

pub struct PointCloudGPU {
    cpu: Arc<PointCloud>,
    gpu_buffer: Arc<CpuAccessibleBuffer<[Vertex]>>,
}

impl PointCloudGPU {
    pub fn from_point_cloud(device: Arc<Device>, pc: Arc<PointCloud>) -> PointCloudGPU {
        let vertex_buffer = CpuAccessibleBuffer::from_iter(
            device,
            BufferUsage::vertex_buffer(),
            false,
            pc.data.clone(),
        )
        .unwrap();

        PointCloudGPU {
            gpu_buffer: vertex_buffer,
            cpu: pc.clone(),
        }
    }

    pub fn cpu(&self) -> &Arc<PointCloud> {
        &self.cpu
    }
    pub fn gpu_buffer(&self) -> &Arc<CpuAccessibleBuffer<[Vertex]>> {
        &self.gpu_buffer
    }
}

#[derive(Debug, Clone, Copy)]
pub struct BoundingBox {
    min: Point3<f32>,
    max: Point3<f32>,
}

impl BoundingBox {
    pub fn new(p1: Point3<f32>, p2: Point3<f32>) -> Self {
        BoundingBox {
            min: p1.zip(p2, |x, y| x.min(y)),
            max: p1.zip(p2, |x, y| x.max(y)),
        }
    }

    pub fn center(&self) -> Point3<f32> {
        self.min.midpoint(self.max)
    }

    pub fn size(&self) -> Vector3<f32> {
        self.max.to_vec() - self.min.to_vec()
    }
}
