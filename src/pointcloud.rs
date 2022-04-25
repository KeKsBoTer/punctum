use std::sync::Arc;

use nalgebra::{center, distance_squared, Point3, RealField, Scalar, Vector3};
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer},
    device::Device,
};

use crate::vertex::{PointPosition, Vertex};

pub struct PointCloud {
    data: Vec<Vertex>,
    bbox: BoundingBox<f32>,
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
        let bbox = BoundingBox::from_points(&points);

        PointCloud {
            data: points,
            bbox: bbox,
        }
    }

    pub fn from_vec(points: &Vec<Vertex>) -> Self {
        let bbox = BoundingBox::from_points(points);
        PointCloud {
            data: points.clone(),
            bbox: bbox,
        }
    }

    // scales all points to fix into a sphere with radius 1 and center at 0.0;
    pub fn scale_to_unit_sphere(&mut self) {
        let center = self
            .data
            .iter()
            .fold(Point3::origin(), |acc, v| acc + v.position.coords)
            / self.data.len() as f32;
        let max_size = self
            .data
            .iter()
            .map(|p| distance_squared(&Point3::from(p.position), &center))
            .reduce(|acum, item| acum.max(item))
            .unwrap()
            .sqrt();
        self.data.iter_mut().for_each(|p| {
            p.position = (p.position - center.coords) / max_size;
        });
        self.bbox = BoundingBox::from_points(&self.data);
    }

    pub fn bounding_box(&self) -> &BoundingBox<f32> {
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
pub struct BoundingBox<T>
where
    T: Scalar + nalgebra::ComplexField + std::cmp::PartialOrd + RealField + Copy,
{
    pub min: Point3<T>,
    pub max: Point3<T>,
}

impl<T> BoundingBox<T>
where
    T: Scalar + nalgebra::ComplexField + std::cmp::PartialOrd + RealField + Copy,
{
    pub fn new(p1: Point3<T>, p2: Point3<T>) -> Self {
        BoundingBox {
            min: BoundingBox::elm_min(&p1, &p2),
            max: BoundingBox::elm_max(&p1, &p2),
        }
    }

    pub fn center(&self) -> Point3<T> {
        center(&self.min, &self.max)
    }

    pub fn size(&self) -> Vector3<T> {
        self.max - self.min
    }

    pub fn contains(&self, p: Point3<T>) -> bool {
        self.min.x <= p.x
            && self.min.y <= p.y
            && self.min.z <= p.z
            && self.max.x >= p.x
            && self.max.y >= p.y
            && self.max.z >= p.z
    }

    fn elm_min(p1: &Point3<T>, p2: &Point3<T>) -> Point3<T> {
        Point3::new(p1.x.min(p2.x), p1.y.min(p2.y), p1.z.min(p2.z))
    }

    fn elm_max(p1: &Point3<T>, p2: &Point3<T>) -> Point3<T> {
        Point3::new(p1.x.max(p2.x), p1.y.max(p2.y), p1.z.max(p2.z))
    }

    pub fn from_points(points: &Vec<impl PointPosition<T>>) -> Self {
        let max_f = T::max_value().unwrap();
        let min_f = T::min_value().unwrap();
        let mut min_corner: Point3<T> = Point3::new(max_f, max_f, max_f);
        let mut max_corner = Point3::new(min_f, min_f, min_f);
        for v in points.iter() {
            let p: &Point3<T> = v.position().into();
            min_corner = BoundingBox::elm_min(&p, &min_corner);
            max_corner = BoundingBox::elm_max(&p, &max_corner);
        }
        BoundingBox {
            min: min_corner,
            max: max_corner,
        }
    }
}
