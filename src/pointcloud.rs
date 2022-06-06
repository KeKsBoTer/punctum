use std::sync::Arc;

use nalgebra::{center, distance_squared, Point3, RealField, Vector3};
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer},
    device::Device,
};

use crate::vertex::{BaseColor, BaseFloat, Vertex};

pub struct PointCloud<F: BaseFloat, C: BaseColor> {
    data: Vec<Vertex<F, C>>,
    bbox: BoundingBox<F>,
}

impl PointCloud<f32, f32> {
    pub fn from_ply_file(filename: &str) -> Self {
        let mut f = std::fs::File::open(filename).unwrap();

        // create a parser
        let p = ply_rs::parser::Parser::<Vertex<f32, f32>>::new();

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
}

impl Into<PointCloud<f32, u8>> for &PointCloud<f32, f32> {
    fn into(self) -> PointCloud<f32, u8> {
        PointCloud {
            data: self
                .data
                .iter()
                .map(|p| (*p).into())
                .collect::<Vec<Vertex<f32, u8>>>(),
            bbox: self.bbox.clone(),
        }
    }
}

impl<F: BaseFloat, C: BaseColor> PointCloud<F, C> {
    pub fn from_vec(points: &Vec<Vertex<F, C>>) -> Self {
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
            .fold(Point3::origin(), |acc, v| acc + &v.position.coords)
            / F::from_usize(self.data.len()).unwrap();
        let mut max_size = self
            .data
            .iter()
            .map(|p| distance_squared(&p.position, &center))
            .reduce(|acum, item| F::max(acum, item))
            .unwrap()
            .sqrt();

        // if we have only one point in the pointcloud we would divide by 0 later
        // so just set it to 1
        if max_size.is_zero() {
            max_size = F::from_subset(&1.);
        }

        self.data.iter_mut().for_each(|p| {
            p.position = (&p.position - &center.coords) / max_size.clone();
        });
        self.bbox = BoundingBox::from_points(&self.data);
    }

    pub fn scale_to_size(&mut self, size: F) {
        let center = self.bbox.center();
        let mut max_size = self.bbox.size().amax();

        // if we have only one point in the pointcloud we would divide by 0 later
        // so just set it to 1
        if max_size.is_zero() {
            max_size = F::from_subset(&1.);
        }

        self.data.iter_mut().for_each(|p| {
            p.position = (&p.position - &center.coords) / max_size.clone() * size;
        });
        self.bbox = BoundingBox::from_points(&self.data);
    }

    pub fn bounding_box(&self) -> &BoundingBox<F> {
        &self.bbox
    }

    pub fn points(&self) -> &Vec<Vertex<F, C>> {
        &self.data
    }
}

pub struct PointCloudGPU {
    cpu: Arc<PointCloud<f32, f32>>,
    gpu_buffer: Arc<CpuAccessibleBuffer<[Vertex<f32, f32>]>>,
}

impl PointCloudGPU {
    pub fn from_point_cloud(device: Arc<Device>, pc: Arc<PointCloud<f32, f32>>) -> Self {
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

    pub fn cpu(&self) -> &Arc<PointCloud<f32, f32>> {
        &self.cpu
    }
    pub fn gpu_buffer(&self) -> &Arc<CpuAccessibleBuffer<[Vertex<f32, f32>]>> {
        &self.gpu_buffer
    }
}

#[derive(Debug, Clone, Copy)]
pub struct BoundingBox<F: BaseFloat> {
    pub min: Point3<F>,
    pub max: Point3<F>,
}

impl<F: BaseFloat> BoundingBox<F> {
    pub fn new(p1: Point3<F>, p2: Point3<F>) -> Self {
        BoundingBox {
            min: BoundingBox::elm_min(&p1, &p2),
            max: BoundingBox::elm_max(&p1, &p2),
        }
    }

    pub fn center(&self) -> Point3<F> {
        center(&self.max, &self.min)
    }

    pub fn size(&self) -> Vector3<F> {
        &self.max - &self.min
    }

    pub fn contains(&self, p: Point3<F>) -> bool {
        self.min.x <= p.x
            && self.min.y <= p.y
            && self.min.z <= p.z
            && self.max.x >= p.x
            && self.max.y >= p.y
            && self.max.z >= p.z
    }

    fn elm_min(p1: &Point3<F>, p2: &Point3<F>) -> Point3<F> {
        Point3::new(
            RealField::min(p1.x, p2.x),
            RealField::min(p1.y, p2.y),
            RealField::min(p1.z, p2.z),
        )
    }

    fn elm_max(p1: &Point3<F>, p2: &Point3<F>) -> Point3<F> {
        Point3::new(
            RealField::max(p1.x, p2.x),
            RealField::max(p1.y, p2.y),
            RealField::max(p1.z, p2.z),
        )
    }

    pub fn from_points<C: BaseColor>(points: &Vec<Vertex<F, C>>) -> Self {
        let max_f = F::max_value().unwrap();
        let min_f = F::min_value().unwrap();
        let mut min_corner: Point3<F> = Point3::new(max_f, max_f, max_f);
        let mut max_corner = Point3::new(min_f, min_f, min_f);
        for v in points.iter() {
            let p: &Point3<F> = &v.position;
            min_corner = BoundingBox::elm_min(&p, &min_corner);
            max_corner = BoundingBox::elm_max(&p, &max_corner);
        }
        BoundingBox {
            min: min_corner,
            max: max_corner,
        }
    }
}
