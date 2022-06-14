use std::sync::Arc;

use nalgebra::{center, distance_squared, Point3, RealField, Vector3};
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer},
    device::Device,
};

use crate::vertex::{BaseColor, BaseFloat, Vertex};

pub struct PointCloud<F: BaseFloat, C: BaseColor> {
    data: Vec<Vertex<F, C>>,
    bbox: CubeBoundingBox<F>,
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
        let bbox = CubeBoundingBox::from_points(&points);

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
        let bbox = CubeBoundingBox::from_points(points);
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
        self.bbox = CubeBoundingBox::from_points(&self.data);
    }

    pub fn scale_to_size(&mut self, size: F) {
        let center = self.bbox.center();
        let mut max_size = self.bbox.size();

        // if we have only one point in the pointcloud we would divide by 0 later
        // so just set it to 1
        if max_size.is_zero() {
            max_size = F::from_subset(&1.);
        }

        self.data.iter_mut().for_each(|p| {
            p.position = (&p.position - &center.coords) / max_size.clone() * size;
        });
        self.bbox = CubeBoundingBox::from_points(&self.data);
    }

    pub fn bounding_box(&self) -> &CubeBoundingBox<F> {
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
pub struct CubeBoundingBox<F: BaseFloat> {
    center: Point3<F>,
    size: F,
}

impl<F: BaseFloat> CubeBoundingBox<F> {
    pub fn new(center: Point3<F>, size: F) -> Self {
        Self { center, size }
    }

    pub fn center(&self) -> &Point3<F> {
        &self.center
    }

    pub fn size(&self) -> F {
        self.size
    }

    pub fn contains(&self, p: Point3<F>) -> bool {
        let half = F::from_subset(&0.5);
        let half_size = Vector3::new(self.size, self.size, self.size) * half;
        let min = self.center - half_size;
        let max = self.center + half_size;
        min.x <= p.x && min.y <= p.y && min.z <= p.z && max.x >= p.x && max.y >= p.y && max.z >= p.z
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
            min_corner = Self::elm_min(&p, &min_corner);
            max_corner = Self::elm_max(&p, &max_corner);
        }
        Self {
            center: center(&min_corner, &max_corner),
            size: (max_corner - min_corner).amax(),
        }
    }

    // pub fn corners(&self) -> [Vector3<F>; 8] {
    //     let size = self.size();
    //     [self.min,
    //     self.min + Vector3::new(size.x,0,0),
    //     self.min + Vector3::new(size.x,size.y,0),
    //     ]
    // }

    // pub fn points_visible(&self, projection: Matrix4<f32>) -> bool {
    //     let points = [];

    //     points
    //         .map(|p| {
    //             let screen_space = view_transform * p.to_homogeneous();
    //             let n_pos = screen_space.xyz() / screen_space.w;
    //             n_pos.abs() <= Vector3::new(1., 1., 1.)
    //         })
    //         .contains(&true)
    // }
}
