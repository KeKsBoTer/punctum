use std::{mem, sync::Arc};

use nalgebra::{center, distance_squared, Matrix4, Point3, RealField, Vector3};
use serde::{Deserialize, Serialize};
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer},
    device::Device,
};

use crate::{
    camera::ViewFrustum,
    vertex::{BaseColor, BaseFloat, Vertex},
};

#[derive(Debug, Clone)]
#[repr(transparent)]
pub struct PointCloud<F: BaseFloat, C: BaseColor>(Vec<Vertex<F, C>>);

impl Into<PointCloud<f32, u8>> for &PointCloud<f32, f32> {
    fn into(self) -> PointCloud<f32, u8> {
        self.0
            .iter()
            .map(|p| (*p).into())
            .collect::<Vec<Vertex<f32, u8>>>()
            .into()
    }
}

impl Into<PointCloud<f32, f32>> for &PointCloud<f64, u8> {
    fn into(self) -> PointCloud<f32, f32> {
        self.0
            .iter()
            .map(|p| (*p).into())
            .collect::<Vec<Vertex<f32, f32>>>()
            .into()
    }
}

impl<F: BaseFloat, C: BaseColor> Into<PointCloud<F, C>> for Vec<Vertex<F, C>> {
    fn into(self) -> PointCloud<F, C> {
        PointCloud(self)
    }
}

impl<'a, F: BaseFloat, C: BaseColor> Into<&'a PointCloud<F, C>> for &'a Vec<Vertex<F, C>> {
    fn into(self) -> &'a PointCloud<F, C> {
        unsafe { mem::transmute::<&Vec<Vertex<F, C>>, &PointCloud<F, C>>(self) }
    }
}

impl<F: BaseFloat, C: BaseColor> Into<Vec<Vertex<F, C>>> for PointCloud<F, C> {
    fn into(self) -> Vec<Vertex<F, C>> {
        self.0
    }
}

impl<F: BaseFloat, C: BaseColor> PointCloud<F, C> {
    // scales all points to fix into a sphere with radius 1 and center at 0.0;
    pub fn scale_to_unit_sphere(&mut self) {
        let center = self
            .0
            .iter()
            .fold(Point3::origin(), |acc, v| acc + &v.position.coords)
            / F::from_usize(self.0.len()).unwrap();
        let mut max_size = self
            .0
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

        self.0.iter_mut().for_each(|p| {
            p.position = (&p.position - &center.coords) / max_size.clone();
        });
    }

    pub fn scale_to_size(&mut self, size: F) {
        let bbox = CubeBoundingBox::from_points(&self.0);
        let center = bbox.center;
        let mut max_size = bbox.size;

        // if we have only one point in the pointcloud we would divide by 0 later
        // so just set it to 1
        if max_size.is_zero() {
            max_size = F::from_subset(&1.);
        }

        self.0.iter_mut().for_each(|p| {
            p.position = (&p.position - &center.coords) / max_size.clone() * size;
        });
    }

    pub fn points(&self) -> &Vec<Vertex<F, C>> {
        &self.0
    }
    pub fn centroid(&self) -> Point3<F> {
        let mut mean = Point3::origin();
        for v in self.0.iter() {
            mean += v.position.coords;
        }
        let length = self.0.len() as f64;
        mean /= F::from_subset(&length);

        return mean;
    }
}

pub struct PointCloudGPU {
    gpu_buffer: Arc<CpuAccessibleBuffer<[Vertex<f32, f32>]>>,
}

impl PointCloudGPU {
    pub fn from_point_cloud(device: Arc<Device>, pc: PointCloud<f32, f32>) -> Self {
        let points: Vec<Vertex<f32, f32>> = pc.into();
        let vertex_buffer =
            CpuAccessibleBuffer::from_iter(device, BufferUsage::vertex_buffer(), false, points)
                .unwrap();

        PointCloudGPU {
            gpu_buffer: vertex_buffer,
        }
    }

    pub fn gpu_buffer(&self) -> &Arc<CpuAccessibleBuffer<[Vertex<f32, f32>]>> {
        &self.gpu_buffer
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CubeBoundingBox<F: BaseFloat> {
    /// the bouding box center
    pub center: Point3<F>,

    /// bounding box size (length of cube edge)
    pub size: F,
}

impl<F: BaseFloat> CubeBoundingBox<F> {
    pub fn new(center: Point3<F>, size: F) -> Self {
        Self { center, size }
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

    pub fn corners(&self) -> [Point3<F>; 8] {
        let size = self.size * F::from_subset(&0.5);
        [
            self.center - Vector3::new(-size, -size, -size),
            self.center - Vector3::new(size, -size, -size),
            self.center - Vector3::new(-size, size, -size),
            self.center - Vector3::new(size, size, -size),
            self.center - Vector3::new(-size, -size, size),
            self.center - Vector3::new(size, -size, size),
            self.center - Vector3::new(-size, size, size),
            self.center - Vector3::new(size, size, size),
        ]
    }

    /// checks if all 8 points that define the bounding box
    /// can be seen by the given projection matrix
    /// IMPORTANT: this does not cover the case where the box is so big,
    /// that none of the points are within the frustum
    pub fn at_least_one_point_visible(&self, projection: &Matrix4<F>) -> bool {
        let one = F::from_subset(&1.);
        let points = self.corners();
        let ones = Vector3::new(one, one, one);
        points
            .map(|p| {
                let screen_space = projection * p.to_homogeneous();
                let n_pos = screen_space.xyz() / screen_space.w;
                n_pos.abs() <= ones
            })
            .contains(&true)
    }

    // TODO something here is not working correctly!
    pub fn within_frustum(&self, frustum: &ViewFrustum<F>) -> bool {
        let radius = self.outer_radius();

        return frustum.sphere_visible(self.center, radius);
    }

    pub fn outer_radius(&self) -> F {
        let two = F::from_subset(&2.);
        let three = F::from_subset(&3.);
        let radius = (self.size / two) * three.sqrt();
        return radius;
    }

    pub fn to_f64(&self) -> CubeBoundingBox<f64> {
        let center = Point3::new(
            self.center.x.to_f64().unwrap(),
            self.center.y.to_f64().unwrap(),
            self.center.z.to_f64().unwrap(),
        );
        let size = self.size.to_f64().unwrap();
        CubeBoundingBox { center, size }
    }
}
