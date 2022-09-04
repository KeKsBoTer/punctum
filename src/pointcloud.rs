use std::{mem, sync::Arc};

use nalgebra::{center, convert, distance_squared, Matrix4, Point3, RealField, Vector3};
use serde::{Deserialize, Serialize};
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer},
    device::Device,
};

use crate::{
    camera::ViewFrustum,
    vertex::{BaseFloat, Vertex},
};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[repr(transparent)]
pub struct PointCloud<F: BaseFloat>(pub Vec<Vertex<F>>);

impl Into<PointCloud<f32>> for &PointCloud<f32> {
    fn into(self) -> PointCloud<f32> {
        self.0
            .iter()
            .map(|p| (*p).into())
            .collect::<Vec<Vertex<f32>>>()
            .into()
    }
}

impl Into<PointCloud<f32>> for &PointCloud<f64> {
    fn into(self) -> PointCloud<f32> {
        self.0
            .iter()
            .map(|p| (*p).into())
            .collect::<Vec<Vertex<f32>>>()
            .into()
    }
}

impl<F: BaseFloat> Into<PointCloud<F>> for Vec<Vertex<F>> {
    fn into(self) -> PointCloud<F> {
        PointCloud(self)
    }
}

impl<'a, F: BaseFloat> Into<&'a PointCloud<F>> for &'a Vec<Vertex<F>> {
    fn into(self) -> &'a PointCloud<F> {
        unsafe { mem::transmute::<&Vec<Vertex<F>>, &PointCloud<F>>(self) }
    }
}

impl<F: BaseFloat> Into<Vec<Vertex<F>>> for PointCloud<F> {
    fn into(self) -> Vec<Vertex<F>> {
        self.0
    }
}

impl<F: BaseFloat> PointCloud<F> {
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

    pub fn points(&self) -> &Vec<Vertex<F>> {
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
    pub fn centroid_and_color(&self) -> (Point3<F>, Vector3<u8>) {
        let mut centroid = Point3::origin();
        let mut color: Vector3<f32> = Vector3::zeros();
        for v in self.0.iter() {
            centroid += v.position.coords;
            color += v.color.cast();
        }
        let length = self.0.len();
        centroid /= convert(length as f64);
        color /= length as f32;

        return (
            centroid,
            Vector3::new(color.x as u8, color.y as u8, color.z as u8),
        );
    }
}

impl PointCloud<f32> {
    pub fn position_color(self) -> (Vec<f32>, Vec<f32>) {
        let mut color = Vec::with_capacity(self.0.len());
        let mut pos = Vec::with_capacity(self.0.len());
        for v in self.0 {
            color.push(v.color.x as f32 / 255.);
            color.push(v.color.y as f32 / 255.);
            color.push(v.color.z as f32 / 255.);

            pos.push(v.position.x);
            pos.push(v.position.y);
            pos.push(v.position.z);
        }
        return (pos, color);
    }
}

pub struct PointCloudGPU {
    gpu_buffer: Arc<CpuAccessibleBuffer<[Vertex<f32>]>>,
}

impl PointCloudGPU {
    pub fn from_point_cloud(device: Arc<Device>, pc: PointCloud<f32>) -> Self {
        let points: Vec<Vertex<f32>> = pc.into();
        let vertex_buffer =
            CpuAccessibleBuffer::from_iter(device, BufferUsage::vertex_buffer(), false, points)
                .unwrap();

        PointCloudGPU {
            gpu_buffer: vertex_buffer,
        }
    }

    pub fn gpu_buffer(&self) -> &Arc<CpuAccessibleBuffer<[Vertex<f32>]>> {
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

    pub fn contains(&self, p: &Point3<F>) -> bool {
        let half = F::from_subset(&0.5);
        let half_size = Vector3::new(self.size, self.size, self.size) * half;
        let min = self.center - half_size;
        let max = self.center + half_size;

        let eps = convert(1e-6);
        min.x - p.x < eps
            && min.y - p.y < eps
            && min.z - p.z < eps
            && p.x - max.x < eps
            && p.y - max.y < eps
            && p.z - max.z < eps
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

    pub fn from_points(points: &Vec<Vertex<F>>) -> Self {
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
        let size = self.size * convert(0.5);
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

        // return self.corners().iter().any(|c| frustum.point_visible(&c));
        return frustum.sphere_visible(self.center, radius);
    }

    pub fn outer_radius(&self) -> F {
        let two = convert(2.);
        let three: F = convert(3.);
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

    pub fn min_corner(&self) -> Point3<F> {
        let size = self.size * convert(0.5);
        self.center - Vector3::new(size, size, size)
    }
    pub fn max_corner(&self) -> Point3<F> {
        let size = self.size * convert(0.5);
        self.center + Vector3::new(size, size, size)
    }
}
