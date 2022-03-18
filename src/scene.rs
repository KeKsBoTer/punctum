use cgmath::Point3;

use crate::camera::Camera;

use crate::pointcloud::PointCloud;

pub struct Scene {
    pub camera: Camera,
    pc: PointCloud,
}

impl Scene {
    pub fn new(pc: PointCloud) -> Self {
        Scene {
            camera: Camera::new(Point3::new(0., 0.1, -0.2)),
            pc: pc,
        }
    }

    pub fn point_cloud(&self) -> &PointCloud {
        &self.pc
    }
}
