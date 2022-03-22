use crate::camera::Camera;

use crate::pointcloud::PointCloud;

pub struct Scene {
    pub camera: Camera,
    pc: PointCloud,
}

impl Scene {
    pub fn new(pc: PointCloud, camera: Camera) -> Self {
        Scene { camera, pc }
    }

    pub fn point_cloud(&self) -> &PointCloud {
        &self.pc
    }
}
