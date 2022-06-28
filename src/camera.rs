use approx::assert_ulps_eq;
use nalgebra::{distance, matrix, vector, Matrix4, Point3, Vector3, Vector4};
use std::{f32::consts::PI, time::Duration};
use winit::{dpi::PhysicalPosition, event::*};

use crate::{pointcloud::CubeBoundingBox, BaseFloat};

pub trait Projection {
    fn projection_matrix(&self, znear: f32, zfar: f32) -> Matrix4<f32>;
}

#[derive(Debug, Clone, Copy)]
pub struct PerspectiveProjection {
    fovy: f32,
    aspect_ratio: f32,
}

impl Projection for PerspectiveProjection {
    fn projection_matrix(&self, znear: f32, zfar: f32) -> Matrix4<f32> {
        Matrix4::new_perspective(self.aspect_ratio, self.fovy, znear, zfar)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct OrthographicProjection {
    width: f32,
    height: f32,
}

impl Projection for OrthographicProjection {
    fn projection_matrix(&self, znear: f32, zfar: f32) -> Matrix4<f32> {
        Matrix4::new_orthographic(
            -self.width / 2.,
            self.width / 2.,
            -self.height / 2.,
            self.height / 2.,
            znear,
            zfar,
        )
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Camera<P: Projection> {
    pos: Point3<f32>,
    rot: Vector3<f32>,

    view: Matrix4<f32>,
    proj: Matrix4<f32>,

    projection: P,
    znear: f32,
    zfar: f32,
}

impl<P: Projection> Camera<P> {
    fn update_proj_matrix(&mut self) {
        self.proj = self.projection.projection_matrix(self.znear, self.zfar);
    }

    pub fn view(&self) -> &Matrix4<f32> {
        &self.view
    }

    pub fn projection(&self) -> &Matrix4<f32> {
        &self.proj
    }

    pub fn position(&self) -> &Point3<f32> {
        &self.pos
    }
    pub fn rotation(&self) -> &Vector3<f32> {
        &self.rot
    }

    pub fn znear(&self) -> &f32 {
        &self.znear
    }

    pub fn zfar(&self) -> &f32 {
        &self.zfar
    }

    fn rot_mat(&self) -> Matrix4<f32> {
        Matrix4::from_euler_angles(self.rot.x, self.rot.y, self.rot.z)
    }

    /// fast frustum plane extraction with Gribb/Hartmann method
    /// see https://www8.cs.umu.se/kurser/5DV051/HT12/lab/plane_extraction.pdf
    pub fn extract_planes_from_projmat(&self, normalize: bool) -> ViewFrustum<f32> {
        let mat = self.proj * self.view;

        let mut left = Vector4::zeros();
        let mut right = Vector4::zeros();
        let mut bottom = Vector4::zeros();
        let mut top = Vector4::zeros();
        let mut near = Vector4::zeros();
        let mut far = Vector4::zeros();

        // Left clipping plane
        left.x = mat[(3, 0)] + mat[(0, 0)];
        left.y = mat[(3, 1)] + mat[(0, 1)];
        left.z = mat[(3, 2)] + mat[(0, 2)];
        left.w = mat[(3, 3)] + mat[(0, 3)];
        // Right clipping plane
        right.x = mat[(3, 0)] - mat[(0, 0)];
        right.y = mat[(3, 1)] - mat[(0, 1)];
        right.z = mat[(3, 2)] - mat[(0, 2)];
        right.w = mat[(3, 3)] - mat[(0, 3)];
        // Top clipping plane
        top.x = mat[(3, 0)] - mat[(1, 0)];
        top.y = mat[(3, 1)] - mat[(1, 1)];
        top.z = mat[(3, 2)] - mat[(1, 2)];
        top.w = mat[(3, 3)] - mat[(1, 3)];
        // Bottom clipping plane
        bottom.x = mat[(3, 0)] + mat[(1, 0)];
        bottom.y = mat[(3, 1)] + mat[(1, 1)];
        bottom.z = mat[(3, 2)] + mat[(1, 2)];
        bottom.w = mat[(3, 3)] + mat[(1, 3)];
        // Near clipping plane
        near.x = mat[(3, 0)] + mat[(2, 0)];
        near.y = mat[(3, 1)] + mat[(2, 1)];
        near.z = mat[(3, 2)] + mat[(2, 2)];
        near.w = mat[(3, 3)] + mat[(2, 3)];
        // Far clipping plane
        far.x = mat[(3, 0)] - mat[(2, 0)];
        far.y = mat[(3, 1)] - mat[(2, 1)];
        far.z = mat[(3, 2)] - mat[(2, 2)];
        far.w = mat[(3, 3)] - mat[(2, 3)];

        if normalize {
            let mag = left.xzy().norm();
            left /= mag;
            let mag = right.xzy().norm();
            right /= mag;
            let mag = top.xzy().norm();
            top /= mag;
            let mag = bottom.xzy().norm();
            bottom /= mag;
            let mag = far.xzy().norm();
            far /= mag;
            let mag = near.xzy().norm();
            near /= mag;
        }

        return ViewFrustum {
            left,
            right,
            bottom,
            top,
            near,
            far,
        };
    }

    pub fn adjust_znear_zfar(&mut self, bbox: &CubeBoundingBox<f32>) {
        let radius = bbox.outer_radius();
        let frustum = self.extract_planes_from_projmat(false);
        let sign = frustum.near.dot(&bbox.center.to_homogeneous());
        let mut d = distance(&bbox.center, &self.pos);
        if sign < 0. {
            d *= -1.;
        }

        let mut zfar = d + radius;
        let mut znear = zfar - 2. * radius;

        if zfar <= 0. {
            zfar = 100.;
        }
        if znear <= 0. || znear >= zfar {
            znear = zfar / 1000.;
        }
        self.znear = znear;
        self.zfar = zfar;
        self.update_proj_matrix();
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ViewFrustum<F: BaseFloat> {
    pub left: Vector4<F>,
    pub right: Vector4<F>,
    pub bottom: Vector4<F>,
    pub top: Vector4<F>,
    pub near: Vector4<F>,
    pub far: Vector4<F>,
}

impl<F: BaseFloat> ViewFrustum<F> {
    // check if point is within frustum
    pub fn point_visible(&self, point: &Point3<F>) -> bool {
        let planes = [
            self.left,
            self.right,
            self.bottom,
            self.top,
            self.near,
            self.far,
        ];
        for p in planes {
            if p.dot(&point.to_homogeneous()) <= F::zero() {
                return false;
            }
        }
        return true;
    }

    /// checks if sphere is intersecting frustum
    /// Important: Frustum planes need to be normed
    pub fn sphere_visible(&self, center: Point3<F>, radius: F) -> bool {
        let planes = [
            self.left,
            self.right,
            self.bottom,
            self.top,
            self.near,
            self.far,
        ];
        for p in planes {
            let d = p.dot(&center.to_homogeneous());
            if d + radius <= F::zero() {
                return false;
            }
        }
        return true;
    }
}

impl Camera<PerspectiveProjection> {
    pub fn new(pos: Point3<f32>, rot: Vector3<f32>, aspect_ratio: f32) -> Self {
        let fovy: f32 = PI / 2.;
        let mut c = Camera {
            pos: pos,
            rot: rot,

            view: Matrix4::identity(),
            proj: Matrix4::identity(),
            projection: PerspectiveProjection {
                fovy: fovy,
                aspect_ratio: aspect_ratio,
            },
            znear: 0.1,
            zfar: 1000.,
        };
        c.update_view_matrix();
        c.update_proj_matrix();
        return c;
    }

    fn update_view_matrix(&mut self) {
        let rot = self.rot_mat();
        let trans = Matrix4::new_translation(&vector!(-self.pos.x, -self.pos.y, self.pos.z));

        self.view = rot * trans;
    }

    pub fn set_near_far(&mut self, znear: f32, zfar: f32) {
        self.znear = znear;
        self.zfar = zfar;
        self.update_proj_matrix();
    }

    // creates a camera that looks at the bounding box (move in z)
    // ensures that the y of the bounding box fits to screen
    pub fn look_at(bbox: CubeBoundingBox<f32>) -> Self {
        let center = bbox.center;
        let size = bbox.size;

        let fovy: f32 = PI / 2.;

        let distance = 0.5 * size / (fovy / 2.0).tan();
        let pos = center - vector!(0., 0., 0.5 * size + distance);

        let mut c = Camera {
            pos: pos,
            rot: vector!(0., 0., 0.),

            view: Matrix4::identity(),
            proj: Matrix4::identity(),
            projection: PerspectiveProjection {
                fovy: fovy,
                aspect_ratio: 1.0,
            },
            znear: 0.001,
            zfar: 2. * (0.5 * size + distance),
        };
        c.update_view_matrix();
        c.update_proj_matrix();
        return c;
    }

    pub fn set_aspect_ratio(&mut self, aspect_ratio: f32) {
        self.projection.aspect_ratio = aspect_ratio;
        self.update_proj_matrix();
    }
}

impl Camera<OrthographicProjection> {
    fn update_view_matrix(&mut self) {
        let rot = self.rot_mat();
        let trans = Matrix4::new_translation(&vector!(-self.pos.x, -self.pos.y, self.pos.z));

        self.view = trans * rot;
    }

    pub fn look_at_ortho_bbox(bbox: CubeBoundingBox<f32>) -> Self {
        let size = bbox.size;
        let center = bbox.center;
        let mut c = Camera {
            pos: Point3::new(center.x, center.y, center.z - size),
            rot: vector!(0., 0., 0.),

            view: Matrix4::identity(),
            proj: Matrix4::identity(),
            projection: OrthographicProjection {
                width: size,
                height: size,
            },
            znear: -10.,
            zfar: 10.,
        };
        c.update_view_matrix();
        c.update_proj_matrix();
        return c;
    }

    pub fn on_unit_sphere(pos: Point3<f32>) -> Self {
        let d: f32 = pos.coords.norm();
        assert_ulps_eq!(d, 1., epsilon = 1e-8);

        let mut c = Camera {
            pos: pos,
            rot: vector!(0., 0., 0.),

            view: Matrix4::identity(),
            proj: Matrix4::identity(),
            projection: OrthographicProjection {
                width: 2.,
                height: 2.,
            },
            znear: -2.0,
            zfar: 2.0,
        };

        // add epsilon z value to avoid NaN in view matrix
        let eye = c.pos + Vector3::new(0., 0., 1e-8);
        c.view = Matrix4::look_at_rh(&eye, &Point3::origin(), &Vector3::y());
        c.update_proj_matrix();
        return c;
    }
}

pub type OrthographicCamera = Camera<OrthographicProjection>;
pub type PerspectiveCamera = Camera<PerspectiveProjection>;

#[derive(Debug)]
pub struct CameraController {
    amount_left: f32,
    amount_right: f32,
    amount_forward: f32,
    amount_backward: f32,
    amount_up: f32,
    amount_down: f32,
    rotate_horizontal: f32,
    rotate_vertical: f32,
    scroll: f32,
    speed: f32,
    sensitivity: f32,
}

impl CameraController {
    pub fn new(speed: f32, sensitivity: f32) -> Self {
        Self {
            amount_left: 0.0,
            amount_right: 0.0,
            amount_forward: 0.0,
            amount_backward: 0.0,
            amount_up: 0.0,
            amount_down: 0.0,
            rotate_horizontal: 0.0,
            rotate_vertical: 0.0,
            scroll: 0.0,
            speed,
            sensitivity,
        }
    }

    pub fn process_keyboard(&mut self, key: VirtualKeyCode, state: ElementState) -> bool {
        let amount = if state == ElementState::Pressed {
            1.0
        } else {
            0.0
        };
        match key {
            VirtualKeyCode::W | VirtualKeyCode::Up => {
                self.amount_forward = amount;
                true
            }
            VirtualKeyCode::S | VirtualKeyCode::Down => {
                self.amount_backward = amount;
                true
            }
            VirtualKeyCode::A | VirtualKeyCode::Left => {
                self.amount_left = amount;
                true
            }
            VirtualKeyCode::D | VirtualKeyCode::Right => {
                self.amount_right = amount;
                true
            }
            VirtualKeyCode::Space => {
                self.amount_up = amount;
                true
            }
            VirtualKeyCode::LShift => {
                self.amount_down = amount;
                true
            }
            _ => false,
        }
    }

    pub fn process_mouse(&mut self, mouse_dx: f64, mouse_dy: f64) {
        self.rotate_horizontal = mouse_dx as f32;
        self.rotate_vertical = mouse_dy as f32;
    }

    pub fn process_scroll(&mut self, delta: &MouseScrollDelta) {
        self.scroll = match delta {
            // I'm assuming a line is about 100 pixels
            MouseScrollDelta::LineDelta(_, scroll) => -scroll * 0.5,
            MouseScrollDelta::PixelDelta(PhysicalPosition { y: scroll, .. }) => -*scroll as f32,
        };
    }

    pub fn update_camera(&mut self, camera: &mut PerspectiveCamera, dt: Duration) -> bool {
        let dt = dt.as_secs_f32();

        let moved = self.amount_forward != 0.
            || self.amount_backward != 0.
            || self.amount_right != 0.
            || self.amount_left != 0.
            || self.amount_up != 0.
            || self.amount_down != 0.
            || self.rotate_horizontal != 0.
            || self.rotate_vertical != 0.;

        // let cam_front = Vector3::new(
        //     camera.rot.x.cos() * camera.rot.y.sin(),
        //     camera.rot.x.sin(),
        //     camera.rot.x.cos() * camera.rot.y.cos(),
        // )
        // .normalize();
        let cam_front = Vector3::new(0., 0., 1.);

        let move_speed = dt * self.speed;

        let cam_left = Vector3::new(-1., 0., 0.); //cam_front.cross(&Vector3::new(0., 1., 0.)).normalize();

        camera.pos += cam_front * self.amount_forward * move_speed;
        camera.pos -= cam_front * self.amount_backward * move_speed;
        camera.pos -= cam_left * self.amount_right * move_speed;
        camera.pos += cam_left * self.amount_left * move_speed;

        camera.pos.y += self.amount_up * move_speed;
        camera.pos.y -= self.amount_down * move_speed;

        let look_speed = dt * self.sensitivity;

        camera.rot.x += self.rotate_vertical * look_speed;
        camera.rot.y += self.rotate_horizontal * look_speed;

        // done processing, reset to 0
        self.rotate_horizontal = 0.;
        self.rotate_vertical = 0.;
        camera.update_view_matrix();
        return moved;
    }
}
