use approx::assert_ulps_eq;
use nalgebra::{vector, Matrix4, Point3, Vector3};
use std::{f32::consts::PI, time::Duration};
use winit::{dpi::PhysicalPosition, event::*};

use crate::pointcloud::BoundingBox;

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

    pub fn znear(&self) -> &f32 {
        &self.znear
    }

    pub fn zfar(&self) -> &f32 {
        &self.zfar
    }

    fn rot_mat(&self) -> Matrix4<f32> {
        Matrix4::from_euler_angles(self.rot.x, self.rot.y, self.rot.z)
    }
}

impl Camera<PerspectiveProjection> {
    pub fn new() -> Self {
        let fovy: f32 = PI / 2.;
        let mut c = Camera {
            pos: Point3::new(0., 0., 0.),
            rot: vector!(0., 0., 0.),

            view: Matrix4::identity(),
            proj: Matrix4::identity(),
            projection: PerspectiveProjection {
                fovy: fovy,
                aspect_ratio: 1.0,
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

    // creates a camera that looks at the bounding box (move in z)
    // ensures that the y of the bounding box fits to screen
    pub fn look_at(bbox: BoundingBox<f32>) -> Self {
        let center = bbox.center();
        let size = bbox.size();

        let fovy: f32 = PI / 2.;

        let distance = 0.5 * size.y / (fovy / 2.0).tan();
        let pos = center - vector!(0., 0., 0.5 * size.z + distance);

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
            zfar: 2. * (0.5 * size.z + distance),
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

    pub fn look_at_ortho_bbox(bbox: BoundingBox<f32>) -> Self {
        let size = bbox.size();
        let center = bbox.center();
        let mut c = Camera {
            pos: Point3::new(center.x, center.y, center.z - size.z),
            rot: vector!(0., 0., 0.),

            view: Matrix4::identity(),
            proj: Matrix4::identity(),
            projection: OrthographicProjection {
                width: size.x,
                height: size.y,
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

    pub fn update_camera(&mut self, camera: &mut PerspectiveCamera, dt: Duration) {
        let dt = dt.as_secs_f32();

        let cam_front = Vector3::new(
            camera.rot.x.cos() * camera.rot.y.sin(),
            camera.rot.x.sin(),
            camera.rot.x.cos() * camera.rot.y.cos(),
        )
        .normalize();

        let move_speed = dt * self.speed;

        let cam_left = cam_front.cross(&Vector3::new(0., 1., 0.)).normalize();

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
    }
}
