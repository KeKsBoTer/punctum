use cgmath::*;
use std::time::Duration;
use winit::{dpi::PhysicalPosition, event::*};

#[derive(Debug)]
pub struct Camera {
    pos: Point3<f32>,
    rot: Vector3<Rad<f32>>,

    fovy: Rad<f32>,
    znear: f32,
    zfar: f32,
    aspect_ratio: f32,

    view: Matrix4<f32>,
    proj: Matrix4<f32>,
    world: Matrix4<f32>,
}

impl Camera {
    pub fn new(position: Point3<f32>) -> Self {
        let mut c = Camera {
            pos: position,
            rot: vec3(Rad::zero(), Rad::zero(), Rad::zero()),

            fovy: Rad(std::f32::consts::FRAC_PI_2),
            znear: 0.01,
            zfar: 100.0,
            aspect_ratio: 1.0,

            view: Matrix4::identity(),
            proj: Matrix4::identity(),
            world: Matrix4::identity(),
        };
        c.update_view_matrix();
        c.update_proj_matrix();
        return c;
    }

    fn rot_mat(&self) -> Matrix4<f32> {
        Matrix4::from_angle_z(self.rot.z)
            * Matrix4::from_angle_y(self.rot.y)
            * Matrix4::from_angle_x(self.rot.x)
    }

    pub fn set_position(&mut self, pos: Point3<f32>) {
        self.pos = pos;
        self.update_view_matrix();
    }

    pub fn position(&self) -> &Point3<f32> {
        &self.pos
    }

    fn update_view_matrix(&mut self) {
        let rot = self.rot_mat();
        let trans = Matrix4::from_translation(vec3(self.pos.x, -self.pos.y, self.pos.z));

        self.view = rot * trans;
    }
    fn update_proj_matrix(&mut self) {
        self.proj = perspective(self.fovy, self.aspect_ratio, self.znear, self.zfar);
    }

    pub fn view(&self) -> &Matrix4<f32> {
        &self.view
    }

    pub fn projection(&self) -> &Matrix4<f32> {
        &self.proj
    }

    pub fn world(&self) -> &Matrix4<f32> {
        &self.world
    }
}

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

    pub fn update_camera(&mut self, camera: &mut Camera, dt: Duration) {
        let dt = dt.as_secs_f32();

        let cam_front = vec3(
            -camera.rot.x.cos() * camera.rot.y.sin(),
            camera.rot.x.sin(),
            camera.rot.x.cos() * camera.rot.y.cos(),
        )
        .normalize();

        let move_speed = dt * self.speed;

        let cam_right = cam_front.cross(vec3(0., 1., 0.)).normalize();

        camera.pos += cam_front * self.amount_forward * move_speed;
        camera.pos -= cam_front * self.amount_backward * move_speed;
        camera.pos += cam_right * self.amount_right * move_speed;
        camera.pos -= cam_right * self.amount_left * move_speed;

        camera.pos.y += self.amount_up * move_speed;
        camera.pos.y -= self.amount_down * move_speed;

        let look_speed = dt * self.sensitivity;

        camera.rot.x += Rad(self.rotate_vertical * look_speed).normalize();
        camera.rot.y += Rad(self.rotate_horizontal * look_speed).normalize();

        // done processing, reset to 0
        self.rotate_horizontal = 0.;
        self.rotate_vertical = 0.;
        camera.update_view_matrix();
    }
}
