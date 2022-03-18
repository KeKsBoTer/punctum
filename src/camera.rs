use cgmath::*;
use std::f32::consts::FRAC_PI_2;
use std::time::Duration;
use winit::{dpi::PhysicalPosition, event::*};

#[derive(Debug)]
pub struct Camera {
    position: Point3<f32>,
    rotation: Vector3<Rad<f32>>,

    fovy: Rad<f32>,
    aspect_ratio: f32,

    view: Matrix4<f32>,
    proj: Matrix4<f32>,
    world: Matrix4<f32>,
}

impl Camera {
    pub fn new(position: Point3<f32>) -> Self {
        let view = Matrix4::look_at_rh(
            position,
            Point3::new(0.0, 0.0, 0.0),
            Vector3::new(0.0, -1.0, 0.0),
        );

        let aspect_ratio = 1.0;
        let fovy = Rad(std::f32::consts::FRAC_PI_2);

        let proj = cgmath::perspective(fovy, aspect_ratio, 0.01, 100.0);

        let world = Matrix4::identity();

        Self {
            position: position,
            rotation: vec3(Rad::zero(), Rad::zero(), Rad::zero()),

            fovy: fovy,
            aspect_ratio: aspect_ratio,

            view: view,
            proj: proj,
            world: world,
        }
    }

    pub fn set_position(&mut self, pos: Point3<f32>) {
        self.position = pos;
        self.update_view_matrix();
    }

    pub fn position(&self) -> &Point3<f32> {
        &self.position
    }

    fn update_view_matrix(&mut self) {
        self.view = Matrix4::look_at_rh(
            self.position,
            Point3::new(0.0, 0.0, 0.0),
            Vector3::new(0.0, -1.0, 0.0),
        );
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

        let proj = camera.view * camera.world;

        let forward: Vector3<f32> = proj.transform_vector(Vector3::new(0., 0., 1.));
        let right: Vector3<f32> = proj.transform_vector(Vector3::new(1., 0., 0.));
        camera.position += forward * (self.amount_forward - self.amount_backward) * self.speed * dt;
        camera.position += right * (self.amount_right - self.amount_left) * self.speed * dt;

        // Move forward/backward and left/right
        let (yaw_sin, yaw_cos) = camera.rotation.y.sin_cos();

        // Move in/out (aka. "zoom")
        // Note: this isn't an actual zoom. The camera's position
        // changes when zooming. I've added this to make it easier
        // to get closer to an object you want to focus on.
        let (pitch_sin, pitch_cos) = camera.rotation.x.sin_cos();
        let scrollward =
            Vector3::new(pitch_cos * yaw_cos, pitch_sin, pitch_cos * yaw_sin).normalize();
        camera.position += scrollward * self.scroll * self.speed * self.sensitivity * dt;
        self.scroll = 0.0;

        // Move up/down. Since we don't use roll, we can just
        // modify the y coordinate directly.
        camera.position.y += (self.amount_up - self.amount_down) * self.speed * dt;

        // Rotate
        camera.rotation.y += Rad(self.rotate_horizontal) * self.sensitivity * dt;
        camera.rotation.x += Rad(-self.rotate_vertical) * self.sensitivity * dt;

        // If process_mouse isn't called every frame, these values
        // will not get set to zero, and the camera will rotate
        // when moving in a non cardinal direction.
        self.rotate_horizontal = 0.0;
        self.rotate_vertical = 0.0;

        // Keep the camera's angle from going too high/low.
        if camera.rotation.x < -Rad(FRAC_PI_2) {
            camera.rotation.x = -Rad(FRAC_PI_2);
        } else if camera.rotation.x > Rad(FRAC_PI_2) {
            camera.rotation.x = Rad(FRAC_PI_2);
        }

        camera.update_view_matrix();
    }
}
