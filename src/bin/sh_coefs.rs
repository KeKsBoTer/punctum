use std::f32::consts::PI;

use nalgebra::{Vector2, Vector3};

fn to_spherical(pos: Vector3<f32>) -> Vector2<f32> {
    assert!(pos.norm_squared() == 1., "only unit vectors!");

    let theta = pos.z.acos();
    let phi = pos.y.atan2(pos.x) + PI;
    return Vector2::new(theta, phi);
}

fn main() {}
