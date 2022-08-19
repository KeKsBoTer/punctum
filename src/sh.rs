use nalgebra::Vector2;
use num_traits::Float;
use rayon::prelude::*;
use std::f32::consts::PI;

fn semifactorial(x: u64) -> u64 {
    let mut result = 1;
    let mut i = x;
    while i > 1 {
        result *= i;
        i -= 2;
    }
    return result;
}

fn pochhammer(x: u64, k: u64) -> u64 {
    (x + 1..x + k).fold(x, |a, b| a * b)
}

fn negative_lpmv(l: u64, m: i64, y: f32) -> f32 {
    if m < 0 {
        y * ((-1i64).pow(m as u32) as f32)
            / (pochhammer((l as i64 + m) as u64 + 1, (-2 * m) as u64) as f32)
    } else {
        y
    }
}

fn lpmv(l: u64, m: i64, x: f32) -> f32 {
    let m_abs = m.abs() as u64;
    if m_abs > l {
        panic!("|m| > l");
    }
    if l == 0 {
        return 1.;
    }

    if m_abs == l {
        let y = (-1i32).pow(m_abs as u32) as f32
            * semifactorial(2 * m_abs as u64 - 1) as f32
            * (1. - x * x).powf(m_abs as f32 / 2.);
        return negative_lpmv(l, m, y);
    }

    let mut y = ((2 * l - 1) as f32 / (l - m_abs) as f32) * x * lpmv(l - 1, m_abs as i64, x);

    if l - m_abs > 1 {
        y -= ((l + m_abs - 1) as f32 / (l - m_abs) as f32) * lpmv(l - 2, m_abs as i64, x);
    }

    if m < 0 {
        y = negative_lpmv(l, m, y)
    }
    return y;
}

fn phi_independent_part(l: u64, m: i64, theta: f32) -> f32 {
    let m_abs = m.abs() as u64;
    let norm = ((2 * l + 1) as f32 / (4. * PI)).sqrt();
    let p = norm * lpmv(l, m_abs as i64, theta.cos());
    return p;
}

fn get_spherical_harmonics_element(l: u64, m: i64, phi: f32, p: f32) -> f32 {
    let m_abs = m.abs() as u64;
    if m == 0 {
        return p;
    } else {
        let pre = (1.).powf(m_abs as f32)
            * (2.0 / pochhammer(l - m_abs + 1, 2 * m_abs) as f32).sqrt()
            * p;
        if m < 0 {
            return pre * (phi * m_abs as f32).sin();
        } else {
            return pre * (phi * m_abs as f32).cos();
        }
    }
}

pub fn sh_at_point(l: u64, m: i64, phi: f32, theta: f32) -> f32 {
    let p = phi_independent_part(l, m, theta);
    return get_spherical_harmonics_element(l, m, phi, p);
}

#[inline]
pub fn lm2flat_index(l: u64, m: i64) -> usize {
    ((l * (l + 1)) as i64 + m as i64) as usize
}

#[inline]
pub fn flat2lm_index(i: usize) -> (u64, i64) {
    let l = (i as f32).sqrt() as u64;
    let m = (l * (l + 1)) as i64 - i as i64;
    return (l, -m);
}

pub fn calc_sh_grid(l_max: u64, resolution: u32) -> Vec<Vec<f32>> {
    let res = resolution;

    let images = (0..lm2flat_index(l_max, l_max as i64) + 1)
        .into_par_iter()
        .map(|i| {
            let (l, m) = flat2lm_index(i as usize);

            let mut buffer: Vec<f32> = Vec::with_capacity((res * res) as usize);
            unsafe { buffer.set_len((res * res) as usize) }

            for j in 0..res {
                let theta = PI * (j as f32 / res as f32);
                let p = phi_independent_part(l, m, theta);
                for i in 0..res {
                    let phi = 2. * PI * (i as f32 / res as f32);
                    let value = get_spherical_harmonics_element(l as u64, m as i64, phi, p);

                    buffer[(j * res + i) as usize] = value;
                }
            }

            return buffer;
        })
        .collect::<Vec<Vec<f32>>>();
    return images;
}

pub fn calc_sh_sparse(l_max: u64, coords: Vec<Vector2<f32>>) -> Vec<Vec<f32>> {
    (0..lm2flat_index(l_max, l_max as i64) + 1)
        .into_par_iter()
        .map(|i| {
            let (l, m) = flat2lm_index(i as usize);

            coords
                .iter()
                .map(|pos| {
                    let theta = pos.x;
                    let phi = pos.y;
                    return sh_at_point(l, m, phi, theta);
                })
                .collect::<Vec<f32>>()
        })
        .collect::<Vec<Vec<f32>>>()
}
#[cfg(test)]
mod tests {
    use image::{io::Reader as ImageReader, ImageBuffer, Luma};
    use std::{
        fs::{self},
        path::{Path, PathBuf},
    };

    use super::{calc_sh_grid, lm2flat_index};

    fn u8_diff(a: u8, b: u8) -> u8 {
        (a as i32 - b as i32) as u8
    }

    #[test]
    fn python_eq_rust() {
        let files = fs::read_dir("tests/shs").unwrap();
        let res = 256;
        let images = files
            .filter_map(|f| f.ok())
            .filter(|f| f.file_type().unwrap().is_file())
            .filter(|f| f.file_name().to_str().unwrap().ends_with(".png"))
            .map(|img| {
                let mut path = PathBuf::from(img.file_name());
                path.set_extension("");
                let filename = path.file_name().unwrap().to_str().unwrap();
                let parts: Vec<_> = filename.split("_").collect();

                let l: u64 = parts[1].parse().unwrap();
                let m: i64 = parts[3].parse().unwrap();
                let img = ImageReader::open(img.path()).unwrap().decode().unwrap();
                if img.width() != res || img.height() != res {
                    panic!(
                        "expected a resolution of {}, got {}x{}",
                        res,
                        img.width(),
                        img.height()
                    );
                }
                let img_grey = img.to_luma8();
                (l, m, img_grey)
            })
            .collect::<Vec<(u64, i64, ImageBuffer<Luma<u8>, Vec<u8>>)>>();

        let l_max = images.iter().max_by_key(|(l, _, _)| *l).unwrap().0;

        let rust_sh = calc_sh_grid(l_max, res);

        for (l, m, py_img) in images {
            let rust_values = rust_sh.get(lm2flat_index(l, m)).unwrap();
            let rust_img: ImageBuffer<Luma<u8>, Vec<_>> = ImageBuffer::from_fn(res, res, |x, y| {
                Luma::from([
                    ((rust_values[(y * res + x) as usize].clamp(-1., 1.) + 1.) / 2. * 255.) as u8,
                ])
            });

            let mut eq = true;
            'y: for i in 0..res {
                for j in 0..res {
                    let py_value = py_img.get_pixel(j, i).0[0];
                    let rust_value = rust_img.get_pixel(j, i).0[0];
                    let diff = u8_diff(py_value, rust_value);
                    if diff > 1 {
                        eq = false;
                        println!("different: l={},m={} (abs_diff: {})", l, m, diff);
                        break 'y;
                    }
                }
            }
            if !eq {
                if !Path::new("tests/sh_errors").exists() {
                    fs::create_dir("tests/sh_errors").unwrap();
                }
                let error: ImageBuffer<Luma<u8>, Vec<u8>> = ImageBuffer::from_vec(
                    res,
                    res,
                    rust_img
                        .iter()
                        .zip(py_img.iter())
                        .map(|(a, b)| u8_diff(*a, *b))
                        .collect::<Vec<u8>>(),
                )
                .unwrap();
                error
                    .save(format!("tests/sh_errors/error_l_{}_m_{}.png", l, m))
                    .unwrap();
                rust_img
                    .save(format!("tests/sh_errors/rust_l_{}_m_{}.png", l, m))
                    .unwrap();
                py_img
                    .save(format!("tests/sh_errors/py_l_{}_m_{}.png", l, m))
                    .unwrap();
            }
        }
    }
}
