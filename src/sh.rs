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

fn phi_independent_part(l: u64, m: i64, thata: f32) -> f32 {
    let m_abs = m.abs() as u64;
    let norm = ((2 * l + 1) as f32 / (4. * PI)).sqrt();
    let p = norm * lpmv(l, m_abs as i64, thata.cos());
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

#[inline]
pub fn lm2flat_index(l: u64, m: u64) -> usize {
    ((l * (l + 1)) as i64 + m as i64) as usize
}

#[inline]
pub fn flat2lm_index(i: usize) -> (u64, i64) {
    let l = (i as f32).sqrt() as u64;
    let m = (l * (l + 1)) as i64 - i as i64;
    return (l, -m);
}

pub fn calc_sh(l_max: u64, resolution: u32) -> Vec<Vec<f32>> {
    let res = resolution;

    let images = (0..lm2flat_index(l_max, l_max) + 1)
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
