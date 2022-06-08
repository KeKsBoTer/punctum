use image::{GrayImage, Luma, Pixel};
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

    let mut y = ((2 * l - 1) / (l - m_abs)) as f32 * x * lpmv(l - 1, m_abs as i64, x);

    if l - m_abs > 1 {
        y -= ((l + m_abs - 1) / (l - m_abs)) as f32 * lpmv(l - 2, m_abs as i64, x);
    }

    if m < 0 {
        y = negative_lpmv(l, m, y)
    }
    return y;
}

fn get_spherical_harmonics_element(l: u64, m: i64, phi: f32, leg: f32) -> f32 {
    let m_abs = m.abs() as u64;
    if m == 0 {
        let n = ((2. * l as f32 + 1.) / (4. * PI)).sqrt();
        return leg * n;
    }

    let mut y = if m > 0 {
        (m as f32 * phi).cos()
    } else {
        (m_abs as f32 * phi).sin()
    };

    y *= leg * (2.0 / pochhammer(l - m_abs + 1, 2 * m_abs) as f32).sqrt();
    return y;
}

fn lm2flat_index(l: i64, m: i64) -> usize {
    (l * (l + 1) + m) as usize
}

fn main() {
    let l_max = 5i64;

    let res = 1000;

    let mut images = (0..lm2flat_index(l_max, l_max) + 1)
        .map(|_| GrayImage::new(res, res))
        .collect::<Vec<GrayImage>>();

    for l in 0..l_max + 1 {
        for m in 0..l + 1 {
            for j in 0..res {
                let theta = PI * (j as f32 / res as f32);
                let leg = lpmv(l as u64, m.abs() as i64, theta.cos());
                for i in 0..res {
                    let phi = 2. * PI * (i as f32 / res as f32);
                    let value = get_spherical_harmonics_element(l as u64, m, phi, leg);

                    images[lm2flat_index(l, m)].put_pixel(
                        i,
                        j,
                        Luma::from([((value + 1.) / 2. * 255.) as u8]),
                    );

                    // // make use of the fact that sperical harmonics are symetric to m (with a shift of pi/2)
                    // if m != 0 {
                    //     images[lm2flat_index(l, -m)].put_pixel(
                    //         (i + res / 4) % res,
                    //         j,
                    //         Luma::from([((value + 1.) / 2. * 255.) as u8]),
                    //     );
                    // }
                }
            }
        }
    }
    for l in 0..l_max + 1 {
        for m in -l..l + 1 {
            let img = &mut images[lm2flat_index(l, m)];
            img.save(format!("sh_tests/l={}_m={}.png", l, m)).unwrap();
        }
    }
}
