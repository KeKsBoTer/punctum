use std::f64::consts::PI;

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

fn negative_lpmv(l: u64, m: i64, y: f64) -> f64 {
    if m < 0 {
        y * ((-1i64).pow(m as u32) as f64)
            / (pochhammer((l as i64 + m) as u64 + 1, (-2 * m) as u64) as f64)
    } else {
        y
    }
}

fn lpmv(l: u64, m: i64, x: f64) -> f64 {
    let m_abs = m.abs() as u64;
    if m_abs > l {
        panic!("|m| > l");
    }
    if l == 0 {
        return 1.;
    }

    if m_abs == l {
        let y = (-1i32).pow(m_abs as u32) as f64
            * semifactorial(2 * m_abs as u64 - 1) as f64
            * (1. - x * x).powf(m_abs as f64 / 2.);
        return negative_lpmv(l, m, y);
    }

    let mut y = ((2 * l - 1) / (l - m_abs)) as f64 * x * lpmv(l - 1, m_abs as i64, x);

    if l - m_abs > 1 {
        y -= ((l + m_abs - 1) / (l - m_abs)) as f64 * lpmv(l - 2, m_abs as i64, x);
    }

    if m < 0 {
        y = negative_lpmv(l, m, y)
    }
    return y;
}

fn get_spherical_harmonics_element(l: u64, m: i64, theta: f64, phi: f64) -> f64 {
    let m_abs = m.abs() as u64;
    assert!(m_abs <= l, "|m| > l");

    let mut N = ((2. * l as f64 + 1.) / (4. * PI)).sqrt();
    let leg = lpmv(l, m_abs as i64, theta.cos());

    if m == 0 {
        return N * leg;
    }

    let mut Y = if m > 0 {
        (m as f64 * phi).cos()
    } else {
        (m_abs as f64 * phi).sin()
    };

    Y *= leg;
    N *= (2.0 / pochhammer(l - m_abs + 1, 2 * m_abs) as f64).sqrt();
    Y *= N;
    return Y;
}

fn main() {
    for x in 0..10 {
        for k in 0..10 {
            println!("{},{} => {}", x, k, pochhammer(x, k))
        }
    }
}
