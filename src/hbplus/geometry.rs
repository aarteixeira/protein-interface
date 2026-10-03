//! Geometry derived from the published HBPLUS manual, section 3.2 and table III.
use std::ops::{Add, Div, Mul, Neg, Sub};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct V(pub f64, pub f64, pub f64);
impl Add for V {
    type Output = Self;
    fn add(self, b: Self) -> Self {
        Self(self.0 + b.0, self.1 + b.1, self.2 + b.2)
    }
}
impl Sub for V {
    type Output = Self;
    fn sub(self, b: Self) -> Self {
        Self(self.0 - b.0, self.1 - b.1, self.2 - b.2)
    }
}
impl Mul<f64> for V {
    type Output = Self;
    fn mul(self, x: f64) -> Self {
        Self(self.0 * x, self.1 * x, self.2 * x)
    }
}
impl Div<f64> for V {
    type Output = Self;
    fn div(self, x: f64) -> Self {
        self * (1.0 / x)
    }
}
impl Neg for V {
    type Output = Self;
    fn neg(self) -> Self {
        self * -1.0
    }
}
impl V {
    pub fn dot(self, b: Self) -> f64 {
        self.0 * b.0 + self.1 * b.1 + self.2 * b.2
    }
    pub fn cross(self, b: Self) -> Self {
        Self(
            self.1 * b.2 - self.2 * b.1,
            self.2 * b.0 - self.0 * b.2,
            self.0 * b.1 - self.1 * b.0,
        )
    }
    pub fn norm(self) -> f64 {
        self.dot(self).sqrt()
    }
    pub fn unit(self) -> Option<Self> {
        let n = self.norm();
        (n > 1e-10).then(|| self / n)
    }
    pub fn distance(self, b: Self) -> f64 {
        (self - b).norm()
    }
}

pub fn angle(a: V, vertex: V, b: V) -> Option<f64> {
    Some(
        (a - vertex)
            .unit()?
            .dot((b - vertex).unit()?)
            .clamp(-1.0, 1.0)
            .acos()
            .to_degrees(),
    )
}

#[derive(Clone, Debug)]
pub enum Locus {
    Points(Vec<V>),
    Circle { center: V, axis: V, radius: f64 },
    Unknown,
}
impl Locus {
    /// Select the closest point on the locus, not a discretized torsion scan.
    pub fn nearest(&self, acceptor: V) -> Option<V> {
        match self {
            Self::Points(points) => points
                .iter()
                .copied()
                .min_by(|a, b| a.distance(acceptor).total_cmp(&b.distance(acceptor))),
            Self::Circle {
                center,
                axis,
                radius,
            } => {
                let v = acceptor - *center;
                // On the circle axis, every azimuth is equivalent. No geometry-derived
                // orientation exists, so leave H unknown rather than invent an axis.
                Some(*center + (v - *axis * v.dot(*axis)).unit()? * *radius)
            }
            Self::Unknown => None,
        }
    }
}

pub fn bisector(d: V, dd1: V, dd2: V, length: f64, toward_first_deg: f64) -> Locus {
    let Some(a) = (dd1 - d).unit() else {
        return Locus::Unknown;
    };
    let Some(b) = (dd2 - d).unit() else {
        return Locus::Unknown;
    };
    let Some(out) = (-(a + b)).unit() else {
        return Locus::Unknown;
    };
    let Some(toward) = (a - out * a.dot(out)).unit() else {
        return Locus::Unknown;
    };
    let t = toward_first_deg.to_radians();
    Locus::Points(vec![d + (out * t.cos() + toward * t.sin()) * length])
}

fn frame(d: V, dd: V, ddd: V) -> Option<(V, V, V)> {
    let axis = (dd - d).unit()?;
    let v = ddd - dd;
    let radial = (v - axis * v.dot(axis)).unit()?;
    Some((axis, radial, axis.cross(radial)))
}

pub fn planar(d: V, dd: V, ddd: V, length: f64, degrees: f64) -> Locus {
    let Some((axis, radial, _)) = frame(d, dd, ddd) else {
        return Locus::Unknown;
    };
    let t = degrees.to_radians();
    Locus::Points(vec![
        d + (axis * t.cos() + radial * t.sin()) * length,
        d + (axis * t.cos() - radial * t.sin()) * length,
    ])
}

pub fn circle(d: V, dd: V, length: f64, degrees: f64) -> Locus {
    let Some(axis) = (dd - d).unit() else {
        return Locus::Unknown;
    };
    let t = degrees.to_radians();
    Locus::Circle {
        center: d + axis * (length * t.cos()),
        axis,
        radius: length * t.sin(),
    }
}

pub fn tetra_three(d: V, dd: V, ddd: V) -> Locus {
    let Some((axis, radial, tangent)) = frame(d, dd, ddd) else {
        return Locus::Unknown;
    };
    // Calibration against released LYS-NZ observations supports 112 degrees,
    // 1.014 A (manual table III rounds these differently). See docs/hbplus.md.
    let t = 112_f64.to_radians();
    Locus::Points(
        (0..3)
            .map(|i| {
                let phi = (180.0 + 120.0 * i as f64).to_radians();
                d + (axis * t.cos() + (radial * phi.cos() + tangent * phi.sin()) * t.sin()) * 1.014
            })
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    fn near(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-8, "{a} != {b}");
    }
    #[test]
    fn ring_n_bisector() {
        let d = V(0., 0., 0.);
        let a = V(-0.5, 3_f64.sqrt() / 2., 0.);
        let b = V(-0.5, -3_f64.sqrt() / 2., 0.);
        let h = bisector(d, a, b, 1., 0.).nearest(V(3., 0., 0.)).unwrap();
        near(h.0, 1.);
        near(angle(a, d, h).unwrap(), 120.);
        near(angle(b, d, h).unwrap(), 120.);
    }
    #[test]
    fn peptide_four_degree_asymmetry() {
        let d = V(0., 0., 0.);
        let ca = V(-0.5, 0.866025403784, 0.);
        let c = V(-0.5, -0.866025403784, 0.);
        let h = bisector(d, ca, c, 1., 2.).nearest(V(3., 0., 0.)).unwrap();
        near(angle(c, d, h).unwrap() - angle(ca, d, h).unwrap(), 4.);
    }
    #[test]
    fn hydroxyl_circle_closest_point() {
        let d = V(0., 0., 0.);
        let dd = V(0., 0., -1.);
        let a = V(2., 1., 2.);
        let h = circle(d, dd, 1., 110.).nearest(a).unwrap();
        near(h.norm(), 1.);
        near(angle(dd, d, h).unwrap(), 110.);
        // Compare with a dense independent circle sweep.
        for k in 0..3600 {
            let t = (k as f64 / 10.).to_radians();
            let x = V(t.cos() * h.0.hypot(h.1), t.sin() * h.0.hypot(h.1), h.2);
            assert!(h.distance(a) <= x.distance(a) + 1e-8);
        }
    }
    #[test]
    fn amide_and_amine_geometry() {
        let d = V(0., 0., 0.);
        let dd = V(0., 0., -1.);
        let ddd = V(1., 0., -1.);
        for (locus, len, ang, n) in [
            (planar(d, dd, ddd, 1., 120.), 1., 120., 2),
            (tetra_three(d, dd, ddd), 1.014, 112., 3),
        ] {
            let Locus::Points(p) = locus else { panic!() };
            assert_eq!(p.len(), n);
            for h in p {
                near(h.norm(), len);
                near(angle(dd, d, h).unwrap(), ang);
            }
        }
    }
    #[test]
    fn degenerate_geometry_is_unknown() {
        assert!(
            bisector(V(0., 0., 0.), V(0., 0., 0.), V(1., 0., 0.), 1., 0.)
                .nearest(V(3., 0., 0.))
                .is_none()
        );
    }
}
