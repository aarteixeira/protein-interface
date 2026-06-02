use std::collections::HashMap;

fn cell(coord: &[f64; 3], cell_size: f64) -> (i64, i64, i64) {
    (
        (coord[0] / cell_size).floor() as i64,
        (coord[1] / cell_size).floor() as i64,
        (coord[2] / cell_size).floor() as i64,
    )
}

pub fn pairs(
    coords_a: &[[f64; 3]],
    coords_b: &[[f64; 3]],
    cutoff: f64,
) -> Vec<(usize, usize, f64)> {
    if coords_a.is_empty() || coords_b.is_empty() {
        return Vec::new();
    }

    let cutoff2 = cutoff * cutoff;
    let cell_size = if cutoff > 0.0 { cutoff } else { 1.0 };
    let mut grid: HashMap<(i64, i64, i64), Vec<usize>> = HashMap::new();
    for (j, coord) in coords_b.iter().enumerate() {
        grid.entry(cell(coord, cell_size)).or_default().push(j);
    }

    let mut out = Vec::new();
    for (i, a) in coords_a.iter().enumerate() {
        let (cx, cy, cz) = cell(a, cell_size);
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(candidates) = grid.get(&(cx + dx, cy + dy, cz + dz)) {
                        for &j in candidates {
                            let b = coords_b[j];
                            let x = a[0] - b[0];
                            let y = a[1] - b[1];
                            let z = a[2] - b[2];
                            let d2 = x * x + y * y + z * z;
                            if d2 <= cutoff2 {
                                out.push((i, j, d2));
                            }
                        }
                    }
                }
            }
        }
    }
    out.sort_unstable_by_key(|(i, j, _)| (*i, *j));
    out
}
