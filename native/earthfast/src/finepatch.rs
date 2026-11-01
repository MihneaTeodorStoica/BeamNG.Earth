//! Fine ground (server finepatch.py): where a terrain block's graded ground is sharp (a cutting's wall, a mountain
//! road's steep side), its 2 m lattice draws a staircase of slanted cells and tilts the lanes beside them. Those cells
//! are holes in the block and a finer mesh of the same graded ground fills them. These are the cells, the holes and the
//! mesh; the heights come from the block's own heights function (Python: the grader is already Rust).
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

/// The cells (between samples [j..j+2, i..i+2]) of heights u (ny, nx) to draw finely: their corners `sharp_dz` or more
/// apart where the graded ground u lies `graded_dz` or more off the natural ground raw at one of them (natural cliffs
/// stay the terrain's), grown by `grow` cells (the slope's foot and top, the lanes beside it). (ny - 1, nx - 1) u8.
#[pyfunction]
pub fn fine_cells<'py>(py: Python<'py>, u: PyReadonlyArray2<'py, f64>, raw: PyReadonlyArray2<'py, f64>, sharp_dz: f64,
                       graded_dz: f64, grow: i64) -> Bound<'py, PyArray2<u8>> {
    let (u, raw) = (u.as_array(), raw.as_array());
    let (ny, nx) = (u.nrows(), u.ncols());
    let (cy, cx) = (ny.saturating_sub(1), nx.saturating_sub(1));
    let mut sharp = vec![false; cy * cx];
    for j in 0..cy {
        for i in 0..cx {
            let c = [u[[j, i]], u[[j, i + 1]], u[[j + 1, i]], u[[j + 1, i + 1]]];
            let hi = c.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let lo = c.iter().cloned().fold(f64::INFINITY, f64::min);
            let moved = [(j, i), (j, i + 1), (j + 1, i), (j + 1, i + 1)].iter()
                .map(|&(a, b)| (u[[a, b]] - raw[[a, b]]).abs()).fold(0.0, f64::max);
            sharp[j * cx + i] = hi - lo >= sharp_dz && moved >= graded_dz;
        }
    }
    let g = grow.max(0) as usize;
    let dilate = |m: &Vec<bool>, r: usize| -> Vec<bool> {
        let mut o = vec![false; cy * cx];
        for j in 0..cy {
            for i in 0..cx {
                if !m[j * cx + i] {
                    continue;
                }
                for a in j.saturating_sub(r)..(j + r + 1).min(cy) {
                    for b in i.saturating_sub(r)..(i + r + 1).min(cx) {
                        o[a * cx + b] = true;
                    }
                }
            }
        }
        o
    };
    let grown = dilate(&sharp, g);
    // closed (dilated by one, then eroded by one): no lone terrain cell inside a patch, which the engine draws with a
    // hole for a corner (pale: its layer index there is the hole's)
    let big = dilate(&grown, 1);
    let mut out = vec![0u8; cy * cx];
    for j in 0..cy {
        for i in 0..cx {
            let mut all = true;
            'n: for a in j.saturating_sub(1)..(j + 2).min(cy) {
                for b in i.saturating_sub(1)..(i + 2).min(cx) {
                    if !big[a * cx + b] {
                        all = false;
                        break 'n;
                    }
                }
            }
            out[j * cx + i] = (grown[j * cx + i] || all) as u8;
        }
    }
    numpy::ndarray::Array2::from_shape_vec((cy, cx), out).unwrap().into_pyarray(py)
}

/// The samples (ny, nx) of a block that are holes for the fine cells (ny - 1, nx - 1): each fine cell's south-west
/// sample (the engine skips a square by its south-west sample's hole flag and draws a kept one with all four corners:
/// tunnel-facts, earthfast tunnel_holes; the 2026-10-25 prototype's "all four cells round a sample" left the fine
/// cells round the patch's rim drawn by the terrain too, pale where a corner was a hole). The block's outermost samples
/// are left alone.
#[pyfunction]
pub fn fine_holes<'py>(py: Python<'py>, cells: PyReadonlyArray2<'py, u8>) -> Bound<'py, PyArray2<u8>> {
    let c = cells.as_array();
    let (cy, cx) = (c.nrows(), c.ncols());
    let (ny, nx) = (cy + 1, cx + 1);
    let mut out = vec![0u8; ny * nx];
    for j in 1..cy {
        for i in 1..cx {
            if c[[j, i]] != 0 {
                out[j * nx + i] = 1;
            }
        }
    }
    numpy::ndarray::Array2::from_shape_vec((ny, nx), out).unwrap().into_pyarray(py)
}

/// The rim of the fine cells (ny - 1, nx - 1): the cells the terrain still draws with a hole for a corner (fine_holes:
/// their east, north or north-east neighbour is fine), pale there. The mesh covers them too, a little over the terrain.
#[pyfunction]
pub fn fine_rim<'py>(py: Python<'py>, cells: PyReadonlyArray2<'py, u8>) -> Bound<'py, PyArray2<u8>> {
    let c = cells.as_array();
    let (cy, cx) = (c.nrows(), c.ncols());
    let f = |j: usize, i: usize| j < cy && i < cx && c[[j, i]] != 0;
    let mut out = vec![0u8; cy * cx];
    for j in 0..cy {
        for i in 0..cx {
            if c[[j, i]] == 0 && (f(j, i + 1) || f(j + 1, i) || f(j + 1, i + 1)) {
                out[j * cx + i] = 1;
            }
        }
    }
    numpy::ndarray::Array2::from_shape_vec((cy, cx), out).unwrap().into_pyarray(py)
}

/// The fine mesh's points (x, y) of cells (n, 2) [j, i] on the lattice whose sample [0, 0] is at (x0, y0), `square`
/// apart, each cell k x k quads: (n * (k + 1)², 2), per cell row a (y) then column b (x).
#[pyfunction]
pub fn fine_points<'py>(py: Python<'py>, x0: f64, y0: f64, square: f64, k: i64, cells: PyReadonlyArray2<'py, i64>)
                        -> Bound<'py, PyArray2<f64>> {
    let c = cells.as_array();
    let k = k.max(1) as usize;
    let m = (k + 1) * (k + 1);
    let mut out = Vec::with_capacity(c.nrows() * m * 2);
    for r in 0..c.nrows() {
        let (j, i) = (c[[r, 0]] as f64, c[[r, 1]] as f64);
        for a in 0..=k {
            for b in 0..=k {
                out.push(x0 + (i + b as f64 / k as f64) * square);
                out.push(y0 + (j + a as f64 / k as f64) * square);
            }
        }
    }
    numpy::ndarray::Array2::from_shape_vec((c.nrows() * m, 2), out).unwrap().into_pyarray(py)
}

/// The fine mesh of cells (n, 2) [j, i] (fine_points' layout): z (n, (k + 1)²) its heights there, u (ny, nx) the
/// lattice's (the terrain's), mask (ny - 1, nx - 1) the fine cells and rim (n) the cells that are their rim (fine_rim). A side the terrain draws (its neighbour not fine)
/// takes the terrain's own edge, linear between its two samples, and every cell's corners the samples: no gap at the
/// holes' rim. Returns (positions (n * (k + 1)², 3), triangles (n * 2k², 3) counter-clockwise from above).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn fine_mesh<'py>(py: Python<'py>, x0: f64, y0: f64, square: f64, k: i64, cells: PyReadonlyArray2<'py, i64>,
                      z: PyReadonlyArray2<'py, f64>, u: PyReadonlyArray2<'py, f64>, mask: PyReadonlyArray2<'py, u8>,
                      rim: numpy::PyReadonlyArray1<'py, u8>, lift: f64)
                      -> (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<i64>>) {
    let (c, z, u, mask, rim) = (cells.as_array(), z.as_array(), u.as_array(), mask.as_array(), rim.as_array());
    let k = k.max(1) as usize;
    let w = k + 1;
    let n = c.nrows();
    let (cy, cx) = (mask.nrows() as i64, mask.ncols() as i64);
    let fine = |j: i64, i: i64| j >= 0 && j < cy && i >= 0 && i < cx && mask[[j as usize, i as usize]] != 0;
    let mut pos = Vec::with_capacity(n * w * w * 3);
    let mut tri = Vec::with_capacity(n * 2 * k * k * 3);
    for r in 0..n {
        let (j, i) = (c[[r, 0]], c[[r, 1]]);
        let (ju, iu) = (j as usize, i as usize);
        let (u00, u10, u01, u11) = (u[[ju, iu]], u[[ju, iu + 1]], u[[ju + 1, iu]], u[[ju + 1, iu + 1]]);
        // (a rim cell (rim[r]): every side the terrain's edge, its inside over the terrain's (bilinear) by lift: the
        // terrain still draws it, pale, under the mesh)
        let is_rim = rim[r] != 0;
        let (rs, rn, rw, re) = if is_rim { (true, true, true, true) }
                               else { (!fine(j - 1, i), !fine(j + 1, i), !fine(j, i - 1), !fine(j, i + 1)) };
        for a in 0..w {
            for b in 0..w {
                let (tx, ty) = (b as f64 / k as f64, a as f64 / k as f64);
                let mut h = z[[r, a * w + b]];
                if is_rim {
                    let bil = (u00 * (1.0 - tx) + u10 * tx) * (1.0 - ty) + (u01 * (1.0 - tx) + u11 * tx) * ty;
                    h = h.max(bil) + lift;
                }
                if a == 0 && rs {
                    h = u00 * (1.0 - tx) + u10 * tx;
                } else if a == k && rn {
                    h = u01 * (1.0 - tx) + u11 * tx;
                }
                if b == 0 && rw {
                    h = u00 * (1.0 - ty) + u01 * ty;
                } else if b == k && re {
                    h = u10 * (1.0 - ty) + u11 * ty;
                }
                h = match (a, b) {
                    (0, 0) => u00,
                    (0, bb) if bb == k => u10,
                    (aa, 0) if aa == k => u01,
                    (aa, bb) if aa == k && bb == k => u11,
                    _ => h,
                };
                pos.push(x0 + (i as f64 + tx) * square);
                pos.push(y0 + (j as f64 + ty) * square);
                pos.push(h);
            }
        }
        let base = (r * w * w) as i64;
        for a in 0..k {
            for b in 0..k {
                let sw = base + (a * w + b) as i64;
                let (se, ne, nw) = (sw + 1, sw + w as i64 + 1, sw + w as i64);
                tri.extend_from_slice(&[sw, se, ne, sw, ne, nw]);
            }
        }
    }
    (numpy::ndarray::Array2::from_shape_vec((n * w * w, 3), pos).unwrap().into_pyarray(py),
     numpy::ndarray::Array2::from_shape_vec((n * 2 * k * k, 3), tri).unwrap().into_pyarray(py))
}
