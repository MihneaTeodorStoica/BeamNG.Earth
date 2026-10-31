//! Lane tapers (server roads.py road_tapers): where a road goes on as one of another width, or onto its own deck, the
//! joint and the width it runs to there. See roads.py TAPER_* for the rules; this is their kernel.
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use std::collections::HashMap;

/// lonlat (P, 2) of the roads' lines, off (n + 1) into it; attrs (n, 5): width (NaN: no ground spec), class code (< 0:
/// none), deck at start (0/1), deck at end (0/1), deck width (NaN: no bridge); own (R, 2): (road, fraction along it) of
/// its own deck's ends inside it; params: m per m, shortest, longest, share of the road, least change, most turn (deg).
/// Returns (T, 5): road, lon, lat, width at the joint, length (m), in road order then end 0, end 1, own decks.
#[pyfunction]
pub fn road_tapers<'py>(py: Python<'py>, lonlat: PyReadonlyArray2<'py, f64>, off: PyReadonlyArray1<'py, i64>,
                        attrs: PyReadonlyArray2<'py, f64>, own: PyReadonlyArray2<'py, f64>,
                        params: PyReadonlyArray1<'py, f64>) -> Bound<'py, PyArray2<f64>> {
    let (ll, off, at, own, p) = (lonlat.as_array(), off.as_array(), attrs.as_array(), own.as_array(), params.as_array());
    let (m_per_m, min_l, max_l, share, min_dw, max_turn) = (p[0], p[1], p[2], p[3], p[4], p[5]);
    let n = off.len().saturating_sub(1);
    let rows = py.allow_threads(|| {
        let mut rows: Vec<[f64; 5]> = Vec::new();
        let ok: Vec<bool> = (0..n).map(|i| (off[i + 1] - off[i]) >= 2 && at[[i, 1]] >= 0.0).collect();
        if !ok.iter().any(|&b| b) {
            return rows;
        }
        // metres per degree at the roads' mean first latitude (as the Python: one scale for the cell)
        let (mut lat_sum, mut cnt) = (0.0, 0.0);
        for i in 0..n {
            if ok[i] {
                lat_sum += ll[[off[i] as usize, 1]];
                cnt += 1.0;
            }
        }
        let lat0 = lat_sum / cnt;
        let (mx, my) = (111320.0 * lat0.to_radians().cos(), 110540.0);
        let key = |lon: f64, lat: f64| ((lon * 1e6).round() as i64, (lat * 1e6).round() as i64);
        let mut ends: HashMap<(i64, i64), Vec<(usize, usize)>> = HashMap::new();
        for i in 0..n {
            if !ok[i] {
                continue;
            }
            let (a, b) = (off[i] as usize, off[i + 1] as usize - 1);
            ends.entry(key(ll[[a, 0]], ll[[a, 1]])).or_default().push((i, 0));
            ends.entry(key(ll[[b, 0]], ll[[b, 1]])).or_default().push((i, 1));
        }
        // unit direction away from end e over its first ~10 m, and the line's length
        let leaving = |i: usize, e: usize| -> (Option<[f64; 2]>, f64) {
            let (a, b) = (off[i] as usize, off[i + 1] as usize);
            let idx = |k: usize| if e == 0 { a + k } else { b - 1 - k };
            let (x0, y0) = (ll[[idx(0), 0]], ll[[idx(0), 1]]);
            let (mut s, mut dir, mut px, mut py_) = (0.0, None, 0.0, 0.0);
            for k in 1..(b - a) {
                let (x, y) = ((ll[[idx(k), 0]] - x0) * mx, (ll[[idx(k), 1]] - y0) * my);
                s += (x - px).hypot(y - py_);
                px = x;
                py_ = y;
                if dir.is_none() && s >= 10.0 {
                    dir = Some([x, y]);
                }
            }
            let v = dir.unwrap_or([px, py_]);
            let l = v[0].hypot(v[1]);
            (if l > 1e-6 { Some([v[0] / l, v[1] / l]) } else { None }, s)
        };
        let cos_max = max_turn.to_radians().cos();
        let span = |dw: f64, length: f64| (dw * m_per_m).clamp(min_l, max_l).min(share * length);
        for i in 0..n {
            let w = at[[i, 0]];
            if !ok[i] || w.is_nan() {
                continue;                       // (all deck: it keeps its width)
            }
            for e in 0..2usize {
                if at[[i, 2 + e]] > 0.5 {
                    continue;                   // (a deck at its end: its width stands, the other road tapers)
                }
                let v = if e == 0 { off[i] as usize } else { off[i + 1] as usize - 1 };
                let (lon, lat) = (ll[[v, 0]], ll[[v, 1]]);
                let (hi, length) = leaving(i, e);
                let Some(hi) = hi else { continue };
                let mut best: Option<(f64, usize, usize)> = None;
                if let Some(list) = ends.get(&key(lon, lat)) {
                    for &(j, ej) in list {
                        if j == i || at[[j, 1]] != at[[i, 1]] {
                            continue;
                        }
                        let (hj, _) = leaving(j, ej);
                        let Some(hj) = hj else { continue };
                        let dot = hi[0] * hj[0] + hi[1] * hj[1];
                        if dot <= -cos_max && best.map_or(true, |b| dot < b.0) {
                            best = Some((dot, j, ej));
                        }
                    }
                }
                let Some((_, j, ej)) = best else { continue };
                let deck = !at[[j, 4]].is_nan() && at[[j, 2 + ej]] > 0.5;
                let wj = if deck { at[[j, 4]] } else { at[[j, 0]] };
                if wj.is_nan() || (w - wj).abs() < min_dw {
                    continue;
                }
                let target = if deck { wj } else { 0.5 * (w + wj) };
                rows.push([i as f64, lon, lat, target, span((w - wj).abs(), length)]);
            }
        }
        // its own deck on a range of it: the ground either side runs to the deck's width
        for r in 0..own.nrows() {
            let (i, f) = (own[[r, 0]] as usize, own[[r, 1]]);
            if i >= n || !ok[i] {
                continue;
            }
            let (w, bw) = (at[[i, 0]], at[[i, 4]]);
            if w.is_nan() || bw.is_nan() || (w - bw).abs() < min_dw {
                continue;
            }
            let (a, b) = (off[i] as usize, off[i + 1] as usize);
            let mut s = vec![0.0f64; b - a];
            for k in 1..(b - a) {
                s[k] = s[k - 1] + ((ll[[a + k, 0]] - ll[[a + k - 1, 0]]) * mx).hypot((ll[[a + k, 1]] - ll[[a + k - 1, 1]]) * my);
            }
            let total = s[b - a - 1];
            let t = f * total;
            let k = s.partition_point(|&x| x < t).clamp(1, b - a - 1);
            let u = if s[k] > s[k - 1] { (t - s[k - 1]) / (s[k] - s[k - 1]) } else { 0.0 };
            let lon = ll[[a + k - 1, 0]] + u * (ll[[a + k, 0]] - ll[[a + k - 1, 0]]);
            let lat = ll[[a + k - 1, 1]] + u * (ll[[a + k, 1]] - ll[[a + k - 1, 1]]);
            rows.push([i as f64, lon, lat, bw, span((w - bw).abs(), total)]);
        }
        rows
    });
    let flat: Vec<f64> = rows.iter().flatten().copied().collect();
    numpy::ndarray::Array2::from_shape_vec((rows.len(), 5), flat).unwrap().into_pyarray(py)
}
