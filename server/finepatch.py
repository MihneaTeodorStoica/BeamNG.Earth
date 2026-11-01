"""Fine ground: where a terrain block's graded ground is sharp (a cutting's wall, a mountain road's steep side, an
abutment's slope), its 2 m lattice draws a staircase of slanted cells (teeth along the Stelvio's hairpins) and its
triangles reach across a road's edge into the slope beside it: the lanes tilt and bump (the drawn ground 0.31 m over
the graded profile in the Stelvio's outer lane, 21 cm bumps; the user, 2026-11-01: "the road feels a bit bumpy",
"would it help if we increased the terrain mesh resolution"). A terrain block has one grid size, so those cells are
holes in the block and a finer mesh of the same graded ground fills them, drawn with the block's own aerial photo
(its roads painted in) and colliding.

The block's own process makes both from the same heights (its .ter's holes and the mesh's pieces: written beside it,
listed in its info `fine`), and the mod makes the pieces in the frame it swaps the .ter in: a hole never shows
without its fill. The cells (earthfast.fine_cells): corners SHARP_DZ_M or more apart where the grading moved the
ground GRADED_DZ_M (natural cliffs stay the terrain's), grown by GROW cells. Their rim is the terrain's own edge
(earthfast.fine_mesh). Pieces of PIECE_M square (each its own small collision mesh: Windows rebuilds one big one in
seconds, collision-engine-facts). The prototype of 2026-10-25 (branch fine-ground) put the mesh in the road tiles.
BNE_FINE=on turns it on (off by default while it is tried out).
"""
import os

import numpy as np

import heightmap

FINE_ON = os.environ.get("BNE_FINE", "off") == "on"
SHARP_DZ_M = 1.5            # m between a cell's highest and lowest corner: a sharp cell ...
GRADED_DZ_M = 0.5           # ... where the grading moved the ground this much at one of its corners
GROW = 1                    # cells round each sharp one drawn finely too (a slope's foot and top, the lanes beside)
K = 2                       # quads per cell side: 1 m (4 at 0.5 m: 4x the triangles, no visible gain)
PIECE_M = 64.0              # m: a piece's square (at most 32 x 32 cells, 8192 triangles)
MAX_CELLS = 40000           # per block (a mountain's worth); more: the steepest kept
EDGE_CELLS = 3              # cells at the block's edge left to the terrain (its edge holes, the next block's seam)
RIM_LIFT_M = 0.04           # m the mesh lies over the terrain inside the rim cells (the terrain draws them, pale)
DETAIL_M = 2.0              # the second UV set's metres per unit (farfield.DETAIL_M: satmat.lua's ground grain)


def apply(frame, gen, raw, keep, cx, cy, size, out_dir, stem, aid, idx, info):
    """The block's fine cells as holes in idx (its .ter's layer indices, None: none yet) and their pieces in
    info["fine"]; returns idx."""
    if not FINE_ON:
        return idx
    import time
    t0 = time.perf_counter()
    holes, fine = block_fine(frame, gen, raw, keep, cx, cy, size, out_dir, stem, aid, layers=idx)
    if fine is None:
        return idx
    idx = np.zeros(holes.shape, np.uint8) if idx is None else idx
    idx[holes] = heightmap.TER_HOLE
    fine["holes"] = int(holes.sum())
    fine["seconds"] = round(time.perf_counter() - t0, 2)
    info["fine"] = fine
    return idx


def material_name(bank_, layer):
    """The Material of a bank's (bank) pieces of surface layer `layer` (surface.MATERIALS index): the mod makes it with the
    block's photo and that layer's detail texture (terrain.lua)."""
    return f"bngEarthFine{int(bank_)}_{int(layer)}"


def bank(cx, cy, size):
    """A block's material bank, 0..3 by its centre's parity on the blocks' centre grid (size / 4 samples of the lattice):
    a block and its neighbours (the next block, whose pieces are made while this one shows) never share one."""
    g = size * heightmap.SQUARE / 8.0
    return int(round(cx / g)) % 2 + 2 * (int(round(cy / g)) % 2)


def block_fine(frame, gen, raw, keep, cx, cy, size, out_dir, stem, aid, layers=None):
    """The fine cells of a terrain block: (holes (ny, nx) bool for its .ter, info {"material", "parts": [{"shape", "pos"}],
    "cells"} or None). gen(lat, lon): the block's heights function (as heightmap.generate drew keep["u"] with), raw: the
    natural ground; keep: heightmap.generate's (u [j, i]); layers: the block's surface layer per sample (surface.MATERIALS
    indices, [j, i]; None: one). The pieces are written to out_dir as <stem>_<px>_<py>.cdae (their names in "parts",
    relative: earthd.terrain_msg makes them the game's paths), each cell drawn with its layer's material (the mod gives
    it the block's photo and the layer's detail texture: the terrain's look, terrain.lua)."""
    import fast
    fast.require(124, "fine_cells")
    import cdae
    u = np.ascontiguousarray(keep["u"], np.float64)
    xs, ys = heightmap.sample_coords(cx, cy, size)
    r = np.ascontiguousarray(heightmap.surface_u(frame, raw, xs, ys), np.float64)
    cells = np.asarray(fast._ef.fine_cells(u, r, SHARP_DZ_M, GRADED_DZ_M, GROW))
    e = EDGE_CELLS
    cells[:e, :] = 0
    cells[-e:, :] = 0
    cells[:, :e] = 0
    cells[:, -e:] = 0
    J, I = np.nonzero(cells)
    if not len(J):
        return np.zeros(u.shape, bool), None
    if len(J) > MAX_CELLS:
        c = np.stack([u[J, I], u[J, I + 1], u[J + 1, I], u[J + 1, I + 1]])
        keep_ = np.sort(np.argsort(-(c.max(0) - c.min(0)), kind="stable")[:MAX_CELLS])
        cells = np.zeros_like(cells)
        cells[J[keep_], I[keep_]] = 1
        J, I = np.nonzero(cells)
    cells8 = np.ascontiguousarray(cells, np.uint8)
    holes = np.asarray(fast._ef.fine_holes(cells8)).astype(bool)
    # (the rim the terrain still draws, pale: the mesh over it too, RIM_LIFT_M up inside)
    rim = np.asarray(fast._ef.fine_rim(cells8))
    rim[:e - 1, :] = 0
    rim[:, :e - 1] = 0
    RJ, RI = np.nonzero(rim)
    nfine = len(J)
    J, I = np.r_[J, RJ], np.r_[I, RI]
    is_rim = np.r_[np.zeros(nfine, np.uint8), np.ones(len(RJ), np.uint8)]
    x0, y0, sq = float(xs[0]), float(ys[0]), heightmap.SQUARE
    ji = np.ascontiguousarray(np.c_[J, I], np.int64)
    pts = np.asarray(fast._ef.fine_points(x0, y0, sq, K, ji))
    la, lo, _ = frame.enu_to_geodetic(pts[:, 0], pts[:, 1], np.zeros(len(pts)))
    la, lo = np.asarray(la, np.float64), np.asarray(lo, np.float64)
    z = np.asarray(frame.geodetic_to_enu(la, lo, np.asarray(gen(la, lo), np.float64))[2], np.float64)
    w = (K + 1) ** 2
    P, T = fast._ef.fine_mesh(x0, y0, sq, K, ji, np.ascontiguousarray(z.reshape(len(J), w)), u, cells8, is_rim,
                              RIM_LIFT_M)
    P, T = np.asarray(P), np.asarray(T)
    extent = size * sq                                  # (the block's photo covers it from its south-west corner)
    import surface as surfacemod
    if layers is not None:
        lay = np.asarray(layers, np.int64)[J, I]
        bad = (lay < 0) | (lay >= len(surfacemod.MATERIALS))        # (a tunnel's hole: 255)
        lay[bad] = surfacemod.DIRT
    else:
        lay = np.full(len(J), surfacemod.DIRT, np.int64)
    used = sorted(set(int(v) for v in np.unique(lay)))
    b = bank(cx, cy, size)
    mats = [material_name(b, k) for k in used]
    tri_layer = np.repeat(np.searchsorted(used, lay), 2 * K * K)    # (each cell's triangles: fine_mesh's order)
    # the pieces: cells by the PIECE_M square their south-west corner lies in (from the block's corner)
    per = int(round(PIECE_M / sq))
    # (the rim cells in pieces of their own, drawn only: the terrain under them collides, the mesh 4 cm over it made
    # ridges across the lanes)
    pk = (J // per) * 100000 + (I // per) + is_rim.astype(np.int64) * 10 ** 10
    parts = []
    tpc = 2 * K * K                                     # triangles per cell (fine_mesh's order: cell by cell)
    for key in np.unique(pk):
        sel = np.flatnonzero(pk == key)
        vi = (sel[:, None] * w + np.arange(w)[None, :]).ravel()
        ti = (sel[:, None] * tpc + np.arange(tpc)[None, :]).ravel()
        remap = np.full(len(P), -1, np.int64)
        remap[vi] = np.arange(len(vi))
        rimp = key >= 10 ** 10
        kk = int(key % 10 ** 10)
        pj, pi = kk // 100000, kk % 100000
        ox, oy = x0 + pi * PIECE_M, y0 + pj * PIECE_M
        pos = P[vi] - [ox, oy, 0.0]
        uv = np.c_[(P[vi, 0] - x0) / extent, (P[vi, 1] - y0) / extent]
        uv2 = pos[:, :2] / DETAIL_M                     # (world-locked: piece corners are on the 2 m lattice)
        name = f"{stem}_{pi}_{pj}{'r' if rimp else ''}.cdae"
        cdae.write(os.path.join(out_dir, name), pos, remap[T[ti]], uvs=uv, uvs2=uv2, material=mats,
                   tri_materials=tri_layer[ti], object_name="fine")
        parts.append({"shape": name, "pos": [round(ox, 3), round(oy, 3), 0.0]} | ({"collide": False} if rimp else {}))
    return holes, {"materials": [[material_name(b, k), surfacemod.MATERIALS[k][0]] for k in used], "bank": b,
                   "parts": parts, "cells": int(nfine), "rim": int(len(RJ))}
