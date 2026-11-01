"""Actual-source tunnel regressions: Obor and held-out tracker #464, plus source splits.

Fixtures contain unchanged Overture 2026-09-23.1 roads (including source record IDs and
linear-referenced flags) and actual zoom-15 DEM samples on an 8 m grid, captured 2026-11-01.
DEM interpolation is fixture-only, not a production fallback or surveyed underground height.
Run: python3 -m unittest discover -s server/tests -p test_tunnel_layering.py
Live-source geometry/mesh report: python3 tools/tunnelprobe.py 44.4499 26.1244 --cache-dir CACHE
Held-out: python3 tools/tunnelprobe.py -31.943509 115.853865 --cache-dir CACHE
These geometry tests do not claim the whole Obor interchange or in-game collision is fixed.
"""
import sys
import unittest
from pathlib import Path

from unittest import mock
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import geodesy
import grading
import mapgeom
import navgraph
import roadmesh
import roads
from test_roads import FRAME, features, rules


OBOR = ("75d3e458-4976-4995-8786-6278090ece97", "7d55c224-9513-4802-8ec1-7621b8d1d8b9")
NORTHBRIDGE = ("235567c9-1d08-4994-b480-4203b19db1fa", "bb7f3e16-ec3b-489b-b564-d9fa8215af7d")


def captured(name):
    with np.load(Path(__file__).with_name("fixtures") / (name + ".npz"), allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    frame = geodesy.EnuFrame(*arrays["center"])
    feats = mapgeom.Features.from_arrays("roads", arrays, "roads.")
    gx, gy, grid = arrays["grid_x"], arrays["grid_y"], arrays["grid_z"]

    def elev(la, lo):
        shape = np.shape(la)
        x, y, _ = frame.geodetic_to_enu(np.asarray(la).ravel(), np.asarray(lo).ravel(), 0.0)
        if np.any((x < gx[0]) | (x > gx[-1]) | (y < gy[0]) | (y > gy[-1])):
            raise ValueError("source probe exceeded captured DEM rectangle")
        ix = np.clip(np.searchsorted(gx, x) - 1, 0, len(gx) - 2)
        iy = np.clip(np.searchsorted(gy, y) - 1, 0, len(gy) - 2)
        tx, ty = (x - gx[ix]) / (gx[ix + 1] - gx[ix]), (y - gy[iy]) / (gy[iy + 1] - gy[iy])
        z = ((1 - tx) * (1 - ty) * grid[iy, ix] + tx * (1 - ty) * grid[iy, ix + 1] +
             (1 - tx) * ty * grid[iy + 1, ix] + tx * ty * grid[iy + 1, ix + 1])
        return z.reshape(shape)
    tp = roads.TunnelProfiles(frame, lambda key: feats, elev)
    tp.raw = elev
    return frame, feats, elev, tp


def selected(feats, ids):
    """Retain exact captured source rows/rules without changing geometry or fractions."""
    rows = sorted(feats.index(fid) for fid in ids)
    remap = {old: new for new, old in enumerate(rows)}
    coords, po, ro, co = [], [0], [0], [0]
    for i in rows:
        for part in feats.parts(i):
            for ring in part:
                coords.extend(ring)
                co.append(len(coords))
            ro.append(len(co) - 1)
        po.append(len(ro) - 1)
    tables = {}
    for name, table in feats.rules.items():
        keep = np.isin(table.feature, rows)
        tables[name] = mapgeom.RuleTable(np.array([remap[int(i)] for i in table.feature[keep]], np.int32),
                                        table.value[keep], table.start[keep], table.end[keep])
    return mapgeom.Features("roads", feats.ids[rows], feats.kind[rows], np.array(po), np.array(ro), np.array(co),
                            np.asarray(coords), attrs={k: v[rows] for k, v in feats.attrs.items()}, rules=tables)


class SourceTunnelLayers(unittest.TestCase):
    def setUp(self):
        self.enterContext(mock.patch("roadz.PROVIDER", None))

    def test_obor_both_mapped_bores_are_continuously_below_the_roof(self):
        frame, feats, elev, tp = captured("obor_tunnels")
        box = (-500.0, -500.0, 500.0, 500.0)
        closed = tp.parts([feats], box)
        for source in OBOR:
            with self.subTest(source=source):
                parts = [p for p in closed if p[0].startswith(source + ".")]
                self.assertTrue(parts, "baseline lower DEM hull erased this mapped bore entirely")
                i = feats.index(source)
                ring = roads.tunnel_parts(feats, i)[0]
                e, n = mapgeom.enu_xy(feats, frame, ring)
                xy = roads.densify(np.c_[e, n], 2.0)
                floors = np.array([roads.tunnel_floor_at(parts, *p) for p in xy], dtype=float)
                self.assertTrue(np.isfinite(floors).all(), "no open gap inside the mapped bore")
                la, lo, _ = frame.enu_to_geodetic(xy[:, 0], xy[:, 1], 0.0)
                ground = frame.geodetic_to_enu(la, lo, elev(la, lo))[2]
                self.assertGreater(float(np.min(ground - floors)), roadmesh.TUNNEL_H + 0.5)
        chunks = roads._tunnel_chunks(tp, [feats], box)
        self.assertTrue(any(c.get("tunnel") and c.get("deck") for c in chunks))
        mesh = roadmesh.tunnel(chunks, (0.0, 0.0, 0.0))
        self.assertIsNotNone(mesh)
        self.assertTrue(np.isfinite(mesh[0]).all())
        # The covered interior cannot punch the surface road out of the terrain.
        xs = np.arange(-400.0, 400.0, 2.0)
        xx, yy = np.meshgrid(xs, xs)
        la, lo, _ = frame.enu_to_geodetic(xx.ravel(), yy.ravel(), 0.0)
        u = frame.geodetic_to_enu(la, lo, elev(la, lo))[2].reshape(xx.shape)
        holes = roads.tunnel_hole_mask(closed, u, 0.0, 0.0, 2.0, tp.heights, tp.mouths)
        for la, lo in ((44.4496548171, 26.1242012230), (44.4498027493, 26.1244340945)):
            x, y, _ = frame.geodetic_to_enu(la, lo, 0.0)
            ix, iy = int(np.floor((x - xs[0]) / 2)), int(np.floor((y - xs[0]) / 2))
            self.assertFalse(holes[iy, ix], "actual Piața Obor surface crossing remains solid")

    def test_northbridge_both_main_bores_exist_and_source_joint_is_not_a_portal(self):
        _frame, feats, _elev, tp = captured("northbridge_tunnels")
        parts = tp.parts([feats], (-300, -300, 300, 300))
        for source in NORTHBRIDGE:
            bore = [p for p in parts if p[0].startswith(source + ".")]
            self.assertTrue(bore, source)
            for fid, _sp, xy, z, _lo in bore:
                self.assertGreater(float(np.min(tp._raw_ground(np.asarray(xy)) - z)), roadmesh.TUNNEL_H + 0.5)
                self.assertTrue(np.isfinite(z).all())
        # The exact source joint at 115.8540771,-31.9435998 joins the long bore to its
        # mapped approach part (72763601...), not a new portal across the carriageway.
        joined = [p for p in parts if p[0].startswith("235567c9-") or p[0].startswith("72763601-")]
        joint = tp.frame.geodetic_to_enu(-31.9435998, 115.8540771, 0.0)
        ends = []
        for fid, _sp, xy, z, _lo in joined:
            for at, mouth in zip((0, -1), tp.mouth(fid)):
                if np.linalg.norm(np.asarray(xy)[at] - joint[:2]) < 0.01:
                    self.assertFalse(mouth)
                    ends.append(float(z[at]))
        self.assertGreaterEqual(len(ends), 2)
        self.assertLess(np.ptp(ends), 1e-5)

    def test_grading_and_meshes_use_the_same_obor_sections(self):
        frame, feats, elev, tp = captured("obor_tunnels")
        feats = selected(feats, (*OBOR, "6d3cc7e9-7354-45d6-80f8-0085c3572f69"))
        tp = roads.TunnelProfiles(frame, lambda key: feats, elev)
        tp.raw = elev
        from types import SimpleNamespace
        planner = grading._Planner(frame, SimpleNamespace(cell=lambda *a, **kw: SimpleNamespace(layers={"roads": feats})), elev)
        secs = planner._tunnel_sections(feats)
        closed = tp.parts([feats], (-500, -500, 500, 500))
        for source in OBOR:
            expected = [p for p in closed if p[0].startswith(source + ".")]
            actual = [s for s in secs if s[0].startswith(source + ".") and s[4]]
            self.assertEqual(len(actual), len(expected))
            for a, b in zip(actual, expected):
                np.testing.assert_allclose(a[2], b[2], atol=1e-6)
                measured = frame.enu_to_geodetic(b[2][:, 0], b[2][:, 1], b[3])[2]
                np.testing.assert_allclose(a[3], measured, atol=1e-6)


class SourceJointAndApproach(unittest.TestCase):
    def setUp(self):
        self.enterContext(mock.patch("roadz.PROVIDER", None))

    def test_split_hill_has_one_floor_and_only_two_outer_portals(self):
        f = features([[(-200, 0), (0, 0)], [(0, 0), (200, 0)]], ["primary"] * 2,
                     flags=[(0, "is_tunnel"), (1, "is_tunnel")])
        def elev(la, lo):
            e, _n, _z = FRAME.geodetic_to_enu(la, lo, 0.0)
            return 10.0 + np.maximum(0.0, 60.0 - np.abs(e) * 0.3)
        tp = roads.TunnelProfiles(FRAME, lambda key: f, elev)
        closed = tp.parts([f], (-300, -100, 300, 100))
        at_joint = []
        for fid, _sp, xy, z, _lon in closed:
            for at, mouth in zip((0, -1), tp.mouth(fid)):
                if np.linalg.norm(xy[at]) < 0.01:
                    self.assertFalse(mouth)
                    at_joint.append(float(z[at]))
        self.assertEqual(len(at_joint), 2)
        self.assertLess(np.ptp(at_joint), 1e-6)
        self.assertLess(max(at_joint), 11.0)

    def test_crossing_constraint_is_not_a_blanket_lowering(self):
        s = np.arange(0.0, 600.1, 2.0)
        ground = np.full(len(s), 20.0)
        mapped = (s >= 200) & (s <= 400)
        kinds = np.zeros(len(s), np.int64)
        floor, closed = roads.mapped_tunnel_floor(s, ground, kinds, mapped, [(290, 310)])
        self.assertTrue((closed[mapped] == 2).all())
        self.assertTrue((closed[~mapped] == 0).all())
        self.assertTrue(np.allclose(floor[mapped], 12.0))
        self.assertEqual(floor[0], 20.0)
        self.assertEqual(floor[-1], 20.0)
        self.assertLessEqual(float(np.max(np.abs(np.diff(floor)) / np.diff(s))), roads.TUNNEL_APPROACH_GRADE + 1e-6)
        untouched, opened = roads.mapped_tunnel_floor(s, ground, kinds, mapped, [])
        np.testing.assert_array_equal(untouched, ground)
        np.testing.assert_array_equal(opened, kinds)

    def test_approach_updates_only_the_source_road_not_a_surface_crossing(self):
        from types import SimpleNamespace
        planner = grading._Planner(FRAME, SimpleNamespace(), lambda la, lo: np.full(np.shape(la), 20.0))
        planner._tunnel_owners = {"bore.0": {"own"}}
        own = grading.Plan(np.array([[-20., 0.], [0., 0.], [20., 0.]]), np.full(3, 20.), 3.5)
        surface = grading.Plan(np.array([[0., -20.], [0., 0.], [0., 20.]]), np.full(3, 20.), 3.5)
        secs = [("bore.0", roads.Spec(7., "paved", False, "primary"), own.xy.copy(), np.array([20., 16., 12.]), 0)]
        planner._tunnel_approaches([("own", own), ("surface", surface)], secs)
        np.testing.assert_allclose(own.z, [20, 16, 12])
        np.testing.assert_allclose(surface.z, [20, 20, 20])

    def test_navigation_uses_solved_floor_without_joining_surface_crossing(self):
        f = features([[(-200, 0), (200, 0)], [(0, -100), (0, 100)]], ["primary", "secondary"],
                     flags=[(0, "is_tunnel", .25, .75)])
        f.rules["connectors"] = rules([(0, "a", 0, 0), (0, "b", 1, 1), (1, "c", 0, 0), (1, "d", 1, 1)])
        floors = {"id000": [(np.array([[-200., 0.], [-100., 0.], [100., 0.], [200., 0.]]),
                              np.array([20., 12., 12., 20.]))]}
        graph = navgraph.build(FRAME, [f], lambda la, lo: np.full(np.shape(la), 20.), (0, 0), tunnel_floors=floors)
        below = [p for name, p in graph["nodes"].items() if "id000" in name and abs(p[0]) <= 100]
        self.assertTrue(below)
        self.assertTrue(all(abs(p[2] - 12.) < .05 for p in below))
        self.assertFalse(any("id000" in e[0] and "id001" in e[1] for e in graph["edges"]))


if __name__ == "__main__":
    unittest.main()
