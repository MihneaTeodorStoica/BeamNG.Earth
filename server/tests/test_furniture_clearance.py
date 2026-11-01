"""Consumer regressions for full, level-aware street-furniture collision clearance (no map downloads)."""
import json
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import furniture  # noqa: E402
import roads  # noqa: E402
from test_furniture import road  # noqa: E402
from test_roads import features  # noqa: E402


def row(group, x, y, z=0.0, yaw=0.0, height=None):
    return [group, x, y, z, yaw, 1.0,
            furniture.COLLISION_BOXES[group][0] if height is None else height, 0.0]


class ColliderClearanceTest(unittest.TestCase):
    def test_sign_face_not_just_pivot_and_yaw(self):
        clear = furniture.RoadClearance([(4.0, np.array([[-100., 0., 0.], [100., 0., 0.]]))])
        # Pivot 0.4 m off the pavement: the turned 0.46 m-wide face intrudes, the unturned face does not.
        self.assertTrue(clear.clear(row('stop', 0., 4.4)))
        self.assertFalse(clear.clear(row('stop', 0., 4.4, yaw=90.)))

    def test_rotated_tower_corner_not_just_pivot(self):
        clear = furniture.RoadClearance([(4.5, np.array([[-100., 0., 0.], [100., 0., 0.]]))])
        self.assertTrue(clear.clear(row('pylon', 0., 10.)))
        self.assertFalse(clear.clear(row('pylon', 0., 10., yaw=45.)))
        self.assertGreater(10., 4.5, 'the pivot is outside both carriageways')

    def test_native_scale_not_the_unused_row_scale(self):
        clear = furniture.RoadClearance([(4., np.array([[-100., 0., 0.], [100., 0., 0.]]))])
        small = row('stop', 0., 4.8, yaw=90.)
        large = row('stop', 0., 4.8, yaw=90., height=6.)
        small[5] = 1000.  # unused by trees.lua:M.pick
        self.assertTrue(clear.clear(small))
        self.assertFalse(clear.clear(large))

    def test_real_overhead_arm_clearance_and_higher_crossing(self):
        # Stock lamp collider's arm starts at z=7.124; its support is outside x=0..8 pavement.
        lamp = row('streetlight', -0.5, 0.)
        ground = [(4., np.array([[4., -100., 0.], [4., 100., 0.]]))]
        self.assertTrue(furniture.RoadClearance(ground).clear(lamp))
        raised = [(4., np.array([[4., -100., 2.], [4., 100., 2.]]))]
        self.assertFalse(furniture.RoadClearance(raised).clear(lamp))

    def test_grade_separated_objects_above_and_below(self):
        ground = [(4., np.array([[-100., 0., 0.], [100., 0., 0.]]))]
        self.assertTrue(furniture.RoadClearance(ground).clear(row('stop', 0., 0., z=10.)))
        self.assertFalse(furniture.RoadClearance(ground).clear(row('stop', 0., 0., z=6.)))
        deck = (np.array([[-100., 0., 50.], [100., 0., 50.]]), 4.)
        self.assertTrue(furniture.RoadClearance([], [deck]).clear(row('pylon', 0., 0.)))
        low = (np.array([[-100., 0., 20.], [100., 0., 20.]]), 4.)
        self.assertFalse(furniture.RoadClearance([], [low]).clear(row('pylon', 0., 0.)))

    def test_all_segments_not_just_the_nearest_branch(self):
        # Closest branch is 30 m overhead, second branch is at grade and clips the sign's face.
        line = np.array([[-100., 0., 30.], [100., 0., 30.], [100., 1., 0.], [-100., 1., 0.]])
        self.assertFalse(furniture.RoadClearance([(1., line)]).clear(row('stop', 0., 0.)))

    def test_rising_segment_clipped_at_collision_height(self):
        # Nearest XY point is too low to hit the hanging lamp; farther along the SAME segment is not.
        line = np.array([[-3., 0., 0.], [3., 0., 14.]])
        self.assertFalse(furniture.RoadClearance([(2., line)]).clear(row('polelamp', 0., 0., z=8.)))

    def test_segment_crossing_rectangle_without_any_endpoint_inside(self):
        a, b = np.array([[-10., 0.]]), np.array([[10., 0.]])
        d = furniture._box_distance2(a, b, np.array([-1., -1.]), np.array([1., 1.]))
        self.assertEqual(d.tolist(), [0.])

    def test_index_omits_distant_roads_but_finds_long_segments(self):
        lines = [(1., np.array([[1000.+i*32., -100., 0.], [1000.+i*32., 100., 0.]])) for i in range(1000)]
        lines.append((4., np.array([[-5000., 0., 0.], [5000., 0., 0.]])))
        clear = furniture.RoadClearance(lines)
        pick = clear.candidates(np.array([-2., -2.]), np.array([2., 2.]))
        self.assertEqual(pick.tolist(), [1000])
        self.assertFalse(clear.clear(row('stop', 0., 0.)))

    def test_catalog_covers_measured_stock_collision_bounds(self):
        fixture = json.loads((Path(__file__).parent / 'fixtures' / 'furniture_collision_bounds.json').read_text())
        for group, asset in fixture['assets'].items():
            if group not in furniture.COLLISION_BOXES:
                continue
            boxes = furniture.COLLISION_BOXES[group][1]
            for source in asset['envelopes']:
                self.assertTrue(any(np.all(np.asarray(b[:3]) <= np.asarray(source[:3]) + 1e-9)
                                    and np.all(np.asarray(b[3:]) >= np.asarray(source[3:]) - 1e-9)
                                    for b in boxes), (group, source))


class PlacementClearanceTest(unittest.TestCase):
    def fixture(self):
        base = road('clearance-fixture', 'residential', True)
        side = 1 if furniture._h01(base[0], 'side') < .5 else -1
        walks = [{'fid': base[0], 'pts': roads.offset_polyline(base[2], side * 5.25),
                  'side': side, 'w': 2.5, 'caps': (False, False)}]
        box = (0., 0., 512., 256.)
        initial = furniture.place([base], walks, box)
        target = next(r for r in initial if r[0] == furniture.G_LAMP)
        crossing = ('crossing', roads.Spec(8., 'paved', False, 'service', False),
                    np.array([[target[1], -100., 0.], [target[1], 300., 0.]]), 0.)
        return base, walks, box, crossing, target

    def test_sidewalk_lamps_clear_crossing_service_road(self):
        base, walks, box, crossing, target = self.fixture()
        placed = furniture.place([base, crossing], walks, box)
        lamps = [r for r in placed if r[0] == furniture.G_LAMP]
        self.assertTrue(lamps, 'unobstructed lamps remain')
        self.assertNotIn(target, lamps, 'old sidewalk placement did not check any carriageway')
        self.assertTrue(all(abs(r[1] - target[1]) >= 4.15 for r in lamps))
        self.assertEqual(placed, furniture.place([crossing, base], walks, box), 'source order does not affect placement')

    def test_tile_ownership_does_not_change_clearance(self):
        base, walks, box, crossing, _ = self.fixture()
        whole = furniture.place([base, crossing], walks, box)
        split = (furniture.place([base, crossing], walks, (0., 0., 256., 256.))
                 + furniture.place([base, crossing], walks, (256., 0., 512., 256.)))
        self.assertEqual(sorted(whole), sorted(split))
        self.assertEqual(len({tuple(r) for r in split}), len(split), 'no duplicate owner at the seam')

    def test_own_hairpin_is_not_exempt(self):
        fid = 'hairpin'
        side = 1 if furniture._h01(fid, 'pside') < .5 else -1
        line = np.array([[-300., 100., 0.], [600., 100., 0.],
                         [600., 100.+side*10., 0.], [-300., 100.+side*10., 0.]])
        bent = (fid, roads.Spec(8., 'paved', False, 'tertiary', False), line, 0.)
        rows = furniture.place([bent], [], (0., 0., 256., 256.), wires=[])
        # These two verges face the 2 m gap between carriageways. A pole 2.4 m off either edge blocks the other.
        self.assertFalse(any(r[0] == furniture.G_POLE for r in rows), rows)
        self.assertTrue(any(r[0] == furniture.G_POLE for r in furniture.place(
            [road(fid, 'tertiary', False, y=100.)], [], (0., 0., 256., 256.))))

    def test_rejected_town_poles_do_not_leave_spans_attached_to_them(self):
        base, walks, box, _, _ = self.fixture()
        baseline_wires = []
        baseline = furniture.place([base], walks, box, town=True, wires=baseline_wires)
        target = next(r for r in baseline if r[0] == furniture.G_POLE)
        crossing = ('crossing', roads.Spec(8., 'paved', False, 'service', False),
                    np.array([[target[1], -100., 0.], [target[1], 300., 0.]]), 0.)
        wires = []
        rows = furniture.place([base, crossing], walks, box, town=True, wires=wires)
        self.assertNotIn(target, rows)
        self.assertTrue(wires)
        # The three wire arms can be offset in x/y; compare the removed pole's original attachment ends instead.
        bad = [w[0] for w in baseline_wires if abs(w[0][0] - target[1]) < 1.]
        for wire in wires:
            self.assertFalse(any(np.allclose(wire[0], p) or np.allclose(wire[-1], p) for p in bad))

    def test_mapped_width_not_road_class_default(self):
        f = features([[(-100., 0.), (100., 0.)]], ['residential'])
        f.attrs['osm_width'] = np.array([24.])
        spec = roads.road_specs(f)[0]
        self.assertEqual(spec.width, 24.)
        clear = furniture.RoadClearance([(spec.width / 2, np.array([[-100., 0., 0.], [100., 0., 0.]]))])
        self.assertFalse(clear.clear(row('stop', 0., 10.)))
        self.assertTrue(clear.clear(row('stop', 0., 13.)))

    def test_long_crossing_segment_is_not_culled_by_its_endpoints(self):
        own = (4., np.array([[0., -100.], [0., 100.]]))
        cross = (4., np.array([[-5000., 6.5], [5000., 6.5]]))
        q = furniture.signal_spot(np.zeros(2), np.array([0., 1.]), np.array([1., 0.]), 4., 4., [own, cross])
        self.assertIsNotNone(q, 'a reachable safe spot exists; blanket omission is not a fix')
        self.assertGreaterEqual(abs(q[1] - 6.5), 4.6)

    def test_relocated_junction_sign_uses_support_height_at_its_spot(self):
        spec = roads.Spec(8., 'paved', True, 'primary', False)
        rs = [('s', spec, np.array([[0., 0., 0.], [0., -100., -20.]]), 0.),
              ('e', spec, np.array([[0., 0., 0.], [100., 0., 0.]]), 0.),
              ('w', spec, np.array([[0., 0., 0.], [-100., 0., 0.]]), 0.)]
        rows = furniture.junction_props(rs, (-50., -50., 50., 50.), signals=np.zeros((1, 2)),
                                        ground=lambda x, y: .2*np.asarray(y))
        self.assertTrue(rows)
        for r in rows:
            self.assertAlmostEqual(r[3], .2*r[2]+.15, places=2)

    def test_true_overhead_wire_stays(self):
        deck = (np.array([[-100., 0., 8.], [100., 0., 8.]]), 5.5)
        high = np.array([[-20., 0., 20.], [20., 0., 20.]])
        low = np.array([[-20., 0., 7.9], [20., 0., 7.9]])
        wires = [high, low]
        furniture.under_decks([], [deck], wires)
        self.assertEqual(len(wires), 1)
        np.testing.assert_array_equal(wires[0], high)


if __name__ == '__main__':
    unittest.main()
