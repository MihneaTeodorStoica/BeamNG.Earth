"""Synthetic bridge scenarios (the user, 2026-10-31: "bridge widths dont always match the road widths"; a friend's
"test city" idea): small made-up road layouts on flat ground through the real spec and deck solvers, each checked for
what a driver sees: a deck as wide as the road it carries, decks of a divided road side by side without overlapping,
decks at the height of the road they continue. A bug found in a real place gets a scenario here (bug -> test -> fix).
.claude/devtools/bridgewidth.py measures the widths over a real area."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import roads  # noqa: E402
from test_approaches import flat_elev, FLAT  # noqa: E402
from test_roads import features, rules, FRAME  # noqa: E402

UNDER = [[(-200, 0), (200, 0)]]         # the street the bridges cross
BRIDGE = (0.45, 0.55)                   # each carried road's bridged range (of its length): over the street


def scenario(lines, classes, names=None, oneway=(), links=(), bridged=None):
    """Features of lines (+ the street under them) with the given classes, names, one-way and link roads; bridged:
    which lines carry a bridge over the street (default all)."""
    lines = list(lines) + UNDER
    classes = list(classes) + ["residential"]
    bridged = range(len(lines) - 1) if bridged is None else bridged
    flags = [(i, "is_bridge", *BRIDGE) for i in bridged]
    f = features(lines, classes, flags=flags, ids=[f"r{i}" for i in range(len(lines))])
    f.attrs["name"] = np.array(list(names or [""] * (len(lines) - 1)) + ["Under St"])
    f.rules["oneway"] = rules([(i, 1.0) for i in oneway], "float")
    f.rules["subclass"] = rules([(i, "link") for i in links])
    return f


def decks(f):
    """{fid: (spec, xy (k, 2), z (k,))} of the solved bridge parts."""
    bp = roads.BridgeProfiles(FRAME, lambda key: f, flat_elev)
    return {fid: (spec, np.asarray(xy, np.float64)[:, :2], np.asarray(z, np.float64))
            for fid, spec, xy, z, _lon in bp.parts([f], (-400, -400, 400, 400))}


class DeckWidthScenarios(unittest.TestCase):
    def assert_deck_as_wide_as_its_road(self, f):
        ground, bridge = roads.road_specs(f), roads.bridge_specs(f)
        n = 0
        for i, b in enumerate(bridge):
            if b is None:
                continue
            n += 1
            self.assertAlmostEqual(b.width, ground[i].width, delta=0.05, msg=f"r{i}: deck {b.width} m, road {ground[i].width} m")
        self.assertGreater(n, 0)
        return ground, bridge

    def test_two_way_road(self):
        self.assert_deck_as_wide_as_its_road(scenario([[(0, -300), (0, 300)]], ["primary"]))

    def test_one_way_ramp(self):
        self.assert_deck_as_wide_as_its_road(scenario([[(0, -300), (40, 300)]], ["motorway"], oneway=[0], links=[0]))

    def test_divided_motorway_mapped_one_way(self):
        f = scenario([[(-9, -300), (-9, 300)], [(9, 300), (9, -300)]], ["motorway"] * 2, names=["I-95", "I-95"],
                     oneway=[0, 1])
        self.assert_deck_as_wide_as_its_road(f)

    def test_divided_road_the_map_does_not_mark_one_way(self):
        """Two same-named carriageways side by side (dual_carriageways): the decks were the class's two-way width, 18.8 m
        decks on 12 m motorway carriageways in Baltimore (bridgewidth.py, 2026-10-31)."""
        f = scenario([[(-8, -300), (-8, 300)], [(8, 300), (8, -300)]], ["primary"] * 2, names=["Main", "Main"])
        ground, bridge = self.assert_deck_as_wide_as_its_road(f)
        self.assertTrue(ground[0].median and bridge[0].median, "(both read as one carriageway of a divided road)")
        self.assertTrue(bridge[0].oneway and bridge[1].oneway)
        gap = 16.0
        self.assertLessEqual((bridge[0].width + bridge[1].width) / 2, gap - roads.DUAL_MIN_MEDIAN_M + 1e-6,
                             "the two decks do not overlap")

    def test_close_carriageways_never_overlap(self):
        """Carriageways 7 m apart: each deck capped so the two keep a barrier's width between them (dual_cap)."""
        f = scenario([[(-3.5, -300), (-3.5, 300)], [(3.5, 300), (3.5, -300)]], ["motorway"] * 2,
                     names=["I-83", "I-83"], oneway=[0, 1])
        _, bridge = self.assert_deck_as_wide_as_its_road(f)
        self.assertLessEqual((bridge[0].width + bridge[1].width) / 2, 7.0 - roads.DUAL_MIN_MEDIAN_M + 1e-6)


class DeckHeightScenarios(unittest.TestCase):
    def test_divided_overpass_decks_side_by_side_at_one_height(self):
        f = scenario([[(-8, -300), (-8, 300)], [(8, 300), (8, -300)]], ["primary"] * 2, names=["Main", "Main"])
        d = decks(f)
        self.assertEqual(set(d), {"r0", "r1"})
        # (the headroom over the street comes with the grading's approach ramps, tested in test_approaches / test_grading:
        # on flat ground without a grader the decks rise only what their own span allows)
        tops = [float(z.max()) for _s, _xy, z in d.values()]
        self.assertGreater(min(tops), FLAT, "raised over the street")
        self.assertLess(abs(tops[0] - tops[1]), 0.5, "a divided road's decks at one height")
        for fid, (spec, xy, z) in d.items():
            self.assertAlmostEqual(z[0], FLAT, delta=0.6, msg=f"{fid} starts on its approach")
            self.assertAlmostEqual(z[-1], FLAT, delta=0.6, msg=f"{fid} ends on its approach")



def chain(widths, classes=None, bridged=(), extra=()):
    """Roads end to end along x (each 200 m, from x = -200 on) with these widths (the map's width rule), + extra lines."""
    lines = [[(-200 + 200 * k, 0), (200 * k, 0)] for k in range(len(widths))] + [ln for ln, _ in extra]
    classes = list(classes or ["primary"] * len(widths)) + [c for _, c in extra]
    f = features(lines, classes, widths=[(k, float(w)) for k, w in enumerate(widths)],
                 flags=[(k, "is_bridge", 0.0, 1.0) for k in bridged], ids=[f"r{k}" for k in range(len(lines))])
    f.attrs["name"] = np.array([""] * len(lines))
    return f


def width_at(f, i, x, y=0.0):
    sp = roads.road_specs(f)[i]
    la, lo, _ = FRAME.enu_to_geodetic(np.array([float(x)]), np.array([float(y)]), 0.0)
    ws = roads.taper_widths(sp.width, roads.road_tapers(f)[i], lo, la)
    return sp.width if ws is None else float(ws[0])


class LaneTaperScenarios(unittest.TestCase):
    """Where the map's lane count changes the drawn width runs over a taper instead of stepping (the user, 2026-10-31:
    "the bridge geometry as lanes merge and split is rough")."""

    def test_two_ground_roads_meet_half_way(self):
        f = chain([10.0, 16.0])
        self.assertAlmostEqual(width_at(f, 0, 0.0), 13.0, delta=0.05, msg="the narrow one widens to half way")
        self.assertAlmostEqual(width_at(f, 1, 0.0), 13.0, delta=0.05, msg="the wide one narrows to half way")
        span = float(np.clip(6.0 * roads.TAPER_M_PER_M, *roads.TAPER_M))
        self.assertAlmostEqual(width_at(f, 0, -span - 1), 10.0, delta=0.01, msg="its own width past the taper")
        self.assertAlmostEqual(width_at(f, 1, span + 1), 16.0, delta=0.01)
        ws = [width_at(f, 0, x) for x in np.linspace(-span, 0.0, 30)]
        self.assertTrue(np.all(np.diff(ws) >= -1e-9), "widening steadily")
        self.assertLess(float(np.abs(np.diff(ws)).max()), 6.0 / 2 / 30 * 2.0, "no step (smoothstep: at most 1.5x the mean rate)")

    def test_ground_road_meets_a_wider_deck_at_its_width(self):
        """The deck keeps its width (its parapets and collision): the road onto it takes all of the change."""
        f = chain([10.0, 16.0], bridged=[1])
        self.assertAlmostEqual(width_at(f, 0, 0.0), roads.bridge_specs(f)[1].width, delta=0.05)
        self.assertEqual(roads.road_tapers(f)[1], (), "no taper on the deck")

    def test_small_change_and_side_roads_keep_their_width(self):
        self.assertEqual(list(roads.road_tapers(chain([10.0, 10.2]))), [(), ()], "under TAPER_MIN_DW: as before")
        f = chain([10.0], extra=[([(0.0, 0.0), (0.0, 200.0)], "primary")])      # a side road from its end, at 90 deg
        self.assertEqual(roads.road_tapers(f)[0], (), "a junction's arm is not a continuation")
        f = chain([10.0, 16.0], extra=[([(0.0, 0.0), (0.0, 200.0)], "residential")])
        self.assertEqual(list(roads.road_tapers(f))[:2], [(), ()], "across a junction: as real streets, no taper")
        f = chain([10.0, 16.0], classes=["primary", "secondary"])
        self.assertEqual(roads.road_tapers(f)[0], (), "another class: a junction, not the same road")

    def test_edge_lines_follow_the_taper(self):
        f = chain([10.0, 16.0], classes=["trunk", "trunk"])
        sp = roads.road_specs(f)[0]
        x = np.linspace(-150.0, 0.0, 76)
        pts = np.c_[x, np.zeros_like(x), np.zeros_like(x)]
        la, lo, _ = FRAME.enu_to_geodetic(x, np.zeros_like(x), 0.0)
        ws = roads.taper_widths(sp.width, roads.road_tapers(f)[0], lo, la)
        self.assertIsNotNone(ws)
        edges = [ln for look, _w, ln in roads.extras(sp, pts, -76.6, ws=ws) if look.get("kind") == "edge"]
        self.assertEqual(len(edges), 2)
        for ln in edges:
            off = np.abs(ln[:, 1])
            self.assertAlmostEqual(float(off[0]), 10.0 / 2 - roads.EDGE_LINE["inset"], delta=0.05)
            self.assertAlmostEqual(float(off[-1]), 13.0 / 2 - roads.EDGE_LINE["inset"], delta=0.05)
        plain = [ln for look, _w, ln in roads.extras(sp, pts, -76.6) if look.get("kind") == "edge"]
        self.assertAlmostEqual(float(np.abs(plain[0][-1, 1])), 10.0 / 2 - roads.EDGE_LINE["inset"], delta=0.05,
                               msg="(without ws: as before)")


def urban(la, lo):
    return np.ones(np.shape(la), bool)


class DeckSidewalkScenarios(unittest.TestCase):
    """A city street's bridge carries its sidewalks (the user, 2026-10-31, East Pratt Street over the Jones Falls): the
    deck wider by them, a concrete strip each side of the carriageway."""

    def test_city_street_bridge_carries_its_sidewalks(self):
        f = features([[(-200, 0), (200, 0)]], ["residential"], flags=[(0, "is_bridge", 0.4, 0.6)], ids=["r0"])
        walk = roads.deck_walks(f, 0, roads.road_specs(f, urban))
        self.assertIsNotNone(walk)
        wm, sides = walk
        self.assertEqual(set(sides), {1, -1})
        spec = roads.bridge_specs(f)[0]
        pts = np.c_[np.linspace(-40, 40, 9), np.zeros(9), np.full(9, 15.0)]
        chunks = roads._deck_chunks([("r0", spec, pts, -76.6)], (-100, -100, 100, 100), walks={"r0": walk})
        deck = [c for c in chunks if c.get("deck") and not c.get("ctx")]
        self.assertTrue(deck)
        self.assertAlmostEqual(deck[0]["w"], spec.width + 2 * wm, delta=0.01, msg="the slab wider by both sidewalks")
        strips = [c for c in chunks if c["m"] == roads.DECK_WALK["m"]]
        self.assertEqual(len(strips), 2)
        for c in strips:
            y = abs(np.asarray(c["n"])[0][1])
            self.assertAlmostEqual(y, spec.width / 2 + wm / 2, delta=0.05, msg="beside the carriageway")
            self.assertAlmostEqual(np.asarray(c["n"])[0][2], 15.0 + roads.DECK_WALK_LIFT_M, delta=1e-3)

    def test_motorway_and_rural_bridges_have_none(self):
        f = features([[(-200, 0), (200, 0)]], ["motorway"], flags=[(0, "is_bridge", 0.4, 0.6)], ids=["r0"])
        self.assertIsNone(roads.deck_walks(f, 0, roads.road_specs(f, urban)))
        f = features([[(-200, 0), (200, 0)]], ["residential"], flags=[(0, "is_bridge", 0.4, 0.6)], ids=["r0"])
        self.assertIsNone(roads.deck_walks(f, 0, roads.road_specs(f)), "rural: no sidewalks on the ground either")

    def test_a_road_all_bridge_takes_the_sidewalks_of_the_street_going_on(self):
        f = features([[(-200, 0), (0, 0)], [(0, 0), (60, 0)]], ["residential"] * 2, flags=[(1, "is_bridge", 0.0, 1.0)],
                     ids=["r0", "r1"])
        specs = roads.road_specs(f, urban)
        self.assertIsNone(specs[1], "(all bridge: no ground spec)")
        walk = roads.deck_walks(f, 1, specs)
        self.assertIsNotNone(walk)
        self.assertEqual(set(walk[1]), {1, -1})

    def test_one_side_offsets_the_deck(self):
        spec = roads.Spec(width=10.0, kind="paved", urban=False, cls="trunk", oneway=True)
        self.assertEqual(roads._deck_frame(spec, (2.0, (-1,))), (12.0, -1.0))
        self.assertEqual(roads._deck_frame(spec, None), (10.0, 0.0))


class LaneFlowScenarios(unittest.TestCase):
    """One-way pieces the map gives no lane count take theirs from what flows in or out (I-95 by the Middle Branch: a
    4-lane deck took a ramp and went on at the 3-lane default width, 11.8 m after 18.8 m)."""

    def flow(self, lines, lanes, links=()):
        f = features(lines, ["motorway"] * len(lines), ids=[f"r{k}" for k in range(len(lines))])
        f.rules["oneway"] = rules([(k, 1.0) for k in range(len(lines))], "float")
        f.rules["subclass"] = rules([(k, "link") for k in links])
        f.attrs["osm_lanes"] = np.array([np.nan if v is None else float(v) for v in lanes])
        return roads.lane_flow(f), roads.road_specs(f)

    def test_merge_adds_the_ramp(self):
        got, specs = self.flow([[(-300, 0), (0, 0)], [(-200, -150), (0, 0)], [(0, 0), (300, 0)]], [4, None, None], links=[1])
        self.assertEqual(got[2], 5, "4 lanes + the ramp's one")
        self.assertGreater(specs[2].width, specs[0].width, "wider than before the merge, not the class default")

    def test_split_and_chain(self):
        got, _ = self.flow([[(-300, 0), (0, 0)], [(0, 0), (150, 0)], [(150, 0), (300, 0)], [(150, 0), (300, -150)]],
                           [5, None, None, None], links=[3])
        self.assertEqual(got[1], 5, "a road going on keeps its lanes")
        self.assertEqual(got[2], 4, "past the exit: what arrives less the ramp")
        self.assertIsNone(got[3], "(a ramp: its own one-lane width)")

    def test_no_counts_at_all_a_ramp_still_adds_a_lane(self):
        got, specs = self.flow([[(-300, 0), (0, 0)], [(-200, -150), (0, 0)], [(0, 0), (300, 0)]], [None, None, None],
                               links=[1])
        dflt = roads.lane_layout(specs[0])[1]
        self.assertEqual(got[2], dflt + 1, "its class's default lanes + the ramp's")
        self.assertGreater(specs[2].width, specs[0].width)

    def test_map_counts_and_two_way_roads_stand(self):
        got, _ = self.flow([[(-300, 0), (0, 0)], [(0, 0), (300, 0)]], [4, 3])
        self.assertEqual(list(got), [None, None], "the map's own counts are not replaced")

if __name__ == "__main__":
    unittest.main()
