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


if __name__ == "__main__":
    unittest.main()
