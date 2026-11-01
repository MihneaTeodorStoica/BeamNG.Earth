"""finepatch: a block's sharp graded cells as holes filled by a finer mesh of the same ground (earthfast fine_*)."""
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import fast  # noqa: E402
import finepatch  # noqa: E402


class FineCellsTest(unittest.TestCase):
    def test_a_graded_step_its_neighbours_and_no_natural_cliff(self):
        """A 3 m cut between columns 5 and 6 (graded: the natural ground was a slope) is fine with a cell round it;
        the same step in the natural ground (raw = u) is the terrain's."""
        u = np.zeros((12, 12))
        u[:, 6:] = 3.0
        raw = np.tile(np.linspace(0, 3, 12), (12, 1))
        c = np.asarray(fast._ef.fine_cells(u, raw, 1.5, 0.5, 1))
        self.assertTrue(c[:, 5].all())                       # the step's cells
        self.assertTrue(c[:, 4].all() and c[:, 6].all())     # grown by one
        self.assertFalse(c[:, 2].any() or c[:, 9].any())
        self.assertFalse(np.asarray(fast._ef.fine_cells(u, u.copy(), 1.5, 0.5, 1)).any(), "a natural cliff")

    def test_holes_at_each_fine_cells_south_west_sample(self):
        """The engine skips a square by its south-west sample: every fine cell's, no other."""
        cells = np.zeros((6, 6), np.uint8)
        cells[1:4, 1:4] = 1
        h = np.asarray(fast._ef.fine_holes(cells))
        self.assertEqual(h.shape, (7, 7))
        self.assertEqual(sorted(zip(*np.nonzero(h))), [(j, i) for j in range(1, 4) for i in range(1, 4)])

    def test_gaps_closed_and_rim(self):
        """A lone terrain cell inside a patch is fine too (closed); the rim: the cells west / south / south-west of
        the patch, whose corners are its holes."""
        u = np.zeros((14, 14))
        raw = u.copy()
        u[6, 5] = u[6, 7] = 4.0                    # two sharp spots, one cell apart
        c = np.asarray(fast._ef.fine_cells(u, raw, 1.5, 0.5, 1))
        self.assertTrue(c[5:8, 4:8].all(), "the gap between them filled")
        rim = np.asarray(fast._ef.fine_rim(c.astype(np.uint8)))
        cov = c.astype(bool) | rim.astype(bool)
        for j, i in zip(*np.nonzero(c)):           # every cell with one of its holes for a corner is covered
            self.assertTrue(cov[j - 1, i] and cov[j, i - 1] and cov[j - 1, i - 1])
        J, I = np.nonzero(c)
        self.assertFalse(rim[J.max() + 1:].any() or rim[:, I.max() + 1:].any(), "none north or east of it")
        self.assertFalse((rim.astype(bool) & c.astype(bool)).any())

    def test_mesh_rim_on_the_terrain_edge(self):
        """A fine cell beside a terrain-drawn one: its shared side is linear between the two samples (no gap), its
        inside the fine heights."""
        u = np.array([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]])          # 2 rows of samples, 2 cells
        mask = np.array([[1, 0]], np.uint8)
        cells = np.array([[0, 0]], np.int64)
        k = 4
        pts = np.asarray(fast._ef.fine_points(0.0, 0.0, 2.0, k, cells))
        z = np.full((1, (k + 1) ** 2), 5.0)                        # the fine heights: all 5
        P, T = (np.asarray(a) for a in fast._ef.fine_mesh(0.0, 0.0, 2.0, k, cells, z, u, mask, np.zeros(1, np.uint8),
                                                           0.04))
        self.assertEqual(len(T), 2 * k * k)
        east = P[np.isclose(P[:, 0], 2.0)]
        self.assertTrue(np.allclose(east[:, 2], east[:, 1]), "the east side: the terrain's edge (z = y here)")
        inner = P[(P[:, 0] > 0.1) & (P[:, 0] < 1.9) & (P[:, 1] > 0.1) & (P[:, 1] < 1.9)]
        self.assertTrue(np.allclose(inner[:, 2], 5.0))
        self.assertTrue(np.allclose(pts, P[:, :2]))


class BlockFineTest(unittest.TestCase):
    def test_pieces_written_and_holes(self):
        import geodesy
        import heightmap
        frame = geodesy.EnuFrame(46.53, 10.45)
        size = 64
        xs, ys = heightmap.sample_coords(0.0, 0.0, size)

        def enu(la, lo):
            e, n, _ = frame.geodetic_to_enu(np.asarray(la), np.asarray(lo), np.zeros(np.shape(la)))
            return np.asarray(e), np.asarray(n)

        def gen(la, lo):        # a road cut 4 m deep into a slope along x = 0
            e, _n = enu(la, lo)
            return 100.0 + np.where(e > 3.0, 4.0 + 0.2 * e, 0.0)

        def raw(la, lo):
            e, _n = enu(la, lo)
            return 100.0 + 0.6 * e
        u = heightmap.surface_u(frame, gen, xs, ys)
        with tempfile.TemporaryDirectory() as d:
            holes, fine = finepatch.block_fine(frame, gen, raw, {"u": u}, 0.0, 0.0, size, d, "b_fine", 1)
            self.assertIsNotNone(fine)
            self.assertTrue(holes.any())
            self.assertTrue(fine["parts"])
            for p in fine["parts"]:
                self.assertTrue(os.path.getsize(os.path.join(d, p["shape"])) > 0)
            b = finepatch.bank(0.0, 0.0, size)
            self.assertEqual(fine["materials"], [[finepatch.material_name(b, __import__('surface').DIRT), "beamngEarthDirt"]])
        flat = heightmap.surface_u(frame, raw, xs, ys)
        with tempfile.TemporaryDirectory() as d:
            holes, fine = finepatch.block_fine(frame, raw, raw, {"u": flat}, 0.0, 0.0, size, d, "b_fine", 1)
            self.assertIsNone(fine, "natural ground: nothing fine")


if __name__ == "__main__":
    unittest.main()


class BankTest(unittest.TestCase):
    def test_neighbours_never_share_a_bank(self):
        for size, g in ((512, 128.0), (1024, 256.0)):
            b = finepatch.bank(0.0, 0.0, size)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx or dy:
                        self.assertNotEqual(finepatch.bank(dx * g, dy * g, size), b)
