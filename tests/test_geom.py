from pathlib import Path
import unittest

from frads import geom
import numpy as np


class TestGeometry(unittest.TestCase):
    def test_merge_window(self):
        """
        0,0,7-------3,0,7   4,0,7-------8,0,7
          |           |       |           |
          |           |       |           |
          |           |       |           |
        0,0,4-------3,0,4   4,0,4-------8,0,4

        0,0,3-------3,0,3   4,0,3-------8,0,3
          |           |       |           |
          |           |       |           |
          |           |       |           |
        0,0,0-------3,0,0   4,0,0-------8,0,0
        """
        pt1 = np.array((0, 0, 0))
        pt2 = np.array((3, 0, 0))
        pt3 = np.array((4, 0, 0))
        pt4 = np.array((8, 0, 0))
        pt5 = np.array((0, 0, 3))
        pt6 = np.array((3, 0, 3))
        pt7 = np.array((4, 0, 3))
        pt8 = np.array((8, 0, 3))
        pt9 = np.array((0, 0, 4))
        pt10 = np.array((3, 0, 4))
        pt11 = np.array((4, 0, 4))
        pt12 = np.array((8, 0, 4))
        pt13 = np.array((0, 0, 7))
        pt14 = np.array((3, 0, 7))
        pt15 = np.array((4, 0, 7))
        pt16 = np.array((8, 0, 7))

        plg1 = geom.Polygon([pt1, pt5, pt6, pt2])
        plg2 = geom.Polygon([pt3, pt7, pt8, pt4])
        plg3 = geom.Polygon([pt9, pt13, pt14, pt10])
        plg4 = geom.Polygon([pt11, pt15, pt16, pt12])

        answer12 = geom.Polygon([pt1, pt5, pt8, pt4])
        answer123 = geom.Polygon([pt1, pt13, pt14, pt8, pt4])
        answer1234 = geom.Polygon([pt1, pt13, pt16, pt4])

        res12 = geom.merge_polygon([plg1, plg2])
        res123 = geom.merge_polygon([plg1, plg2, plg3])
        res1234 = geom.merge_polygon([plg1, plg2, plg3, plg4])

        self.assertEqual(res12, answer12)
        self.assertEqual(res123, answer123)
        self.assertEqual(res1234, answer1234)

    def test_trim_window(self):
        wall_width = 50
        wall_height = 3
        window_width = 50
        window_height = 1.2
        sill_height = 1.2
        south_wall = geom.Polygon([
            np.array((0, 0, 0)),
            np.array((wall_width, 0, 0)),
            np.array((wall_width, 0, wall_height)),
            np.array((0, 0, wall_height)),
        ])
        window = geom.Polygon([
            np.array((0, 0, sill_height)),
            np.array((0, 0, sill_height + window_height)),
            np.array((window_width, 0, sill_height + window_height)),
            np.array((window_width, 0, sill_height)),
        ])
        trimmed_wall = south_wall - window
        print(trimmed_wall)
        print(trimmed_wall.normal)
        self.assertEqual(trimmed_wall.normal[0], 0)
        self.assertEqual(trimmed_wall.normal[1], -1)
        self.assertEqual(trimmed_wall.normal[2], 0)

    def test_polygon_center_box_zone(self):
        """polygon_center works for the six faces of a box-shaped zone.

        The six face centroids of a box are arranged symmetrically around
        the zone center, so their Newell normal vanishes. Wrapping them in a
        Polygon (previous implementation) raised 'Cannot compute normal -
        degenerate polygon'; the center must instead be the centroid mean.
        """
        lx, ly, lz = 2.0, 4.0, 3.0
        p0 = np.array((0, 0, 0))
        faces = [
            geom.Polygon([p0, (lx, 0, 0), (lx, ly, 0), (0, ly, 0)]),          # floor
            geom.Polygon([(0, 0, lz), (lx, 0, lz), (lx, ly, lz), (0, ly, lz)]),  # ceiling
            geom.Polygon([p0, (lx, 0, 0), (lx, 0, lz), (0, 0, lz)]),          # south
            geom.Polygon([p0, (0, ly, 0), (0, ly, lz), (0, 0, lz)]),          # west
            geom.Polygon([(lx, 0, 0), (lx, ly, 0), (lx, ly, lz), (lx, 0, lz)]),  # east
            geom.Polygon([(0, ly, 0), (lx, ly, 0), (lx, ly, lz), (0, ly, lz)]),  # north
        ]
        center = geom.polygon_center(*faces)
        np.testing.assert_allclose(center, [lx / 2, ly / 2, lz / 2])

    def test_polygon_center_equals_centroid_mean(self):
        """polygon_center is the arithmetic mean of the polygon centroids."""
        square = geom.Polygon(
            [np.array((0, 0, 0)), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
        )
        shifted = geom.Polygon(
            [np.array((2, 0, 1)), (3, 0, 1), (3, 1, 1), (2, 1, 1)]
        )
        center = geom.polygon_center(square, shifted)
        expected = np.mean([square.centroid, shifted.centroid], axis=0)
        np.testing.assert_allclose(center, expected)


if __name__== "__main__":
    unittest.main()
