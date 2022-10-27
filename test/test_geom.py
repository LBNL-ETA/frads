import os
from pathlib import Path
import sys
sys.path.append('.')
from frads import geom


def test_merge_window():
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
    pt1 = geom.Vector(0, 0, 0)
    pt2 = geom.Vector(3, 0, 0)
    pt3 = geom.Vector(4, 0, 0)
    pt4 = geom.Vector(8, 0, 0)
    pt5 = geom.Vector(0, 0, 3)
    pt6 = geom.Vector(3, 0, 3)
    pt7 = geom.Vector(4, 0, 3)
    pt8 = geom.Vector(8, 0, 3)
    pt9 = geom.Vector(0, 0, 4)
    pt10 = geom.Vector(3, 0, 4)
    pt11 = geom.Vector(4, 0, 4)
    pt12 = geom.Vector(8, 0, 4)
    pt13 = geom.Vector(0, 0, 7)
    pt14 = geom.Vector(3, 0, 7)
    pt15 = geom.Vector(4, 0, 7)
    pt16 = geom.Vector(8, 0, 7)

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

    assert res12 == answer12
    assert res123 == answer123
    assert res1234 == answer1234

