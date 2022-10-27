from pathlib import Path
import sys
sys.path.append('.')

from frads import ncp
from frads import parsers
from frads import geom




def test_gen_ports_from_window_ncp():
    """Generate sun sources for matrix generation."""
    window_polygon = geom.Polygon([geom.Vector(0, 0, 0),
                                   geom.Vector(0, 0, 4),
                                   geom.Vector(3, 0, 4),
                                   geom.Vector(3, 0, 0)])
    awning_polygon = [geom.Polygon([geom.Vector(0, -1, 2),
                                    geom.Vector(0, 0, 4),
                                    geom.Vector(3, 0, 4),
                                    geom.Vector(3, -1, 2)])]
    ports = ncp.gen_ports_from_window_ncp(window_polygon, awning_polygon)
    assert len(ports) == 5
    assert sum([p.area for p in ports]) == 26

