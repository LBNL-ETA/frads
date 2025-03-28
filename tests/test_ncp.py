from pathlib import Path
import numpy as np

from frads import ncp
from frads import geom




def test_gen_ports_from_window_ncp():
    """Generate sun sources for matrix generation."""
    window_polygon = geom.Polygon([np.array((0, 0, 0)),
                                   np.array((0, 0, 4)),
                                   np.array((3, 0, 4)),
                                   np.array((3, 0, 0))])
    awning_polygon = [geom.Polygon([np.array((0, -1, 2)),
                                    np.array((0, 0, 4)),
                                    np.array((3, 0, 4)),
                                    np.array((3, -1, 2))])]
    ports = ncp.gen_ports_from_window_ncp(window_polygon, awning_polygon)
    assert len(ports) == 5
    # assert sum([p.area for p in ports]) == 26

