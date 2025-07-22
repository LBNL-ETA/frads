"""
This module contains definitions of Vector and Polygon objects
and other geometry related routines.
"""

from typing import Sequence

import numpy as np
from pyradiance import Primitive


class Polygon:
    """Polygon class."""

    def __init__(self, vertices: list[np.ndarray]):
        """
        Initialize a polygon.
        Args:
            vertices: a list of vertices (numpy array)
        """
        self._vertices = np.array(vertices)
        self._normal = self._calculate_normal()
        self._area = self._calculate_area()
        self._centroid = self._calculate_centroid()

    def __repr__(self):
        """Create a stable string representation of the polygon."""
        precision = 8
        vertices_repr = [
            tuple(round(float(coord), precision) for coord in vertex)
            for vertex in self._vertices
        ]
        return f"Polygon(vertices={vertices_repr})"

    @property
    def vertices(self):
        return self._vertices

    @vertices.setter
    def vertices(self, new_vertices: list[np.ndarray]):
        self._vertices = np.array(new_vertices)
        self._normal = self._calculate_normal()
        self._area = self._calculate_area()
        self._centroid = self._calculate_centroid()

    @property
    def normal(self):
        return self._normal

    @property
    def area(self):
        return self._area

    @property
    def centroid(self):
        return self._centroid

    @property
    def coordinates(self):
        return self._vertices.flatten().tolist()

    def __sub__(self, other: "Polygon") -> "Polygon":
        """Polygon subtraction.

        Args:
            other: subtract this polygon

        Returns:
            Clipped polygon (Polygon)

        """
        pt1 = self._vertices[0]
        distances1 = np.linalg.norm(other._vertices - pt1, axis=1)
        idx_min = np.argmin(distances1)
        new_other_vert = np.concatenate(
            (other._vertices[idx_min:], other._vertices[:idx_min])
        )
        results = [pt1]
        results.append(new_other_vert[0])
        results.extend(new_other_vert[1:])
        results.append(new_other_vert[0])
        results.extend(self._vertices)
        return Polygon(results)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Polygon):
            return NotImplemented
        if len(other._vertices) != len(self._vertices):
            return False
        if set(map(tuple, self._vertices)) == set(map(tuple, other.vertices)):
            return True
        return False

    def _calculate_normal(self):
        normal = np.zeros(3)
        for i in range(len(self._vertices)):
            v1 = self._vertices[i]
            v2 = self._vertices[(i + 1) % len(self._vertices)]
            normal[0] += (v1[1] - v2[1]) * (v1[2] + v2[2])
            normal[1] += (v1[2] - v2[2]) * (v1[0] + v2[0])
            normal[2] += (v1[0] - v2[0]) * (v1[1] + v2[1])
        norm = np.linalg.norm(normal)
        if norm < np.finfo(float).eps:
            raise ValueError("Cannot compute normal - degenerate polygon")
        return normal / norm

    def _calculate_area(self) -> np.ndarray:
        total = np.array((0.0, 0.0, 0.0))
        for idx in range(1, len(self._vertices) - 1):
            total += np.cross(
                (self._vertices[idx] - self._vertices[0]),
                (self._vertices[idx + 1] - self._vertices[0]),
            )
        return total * np.array((0.5, 0.5, 0.5))

    def _calculate_centroid(self):
        return np.mean(self._vertices, axis=0)

    def scale(self, scale_vect: np.ndarray, center: np.ndarray) -> "Polygon":
        """Scale the polygon.

        Args:
            scale_vect: scale along x, y, z;
            center: center of scaling
        Returns:
            Scaled polygon
        """

        new_vertices = []
        for vert in self._vertices:
            sx = center[0] + (vert[0] - center[0]) * scale_vect[0]
            sy = center[1] + (vert[1] - center[1]) * scale_vect[1]
            sz = center[2] + (vert[2] - center[2]) * scale_vect[2]
            new_vertices.append(np.array([sx, sy, sz]))
        return Polygon(new_vertices)

    @property
    def extreme(self) -> tuple[float, ...]:
        """
        Return the extreme values of the polygon.
        """
        xmin = self._vertices[:, 0].min()
        xmax = self._vertices[:, 0].max()
        ymin = self._vertices[:, 1].min()
        ymax = self._vertices[:, 1].max()
        zmin = self._vertices[:, 2].min()
        zmax = self._vertices[:, 2].max()
        return xmin, xmax, ymin, ymax, zmin, zmax

    def rotate(self, center: np.ndarray, vector: np.ndarray, angle: float) -> "Polygon":
        """
        Rotate the polygon.

        Args:
            center (Vector): center of rotation;
            vector (Vector): rotation axis;
            angle (float): rotation angle in radians;
        Returns:
            Polygon: rotated polygon;
        """
        ro_pts = [rotate_3d(v, center, vector, angle) for v in self._vertices]
        return Polygon(ro_pts)

    def move(self, vector) -> "Polygon":
        """Return the moved polygon along a vector."""
        # mo_pts = [v + vector for v in self.vertices]
        mo_pts = self._vertices + vector
        return Polygon(mo_pts)

    def extrude(self, vector: np.ndarray) -> list:
        """Extrude the polygon.

        Args:
            vector (Vector): extrude along the vector;

        Returns:
            Polygon (list): a list of polygons;

        """
        polygons: list[Polygon] = [self]
        polygon2 = Polygon(([i + vector for i in self._vertices]))
        polygons.append(polygon2)
        for i in range(len(self._vertices) - 1):
            polygons.append(
                Polygon(
                    [
                        self._vertices[i],
                        polygon2._vertices[i],
                        polygon2._vertices[i + 1],
                        self._vertices[i + 1],
                    ]
                )
            )
        polygons.append(
            Polygon(
                [
                    self._vertices[-1],
                    polygon2._vertices[-1],
                    polygon2._vertices[0],
                    self._vertices[0],
                ]
            )
        )
        return polygons

    def flip(self) -> "Polygon":
        """Reverse the vertices order, thus reversing the normal."""
        return Polygon(list(self._vertices[::-1, :]))

    @classmethod
    def rectangle3pts(cls, pt1, pt2, pt3) -> "Polygon":
        """."""
        vect21 = pt2 - pt1
        vect32 = pt3 - pt2
        err_msg = "Three points are not at a right angle"
        assert angle_between(vect21, vect32) == np.pi / 2, err_msg
        vect12 = pt1 - pt2
        pt4 = pt3 + vect12
        return cls([pt1, pt2, pt3, pt4])


def angle_between(vec1: np.ndarray, vec2: np.ndarray, degree=False) -> float:
    """Return angle between two vectors in radians."""
    dot_prod = np.dot(vec1, vec2)
    angle = np.arccos(dot_prod / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
    if degree:
        angle = np.degrees(angle)
    return angle


def rotate_3d(
    point: np.ndarray, center: np.ndarray, axis: np.ndarray, angle: float
) -> np.ndarray:
    """
    Rotate a point around a center and axis
    Args:
        point: the point to rotate
        center: the rotation center
        axis: the rotation axis
        angle: the rotation angle in randians
    Returns:
        the rotated point
    """
    translated_point = point - center
    ct = np.cos(angle)
    st = np.sin(angle)
    axisx, axisy, axisz = axis
    rotation_matrix = np.array(
        [
            [
                axisx**2 + (1 - axisx**2) * ct,
                axisx * axisy * (1 - ct) - axisz * st,
                axisx * axisz * (1 - ct) + axisy * st,
            ],
            [
                axisx * axisy * (1 - ct) + axisz * st,
                axisy**2 + (1 - axisy**2) * ct,
                axisy * axisz * (1 - ct) - axisx * st,
            ],
            [
                axisx * axisz * (1 - ct) - axisy * st,
                axisy * axisz * (1 - ct) + axisx * st,
                axisz**2 + (1 - axisz**2) * ct,
            ],
        ]
    )
    rotated = np.dot(rotation_matrix, translated_point)
    return rotated + center


def convexhull(points: list[np.ndarray], normal: np.ndarray) -> Polygon:
    """Convex hull on coplanar points.

    Args:
        points (list): list of Point;
        normal (Vector): plane's normal
    Returns:
        Polygon: convex hull polygon;
    """

    def toleft(u, v, points):
        """."""
        vect2 = v - u
        pts = []
        for p in points:
            vect1 = p - u
            cross_prod = np.cross(vect1, vect2)
            cross_val = np.dot(cross_prod, normal)
            if cross_val < 0:
                pts.append(p)
        return pts

    def extend_pts(u, v, points):
        """."""
        if not points:
            return []

        vect2 = v - u
        w = min(points, key=lambda p: np.dot(np.cross((p - u), vect2), normal))
        p1, p2 = toleft(w, v, points), toleft(u, w, points)
        return extend_pts(w, v, p1) + [w] + extend_pts(u, w, p2)

    u = min(points, key=lambda p: p[0])
    v = max(points, key=lambda p: p[0])
    if np.array_equal(u, v):
        u = min(points, key=lambda p: p[1])
        v = max(points, key=lambda p: p[1])
    left, right = toleft(u, v, points), toleft(v, u, points)
    points = [v] + extend_pts(u, v, left) + [u] + extend_pts(v, u, right)
    points.append(points[0])
    points.append(points[1])
    to_remove = []
    for i, p in enumerate(points):
        if i < len(points) - 2:
            p12 = np.linalg.norm(p - points[i + 1])
            p23 = np.linalg.norm(points[i + 1] - points[i + 2])
            p13 = np.linalg.norm(p - points[i + 2])
            if (p12 + p23) == p13:
                to_remove.append(points[i + 1])
    points.pop()
    points.pop()
    for r in to_remove:
        points.remove(r)
    return Polygon(points)


def polygon_center(*polygons):
    """Calculate the center from polygons."""
    centroids = [p.centroid for p in polygons]
    return Polygon(centroids).centroid


def get_polygon_limits(polygon_list: Sequence[Polygon], offset: float = 0.0):
    """Get the x,y,z limits from a list of polygons."""
    extreme_list = [p.extreme for p in polygon_list]
    lim = list(zip(*extreme_list))
    xmin = min(lim[0]) - offset
    xmax = max(lim[1]) + offset
    ymin = min(lim[2]) - offset
    ymax = max(lim[3]) + offset
    zmin = min(lim[4]) - offset
    zmax = max(lim[5]) + offset
    return xmin, xmax, ymin, ymax, zmin, zmax


def getbbox(polygons: Sequence[Polygon], offset: float = 0.0) -> list[Polygon]:
    """Get a bounding box for a list of polygons.

    Return a list of polygon that is the
    orthogonal bounding box of a list of polygon.

    Args:
        polygons: list of polygons
        offset: make the box smaller or bigger

    Returns:
        A list of polygon that is the bounding box.
    """
    xmin, xmax, ymin, ymax, zmin, zmax = get_polygon_limits(polygons, offset=offset)
    fp1 = np.array((xmin, ymin, zmin))
    fp2 = np.array((xmax, ymin, zmin))
    fp3 = np.array((xmax, ymax, zmin))
    fpg = Polygon.rectangle3pts(fp1, fp2, fp3)  # -Z
    cp1 = np.array((xmin, ymin, zmax))
    cp2 = np.array((xmax, ymin, zmax))
    cp3 = np.array((xmax, ymax, zmax))
    cpg = Polygon.rectangle3pts(cp3, cp2, cp1)  # +Z
    swpg = Polygon.rectangle3pts(cp2, fp2, fp1)  # -Y
    ewpg = Polygon.rectangle3pts(fp3, fp2, cp2)  # +X
    s2n_vec = np.array((0, ymax - ymin, 0))
    nwpg = Polygon([v + s2n_vec for v in swpg.vertices]).flip()  # +Y
    e2w_vec = np.array((xmax - xmin, 0, 0))
    wwpg = Polygon([v - e2w_vec for v in ewpg.vertices]).flip()  # -X

    return [fpg, cpg, ewpg, swpg, wwpg, nwpg]


def merge_polygon(polygons: Sequence[Polygon]) -> Polygon:
    """Merge polygons into a polygon using Convex Hull.

    Args:
        polygons: Polygons to be merged
    Returns:
        Merged polygon
    Raises:
        ValueError: if windows are not co-planar
    """
    normals = [p.normal for p in polygons]
    if len(set(map(tuple, normals))) > 1:
        raise ValueError("Windows not co-planar")
    points = [i for p in polygons for i in p.vertices]
    return convexhull(points, normals[0])


def polygon_primitive(polygon: Polygon, modifier: str, identifier: str) -> Primitive:
    """
    Generate a primitive from a polygon.
    Args:
        polygon: a Polygon object
        modifier: a Radiance primitive modifier
        identifier: a Radiance primitive identifier
    Returns:
        A Primitive object
    """
    return Primitive(modifier, "polygon", identifier, [], polygon.coordinates)


def parse_polygon(primitive: Primitive) -> Polygon:
    """
    Parse a primitive into a polygon.

    Args:
        primitive: a dictionary object containing a primitive
    Returns:
        A Polygon object
    """
    if primitive.ptype != "polygon":
        raise ValueError("Not a polygon: ", primitive.identifier)
    vertices = [
        np.array(primitive.fargs[i : i + 3]) for i in range(0, len(primitive.fargs), 3)
    ]
    return Polygon(vertices)


def pt_inclusion(pt: np.ndarray, polygon_pts: list[np.ndarray]) -> int:
    """Test whether a point is inside a polygon
    using winding number algorithm."""

    def isLeft(pt0, pt1, pt2):
        """Test whether a point is left to a line."""
        return (pt1[0] - pt0[0]) * (pt2[1] - pt0[1]) - (pt2[0] - pt0[0]) * (
            pt1[1] - pt0[1]
        )

    # Close the polygon for looping
    # polygon_pts.append(polygon_pts[0])
    polygon_pts = [*polygon_pts, polygon_pts[0]]
    wn = 0
    for i in range(len(polygon_pts) - 1):
        if polygon_pts[i][1] <= pt[1]:
            if polygon_pts[i + 1][1] > pt[1]:
                if isLeft(polygon_pts[i], polygon_pts[i + 1], pt) > 0:
                    wn += 1
        else:
            if polygon_pts[i + 1][1] <= pt[1]:
                if isLeft(polygon_pts[i], polygon_pts[i + 1], pt) < 0:
                    wn -= 1
    return wn


def gen_grid(polygon: Polygon, height: float, spacing: float) -> list[list[float]]:
    """Generate a grid of points for orthogonal planar surfaces.

    Args:
        polygon: a polygon object
        height: points' distance from the surface in its normal direction
        spacing: distance between the grid points
    Returns:
        List of the points as list
    """
    vertices = polygon.vertices
    plane_height = sum(i[2] for i in vertices) / len(vertices)
    imin, imax, jmin, jmax, _, _ = polygon.extreme
    xlen_spc = (imax - imin) / spacing
    ylen_spc = (jmax - jmin) / spacing
    xstart = (xlen_spc - int(xlen_spc) + 1) * spacing / 2
    ystart = (ylen_spc - int(ylen_spc) + 1) * spacing / 2
    x0 = np.arange(imin, imax, spacing) + xstart
    y0 = np.arange(jmin, jmax, spacing) + ystart
    grid_dir = polygon.normal * -1
    grid_hgt = np.array((0, 0, plane_height)) + grid_dir * height
    raw_pts = [
        np.array((round(i, 3), round(j, 3), round(grid_hgt[2], 3)))
        for i in x0
        for j in y0
    ]
    if np.array_equal(polygon.normal, np.array((0, 0, 1))):
        _grid = [p for p in raw_pts if pt_inclusion(p, vertices) > 0]
    else:
        _grid = [p for p in raw_pts if pt_inclusion(p, vertices[::-1]) > 0]
    grid = [p.tolist() + grid_dir.tolist() for p in _grid]
    return grid
