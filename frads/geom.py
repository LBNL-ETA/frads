"""
This module contains definitions of Vector and Polygon objects
and other geometry related routines.
"""

from typing import List, Tuple, Sequence

import numpy as np


class Polygon:
    """Polygon class."""

    def __init__(self, vertices: List[np.ndarray]):
        """
        Initialize a polygon.
        Args:
            vertices: a list of vertices (numpy array)
        """
        self._vertices = np.array(vertices)
        self._normal = self._calculate_normal()
        self._area = self._calculate_area()
        self._centroid = self._calculate_centroid()

    @property
    def vertices(self):
        return self._vertices

    @vertices.setter
    def vertices(self, new_vertices: List[np.ndarray]):
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
        results.extend(reversed(new_other_vert[1:]))
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
        vector1 = self._vertices[1] - self._vertices[0]
        normal_vector = np.array((0.0, 0.0, 0.0))
        for i in range(2, len(self._vertices)):
            vector2 = self._vertices[i] - self._vertices[0]
            normal_vector = np.cross(vector1, vector2)
            if np.linalg.norm(normal_vector) > 0:
                break
        return normal_vector / np.linalg.norm(normal_vector)

    def _calculate_area(self) -> np.ndarray:
        total = np.array((0.0, 0.0, 0.0))
        for idx in range(1, len(self._vertices) - 1):
            total += np.cross(
                (self._vertices[idx] - self._vertices[0]),
                (self._vertices[idx + 1] - self._vertices[0]),
            )
        return abs(total * np.array((0.5, 0.5, 0.5)))

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
    def extreme(self) -> Tuple[float, ...]:
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
        polygons: List[Polygon] = [self]
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


def angle_between(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Return angle between two vectors in radians."""
    dot_prod = np.dot(vec1, vec2)
    angle = np.arccos(dot_prod / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
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


def convexhull(points: List[np.ndarray], normal: np.ndarray) -> Polygon:
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


def getbbox(polygons: Sequence[Polygon], offset: float = 0.0) -> List[Polygon]:
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
