"""
This module contains definitions of Vector and Polygon objects
and other geometry related routines.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Union, List, Tuple, Sequence


@dataclass(frozen=True)
class Vector:
    """3D vector class.

    Attributes:
        x: x coordinate
        y: y coordinate
        z: z coordinate
    """

    x: float = 0
    y: float = 0
    z: float = 0

    def __str__(self) -> str:
        """Class string representation.

        Returns:
            The string representation of the vector(str)

        """
        return f"{self.x:02f} {self.y:02f} {self.z:02f}"

    def __add__(self, other) -> Vector:
        """Add the two vectors.
        Args:
            other(Vector): vector to add

        Returns:
            The added vector(Vector)

        """
        return Vector(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: Vector) -> Vector:
        return Vector(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, other: Vector) -> float:
        """Return the dot produce between two vectors."""
        return self.x * other.x + self.y * other.y + self.z * other.z

    @property
    def length(self) -> float:
        """Get vector distance from origin."""
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def cross(self, other: Vector) -> Vector:
        """Return the cross product of the two vectors.

        Args:
            other: the vector to take a cross product

        Returns:
            The resulting vector

        """
        return Vector(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def distance_from(self, other: Vector) -> float:
        """Calculate the distance between two points."""
        dx = math.fabs(self.x - other.x)
        dy = math.fabs(self.y - other.y)
        dz = math.fabs(self.z - other.z)
        return math.sqrt(dx**2 + dy**2 + dz**2)

    def normalize(self) -> Vector:
        """Return the unit vector."""
        magnitude = math.sqrt(self.x**2 + self.y**2 + self.z**2)
        return Vector(self.x / magnitude, self.y / magnitude, self.z / magnitude)

    def reverse(self) -> Vector:
        """Return the reversed vector."""
        return Vector(self.x * -1, self.y * -1, self.z * -1)

    def scale(self, factor: float) -> Vector:
        """Scale the vector by a scalar."""
        return Vector(self.x * factor, self.y * factor, self.z * factor)

    def angle_from(self, other: "Vector") -> float:
        """."""
        dot_prod = self * other
        angle = math.acos(dot_prod / (self.length * other.length))
        return angle

    def rotate_3d(self, vector: Vector, theta: float) -> Vector:
        """Rotate the point around the vector theta radians.

        Args:
            vector: rotation axis
            theta: rotation radians
        Returns:
            the rotated point

        """
        cosa = math.cos(theta)
        sina = math.sin(theta)

        row1 = [
            (vector.x * vector.x) + ((1 - (vector.x * vector.x)) * cosa),
            (vector.x * vector.y * (1 - cosa)) - (vector.z * sina),
            (vector.x * vector.z * (1 - cosa)) + (vector.y * sina),
            0.0,
        ]
        row2 = [
            (vector.x * vector.y * (1 - cosa)) + (vector.z * sina),
            (vector.y * vector.y) + ((1 - (vector.y * vector.y)) * cosa),
            (vector.y * vector.z * (1 - cosa)) - (vector.x * sina),
            0.0,
        ]
        row3 = [
            (vector.x * vector.z * (1.0 - cosa)) - (vector.y * sina),
            (vector.y * vector.z * (1.0 - cosa)) + (vector.x * sina),
            (vector.z * vector.z) + ((1.0 - (vector.z * vector.z)) * cosa),
            0.0,
        ]

        row4 = [0.0] * 4

        rx = (self.x * row1[0]) + (self.y * row2[0]) + (self.z * row3[0]) + row4[0]
        ry = (self.x * row1[1]) + (self.y * row2[1]) + (self.z * row3[1]) + row4[1]
        rz = (self.x * row1[2]) + (self.y * row2[2]) + (self.z * row3[2]) + row4[2]

        return Vector(rx, ry, rz)

    def to_sphr(self):
        """Convert cartesian to spherical coordinates."""
        r = self.distance_from(Vector())
        theta = math.atan(self.y / self.x)
        phi = math.acos(self.z / r)
        return theta, phi, r

    def coplanar(self, other1, other2) -> bool:
        """Test if the vector is coplanar with the other two vectors."""
        triple_prod = self * other1.cross(other2)
        return triple_prod == 0

    def to_list(self) -> List[float]:
        """Return a list containing the coordinates."""
        return [self.x, self.y, self.z]

    def to_tuple(self) -> Tuple[float, ...]:
        """Return a tuple containing the coordinates."""
        return self.x, self.y, self.z

    @classmethod
    def spherical(
        cls,
        theta: float,
        phi: float,
        r: float,
    ) -> Vector:
        """Construct a vector using spherical coordinates."""
        xcoord = math.sin(theta) * math.cos(phi) * r
        ycoord = math.sin(theta) * math.sin(phi) * r
        zcoord = math.cos(theta) * r
        return cls(xcoord, ycoord, zcoord)


@dataclass(frozen=True)
class Polygon:
    """3D polygon class

    Attributes:
        vertices(List[Vector]): list of vertices of the polygon.
    """

    vertices: List[Vector]

    def __post_init__(self) -> None:
        """."""
        if len(self.vertices) < 3:
            raise ValueError("Need more than 2 vertices to make a polygon.")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Polygon):
            return NotImplemented
        if len(other.vertices) != len(self.vertices):
            return False
        for vert in self.vertices:
            if vert not in other.vertices:
                return False
        return True

    def __add__(self, other: Polygon) -> Polygon:
        """Merge two polygons."""
        sp, index = self.shared_pts(other)
        if sp != 2:
            raise ValueError("Trying to merge two polygons without shared sides")
        t1 = self.vertices[index[0] :] + self.vertices[: index[0]]
        oid = other.vertices.index(self.vertices[index[0]])
        t2 = other.vertices[oid:] + other.vertices[:oid]
        if t1[-1] == t2[1]:
            return Polygon(t1 + t2[2:])
        return Polygon(t2 + t1[2:])

    def __sub__(self, other: Polygon) -> Polygon:
        """Polygon subtraction.

        Args:
            other: subtract this polygon

        Returns:
            Clipped polygon (Polygon)

        """
        pt1 = self.vertices[0]
        # opposite = False if self.normal() == other.normal() else True
        opposite = False
        distances1 = [pt1.distance_from(i) for i in other.vertices]
        idx_min = distances1.index(min(distances1))
        new_other_vert = other.vertices[idx_min:] + other.vertices[:idx_min]
        results = [pt1]
        if opposite:
            results.extend(new_other_vert)
            results.append(other.vertices[idx_min])
        else:
            results.append(new_other_vert[0])
            results.extend(reversed(new_other_vert[1:]))
            results.append(new_other_vert[0])
        results.extend(self.vertices)
        return Polygon(results)

    @property
    def normal(self) -> Vector:
        """Calculate the polygon normal."""
        normal = Vector(0, 0, 0)
        for idx in range(len(self.vertices) - 2):
            normal += (self.vertices[idx + 1] - self.vertices[idx]).cross(
                self.vertices[idx + 2] - self.vertices[idx + 1]
            )
        return normal.normalize()

    @property
    def centroid(self) -> Vector:
        """Return the geometric center point."""
        return sum(self.vertices, Vector()).scale(1 / len(self.vertices))

    @property
    def area(self) -> float:
        """Calculate the area of the polygon."""
        total = Vector(0, 0, 0)
        for idx in range(1, len(self.vertices) - 1):
            total += (self.vertices[idx] - self.vertices[0]).cross(
                self.vertices[idx + 1] - self.vertices[0]
            )
        return abs(total * Vector(0.5, 0.5, 0.5))

    @property
    def extreme(self) -> Tuple[float, ...]:
        """."""
        xs = [v.x for v in self.vertices]
        ys = [v.y for v in self.vertices]
        zs = [v.z for v in self.vertices]
        return min(xs), max(xs), min(ys), max(ys), min(zs), max(zs)

    def flip(self) -> "Polygon":
        """Reverse the vertices order, thus reversing the normal."""
        return Polygon(self.vertices[::-1])
        # return Polygon([self.vertices[0]] + self.vertices[:0:-1])

    def scale(self, scale_vect, center) -> "Polygon":
        """Scale the polygon.

        Parameters:
            scale_vect (Vector): scale along x, y, z;
            center (Vector): center of scaling
        Return:
            Scaled polygon (Polygon)

        """
        new_vertices = []
        for vert in self.vertices:
            sx = center.x + (vert.x - center.x) * scale_vect.x
            sy = center.y + (vert.y - center.y) * scale_vect.y
            sz = center.z + (vert.z - center.z) * scale_vect.z
            new_vertices.append(Vector(sx, sy, sz))
        return Polygon(new_vertices)

    def extrude(self, vector: Vector) -> list:
        """Extrude the polygon.

        Args:
            vector (Vector): extrude along the vector;

        Returns:
            Polygon (list): a list of polygons;

        """
        polygons = [self]
        polygon2 = Polygon(([i + vector for i in self.vertices]))
        polygons.append(polygon2)
        for i in range(len(self.vertices) - 1):
            polygons.append(
                Polygon(
                    [
                        self.vertices[i],
                        polygon2.vertices[i],
                        polygon2.vertices[i + 1],
                        self.vertices[i + 1],
                    ]
                )
            )
        polygons.append(
            Polygon(
                [
                    self.vertices[-1],
                    polygon2.vertices[-1],
                    polygon2.vertices[0],
                    self.vertices[0],
                ]
            )
        )
        return polygons

    def shared_pts(self, other) -> Tuple[int, List[int]]:
        """Return the total number of share points between two polygons."""
        cnt = 0
        index = []
        for pid, val in enumerate(self.vertices):
            if val.to_list() in other.to_list():
                cnt += 1
                index.append(pid)
        return cnt, index

    def rotate(self, vector, angle) -> "Polygon":
        """."""
        ro_pts = [v.rotate_3d(vector, angle) for v in self.vertices]
        return Polygon(ro_pts)

    def move(self, vector) -> "Polygon":
        """Return the moved polygon along a vector."""
        mo_pts = [v + vector for v in self.vertices]
        return Polygon(mo_pts)

    def to_list(self):
        """Return a list of tuples."""
        return [p.to_tuple() for p in self.vertices]

    def to_real(self) -> Union[List[float], List[int]]:
        """Convert the vertices to real arg string format."""
        real_arg = []
        for vert in self.vertices:
            real_arg.append(vert.x)
            real_arg.append(vert.y)
            real_arg.append(vert.z)
        return real_arg

    @classmethod
    def rectangle3pts(cls, pt1, pt2, pt3) -> "Polygon":
        """."""
        vect21 = pt2 - pt1
        vect32 = pt3 - pt2
        err_msg = "Three points are not at a right angle"
        assert vect21.angle_from(vect32) == math.pi / 2, err_msg
        vect12 = pt1 - pt2
        pt4 = pt3 + vect12
        return cls((pt1, pt2, pt3, pt4))


def convexhull(points: List[Vector], normal: Vector) -> Polygon:
    """Convex hull on coplanar points.

    Parameters:
        points (list): list of Point;
        normal (Vector): plane's normal
    """

    def toleft(u, v, points):
        """."""
        vect2 = v - u
        pts = []
        for p in points:
            vect1 = p - u
            cross_prod = vect1.cross(vect2)
            cross_val = cross_prod * normal
            if cross_val < 0:
                pts.append(p)
        return pts

    def extend_pts(u, v, points):
        """."""
        if not points:
            return []

        vect2 = v - u
        w = min(points, key=lambda p: (p - u).cross(vect2) * normal)
        p1, p2 = toleft(w, v, points), toleft(u, w, points)
        return extend_pts(w, v, p1) + [w] + extend_pts(u, w, p2)

    u = min(points, key=lambda p: p.x)
    v = max(points, key=lambda p: p.x)
    if u == v:
        u = min(points, key=lambda p: p.y)
        v = max(points, key=lambda p: p.y)
    left, right = toleft(u, v, points), toleft(v, u, points)
    points = [v] + extend_pts(u, v, left) + [u] + extend_pts(v, u, right)
    points.append(points[0])
    points.append(points[1])
    to_remove = []
    for i, p in enumerate(points):
        if i < len(points) - 2:
            p12 = p.distance_from(points[i + 1])
            p23 = points[i + 1].distance_from(points[i + 2])
            p13 = p.distance_from(points[i + 2])
            if (p12 + p23) == p13:
                to_remove.append(points[i + 1])
    points.pop()
    points.pop()
    for r in to_remove:
        points.remove(r)
    return Polygon(points)


def polygon_center(*polygons):
    """Calculate the center from polygons."""
    vertices = [v for p in polygons for v in p.vertices]
    return sum(vertices, Vector()).scale(1 / len(vertices))


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


def getbbox(polygons: Sequence[Polygon], offset: float = 0.0):
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
    fp1 = Vector(xmin, ymin, zmin)
    fp2 = Vector(xmax, ymin, zmin)
    fp3 = Vector(xmax, ymax, zmin)
    fpg = Polygon.rectangle3pts(fp1, fp2, fp3)  # -Z
    cp1 = Vector(xmin, ymin, zmax)
    cp2 = Vector(xmax, ymin, zmax)
    cp3 = Vector(xmax, ymax, zmax)
    cpg = Polygon.rectangle3pts(cp3, cp2, cp1)  # +Z
    swpg = Polygon.rectangle3pts(cp2, fp2, fp1)  # -Y
    ewpg = Polygon.rectangle3pts(fp3, fp2, cp2)  # +X
    s2n_vec = Vector(0, ymax - ymin, 0)
    nwpg = Polygon([v + s2n_vec for v in swpg.vertices]).flip()  # +Y
    e2w_vec = Vector(xmax - xmin, 0, 0)
    wwpg = Polygon([v - e2w_vec for v in ewpg.vertices]).flip()  # -X

    return [fpg, cpg, ewpg, swpg, wwpg, nwpg]


def merge_polygon(polygons: Sequence[Polygon]) -> Polygon:
    """Merge polygons into a polygon using Convex Hull.

    Args:
        polygons: Polygons to be merged
    Returns:
        Merged polygon
    Raises:
        ValueError if window normals are not the same
    """
    normals = [p.normal for p in polygons]
    if len(set(normals)) > 1:
        raise ValueError("Windows not co-planar")
    points = [i for p in polygons for i in p.vertices]
    return convexhull(points, normals[0])
