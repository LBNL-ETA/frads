"""
Utilities for handling geometries.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List


@dataclass(frozen=True)
class Vector:
    """3D vector class.

    Attributes:
        x: x coordinate
        y: y coordinate
        z: z coordinate
        length: length of the vector
    """
    x: float = 0
    y: float = 0
    z: float = 0

    def __post_init__(self):
        """Initialize vector."""

    def __str__(self) -> str:
        """Class string representation.

        Returns:
            The string representation of the vector(str)

        """
        return "{:02f} {:02f} {:02f}".format(self.x, self.y, self.z)

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

    def __eq__(self, other: object) -> bool:
        """Check if two vectors are the same."""
        if not isinstance(other, Vector):
            return NotImplemented
        return (self.x, self.y, self.z) == (other.x, other.y, other.z)

    def __hash__(self):
        return hash((self.x, self.y, self.z))

    def length(self):
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def cross(self, other: Vector) -> Vector:
        """Return the cross product of the two vectors.

        Args:
            other: the vector to take a cross product

        Returns:
            The resulting vector

        """
        x_ = self.y * other.z - self.z * other.y
        y_ = self.z * other.x - self.x * other.z
        z_ = self.x * other.y - self.y * other.x
        return Vector(x_, y_, z_)

    def distance_from(self, other: Vector):
        """Calculate the distance between two points."""
        dx = math.fabs(self.x - other.x)
        dy = math.fabs(self.y - other.y)
        dz = math.fabs(self.z - other.z)
        return math.sqrt(dx**2 + dy**2 + dz**2)

    def normalize(self) -> Vector:
        """Return the unit vector."""
        magnitude = math.sqrt(self.x**2 + self.y**2 + self.z**2)
        return Vector(self.x / magnitude, self.y / magnitude,
                      self.z / magnitude)

    def reverse(self) -> Vector:
        """Return the reversed vector."""
        return Vector(self.x * -1, self.y * -1, self.z * -1)

    def scale(self, factor) -> Vector:
        """Scale the vector by a scalar."""
        return Vector(self.x * factor, self.y * factor, self.z * factor)

    def angle_from(self, other) -> float:
        """."""
        dot_prod = self * other
        angle = math.acos(dot_prod / (self.length() * other.length()))
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

        row1 = [(vector.x * vector.x) + ((1 - (vector.x * vector.x)) * cosa),
                (vector.x * vector.y * (1 - cosa)) - (vector.z * sina),
                (vector.x * vector.z * (1 - cosa)) + (vector.y * sina), 0.0]
        row2 = [(vector.x * vector.y * (1 - cosa)) + (vector.z * sina),
                (vector.y * vector.y) + ((1 - (vector.y * vector.y)) * cosa),
                (vector.y * vector.z * (1 - cosa)) - (vector.x * sina), 0.0]
        row3 = [(vector.x * vector.z * (1.0 - cosa)) - (vector.y * sina),
                (vector.y * vector.z * (1.0 - cosa)) + (vector.x * sina),
                (vector.z * vector.z) + ((1.0 - (vector.z * vector.z)) * cosa),
                0.0]

        row4 = [0.0] * 4

        rx = (self.x * row1[0]) + (self.y * row2[0]) + (self.z *
                                                        row3[0]) + row4[0]
        ry = (self.x * row1[1]) + (self.y * row2[1]) + (self.z *
                                                        row3[1]) + row4[1]
        rz = (self.x * row1[2]) + (self.y * row2[2]) + (self.z *
                                                        row3[2]) + row4[2]

        return Vector(rx, ry, rz)

    def to_sphr(self):
        """Convert cartesian to spherical coordinates."""
        r = self.distance_from(Vector())
        theta = math.atan(self.y / self.x)
        phi = math.atan(math.sqrt(self.x**2 + self.y**2) / self.z)
        return theta, phi, r

    def coplanar(self, other1, other2):
        """Test if the vector is coplanar with the other two vectors."""
        triple_prod = self * other1.cross(other2)
        return triple_prod==0

    def to_list(self):
        """Return a list containing the coordinates."""
        return [self.x, self.y, self.z]

    def to_tuple(self):
        """Return a tuple containing the coordinates, and round to third decimal place."""
        return (round(self.x,3), round(self.y, 3), round(self.z, 3))

    @classmethod
    def spherical(cls, theta, phi, r) -> Vector:
        """Construct a vector using spherical coordinates."""
        xcoord = math.sin(theta) * math.cos(phi) * r
        ycoord = math.sin(theta) * math.sin(phi) * r
        zcoord = math.cos(theta) * r
        return cls(xcoord, ycoord, zcoord)


class Polygon:
    """3D polygon class."""

    def __init__(self, vertices):
        """."""
        self.vert_cnt = len(vertices)
        assert self.vert_cnt > 2, "Need more than 2 vertices to make a polygon."
        self.vertices = vertices

    def __sub__(self, other: Polygon) -> Polygon:
        """Polygon subtraction.

        Args:
            other: subtract this polygon

        Returns:
            Clipped polygon (Polygon)

        """
        pt1 = self.vertices[0]
        #opposite = False if self.normal() == other.normal() else True
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

    def flip(self):
        """Reverse the vertices order, thus reversing the normal."""
        return Polygon([self.vertices[0]] + self.vertices[:0:-1])

    def normal(self):
        """Calculate the polygon normal."""
        vect21 = self.vertices[1] - self.vertices[0]
        vect32 = self.vertices[2] - self.vertices[1]
        normal = vect21.cross(vect32)
        normal_u = normal.normalize()
        return normal_u

    def centroid(self):
        """Return the geometric center point."""
        return sum(self.vertices, Vector()).scale(1/self.vert_cnt)

    def area(self):
        """Calculate the area of the polygon.

        Returns:
            polygon area(float)

        """

        total = Vector()
        for i in range(self.vert_cnt):
            vect1 = self.vertices[i]
            if i == self.vert_cnt - 1:
                vect2 = self.vertices[0]
            else:
                vect2 = self.vertices[i + 1]
            prod = vect1.cross(vect2)
            total += prod
        area = abs(total * self.normal() / 2)
        return area

    def scale(self, scale_vect, center):
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
        polygon2 = Polygon([i + vector for i in self.vertices])
        polygons.append(polygon2)
        for i in range(len(self.vertices) - 1):
            polygons.append(
                Polygon([self.vertices[i], polygon2.vertices[i],
                         polygon2.vertices[i + 1], self.vertices[i + 1]]))
        polygons.append(
            Polygon([self.vertices[-1], polygon2.vertices[-1],
                     polygon2.vertices[0], self.vertices[0]]))
        return polygons

    def __add__(self, other: Polygon) -> Polygon:
        """Merge two polygons."""
        sp, index = self.shared_pts(other)
        if sp != 2:
            raise ValueError(
                "Trying to merge two polygons without shared sides")
        t1 = self.vertices[index[0]:] + self.vertices[:index[0]]
        oid = other.vertices.index(self.vertices[index[0]])
        t2 = other.vertices[oid:] + other.vertices[:oid]
        if t1[-1] == t2[1]:
            return Polygon(t1 + t2[2:])
        return Polygon(t2 + t1[2:])

    def shared_pts(self, other):
        """Return the total number of share points between two polygons."""
        cnt = 0
        index = []
        for pid in range(len(self.vertices)):
            if self.vertices[pid].to_list() in other.to_list():
                cnt += 1
                index.append(pid)
        return cnt, index

    def rotate(self, vector, angle):
        """."""
        ro_pts = [v.rotate_3d(vector, angle) for v in self.vertices]
        return Polygon(ro_pts)

    def move(self, vector):
        """Return the moved polygon along a vector."""
        mo_pts = [v + vector for v in self.vertices]
        return Polygon(mo_pts)

    def extreme(self):
        """."""
        xs = [v.x for v in self.vertices]
        ys = [v.y for v in self.vertices]
        zs = [v.z for v in self.vertices]
        return min(xs), max(xs), min(ys), max(ys), min(zs), max(zs)

    def to_list(self):
        """Return a list of tuples."""
        return [p.to_tuple() for p in self.vertices]

    def to_real(self):
        """Convert the vertices to real arg string format."""
        real_str = "{} ".format(3 * len(self.vertices))
        vert_str = ' '.join([str(i) for i in self.vertices])
        return real_str + vert_str

    @classmethod
    def rectangle3pts(cls, pt1, pt2, pt3):
        """."""
        vect21 = pt2 - pt1
        vect32 = pt3 - pt2
        err_msg = "Three points are not at a right angle"
        assert vect21.angle_from(vect32) == math.pi / 2, err_msg
        vect12 = pt1 - pt2
        pt4 = pt3 + vect12
        return cls([pt1, pt2, pt3, pt4])


def convexhull(points: List[Vector], normal: Vector):
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
    if u.__str__() == v.__str__():
        u = min(points, key=lambda p: p.y)
        v = max(points, key=lambda p: p.y)
    left, right = toleft(u, v, points), toleft(v, u, points)
    points = [v] + extend_pts(u, v, left) + [u] + extend_pts(v, u, right)
    return Polygon(points)


def polygon_center(*polygons):
    """Calculate the center from polygons."""
    vertices = [v for p in polygons for v in p.vertices]
    return sum(vertices, Vector()).scale(1 / len(vertices))


def get_polygon_limits(polygon_list: list, offset=0.0):
    """Get the x,y,z limits from a list of polygons."""
    extreme_list = [p.extreme() for p in polygon_list]
    lim = list(zip(*extreme_list))
    xmin = min(lim[0]) - offset
    xmax = max(lim[1]) + offset
    ymin = min(lim[2]) - offset
    ymax = max(lim[3]) + offset
    zmin = min(lim[4]) - offset
    zmax = max(lim[5]) + offset
    return xmin, xmax, ymin, ymax, zmin, zmax


def getbbox(polygons: list, offset: float = 0.0):
    """Get a bounding box for a list of polygons.

    Return a list of polygon that is the
    orthogonal bounding box of a list of polygon.

    Args:
        polygons: list of polygons
        offset: make the box smaller or bigger

    Returns:
        A list of polygon that is the bounding box.
    """
    xmin, xmax, ymin, ymax, zmin, zmax = get_polygon_limits(
        polygons, offset=offset)
    fp1 = Vector(xmin, ymin, zmin)
    fp2 = Vector(xmax, ymin, zmin)
    fp3 = Vector(xmax, ymax, zmin)
    fpg = Polygon.rectangle3pts(fp1, fp2, fp3) #-Z
    cp1 = Vector(xmin, ymin, zmax)
    cp2 = Vector(xmax, ymin, zmax)
    cp3 = Vector(xmax, ymax, zmax)
    cpg = Polygon.rectangle3pts(cp3, cp2, cp1) #+Z
    swpg = Polygon.rectangle3pts(cp2, fp2, fp1) #-Y
    ewpg = Polygon.rectangle3pts(fp3, fp2, cp2) #+X
    s2n_vec = Vector(0, ymax - ymin, 0)
    nwpg = Polygon([v + s2n_vec for v in swpg.vertices]).flip() #+Y
    e2w_vec = Vector(xmax - xmin, 0, 0)
    wwpg = Polygon([v - e2w_vec for v in ewpg.vertices]).flip() #-X

    return [fpg, cpg, ewpg, swpg, wwpg, nwpg]
