"""
Handle geometries.

T.Wang
"""

import math


class Vector(object):
    """3D vector class."""

    def __init__(self, x=0, y=0, z=0):
        """Initialize vector."""
        errmsg = "float or integer required to define a vector"
        assert all([type(i) in [float, int] for i in [x, y, z]]), errmsg
        self.x = x
        self.y = y
        self.z = z
        self.length = math.sqrt(x**2 + y**2 + z**2)

    def __str__(self):
        """Class string representation."""
        return "{}\t{}\t{}\t".format(self.x, self.y, self.z)

    def __add__(self, other):
        """Add the two vectors."""
        return Vector(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, other):
        """Return the dot produce between two vectors."""
        return self.x * other.x + self.y * other.y + self.z * other.z

    def __eq__(self, other):
        return self.to_list() == other.to_list()

    def cross(self, other):
        """Return the cross product of the two vectors."""
        x_ = self.y * other.z - self.z * other.y
        y_ = self.z * other.x - self.x * other.z
        z_ = self.x * other.y - self.y * other.x
        return Vector(x_, y_, z_)

    def distance_from(self, other):
        """Calculate the distance between two points."""
        dx = math.fabs(self.x - other.x)
        dy = math.fabs(self.y - other.y)
        dz = math.fabs(self.z - other.z)
        return math.sqrt(dx**2 + dy**2 + dz**2)

    def unitize(self):
        """Return the unit vector."""
        magnitude = math.sqrt(self.x**2 + self.y**2 + self.z**2)
        return Vector(self.x / magnitude, self.y / magnitude,
                      self.z / magnitude)

    def reverse(self):
        """Return the reversed vector."""
        return Vector(self.x * -1, self.y * -1, self.z * -1)

    def scale(self, factor):
        return Vector(self.x * factor, self.y * factor, self.z * factor)

    def distance_from(self, other):
        """Calculate the distance between two points."""
        dx = math.fabs(self.x - other.x)
        dy = math.fabs(self.y - other.y)
        dz = math.fabs(self.z - other.z)
        return math.sqrt(dx**2 + dy**2 + dz**2)

    def angle_from(self, other):
        """."""
        dot_prod = self * other
        angle = math.acos(dot_prod / (self.length * other.length))
        return angle

    def rotate3D(self, vector, theta):
        """Rotate the point around the vector theta radians.

        Parameters:
            vector (Vector): rotation axis;
            theta (float): rotation radians;
        Return:
            the rotated point (Point)

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
        r = self.distance_from(Vector())
        theta = math.atan(self.y / self.x)
        phi = math.atan(math.sqrt(self.x**2 + self.y**2) / self.z)
        return VecSph(theta, phi, r)

    def coplanar(self, other1, other2):
        """Test if the vector is coplanar with the other two vectors."""
        triple_prod = self * other1.cross(other2)
        cp = True if triple_prod == 0 else False
        return cp

    def to_list(self):
        return [self.x, self.y, self.z]


class VecSph(Vector):
    """Define a vector in spherical coordinate."""

    def __init__(self, theta=0, phi=0, r=0):
        self.theta = theta
        self.phi = phi
        self.r = r
        self.x = math.sin(theta) * math.cos(phi) * r
        self.y = math.sin(theta) * math.cos(phi) * r
        self.z = math.cos(phi) * r


class Polygon(object):
    """3D polygon class."""

    def __init__(self, vertices):
        """."""
        self.vert_cnt = len(vertices)
        assert self.vert_cnt > 2, "Need more than 2 vertices to make a polygon."
        self.vertices = vertices

    def __sub__(self, other):
        """Polygon subtraction.

        Parameter:
            other (Polygon): subtract this polygon;
        Return:
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
        return Polygon([self.vertices[0]] + self.vertices[:0:-1])

    def normal(self):
        """Calculate the polygon normal."""
        vect21 = self.vertices[1] - self.vertices[0]
        vect32 = self.vertices[2] - self.vertices[1]
        normal = vect21.cross(vect32)
        normal_u = normal.unitize()
        return normal_u

    def centroid(self):
        ctr = [
            sum(i) / self.vert_cnt
            for i in zip(*[i.to_list() for i in self.vertices])
        ]
        return Vector(*ctr)

    def area(self):
        """Calculate the area of the polygon."""
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

    def extrude(self, vector):
        """Extrude the polygon.

        Parameter:
            vector (Vector): extrude along the vector;
        Return:
            Polygon (list): a list of polygons;

        """
        polygons = [self]
        polygon2 = Polygon([i + vector for i in self.vertices])
        polygons.append(polygon2)
        for i in range(len(self.vertices) - 1):
            polygons.append(
                Polygon([
                    self.vertices[i], polygon2.vertices[i],
                    polygon2.vertices[i + 1], self.vertices[i + 1]
                ]))
        polygons.append(
            Polygon([
                self.vertices[-1], polygon2.vertices[-1], polygon2.vertices[0],
                self.vertices[0]
            ]))
        return polygons

    def __add__(self, other):
        sp, index = self.shared_pts(other)
        print(sp)
        if sp == 2:
            t1 = self.vertices[index[0]:] + self.vertices[:index[0]]
            oid = other.vertices.index(self.vertices[index[0]])
            t2 = other.vertices[oid:] + other.vertices[:oid]
            if t1[-1] == t2[1]:
                return Polygon(t1 + t2[2:])
            else:
                return Polygon(t2 + t1[2:])
        else:
            raise "{} and {} don't share a side".format(
                self.vertices, other.vertices)

    def shared_pts(self, other):
        cnt = 0
        index = []
        for pid in range(len(self.vertices)):
            if self.vertices[pid].to_list() in other.to_list():
                cnt += 1
                index.append(pid)
        return cnt, index

    def rotate(self, vector, angle):
        """."""
        ro_pts = [v.rotate3D(vector, angle) for v in self.vertices]
        return Polygon(ro_pts)

    def extreme(self):
        """."""
        xs = [v.x for v in self.vertices]
        ys = [v.y for v in self.vertices]
        zs = [v.z for v in self.vertices]
        return min(xs), max(xs), min(ys), max(ys), min(zs), max(zs)

    def to_list(self):
        return [p.to_list() for p in self.vertices]

    def to_real(self):
        """Convert the vertices to real arg string format."""
        real_str = "{}\n".format(3 * len(self.vertices))
        vert_str = ''.join([str(i) for i in self.vertices])
        return real_str + vert_str


class Rectangle3P(Polygon):
    """Rectangle from three points."""

    def __init__(self, pt1, pt2, pt3):
        """Define a rectangle with three consective vertices."""
        self.pt1 = pt1
        self.pt2 = pt2
        self.pt3 = pt3
        self.vect21 = pt2 - pt1
        self.vect32 = pt3 - pt2
        err_msg = "Three points are not at a right angle"
        assert self.vect21.angle_from(self.vect32) == math.pi / 2, err_msg
        vect12 = pt1 - pt2
        self.pt4 = pt3 + vect12
        self.vertices = [self.pt1, self.pt2, self.pt3, self.pt4]
        self.vert_cnt = 4

    def area(self):
        length = self.vect21.length
        width = self.vect32.length
        return length * width


class Triangle(Polygon):
    """Triangle"""

    def __init__(self, pt1, pt2, pt3):
        self.pt1 = pt1
        self.pt2 = pt2
        self.pt3 = pt3
        self.vertices = [pt1, pt2, pt3]


class Convexhull(object):
    """Convex hull on coplanar points."""

    def __init__(self, points, normal):
        """Convex hull on coplanar points.

        Parameters:
            points (list): list of Point;
            normal (Vector): plane's normal
        """
        self.normal = normal
        vectors = [p.as_vector() for p in points]
        projected = [v * normal for v in vectors]
        u = min(points, key=lambda p: p.x)
        v = max(points, key=lambda p: p.x)
        if u.__str__() == v.__str__():
            u = min(points, key=lambda p: p.y)
            v = max(points, key=lambda p: p.y)
        left, right = self.toleft(u, v, points), self.toleft(v, u, points)
        points = [v] + self.extend(u, v, left) + [u] + self.extend(v, u, right)
        self.hull = Polygon(points)

    def toleft(self, u, v, points):
        """."""
        vect2 = v - u
        pts = []
        for p in points:
            vect1 = p - u
            cross_prod = vect1.cross(vect2)
            cross_val = cross_prod * self.normal
            if cross_val < 0:
                pts.append(p)
        return pts

    def extend(self, u, v, points):
        """."""
        if not points:
            return []

        vect2 = v - u
        w = min(points, key=lambda p: (p - u).cross(vect2) * self.normal)
        p1, p2 = self.toleft(w, v, points), self.toleft(u, w, points)
        return self.extend(w, v, p1) + [w] + self.extend(u, w, p2)
