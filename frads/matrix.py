"""
This module contains routines to generate sender and receiver objects, generate
matrices by calling either rfluxmtx or rcontrib.
"""

import gc
import logging
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, List, Optional, Union

import numpy as np
import pyradiance as pr
from scipy.sparse import csr_matrix, lil_matrix

from frads.geom import Polygon
from frads.utils import (
    parse_polygon,
    parse_rad_header,
    polygon_primitive,
    random_string,
)

logger: logging.Logger = logging.getLogger("frads.matrix")


BASIS_DIMENSION = {
    "kf": 145,
    "kq": 41,
    "kh": 86,
    "r1": 145,
    "r2": 577,
    "r4": 2305,
    "r6": 5185,
    "sc2": 16,
    "sc4": 256,
    "sc5": 1024,
    "sc6": 4096,
    "sc7": 16384,
    "u": 1,
}


class SensorSender:
    """Sender object as a list of sensors.

    Attributes:
        sensors: A list of sensors, each sensor is a list of 6 numbers
        content: Sensor encoded for Radiance
        yres: Number of sensors
    """

    def __init__(self, sensors: List[List[float]], ray_count: int = 1):
        """Initialize a sender object with a list of sensors:

        Args:
            sensors: A list of sensors, each sensor is a list of 6 numbers
            ray_count: Number of rays per sensor
        """
        self.sensors = [i for i in sensors for _ in range(ray_count)]
        self.content = (
            os.linesep.join([" ".join(map(str, li)) for li in self.sensors])
            + os.linesep
        ).encode()
        self.yres = len(sensors)

    def __eq__(self, other):
        return self.content == other.content


class ViewSender:
    """Sender object as a view.

    Attributes:
        view: A view object
        content: View encoded for Radiance
        xres: x resolution
        yres: y resolution
    """

    def __init__(self, view: pr.View, ray_count=1, xres=800, yres=800):
        self.view = view
        self.xres = xres
        self.yres = yres
        res_eval = pr.vwrays(
            view=view.args(), xres=xres, yres=yres, dimensions=True
        ).split()
        new_xres, new_yres = int(res_eval[1]), int(res_eval[3])
        vwrays_proc = pr.vwrays(
            view=view.args(),
            outform="f",
            xres=new_xres,
            yres=new_yres,
            ray_count=ray_count,
            pixel_jitter=0.7,
        )
        if view.vtype == "a":
            ray_flush_exp = (
                f"DIM:{xres};CNT:{ray_count};"
                "pn=(recno-1)/CNT+.5;"
                "frac(x):x-floor(x);"
                "xpos=frac(pn/DIM);ypos=pn/(DIM*DIM);"
                "incir=if(.25-(xpos-.5)*(xpos-.5)-(ypos-.5)*(ypos-.5),1,0);"
                "$1=$1;$2=$2;$3=$3;$4=$4*incir;$5=$5*incir;$6=$6*incir;"
            )
            flushed_rays = pr.rcalc(
                vwrays_proc, inform="f", incount=6, outform="f", expr=ray_flush_exp
            )
            vrays = flushed_rays
        else:
            vrays = vwrays_proc
        self.content = vrays

    def __eq__(self, other):
        return (
            self.view == other.view
            and self.xres == other.xres
            and self.yres == other.yres
        )


class SurfaceSender:
    """Sender object as a list of surface primitives.

    Attributes:
        surfaces: A list of surface primitives
        basis: Sender sampling basis
        content: Surface encoded for Radiance
    """

    def __init__(
        self,
        surfaces: List[pr.Primitive],
        basis: str,
        left_hand: bool = False,
        offset: bool = None,
    ):
        """Instantiate a SurfaceSender object.

        Args:
            surfaces: A list of surface primitives
            basis: Sender sampling basis
            left_hand: Whether to use left-hand coordinate system
            offset: Offset of the sender surface
        """
        self.surfaces = [s for s in surfaces if s.ptype in ("polygon", "ring")]
        self.basis = basis
        self.content = rfluxmtx_markup(
            self.surfaces, basis, left_hand=left_hand, offset=offset
        )

    def __eq__(self, other):
        return self.content == other.content


class Receiver:
    """Base class for receiver objects.

    Attributes:
        basis: Receiver sampling basis
        content: Receiver encoded for Radiance
    """

    def __init__(self, basis):
        check_index = 0
        if basis[0] == "-":
            check_index = 1
        if basis[check_index] not in ("u", "k", "r", "s"):
            raise ValueError("Invalid basis")
        if basis[check_index] == "k" and basis[check_index + 1] not in ("f", "q", "h"):
            raise ValueError("Invalid Klems basis", basis)
        if basis[check_index] == "r" and not basis[check_index + 1].isdigit():
            raise ValueError("Invalid Reinhart/Treganza basis", basis)
        if basis[check_index] == "s" and not basis[check_index + 2].isdigit():
            raise ValueError("Invalid Shirley-Chiu basis", basis)
        self.basis = basis
        self.content = ""


class SurfaceReceiver(Receiver):
    """Receiver object as a list of surface primitives.

    Attributes:
        surfaces: A list of surface primitives
        basis: Receiver sampling basis
        content: Surface encoded for Radiance
    """

    def __init__(
        self,
        surfaces: List[pr.Primitive],
        basis: str,
        left_hand=False,
        offset=None,
        source="glow",
        out=None,
    ):
        super().__init__(basis)
        if not isinstance(surfaces[0], pr.Primitive):
            raise ValueError("Surface must be a primitive", surfaces)
        self.surfaces = [s for s in surfaces if s.ptype in ("polygon", "ring")]
        self.content = rfluxmtx_markup(
            self.surfaces,
            basis=basis,
            left_hand=left_hand,
            offset=offset,
            source=source,
            out=out,
        )


class SkyReceiver(Receiver):
    """Sky as receiver object.

    Attributes:
        basis: Receiver sampling basis
        content: Sky encoded for Radiance
    """

    def __init__(self, basis, out=None):
        super().__init__(basis)
        self.content = ""
        if out is not None:
            self.content += f"#@rfluxmtx o={out}\n"
        self.content += (
            f"#@rfluxmtx h=u\n"
            "void glow groundglow 0 0 4 1 1 1 0\n"
            "groundglow source groundplane 0 0 4 0 0 -1 180\n"
            f"#@rfluxmtx h={basis} u=+Y\n"
            "void glow skyglow 0 0 4 1 1 1 0\n"
            "skyglow source skydome 0 0 4 0 0 1 180\n"
        )


class SunReceiver(Receiver):
    """Sun as receiver object.
    The number of suns is reduced depending on the input.
    If an annual sun_matrix is provided, only the non-zero
    solar position is used. If window_normals are provided,
    only the suns that are visible through the window are used.

    Attributes:
        basis: Receiver sampling basis
        content: Sun encoded for Radiance
        modifiers: Sun modifier names
    """

    def __init__(
        self,
        basis: str,
        sun_matrix: Optional[np.ndarray] = None,
        window_normals: Optional[List[np.ndarray]] = None,
        full_mod: bool = False,
    ):
        """Initialize a sun receiver.

        Args:
            basis: Sampling basis
            sun_matrix: Annual sun matrix
            window_normals: List of window normals
            full_mod: Use full set of sun modifiers
        """
        super().__init__(basis)
        if not basis.startswith("r") and not basis[-1].isdigit():
            raise ValueError("Invalid Reinhart/Treganza basis", basis)
        mf = int(basis[-1])
        nbins = 144 * mf**2 + 1
        res = pr.rcalc(
            inp=pr.cnt(nbins),
            outform="f",
            expr=f"MF:{mf};Rbin=recno;$1=Dx;$2=Dy;$3=Dz",
            source="reinsrc.cal",
        )
        sundirs = np.frombuffer(res, dtype=np.single).reshape(nbins, 3)
        sunvals = np.ones(BASIS_DIMENSION[basis])
        if sun_matrix is not None:
            sunvals[np.sum(sun_matrix[1:, :, 0], axis=1) == 0] = 0
        if window_normals is not None:
            sunvals[np.dot(sundirs, np.array(window_normals).T).flatten() >= 0] = 0
        self.content = "\n".join(
            f"void light sol{i} 0 0 3 {d} {d} {d} sol{i} source sun 0 0 4 {sundirs[i][0]:.6g} {sundirs[i][1]:.6g} {sundirs[i][2]:.6g} 0.533"
            for i, d in enumerate(sunvals)
        )
        if full_mod:
            self.modifiers = [f"sol{i}" for i in range(nbins)]
        else:
            self.modifiers = [f"sol{i}" for i in np.where(sunvals > 0)[0]]


class Matrix:
    """Base Matrix object.

    Attributes:
        sender: Sender object
        receivers: List of receiver objects
        array: Matrix array, this is usually where the matrix data is stored
        ncols: Number of columns
        nrows: Number of rows
        dtype: Matrix data type
        ncomp: Number of components
        octree: Octree file used to generate the matrix
        surfaces: List of environment surface primitives
    """

    def __init__(
        self,
        sender: Union[SensorSender, ViewSender, SurfaceSender],
        receivers: Union[List[SkyReceiver], List[SurfaceReceiver], List[SunReceiver]],
        octree: Optional[str] = None,
        surfaces: Optional[List[pr.Primitive]] = None,
    ):
        """Initialize a matrix.

        Args:
            sender: Sender object
            receivers: List of receiver objects
            octree: Octree file used to generate the matrix
            surfaces: List of environment surface file paths
        """
        if not isinstance(sender, (SensorSender, ViewSender, SurfaceSender)):
            raise ValueError(
                "Sender must be a SensorSender, ViewSender, or SurfaceSender"
            )
        if not isinstance(receivers[0], (SurfaceReceiver, SkyReceiver, SunReceiver)):
            raise ValueError(
                "Receiver must be a SurfaceReceiver, SkyReceiver, or SunReceiver"
            )
        if len(receivers) > 1:
            if not all(isinstance(r, SurfaceReceiver) for r in receivers):
                raise ValueError("All receivers must be SurfaceReceivers")
        self.sender = sender
        self.receivers = receivers
        self.array = None
        self.ncols = None
        self.nrows: int
        self.dtype = "d"
        self.ncomp = 3
        if isinstance(sender, SensorSender):
            self.nrows = sender.yres
        elif isinstance(sender, SurfaceSender):
            self.nrows = BASIS_DIMENSION[sender.basis]
        elif isinstance(sender, ViewSender):
            self.nrows = sender.yres * sender.xres
        if isinstance(receivers[0], SurfaceReceiver):
            self.ncols = [BASIS_DIMENSION[r.basis] for r in receivers]
        elif isinstance(receivers[0], SkyReceiver):
            self.ncols = BASIS_DIMENSION[receivers[0].basis] + 1
        self.octree = octree
        self.surfaces = surfaces

    def generate(
        self,
        params: List[str],
        nproc: int = 1,
        sparse: bool = False,
        to_file: bool = False,
        memmap: bool = False,
    ) -> None:
        """Call rfluxmtx to generate the matrix.

        Args:
            params: List of rfluxmtx parameters
            nproc: Number of processes to use
            sparse: Use sparse matrix format
            to_file: Indicate the matrix will be written to file
                directly as specified by rfluxmtx markup. As a
                result, the matrix will not be stored in memory
                as numpy array.
            memmap: Use memory mapping (out-of-core) to store
                the matrix, which is useful for large matrices.
        """
        surface_file = None
        rays = None
        params.append("-n")
        params.append(f"{nproc}")
        logger.info("Generating matrix...")
        if logger.getEffectiveLevel() > 20:
            params.append("-w")
        if not isinstance(self.sender, SurfaceSender):
            rays = self.sender.content
            params.append("-y")
            params.append(str(self.sender.yres))
            if isinstance(self.sender, SensorSender):
                params.append("-I")
                params.append("-fad")
            elif isinstance(self.sender, ViewSender):
                params.append("-x")
                params.append(str(self.sender.xres))
                params.append("-ffd")
        with TemporaryDirectory() as tmpdir:
            receiver_file = os.path.join(tmpdir, "receiver")
            with open(receiver_file, "w") as f:
                [f.write(r.content) for r in self.receivers]
            env_file = None
            if self.surfaces is not None:
                env_file = os.path.join(tmpdir, "scene")
                with open(env_file, "wb") as f:
                    f.write(b" ".join(s.bytes for s in self.surfaces))
                env_file = [env_file]
            if isinstance(self.sender, SurfaceSender):
                params.append("-ffd")
                surface_file = os.path.join(tmpdir, "surface")
                with open(surface_file, "w") as f:
                    f.write(self.sender.content)
            matrix = pr.rfluxmtx(
                receiver_file,
                surface=surface_file,
                rays=rays,
                params=params,
                octree=self.octree,
                scene=env_file,
            )
        if not to_file:
            _ncols = sum(self.ncols) if isinstance(self.ncols, list) else self.ncols
            _array = load_binary_matrix(
                matrix,
                self.nrows,
                _ncols,
                self.ncomp,
                self.dtype,
                header=True,
            )
            del matrix
            gc.collect()
            if memmap:
                _dtype = np.float64 if self.dtype.startswith("d") else np.float32
                self.array = np.memmap(
                    f"{random_string(5)}.dat",
                    dtype=_dtype,
                    shape=(self.nrows, _ncols, self.ncomp),
                    order="F",
                    mode="w+",
                )
                self.array[:] = _array
                self.array.flush()
            else:
                self.array = _array
            del _array
            gc.collect()
            # If multiple receivers, split the array horizontally
            if isinstance(self.ncols, list):
                self.array = np.hsplit(self.array, np.cumsum(self.ncols)[:-1])
                if sparse:
                    self.array = np.array(
                        [
                            np.array(
                                (
                                    csr_matrix(a[:, :, 0]),
                                    csr_matrix(a[:, :, 1]),
                                    csr_matrix(a[:, :, 2]),
                                )
                            )
                            for a in self.array
                        ]
                    )
            elif sparse:
                self.array = np.array(
                    (
                        csr_matrix(self.array[:, :, 0]),
                        csr_matrix(self.array[:, :, 1]),
                        csr_matrix(self.array[:, :, 2]),
                    )
                )


class SunMatrix(Matrix):
    """Sun Matrix object, specialized for sun-only matrices.

    Attributes:
        sender: Sender object
        receiver: Receiver object
        octree: Octree file
        surfaces: List of surface files
        array: Matrix array
        nrows: Number of rows
        ncols: Number of columns
        dtype: Data type
        ncomp: Number of components
    """

    def __init__(
        self,
        sender: Union[SensorSender, ViewSender],
        receiver: SunReceiver,
        octree: Optional[str],
        surfaces: Optional[List[str]] = None,
    ):
        if isinstance(sender, SurfaceSender):
            raise TypeError("SurfaceSender cannot be used with SunMatrix")
        super().__init__(sender, [receiver], octree, surfaces=surfaces)
        self.surfaces = [] if surfaces is None else surfaces
        self.receiver = receiver
        self.ncols = BASIS_DIMENSION[receiver.basis] + 1

    def generate(
        self,
        parameters: List[str],
        nproc: int = 1,
        radmtx: bool = False,
        sparse: bool = False,
    ) -> Optional[bytes]:
        """Call rcontrib to generate a matrix and store it in memory
        as a numpy array.

        Args:
            parameters: List of rcontrib parameters
            nproc: Number of processes to use
            radmtx: If true, return the matrix bytes from rcontrib directly
            sparse: Use sparse matrix format, this is usually a good idea
                given the nature of sun-only matrices.
        """
        if not isinstance(self.receiver, SunReceiver):
            raise TypeError("SunMatrix must have a SunReceiver")
        xres, yres = None, None
        inform = "a"
        parameters.append("-n")
        parameters.append(f"{nproc}")
        parameters.append("-h")
        if logger.getEffectiveLevel() > 20:
            parameters.append("-w")
        logger.info("Generating matrix...")
        with TemporaryDirectory() as tmpdir:
            octree_file = os.path.join(tmpdir, "octree")
            receiver_file = os.path.join(tmpdir, "receiver")
            modifier_file = os.path.join(tmpdir, "modifier")
            with open(modifier_file, "w") as f:
                f.write("\n".join(self.receiver.modifiers))
            with open(receiver_file, "w") as f:
                f.write(self.receiver.content)
            with open(octree_file, "wb") as f:
                f.write(pr.oconv(receiver_file, *self.surfaces, octree=self.octree))
            if isinstance(self.sender, SensorSender):
                parameters.append("-I+")
                yres = self.sender.yres
            elif isinstance(self.sender, ViewSender):
                inform = "f"
                xres = self.sender.xres
                yres = self.sender.yres
            modifier = pr.RcModifier()
            modifier.modifier_path = modifier_file
            modifier.xres = xres
            modifier.yres = yres
            matrix = pr.rcontrib(
                self.sender.content,
                octree_file,
                [modifier],
                inform=inform,
                outform="d",
                params=parameters,
            )
        if radmtx:
            return matrix
        _array = load_binary_matrix(
            matrix,
            self.nrows,
            min(self.ncols, len(self.receiver.modifiers)),
            self.ncomp,
            self.dtype,
            header=False,
        )
        # Convert to sparse matrix
        sparse_r = lil_matrix(_array[:, :, 0])
        sparse_g = lil_matrix(_array[:, :, 1])
        sparse_b = lil_matrix(_array[:, :, 2])
        del matrix, _array
        gc.collect()
        # Fill the matrix back up to full basis size if culled
        if len(self.receiver.modifiers) < self.ncols:
            indices = [int(m.lstrip("sol")) for m in self.receiver.modifiers]
            padded_sparse_r = lil_matrix((self.nrows, self.ncols))
            padded_sparse_g = lil_matrix((self.nrows, self.ncols))
            padded_sparse_b = lil_matrix((self.nrows, self.ncols))
            padded_sparse_r[:, indices] = sparse_r
            padded_sparse_g[:, indices] = sparse_g
            padded_sparse_b[:, indices] = sparse_b
            self.array = np.array(
                (
                    padded_sparse_r.tocsr(),
                    padded_sparse_g.tocsr(),
                    padded_sparse_b.tocsr(),
                )
            )
        else:
            self.array = np.array(
                (sparse_r.tocsr(), sparse_g.tocsr(), sparse_b.tocsr())
            )


def load_matrix(file: Union[bytes, str, Path], dtype: str = "float") -> np.ndarray:
    """Load a Radiance matrix file into numpy array.

    Args:
        file: a file path
        dtype: data type

    Returns:
        A numpy array
    """
    npdtype = np.double if dtype.startswith("d") else np.single
    mtx = pr.rmtxop(file, outform=dtype[0].lower())
    nrows, ncols, ncomp, _ = parse_rad_header(pr.getinfo(mtx).decode())
    return np.frombuffer(pr.getinfo(mtx, strip_header=True), dtype=npdtype).reshape(
        nrows, ncols, ncomp
    )


def load_binary_matrix(
    buffer: bytes, nrows: int, ncols: int, ncomp: int, dtype: str, header: bool = False
) -> np.ndarray:
    """Load a matrix in binary format into a numpy array.

    Args:
        buffer: buffer to read from
        nrows: number of rows
        ncols: number of columns
        ncomp: number of components
        dtype: data type
        header: if True, strip header

    Returns:
        The matrix as a numpy array
    """
    npdtype = np.double if dtype.startswith("d") else np.single
    if header:
        buffer = pr.getinfo(buffer, strip_header=True)
    return np.frombuffer(buffer, dtype=npdtype).reshape(nrows, ncols, ncomp)


def matrix_multiply_rgb(
    *mtx: np.ndarray, weights: Optional[List[float]] = None
) -> np.ndarray:
    """Multiply matrices as numpy ndarray.
    linalg.multi_dot figures out the multiplication order
    but uses more memory.

    Args:
        mtx: matrices to multiply
        weights: weights for each component

    Returns:
        The result as a numpy array
    """
    resr = np.linalg.multi_dot([m[:, :, 0] for m in mtx])
    resg = np.linalg.multi_dot([m[:, :, 1] for m in mtx])
    resb = np.linalg.multi_dot([m[:, :, 2] for m in mtx])
    if weights:
        if len(weights) != 3:
            raise ValueError("Weights should have 3 values")
        return resr * weights[0] + resg * weights[1] + resb * weights[2]
    return np.dstack((resr, resg, resb))


def sparse_matrix_multiply_rgb_vtds(
    vmx: np.ndarray,
    tmx: np.ndarray,
    dmx: np.ndarray,
    smx: np.ndarray,
    weights: Optional[List[float]] = None,
) -> np.ndarray:
    """Multiply sparse view, transmission, daylight,
    and sky matrices. (ThreePhaseMethod)

    Args:
        vmx: view matrix (sparse)
        tmx: transmission matrix (non-sparse)
        dmx: daylight matrix (sparse)
        smx: sky matrix (sparse)
        weights: weights for RGB channels

    Returns:
        The result as a numpy array (non sparse)
    """
    if weights is not None:
        if len(weights) != vmx.shape[0]:
            raise ValueError("Mismatch between weights and matrix channels")
    # Looping through each of the RGB channels
    _res = []
    for c in range(vmx.shape[0]):
        td_mtx = np.dot(csr_matrix(tmx[:, :, c]), dmx[c])
        tds_mtx = np.dot(td_mtx, smx[c])
        _res.append(np.dot(vmx[c], tds_mtx))
    if weights is not None:
        res = np.zeros((vmx[0].shape[0], smx[0].shape[1]))
        for i, w in enumerate(weights):
            res += _res[i] * w
    else:
        res = np.dstack(_res)
    return res


def to_sparse_matrix3(mtx: np.ndarray, mtype: str = "csr") -> np.ndarray:
    """Convert a three-channel matrix to sparse matrix.

    Args:
        mtx: a three-channel matrix
        mtype: matrix type

    Returns:
        An array of sparse matrix
    """
    sparser = {
        "csr": csr_matrix,
        "lil": lil_matrix,
    }
    if mtx.shape[2] != 3:
        raise ValueError("Matrix must have 3 channels")
    return np.array(
        (
            sparser[mtype](mtx[:, :, 0]),
            sparser[mtype](mtx[:, :, 1]),
            sparser[mtype](mtx[:, :, 2]),
        )
    )


def rfluxmtx_markup(
    surfaces: List[pr.Primitive],
    basis: str,
    left_hand: bool = False,
    offset: Optional[float] = None,
    source: str = "glow",
    out: Optional[str] = None,
) -> str:
    """Mark up surfaces for rfluxmtx.

    Args:
        surfaces: list of surfaces
        basis: basis type
        left_hand: left hand coordinate system
        offset: offset
        source: source type
        out: output file

    Returns:
        Marked up primitives as strings (to be written to a file for rfluxmtx)
    """
    modifier_set = {p.modifier for p in surfaces}
    source_modifier = f"rflx{surfaces[0].modifier}{random_string(10)}"
    if left_hand:
        basis = "-" + basis
    if source not in ("glow", "light"):
        raise ValueError("Invalid source")
    primitives = [p for p in surfaces if p.ptype in ("polygon", "ring")]
    surface_normal = np.zeros(3)
    for primitive in primitives:
        polygon = parse_polygon(primitive)
        surface_normal += polygon.area
    sampling_direction = surface_normal * (1 / len(primitives))
    sampling_direction = sampling_direction / np.linalg.norm(sampling_direction)
    if np.array_equal(sampling_direction, np.array([0, 0, 1])):
        up_vector = np.array([0, 1, 0])
    elif np.array_equal(sampling_direction, np.array([0, 0, -1])):
        up_vector = np.array([0, -1, 0])
    else:
        up_vector = np.cross(
            sampling_direction, np.cross(np.array([0, 0, 1]), sampling_direction)
        )
        up_vector = up_vector / np.linalg.norm(up_vector)
    if left_hand:
        up_vector = -up_vector
    up_vector = ",".join(map(str, up_vector.tolist()))
    if len(modifier_set) > 1:
        raise ValueError("Multiple modifiers")
    header = f"#@rfluxmtx h={basis} u={up_vector}\n"
    if out is not None:
        header += f'#@rfluxmtx o="{out}"\n\n'
    if source == "glow":
        source_prim = pr.Primitive("void", source, source_modifier, [], [1, 1, 1, 0])
        header += str(source_prim)
    elif source == "light":
        source_prim = pr.Primitive("void", source, source_modifier, [], [1, 1, 1])
        header += str(source_prim)
    content = ""
    for prim in surfaces:
        if prim.identifier in modifier_set:
            _identifier = "discarded"
        else:
            _identifier = prim.identifier
        _modifier = source_modifier
        if offset is not None:
            poly = parse_polygon(prim)
            offset_vec = poly.normal * offset
            moved_pts = [pt + offset_vec for pt in poly.vertices]
            _real_args = Polygon(moved_pts).vertices.flatten().tolist()
        else:
            _real_args = prim.fargs
        new_prim = pr.Primitive(
            _modifier, prim.ptype, _identifier, prim.sargs, _real_args
        )
        content += str(new_prim) + "\n"

    return header + content


def surfaces_view_factor(
    surfaces: List[pr.Primitive],
    env: List[pr.Primitive],
    ray_count: int = 10000,
) -> Dict[str, Dict[str, List[float]]]:
    """Calculate surface to surface view factor using rfluxmtx.

    Args:
        surfaces: list of surface primitives that we want to calculate view factor for.
            Surface normal needs to be facing outward.
        env: list of environment primitives that our surfaces will be exposed to.
            Surface normal needs to be facing inward.
        ray_count: number of rays spawned for each surface.

    Returns:
        A dictionary of view factors, where the key is the surface identifier.
    """
    view_factors = {}
    surfaces_env = env + surfaces
    for idx, surface in enumerate(surfaces):
        sender = SurfaceSender([surface], basis="u")
        rest_of_surfaces = surfaces[:idx] + surfaces[idx + 1 :]
        rest_of_surface_polygons = [parse_polygon(s) for s in rest_of_surfaces]
        rest_of_surfaces_flipped = [
            polygon_primitive(p.flip(), s.modifier, s.identifier)
            for s, p in zip(rest_of_surfaces, rest_of_surface_polygons)
        ]
        receivers = [
            SurfaceReceiver([s], basis="u")
            for s in env + rest_of_surfaces_flipped
            if s.ptype in ("polygon", "ring")
        ]
        mat = Matrix(sender, receivers, octree=None, surfaces=surfaces_env)
        mat.generate(params=["-c", f"{ray_count}"])
        view_factors[surface.identifier] = {
            r.surfaces[0].identifier: [i][0] for r, i in zip(mat.receivers, mat.array)
        }
    return view_factors
