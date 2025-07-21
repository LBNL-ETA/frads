"""Matrix generation and manipulation for Radiance simulations.

This module provides classes and functions for creating sender and receiver objects,
generating view factor and daylight matrices using Radiance tools (rfluxmtx, rcontrib),
and performing matrix operations for lighting simulations.

Key Components:
    Sender Classes:
        - SensorSender: Point sensors for illuminance calculations
        - ViewSender: Camera views for luminance images
        - SurfaceSender: Surface-based senders with sampling basis

    Receiver Classes:
        - SurfaceReceiver: Surface-based receivers with sampling basis
        - SkyReceiver: Sky dome receiver for daylight calculations
        - SunReceiver: Solar disc receiver for direct sun calculations

    Matrix Classes:
        - Matrix: General matrix for view factor and daylight calculations
        - SunMatrix: Specialized matrix for sun-only calculations

The module supports various sampling bases (Klems, Reinhart-Tregenza, Shirley-Chiu)
and can generate both dense and sparse matrices for memory efficiency.
"""

import gc
import logging
import os
from pathlib import Path
import re
from tempfile import TemporaryDirectory

from frads.geom import parse_polygon, polygon_primitive, Polygon
from frads.utils import random_string
import numpy as np
import pyradiance as pr
from scipy.sparse import csr_matrix, lil_matrix


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
    """Sender object representing a collection of point sensors.

    This class creates a sender object from a list of point sensors, where each
    sensor is defined by position and direction vectors. Used for illuminance
    calculations at specific points in space.

    Attributes:
        sensors: List of sensors, each containing 6 float values [x, y, z, dx, dy, dz]
            representing position (x,y,z) and direction (dx,dy,dz) vectors.
        content: Sensor data encoded as bytes for Radiance input.
        yres: Number of unique sensor locations (before ray multiplication).
    """

    def __init__(self, sensors: list[list[float]], ray_count: int = 1):
        """Initialize a sensor sender object.

        Args:
            sensors: List of sensors, each containing 6 float values [x, y, z, dx, dy, dz]
                representing position and direction vectors.
            ray_count: Number of rays to generate per sensor for Monte Carlo sampling.
                Higher values improve accuracy but increase computation time.
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
    """Sender object representing a camera view for luminance calculations.

    This class creates a sender object from a camera view definition, generating
    rays for each pixel in the view. Used for creating luminance images and
    view-based daylight analysis.

    Attributes:
        view: Pyradiance View object defining camera parameters (position, direction, etc.).
        content: View rays encoded as bytes for Radiance input.
        xres: Horizontal resolution (number of pixels in x-direction).
        yres: Vertical resolution (number of pixels in y-direction).
    """

    def __init__(
        self, view: pr.View, ray_count: int = 1, xres: int = 800, yres: int = 800
    ):
        """Initialize a view sender object.

        Args:
            view: Pyradiance View object defining camera parameters.
            ray_count: Number of rays per pixel for anti-aliasing and Monte Carlo sampling.
            xres: Horizontal resolution in pixels.
            yres: Vertical resolution in pixels.
        """
        self.view = view
        self.xres = xres
        self.yres = yres
        view_args = pr.get_view_args(view)
        res_eval = pr.vwrays(
            view=view_args, xres=xres, yres=yres, dimensions=True
        ).split()
        new_xres, new_yres = int(res_eval[1]), int(res_eval[3])
        vwrays_proc = pr.vwrays(
            view=view_args,
            outform="f",
            xres=new_xres,
            yres=new_yres,
            ray_count=ray_count,
            pixel_jitter=0.7,
        )
        if view.type == "a":
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
    """Sender object representing surfaces with directional sampling.

    This class creates a sender object from surface primitives using a specified
    sampling basis. Used for calculating inter-surface view factors and daylight
    transmission through surfaces like windows.

    Attributes:
        surfaces: List of surface primitives (polygons or rings) to use as senders.
        basis: Sampling basis string (e.g., 'kf', 'r4', 'sc6') defining directional
            discretization for the surface.
        content: Surface data encoded as string for Radiance rfluxmtx input.
    """

    def __init__(
        self,
        surfaces: list[pr.Primitive],
        basis: str,
        left_hand: bool = False,
        offset: float | None = None,
    ):
        """Initialize a surface sender object.

        Args:
            surfaces: List of surface primitives (polygons or rings) to use as senders.
            basis: Sampling basis string defining directional discretization
                (e.g., 'kf' for Klems full, 'r4' for Reinhart 4-division).
            left_hand: Whether to use left-hand coordinate system for basis orientation.
            offset: Distance to offset the sender surface along its normal vector.
                Useful for avoiding self-intersection in calculations.
        """
        self.surfaces = [s for s in surfaces if s.ptype in ("polygon", "ring")]
        self.basis = basis
        self.content = rfluxmtx_markup(
            self.surfaces, basis, left_hand=left_hand, offset=offset
        )

    def __eq__(self, other):
        return self.content == other.content


class Receiver:
    """Base class for all receiver objects in matrix calculations.

    This abstract base class defines the common interface for receiver objects
    that collect light from senders. All receiver types inherit from this class
    and implement specific encoding for different receiver geometries.

    Attributes:
        basis: Sampling basis string defining directional discretization for the receiver.
        content: Receiver data encoded as string for Radiance input.
    """

    def __init__(self, basis: str):
        """Initialize a receiver object.

        Args:
            basis: Sampling basis string (e.g., 'kf', 'r4', 'sc6', 'u') defining
                the directional discretization scheme for the receiver.

        Raises:
            ValueError: If the basis string format is invalid.
        """
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
    """Receiver object representing surfaces that collect light.

    This class creates a receiver object from surface primitives using a specified
    sampling basis. Used for calculating how much light surfaces receive from
    various sources in daylight and artificial lighting simulations.

    Attributes:
        surfaces: List of surface primitives (polygons or rings) acting as receivers.
        basis: Sampling basis string defining directional discretization.
        content: Surface data encoded as string for Radiance rfluxmtx input.
    """

    def __init__(
        self,
        surfaces: list[pr.Primitive],
        basis: str,
        left_hand: bool = False,
        offset: float | None = None,
        source: str = "glow",
        out: str | None = None,
    ):
        """Initialize a surface receiver object.

        Args:
            surfaces: List of surface primitives (polygons or rings) to use as receivers.
            basis: Sampling basis string defining directional discretization.
            left_hand: Whether to use left-hand coordinate system for basis orientation.
            offset: Distance to offset the receiver surface along its normal vector.
            source: Light source type for the receiver ('glow' or 'light').
            out: Optional output file path for matrix results.

        Raises:
            ValueError: If surfaces are not primitives or contain invalid types.
        """
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
    """Receiver object representing the sky dome for daylight calculations.

    This class creates a receiver object that represents the entire sky dome,
    including both sky and ground hemispheres. Used in daylight matrix calculations
    to capture contributions from all sky directions.

    Attributes:
        basis: Sampling basis string defining sky patch discretization.
        content: Sky dome definition encoded as string for Radiance input.
    """

    def __init__(self, basis: str, out: str | None = None):
        """Initialize a sky receiver object.

        Args:
            basis: Sampling basis string defining sky patch discretization
                (e.g., 'kf' for Klems full, 'r4' for Reinhart 4-division).
            out: Optional output file path for matrix results.
        """
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
    """Receiver object representing discrete solar positions for direct sun calculations.

    This class creates a receiver object that represents the sun at discrete positions
    based on a Reinhart-Tregenza sky division. The number of active sun positions
    can be reduced based on annual solar data or window visibility constraints.

    Attributes:
        basis: Sampling basis string (must be Reinhart-Tregenza format like 'r4').
        content: Sun positions encoded as Radiance light sources.
        modifiers: List of sun modifier names for active solar positions.
    """

    def __init__(
        self,
        basis: str,
        sun_matrix: None | np.ndarray = None,
        window_normals: None | list[np.ndarray] = None,
        full_mod: bool = False,
    ):
        """Initialize a sun receiver object.

        Args:
            basis: Sampling basis string (must be Reinhart-Tregenza format like 'r4').
            sun_matrix: Optional annual sun matrix to filter out zero-contribution
                solar positions. Shape should be (time_steps, sun_positions, components).
            window_normals: Optional list of window normal vectors to filter out
                sun positions not visible through windows.
            full_mod: If True, include all sun modifiers regardless of filtering.
                If False, only include modifiers for active sun positions.

        Raises:
            ValueError: If basis is not a valid Reinhart-Tregenza format.
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
    """General matrix object for daylight and view factor calculations.

    This class represents a transfer matrix between senders and receivers, encoding
    how light travels from source points/surfaces to destination points/surfaces.
    Used for daylight simulations, view factor calculations, and lighting analysis.

    Attributes:
        sender: Sender object (sensors, view, or surface) that emits light.
        receivers: List of receiver objects that collect light.
        array: Numpy array or list of arrays storing the matrix data with shape
            (nrows, ncols, ncomp) where ncomp is typically 3 for RGB.
        ncols: Number of columns (receiver basis size) or list of sizes for multiple receivers.
        nrows: Number of rows (sender basis size or sensor count).
        dtype: Matrix data type ('d' for double, 'f' for float).
        ncomp: Number of color components (typically 3 for RGB).
        octree: Path to octree file used for ray tracing acceleration.
        surfaces: List of environment surface primitives for the scene.
    """

    def __init__(
        self,
        sender: SensorSender | ViewSender | SurfaceSender,
        receivers: list[SkyReceiver] | list[SurfaceReceiver] | list[SunReceiver],
        octree: None | str = None,
        surfaces: None | list[pr.Primitive] = None,
    ):
        """Initialize a matrix object.

        Args:
            sender: Sender object (SensorSender, ViewSender, or SurfaceSender).
            receivers: List of receiver objects (SurfaceReceiver, SkyReceiver, or SunReceiver).
                Multiple receivers are only supported for SurfaceReceiver types.
            octree: Optional path to octree file for ray tracing.
            surfaces: Optional list of environment surface primitives for the scene.

        Raises:
            ValueError: If sender/receiver types are invalid or incompatible.
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
        params: list[str],
        nproc: int = 1,
        sparse: bool = False,
        to_file: bool = False,
        memmap: bool = False,
    ) -> None:
        """Generate the matrix using Radiance rfluxmtx tool.

        Args:
            params: List of rfluxmtx command-line parameters (e.g., ['-ab', '3', '-ad', '1000']).
            nproc: Number of parallel processes to use for calculation.
            sparse: If True, convert result to sparse matrix format for memory efficiency.
            to_file: If True, write matrix directly to file without storing in memory.
                Useful for very large matrices that exceed available RAM.
            memmap: If True, use memory mapping to store matrix data on disk.
                Allows handling of matrices larger than available RAM.
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
    """Specialized matrix object for direct solar radiation calculations.

    This class extends the base Matrix class to handle sun-only calculations using
    the Radiance rcontrib tool. It's optimized for sparse matrices since most
    solar positions contribute zero radiation at any given time.

    Attributes:
        sender: Sender object (SensorSender or ViewSender only).
        receiver: SunReceiver object representing discrete solar positions.
        octree: Path to octree file for ray tracing acceleration.
        surfaces: List of environment surface file paths.
        array: Sparse matrix array storing sun contribution data.
        nrows: Number of rows (sender basis size or sensor count).
        ncols: Number of columns (solar position count).
        dtype: Matrix data type ('d' for double, 'f' for float).
        ncomp: Number of color components (typically 3 for RGB).
    """

    def __init__(
        self,
        sender: SensorSender | ViewSender,
        receiver: SunReceiver,
        octree: str | None,
        surfaces: list[str] | None = None,
    ):
        """Initialize a sun matrix object.

        Args:
            sender: Sender object (SensorSender or ViewSender only).
            receiver: SunReceiver object representing discrete solar positions.
            octree: Optional path to octree file for ray tracing acceleration.
            surfaces: Optional list of environment surface file paths.

        Raises:
            TypeError: If sender is a SurfaceSender (not supported for sun matrices).
        """
        if isinstance(sender, SurfaceSender):
            raise TypeError("SurfaceSender cannot be used with SunMatrix")
        super().__init__(sender, [receiver], octree, surfaces=surfaces)
        self.surfaces = [] if surfaces is None else surfaces
        self.receiver = receiver
        self.ncols = BASIS_DIMENSION[receiver.basis] + 1

    def generate(
        self,
        parameters: list[str],
        nproc: int = 1,
        radmtx: bool = False,
        sparse: bool = False,
    ) -> None | bytes:
        """Generate the sun matrix using Radiance rcontrib tool.

        Args:
            parameters: List of rcontrib command-line parameters (e.g., ['-ab', '1', '-ad', '512']).
            nproc: Number of parallel processes to use for calculation.
            radmtx: If True, return raw matrix bytes instead of processing into numpy array.
            sparse: If True, store result as sparse matrix format (recommended for sun matrices
                due to their inherently sparse nature).

        Returns:
            Raw matrix bytes if radmtx=True, otherwise None (matrix stored in self.array).

        Raises:
            TypeError: If receiver is not a SunReceiver object.
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


def parse_rad_header(header_str: str) -> tuple[int, int, int, str]:
    """Parse a Radiance matrix file header to extract matrix dimensions and format.

    Args:
        header_str: Header string from a Radiance matrix file containing metadata.

    Returns:
        Tuple containing (nrows, ncols, ncomp, dtype) where:
            - nrows: Number of matrix rows
            - ncols: Number of matrix columns
            - ncomp: Number of color components (typically 3 for RGB)
            - dtype: Data type string (e.g., 'double', 'float')

    Raises:
        ValueError: If any required header entries (NROWS, NCOLS, NCOMP, FORMAT) are missing.
    """
    compiled = re.compile(
        r" NROWS=(.*) | NCOLS=(.*) | NCOMP=(.*) | FORMAT=(.*) ", flags=re.X
    )
    matches = compiled.findall(header_str)
    if len(matches) != 4:
        raise ValueError("Can't find one of the header entries.")
    nrow = int([mat[0] for mat in matches if mat[0] != ""][0])
    ncol = int([mat[1] for mat in matches if mat[1] != ""][0])
    ncomp = int([mat[2] for mat in matches if mat[2] != ""][0])
    dtype = [mat[3] for mat in matches if mat[3] != ""][0].strip()
    return nrow, ncol, ncomp, dtype


def load_matrix(file: bytes | str | Path, dtype: str = "float") -> np.ndarray:
    """Load a Radiance matrix file into a numpy array.

    Args:
        file: Path to Radiance matrix file or raw matrix bytes.
        dtype: Data type for the output array ('float' or 'double').

    Returns:
        Numpy array with shape (nrows, ncols, ncomp) containing the matrix data.
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
    """Load a binary matrix buffer into a numpy array.

    Args:
        buffer: Raw binary data containing the matrix.
        nrows: Number of matrix rows.
        ncols: Number of matrix columns.
        ncomp: Number of color components (typically 3 for RGB).
        dtype: Data type string ('d' for double, 'f' for float).
        header: If True, strip Radiance header from buffer before parsing.

    Returns:
        Numpy array with shape (nrows, ncols, ncomp) containing the matrix data.
    """
    npdtype = np.double if dtype.startswith("d") else np.single
    if header:
        buffer = pr.getinfo(buffer, strip_header=True)
    return np.frombuffer(buffer, dtype=npdtype).reshape(nrows, ncols, ncomp)


def matrix_multiply_rgb(
    *mtx: np.ndarray, weights: list[float] | None = None
) -> np.ndarray:
    """Multiply RGB matrices using optimized matrix multiplication.

    Performs matrix multiplication on RGB matrices by multiplying each color
    channel separately. Uses numpy's multi_dot for optimal multiplication order.

    Args:
        *mtx: Variable number of matrices to multiply, each with shape (..., ..., 3).
        weights: Optional RGB weights [r_weight, g_weight, b_weight] to combine
            color channels into a single weighted result.

    Returns:
        Result matrix with same shape as input matrices. If weights provided,
        returns single-channel weighted combination, otherwise returns RGB matrix.

    Raises:
        ValueError: If weights list doesn't contain exactly 3 values.
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
    weights: list[float] | None = None,
) -> np.ndarray:
    """Multiply view, transmission, daylight, and sky matrices for three-phase method.

    Performs the matrix multiplication sequence V × T × D × S for three-phase daylight
    calculations, handling sparse matrices efficiently for memory optimization.

    Args:
        vmx: View matrix (sparse) with shape (3, view_points, surface_patches).
        tmx: Transmission matrix (dense) with shape (surface_patches, sky_patches, 3).
        dmx: Daylight matrix (sparse) with shape (3, sky_patches, sky_directions).
        smx: Sky matrix (sparse) with shape (3, sky_directions, time_steps).
        weights: Optional RGB weights to combine color channels into weighted result.

    Returns:
        Dense numpy array with final illuminance/luminance values. Shape depends on
        whether weights are provided (single channel vs RGB).

    Raises:
        ValueError: If weights length doesn't match number of matrix channels.
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
    """Convert a three-channel RGB matrix to sparse matrix format.

    Args:
        mtx: Dense matrix with shape (..., ..., 3) representing RGB data.
        mtype: Sparse matrix type ('csr' for compressed sparse row,
            'lil' for list of lists format).

    Returns:
        Numpy array containing 3 sparse matrices, one for each RGB channel.

    Raises:
        ValueError: If matrix doesn't have exactly 3 channels.
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
    surfaces: list[pr.Primitive],
    basis: str,
    left_hand: bool = False,
    offset: float | None = None,
    source: str = "glow",
    out: str | None = None,
) -> str:
    """Generate rfluxmtx markup for surface primitives.

    Creates properly formatted Radiance scene description with rfluxmtx directives
    for matrix generation. Handles basis orientation, surface offsetting, and
    source material assignment.

    Args:
        surfaces: List of surface primitives (polygons or rings) to mark up.
        basis: Sampling basis string (e.g., 'kf', 'r4', 'sc6') for directional discretization.
        left_hand: If True, use left-hand coordinate system for basis orientation.
        offset: Distance to offset surfaces along their normal vectors.
        source: Light source material type ('glow' or 'light').
        out: Optional output file path for matrix results.

    Returns:
        String containing formatted Radiance scene with rfluxmtx directives.

    Raises:
        ValueError: If multiple surface modifiers found or invalid source type.
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
    surfaces: list[pr.Primitive],
    env: list[pr.Primitive],
    ray_count: int = 10000,
) -> dict[str, dict[str, list[float]]]:
    """Calculate surface-to-surface view factors using rfluxmtx.

    Computes geometric view factors between surfaces, representing the fraction
    of radiation leaving one surface that directly reaches another surface.

    Args:
        surfaces: List of surface primitives to calculate view factors for.
            Surface normals must face outward (away from the surface).
        env: List of environment surface primitives that surfaces are exposed to.
            Surface normals must face inward (toward the calculation domain).
        ray_count: Number of rays to spawn from each surface for Monte Carlo integration.
            Higher values improve accuracy but increase computation time.

    Returns:
        Nested dictionary where outer keys are source surface identifiers and
        inner keys are target surface identifiers, with values being RGB view factors.
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
