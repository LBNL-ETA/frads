"""Typical Radiance matrix-based simulation workflows
"""

from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import logging
import os
from pathlib import Path
from typing import Any, ByteString, Dict, List, Optional, Union
from shutil import rmtree

from frads.matrix import (
    BASIS_DIMENSION,
    load_matrix,
    load_binary_matrix,
    Matrix,
    matrix_multiply_rgb,
    SensorSender,
    SkyReceiver,
    SunReceiver,
    SunMatrix,
    SurfaceSender,
    SurfaceReceiver,
    ViewSender,
    sparse_matrix_multiply_rgb_vtds,
    to_sparse_matrix3,
)
from frads.sky import (
    WeaData,
    WeaMetaData,
    gen_perez_sky,
    parse_epw,
    parse_wea,
)
from frads.utils import (
    minutes_to_datetime,
    parse_polygon,
    parse_rad_header,
    random_string,
)
import numpy as np
import pyradiance as pr
from pyradiance.model import parse_view
from scipy.sparse import csr_matrix


logger: logging.Logger = logging.getLogger("frads.methods")


@dataclass
class SceneConfig:
    """
    SceneConfig is a dataclass that holds the information needed to generate
    a Radiance scene. It can be initialized with either a raw data string or a list
    of files. If a list of files is provided, they will be concatenated in
    the order they are provided.
    """

    files: List[Path] = field(default_factory=list)
    bytes: ByteString = b""
    files_mtime: List[float] = field(init=False, default_factory=list)

    def __post_init__(self):
        if len(self.files) > 0:
            for fpath in self.files:
                self.files_mtime.append(os.path.getmtime(fpath))


@dataclass
class MaterialConfig:
    files: List[Path] = field(default_factory=list)
    bytes: ByteString = b""
    files_mtime: List[float] = field(init=False, default_factory=list)

    def __post_init__(self):
        if len(self.files) > 0:
            for fpath in self.files:
                self.files_mtime.append(os.path.getmtime(fpath))


@dataclass
class WindowConfig:
    file: Union[str, Path] = ""
    bytes: ByteString = b""
    matrix_file: Union[str, Path] = ""
    matrix_data: Optional[List[List[float]]] = field(default_factory=list)
    shading_geometry_file: Union[str, Path] = ""
    shading_geometry_bytes: Optional[ByteString] = None
    tensor_tree_file: Union[str, Path] = ""
    files_mtime: List[float] = field(init=False, default_factory=list)

    def __post_init__(self):
        if os.path.exists(self.file):
            self.files_mtime.append(os.path.getmtime(self.file))
            if not isinstance(self.file, Path):
                self.file = Path(self.file)
        if os.path.exists(self.matrix_file): 
            self.files_mtime.append(os.path.getmtime(self.matrix_file))
            if not isinstance(self.matrix_file, Path):
                self.matrix_file = Path(self.matrix_file)
        if os.path.exists(self.shading_geometry_file): 
            self.files_mtime.append(os.path.getmtime(self.shading_geometry_file))
            if not isinstance(self.shading_geometry_file, Path):
                self.shading_geometry_file = Path(self.shading_geometry_file)
        if os.path.exists(self.tensor_tree_file):
            self.files_mtime.append(os.path.getmtime(self.tensor_tree_file))
            if not isinstance(self.tensor_tree_file, Path):
                self.tensor_tree_file = Path(self.tensor_tree_file)
        if self.bytes == b"":
            with open(self.file, "rb") as f:
                self.bytes = f.read()


@dataclass
class SensorConfig:
    file: str = ""
    data: List[List[float]] = field(default_factory=list)
    file_mtime: float = field(init=False, default=0.0)

    def __post_init__(self):
        if self.file != "":
            self.file_mtime = os.path.getmtime(self.file)
        if len(self.data) == 0:
            if self.file != "":
                with open(self.file) as f:
                    self.data = [
                        [float(i) for i in line.split()] for line in f.readlines()
                    ]
            else:
                raise ValueError("SensorConfig must have either file or data")


@dataclass
class ViewConfig:
    file: Union[str, Path] = ""
    view: Union[pr.View, str] = field(default_factory=str)
    xres: int = 512
    yres: int = 512
    file_mtime: float = field(init=False, default=0.0)

    def __post_init__(self):
        if self.file != "":
            self.file_mtime = os.path.getmtime(self.file)
        if not isinstance(self.file, Path):
            self.file = Path(self.file)
        if self.file.exists() and self.view == "":
            self.view = pr.load_views(self.file)[0]
        elif not isinstance(self.view, pr.View):
            self.view = parse_view(self.view)


@dataclass
class Settings:
    name: str = field(default="")
    num_processors: int = 1
    method: str = field(default="3phase")
    overwrite: bool = False
    save_matrices: bool = False
    sky_basis: str = field(default="r1")
    window_basis: str = field(default="kf")
    non_coplanar_basis: str = field(default="kf")
    sun_basis: str = field(default="r6")
    sun_culling: bool = field(default=True)
    separate_direct: bool = field(default=False)
    epw_file: str = field(default="")
    wea_file: str = field(default="")
    start_hour: int = field(default=8)
    end_hour: int = field(default=18)
    daylight_hours_only: bool = True
    latitude: int = field(default=37)
    longitude: int = field(default=122)
    time_zone: int = field(default=120)
    orientation: int = field(default=0)
    site_elevation: int = field(default=100)
    sensor_sky_matrix: List[str] = field(
        default_factory=lambda: ["-ab", "6", "-ad", "8192", "-lw", "5e-5"]
    )
    sensor_sun_matrix: List[str] = field(
        default_factory=lambda: [
            "-ab",
            "1",
            "-ad",
            "256",
            "-lw",
            "1e-3",
            "-dj",
            "0",
            "-st",
            "0",
        ]
    )
    view_sun_matrix: List[str] = field(
        default_factory=lambda: [
            "-ab",
            "1",
            "-ad",
            "256",
            "-lw",
            "1e-3",
            "-dj",
            "0",
            "-st",
            "0",
        ]
    )
    view_sky_matrix: List[str] = field(
        default_factory=lambda: ["-ab", "6", "-ad", "8192", "-lw", "5e-5"]
    )
    daylight_matrix: List[str] = field(
        default_factory=lambda: ["-ab", "2", "-c", "5000"]
    )
    sensor_window_matrix: List[str] = field(
        default_factory=lambda: ["-ab", "5", "-ad", "8192", "-lw", "5e-5"]
    )
    view_window_matrix: List[str] = field(
        default_factory=lambda: ["-ab", "5", "-ad", "8192", "-lw", "5e-5"]
    )
    files_mtime: List[float] = field(init=False, default_factory=list)

    def __post_init__(self):
        if self.wea_file != "":
            self.files_mtime.append(os.path.getmtime(self.wea_file))
        if self.epw_file != "":
            self.files_mtime.append(os.path.getmtime(self.epw_file))


@dataclass
class Model:
    scene: "SceneConfig"
    windows: Dict[str, "WindowConfig"]
    materials: "MaterialConfig"
    sensors: Dict[str, "SensorConfig"]
    views: Dict[str, "ViewConfig"]

    # Make Path() out of all path strings
    def __post_init__(self):
        if isinstance(self.scene, dict):
            self.scene = SceneConfig(**self.scene)
        if isinstance(self.materials, dict):
            self.materials = MaterialConfig(**self.materials)
        for k, v in self.windows.items():
            if isinstance(v, dict):
                self.windows[k] = WindowConfig(**v)
        for k, v in self.sensors.items():
            if isinstance(v, dict):
                self.sensors[k] = SensorConfig(**v)
        for k, v in self.views.items():
            if isinstance(v, dict):
                self.views[k] = ViewConfig(**v)


@dataclass
class WorkflowConfig:
    settings: "Settings"
    model: "Model"
    hash_str: str = field(init=False)

    def __post_init__(self):
        if isinstance(self.settings, dict):
            self.settings = Settings(**self.settings)
        if isinstance(self.model, dict):
            self.model = Model(**self.model)
        self.hash_str = hashlib.md5(str(self.__dict__).encode()).hexdigest()[:16]

    @staticmethod
    def from_dict(obj: Dict[str, Any]) -> "WorkflowConfig":
        settings = Settings(**obj["settings"])
        model = Model(**obj["model"])
        return WorkflowConfig(settings, model)


class PhaseMethod:
    """
    Base class for phase methods.
    This class is not meant to be used directly.
    """

    def __init__(self, config):
        """
        Initialize a phase method.

        Args:
            config: A WorkflowConfig object.

        """
        self.config = config

        # Setup the view and sensor senders
        self.view_senders = {}
        self.sensor_senders = {}
        for name, sensors in self.config.model.sensors.items():
            self.sensor_senders[name] = SensorSender(sensors.data)
        for name, view in self.config.model.views.items():
            self.view_senders[name] = ViewSender(
                view.view, xres=view.xres, yres=view.yres
            )

        # Setup the sky receiver object
        self.sky_receiver = SkyReceiver(self.config.settings.sky_basis)

        # Figure out the weather related stuff
        if self.config.settings.epw_file != "":
            with open(self.config.settings.epw_file) as f:
                self.wea_metadata, self.wea_data = parse_epw(f.read())
            self.wea_header = self.wea_metadata.wea_header()
            self.wea_str = self.wea_header + "\n".join(str(d) for d in self.wea_data)
        elif self.config.settings.wea_file != "":
            with open(self.config.settings.wea_file) as f:
                self.wea_metadata, self.wea_data = parse_wea(f.read())
            self.wea_header = self.wea_metadata.wea_header()
            self.wea_str = self.wea_header + "\n".join(str(d) for d in self.wea_data)
        else:
            if (
                self.config.settings.latitude is None
                or self.config.settings.longitude is None
            ):
                raise ValueError(
                    "Latitude and longitude must be specified if no weather file is given"
                )
            self.wea_header = (
                f"place {self.config.settings.latitude}_{self.config.settings.longitude}\n"
                f"latitude {self.config.settings.latitude}\n"
                f"longitude {self.config.settings.longitude}\n"
                f"time_zone {self.config.settings.time_zone}\n"
                f"site_elevation {self.config.settings.site_elevation}\n"
                f"weather_data_file_units 1\n"
            )
            self.wea_metadata = WeaMetaData(
                "city",
                "country",
                self.config.settings.latitude,
                self.config.settings.longitude,
                self.config.settings.time_zone,
                self.config.settings.site_elevation,
            )

        # Setup Temp and Octrees directory
        self.tmpdir = Path("Temp")
        self.tmpdir.mkdir(exist_ok=True)
        self.octdir = Path("Octrees")
        self.octdir.mkdir(exist_ok=True)
        self.mtxdir = Path("Matrices")
        self.mtxdir.mkdir(exist_ok=True)
        self.mfile = (self.mtxdir / self.config.hash_str).with_suffix(".npz")

        # Generate a base octree
        self.octree = self.octdir / f"{random_string(5)}.oct"

    def __enter__(self):
        """
        Context manager enter method. This method is called when
        the class is used as a context manager. Anything happens
        after the with statement is run.
        """
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        """
        Context manager exit method. This method is called when
        the class is used as a context manager. Cleans up the
        Temp and Octrees directory.
        """
        rmtree("Octrees")
        rmtree("Matrices")
        rmtree("Temp")

    def generate_matrices(self):
        raise NotImplementedError

    def calculate_view(self, view, time, dni, dhi):
        raise NotImplementedError

    def calculate_sensor(self, sensor, time, dni, dhi):
        raise NotImplementedError

    def get_sky_matrix(self, time, dni, dhi):
        _wea = self.wea_header
        _wea += str(WeaData(time, dni, dhi))
        smx = pr.gendaymtx(
            _wea.encode(),
            outform="d",
            mfactor=int(self.config.settings.sky_basis[-1]),
            header=False,
        )
        return load_binary_matrix(
            smx,
            nrows=BASIS_DIMENSION[self.config.settings.sky_basis] + 1,
            ncols=1,
            ncomp=3,
            dtype="d",
        )

    def get_sky_matrix_from_wea(self, mfactor: int, sun_only=False, onesun=False):
        if self.wea_str is None:
            raise ValueError("No weather string available")
        _sun_str = pr.gendaymtx(
            self.wea_str.encode(),
            sun_file="-",
            dryrun=True,
            daylight_hours_only=True,
        )
        prims = pr.parse_primitive(_sun_str.decode())
        _datetimes = [
            minutes_to_datetime(2023, int(p.identifier.lstrip("solar")))
            for p in prims
            if p.ptype == "light"
        ]
        _matrix = pr.gendaymtx(
            self.wea_str.encode(),
            sun_only=sun_only,
            onesun=onesun,
            outform="d",
            daylight_hours_only=True,
            mfactor=mfactor,
        )
        _nrows, _ncols, _ncomp, _dtype = parse_rad_header(pr.getinfo(_matrix).decode())
        return load_binary_matrix(
            _matrix,
            nrows=_nrows,
            ncols=_ncols,
            ncomp=_ncomp,
            dtype=_dtype,
            header=True,
        )

    def save_matrices(self):
        raise NotImplementedError


class TwoPhaseMethod(PhaseMethod):
    """
    Implements two phase method
    """

    def __init__(self, config):
        """
        Initializes the two phase method
        Args:
            config: A WorkflowConfig object
        """
        super().__init__(config)
        oct_stdin = config.model.materials.bytes + config.model.scene.bytes
        for window in config.model.windows.values():
            oct_stdin += window.bytes
        with open(self.octree, "wb") as f:
            f.write(
                pr.oconv(
                    *config.model.materials.files,
                    *config.model.scene.files,
                    stdin=oct_stdin,
                )
            )
        self.view_sky_matrices = {}
        self.sensor_sky_matrices = {}
        for vs in self.view_senders:
            self.view_sky_matrices[vs] = Matrix(
                self.view_senders[vs], [self.sky_receiver], self.octree
            )
        for ss in self.sensor_senders:
            self.sensor_sky_matrices[ss] = Matrix(
                self.sensor_senders[ss], [self.sky_receiver], self.octree
            )

    def generate_matrices(self) -> None:
        """
        Generate matrices for all view and sensor points
        Args:
            save: Save matrices to a .npz file
            overwrite: Overwrite existing matrices
        """
        # First check if matrices files already exist
        if self.mfile.exists() and (not self.config.settings.overwrite):
            self.load_matrices()
            return
        # Then check if overwrite is set to True
        for _, mtx in self.view_sky_matrices.items():
            mtx.generate(
                self.config.settings.view_sky_matrix,
                nproc=self.config.settings.num_processors,
            )
        for _, mtx in self.sensor_sky_matrices.items():
            mtx.generate(
                self.config.settings.sensor_sky_matrix,
                nproc=self.config.settings.num_processors,
            )
        if self.config.settings.save_matrices:
            self.save_matrices()

    def load_matrices(self):
        """
        Load matrices from a .npz file
        """
        logger.info(f"Loading matrices from {self.mfile}")
        if not self.mfile.exists():
            raise FileNotFoundError("Matrices file not found")
        mdata = np.load(self.mfile)
        for view, mtx in self.view_sky_matrices.items():
            mtx.array = mdata[f"{view}_sky_matrix"]
        for sensor, mtx in self.sensor_sky_matrices.items():
            mtx.array = mdata[f"{sensor}_sky_matrix"]

    def calculate_view(self, view, time, dni, dhi):
        sky_matrix = self.get_sky_matrix(time, dni, dhi)
        return matrix_multiply_rgb(self.view_sky_matrices[view].array, sky_matrix)

    def calculate_sensor(self, sensor, time, dni, dhi):
        sky_matrix = self.get_sky_matrix(time, dni, dhi)
        return matrix_multiply_rgb(
            self.sensor_sky_matrices[sensor].array,
            sky_matrix,
            weights=[47.4, 119.9, 11.6],
        )

    def calculate_view_from_wea(self, view):
        if self.wea_data is None:
            raise ValueError("No wea data available")
        sky_matrix = self.get_sky_matrix_from_wea(
            int(self.config.settings.sky_basis[-1])
        )
        # arbitrary chunksize
        chunksize = 300
        shape = (
            self.view_sky_matrices[view].nrows,
            sky_matrix.shape[1],
            3,
        )
        final = np.memmap(
            f"{view}_2ph.dat", shape=shape, dtype=np.float64, mode="w+", order="F"
        )
        for idx in range(0, sky_matrix.shape[1], chunksize):
            end = min(idx + chunksize, sky_matrix.shape[1])
            _chunksize = end - idx
            res = matrix_multiply_rgb(
                self.view_sky_matrices[view].array,
                sky_matrix[:, idx:end, :],
            )
            final[:, idx:end, 0] = res[:, :, 0]
            final[:, idx:end, 1] = res[:, :, 1]
            final[:, idx:end, 2] = res[:, :, 2]
            final.flush()
        return final

    def calculate_sensor_from_wea(self, sensor):
        if self.wea_data is None:
            raise ValueError("No wea data available")
        return matrix_multiply_rgb(
            self.sensor_sky_matrices[sensor].array,
            self.get_sky_matrix_from_wea(int(self.config.settings.sky_basis[-1])),
            weights=[47.4, 119.9, 11.6],
        )

    def save_matrices(self):
        """
        """
        matrices = {}
        for view, mtx in self.view_sky_matrices.items():
            matrices[f"{view}_sky_matrix"] = mtx.array
        for sensor, mtx in self.sensor_sky_matrices.items():
            matrices[f"{sensor}_sky_matrix"] = mtx.array
        np.savez(self.mtxdir / self.config.hash_str, **matrices)


class ThreePhaseMethod(PhaseMethod):
    def __init__(self, config):
        super().__init__(config)
        with open(self.octree, "wb") as f:
            f.write(
                pr.oconv(
                    *config.model.materials.files,
                    *config.model.scene.files,
                    stdin=config.model.materials.bytes + config.model.scene.bytes,
                )
            )
        self.window_senders = {}
        self.window_receivers = {}
        self.window_bsdfs = {}
        self.daylight_matrices = {}
        for _name, window in self.config.model.windows.items():
            _prims = pr.parse_primitive(window.bytes.decode())
            if window.matrix_file != "":
                self.window_bsdfs[_name] = load_matrix(window.matrix_file)
                window_basis = [
                    k
                    for k, v in BASIS_DIMENSION.items()
                    if v == self.window_bsdfs[_name].shape[0]
                ][0]
            elif window.matrix_data != []:
                self.window_bsdfs[_name] = np.array(window.matrix_data)
            else:
                # raise ValueError("No matrix data or file available", _name)
                logger.warning("No matrix data or file available", _name)
            if _name in self.window_bsdfs:
                window_basis = [
                    k
                    for k, v in BASIS_DIMENSION.items()
                    if v == self.window_bsdfs[_name].shape[0]
                ][0]
            else:
                window_basis = self.config.settings.window_basis
            self.window_receivers[_name] = SurfaceReceiver(
                _prims,
                window_basis,
            )
            self.window_senders[_name] = SurfaceSender(_prims, window_basis)
            self.daylight_matrices[_name] = Matrix(
                self.window_senders[_name],
                [self.sky_receiver],
                self.octree,
            )
        self.view_window_matrices = {}
        self.sensor_window_matrices = {}
        for _v, sender in self.view_senders.items():
            self.view_window_matrices[_v] = Matrix(
                sender, list(self.window_receivers.values()), self.octree
            )
        for _s, sender in self.sensor_senders.items():
            self.sensor_window_matrices[_s] = Matrix(
                sender, list(self.window_receivers.values()), self.octree
            )

    def generate_matrices(self, view_matrices=True):
        """
        view_matrices: Toggle to generate view matrices. Toggle it off can be useful for
        not needing the view matrices but still need the view data for things like
        edgps calculation.
        """
        if self.mfile.exists() and (not self.config.settings.overwrite):
            self.load_matrices()
            return
        if view_matrices:
            for _, mtx in self.view_window_matrices.items():
                mtx.generate(self.config.settings.view_window_matrix)
        for _, mtx in self.sensor_window_matrices.items():
            mtx.generate(self.config.settings.sensor_window_matrix)
        for _, mtx in self.daylight_matrices.items():
            mtx.generate(self.config.settings.daylight_matrix)
        if self.config.settings.save_matrices:
            self.save_matrices()

    def load_matrices(self):
        """
        """
        logger.info(f"Loading matrices from {self.mfile}")
        mdata = np.load(self.mfile)
        for view, mtx in self.view_window_matrices.items():
            if (key := f"{view}_window_matrix") in mdata:
                mtx.array = mdata[key]
        for sensor, mtx in self.sensor_window_matrices.items():
            mtx.array = mdata[f"{sensor}_window_matrix"]
        for name, mtx in self.daylight_matrices.items():
            mtx.array = mdata[f"{name}_daylight_matrix"]


    def calculate_view(
        self,
        view: str,
        bsdf: np.ndarray,
        time: datetime,
        dni: float,
        dhi: float,
    ):
        sky_matrix = self.get_sky_matrix(time, dni, dhi)
        res = []
        if isinstance(bsdf, list):
            if len(bsdf) != len(self.config.model.windows):
                raise ValueError("Number of BSDF should match number of windows.")
        for idx, _name in enumerate(self.config.model.windows):
            _bsdf = bsdf[idx] if isinstance(bsdf, list) else bsdf
            res.append(
                matrix_multiply_rgb(
                    self.view_window_matrices[view].array[idx],
                    _bsdf,
                    self.daylight_matrices[_name].array,
                    sky_matrix,
                )
            )
        return np.sum(res, axis=0)

    def calculate_sensor(
        self,
        sensor: str,
        bsdf: Union[np.ndarray, List[np.ndarray]],
        time: datetime,
        dni: float,
        dhi: float,
    ):
        sky_matrix = self.get_sky_matrix(time, dni, dhi)
        res = []
        if isinstance(bsdf, list):
            if len(bsdf) != len(self.config.model.windows):
                raise ValueError("Number of BSDF should match number of windows.")
        for idx, _name in enumerate(self.config.model.windows):
            _bsdf = bsdf[idx] if isinstance(bsdf, list) else bsdf
            res.append(
                matrix_multiply_rgb(
                    self.sensor_window_matrices[sensor].array[idx],
                    _bsdf,
                    self.daylight_matrices[_name].array,
                    sky_matrix,
                    weights=[47.4, 119.9, 11.6],
                )
            )
        return np.sum(res, axis=0)

    def calculate_view_from_wea(self, view: str):
        if self.wea_data is None:
            raise ValueError("No wea data available")
        sky_matrix = self.get_sky_matrix_from_wea(
            int(self.config.settings.sky_basis[-1])
        )
        # arbitrary chunksize
        chunksize = 300
        shape = (
            self.view_senders[view].xres * self.view_senders[view].yres,
            sky_matrix.shape[1],
            3,
        )
        final = np.memmap(
            f"{view}_3ph.dat", shape=shape, dtype=np.float64, mode="w+", order="F"
        )
        for idx in range(0, sky_matrix.shape[1], chunksize):
            end = min(idx + chunksize, sky_matrix.shape[1])
            _chunksize = end - idx
            res = np.zeros(
                (
                    self.view_senders[view].xres * self.view_senders[view].yres,
                    _chunksize,
                    3,
                )
            )
            for widx, _name in enumerate(self.config.model.windows):
                res += matrix_multiply_rgb(
                    self.view_window_matrices[view].array[widx],
                    self.window_bsdfs[_name],
                    self.daylight_matrices[_name].array,
                    sky_matrix[:, idx:end, :],
                )
            final[:, idx:end, 0] = res[:, :, 0]
            final[:, idx:end, 1] = res[:, :, 1]
            final[:, idx:end, 2] = res[:, :, 2]
            final.flush()
        return final

    def calculate_sensor_from_wea(self, sensor: str):
        if self.wea_data is None:
            raise ValueError("No wea data available")
        sky_matrix = self.get_sky_matrix_from_wea(
            int(self.config.settings.sky_basis[-1])
        )
        res = np.zeros((self.sensor_senders[sensor].yres, sky_matrix.shape[1]))
        for idx, _name in enumerate(self.config.model.windows):
            res += matrix_multiply_rgb(
                self.sensor_window_matrices[sensor].array[idx],
                self.window_bsdfs[_name],
                self.daylight_matrices[_name].array,
                sky_matrix,
                weights=[47.4, 119.9, 11.6],
            )
        return res

    def calculate_edgps(
        self,
        view: str,
        shades: Union[List[pr.Primitive], List[str]],
        bsdf,
        date_time,
        dni,
        dhi,
        ambient_bounce=1,
    ):
        """
        Calculate enhanced simplified daylight glare probability (EDGPs) for a view.
        Args:
            view: view name, must be in config.model.views
            shades: list of shades, either primitves or file paths. This is used
            for high resolution direct sun calculation.
            bsdf: bsdf matrix, either a single matrix or a list of matrices depending
            on the number of windows This is used to calculate the vertical illuminance.
            date_time: datetime object
            dni: direct normal irradiance
            dhi: diffuse horizontal irradiance
            ambient_bounce: ambient bounce, default to 1. Could be set to zero for
            macroscopic non-scattering systems.
        Returns:
            EDGPs
        """
        # generate octree with bsdf
        stdin = b""
        stdin += gen_perez_sky(
            date_time,
            self.wea_metadata.latitude,
            self.wea_metadata.longitude,
            self.wea_metadata.timezone,
            dni,
            dhi,
        )
        if isinstance(shades[0], pr.Primitive):
            for shade in shades:
                stdin += shade.bytes
        elif isinstance(shades[0], (str, Path)):
            _shades = shades
        else:
            _shades = []
        octree = "test.oct"
        with open(octree, "wb") as f:
            f.write(pr.oconv(*shades, stdin=stdin, octree=self.octree))
        # render image with -ab 1
        params = ["-ab", f"{ambient_bounce}"]
        hdr = pr.rpict(
            self.view_senders[view].view.args(),
            octree,
            # fix resolution. Evalglare would complain if resolution too small
            xres=800,
            yres=800,
            params=params,
        )
        ev = self.calculate_sensor(
            view,
            bsdf,
            date_time,
            dni,
            dhi,
        )
        res = pr.evalglare(hdr, ev=float(ev))
        edgps = float(res.split(b":")[1].split()[0])
        return edgps

    def save_matrices(self):
        matrices = {}
        for view, mtx in self.view_window_matrices.items():
            matrices[f"{view}_window_matrix"] = mtx.array
        for sensor, mtx in self.sensor_window_matrices.items():
            matrices[f"{sensor}_window_matrix"] = mtx.array
        for window, mtx in self.daylight_matrices.items():
            matrices[f"{window}_daylight_matrix"] = mtx.array
        np.savez(self.mfile, **matrices)


class FivePhaseMethod(PhaseMethod):
    def __init__(self, config):
        super().__init__(config)
        with open(self.octree, "wb") as f:
            f.write(
                pr.oconv(
                    *config.model.materials.files,
                    *config.model.scene.files,
                    stdin=(config.model.materials.bytes + config.model.scene.bytes),
                )
            )
        self.blacked_out_octree = self.octdir / f"{random_string(5)}.oct"
        self.vmap_oct = self.octdir / f"vmap_{random_string(5)}.oct"
        self.cdmap_oct = self.octdir / f"cdmap_{random_string(5)}.oct"
        self.window_senders = {}
        self.window_receivers = {}
        self.window_bsdfs = {}
        self.view_window_matrices = {}
        self.sensor_window_matrices = {}
        self.daylight_matrices = {}
        self.view_window_direct_matrices = {}
        self.sensor_window_direct_matrices = {}
        self.daylight_direct_matrices = {}
        self.sensor_sun_direct_matrices = {}
        self.view_sun_direct_matrices = {}
        self.view_sun_direct_illuminance_matrices = {}
        self.vmap = {}
        self.cdmap = {}
        self.direct_sun_matrix = self.get_sky_matrix_from_wea(
            mfactor=int(self.config.settings.sun_basis[-1]), onesun=True, sun_only=True
        )
        self._prepare_window_objects()
        self._prepare_sun_receivers()
        self.direct_sun_matrix = to_sparse_matrix3(self.direct_sun_matrix)
        self._gen_blacked_out_octree()
        self._prepare_mapping_octrees()
        self._prepare_view_sender_objects()
        self._prepare_sensor_sender_objects()

    def _gen_blacked_out_octree(self):
        black_scene = b"\n".join(
            pr.xform(s, modifier="black") for s in self.config.model.scene.files
        )
        if self.config.model.scene.bytes != b"":
            black_scene += pr.xform(
                self.config.model.scene.data.encode(), modifier="black"
            )
        black = pr.Primitive("void", "plastic", "black", [], [0, 0, 0, 0, 0])
        glow = pr.Primitive("void", "glow", "glowing", [], [1, 1, 1, 0])
        with open(self.blacked_out_octree, "wb") as f:
            f.write(
                pr.oconv(
                    *self.config.model.materials.files,
                    # *self.config.model.windows,
                    stdin=self.config.model.materials.bytes
                    + str(glow).encode()
                    + str(black).encode()
                    + black_scene,
                )
            )

    def _prepare_window_objects(self):
        for _name, window in self.config.model.windows.items():
            _prims = pr.parse_primitive(window.bytes.decode())
            self.window_receivers[_name] = SurfaceReceiver(
                _prims, self.config.settings.window_basis
            )
            self.window_senders[_name] = SurfaceSender(
                _prims, self.config.settings.window_basis
            )
            if window.matrix_file != "":
                self.window_bsdfs[_name] = load_matrix(window.matrix_file)
            elif window.matrix_data != []:
                self.window_bsdfs[_name] = np.array(window.matrix_data)
            else:
                raise ValueError("No matrix data or file available", _name)
            self.daylight_matrices[_name] = Matrix(
                self.window_senders[_name],
                [self.sky_receiver],
                self.octree,
            )
            self.daylight_direct_matrices[_name] = Matrix(
                self.window_senders[_name],
                [self.sky_receiver],
                self.blacked_out_octree,
            )

    def _prepare_view_sender_objects(self):
        for _v, sender in self.view_senders.items():
            self.vmap[_v] = load_binary_matrix(
                pr.rtrace(
                    sender.content,
                    params=["-ffd", "-av", ".31831", ".31831", ".31831"],
                    octree=self.vmap_oct,
                ),
                nrows=sender.xres * sender.yres,
                ncols=1,
                ncomp=3,
                dtype="d",
                header=True,
            )
            self.cdmap[_v] = load_binary_matrix(
                pr.rtrace(
                    sender.content,
                    params=["-ffd", "-av", ".31831", ".31831", ".31831"],
                    octree=self.cdmap_oct,
                ),
                nrows=sender.xres * sender.yres,
                ncols=1,
                ncomp=3,
                dtype="d",
                header=True,
            )
            self.view_window_matrices[_v] = Matrix(
                sender, list(self.window_receivers.values()), self.octree
            )
            self.view_window_direct_matrices[_v] = Matrix(
                sender, list(self.window_receivers.values()), self.blacked_out_octree
            )
            self.view_sun_direct_matrices[_v] = SunMatrix(
                sender, self.view_sun_receiver, self.blacked_out_octree
            )
            self.view_sun_direct_illuminance_matrices[_v] = SunMatrix(
                sender, self.view_sun_receiver, self.blacked_out_octree
            )

    def _prepare_sensor_sender_objects(self):
        for _s, sender in self.sensor_senders.items():
            self.sensor_window_matrices[_s] = Matrix(
                sender, list(self.window_receivers.values()), self.octree
            )
            self.sensor_window_direct_matrices[_s] = Matrix(
                sender, list(self.window_receivers.values()), self.blacked_out_octree
            )
            self.sensor_sun_direct_matrices[_s] = SunMatrix(
                sender, self.sensor_sun_receiver, self.blacked_out_octree
            )

    def _prepare_sun_receivers(self):
        if self.config.settings.sun_culling:
            window_normals = [
                parse_polygon(r.surfaces[0]).normal.tobytes()
                for r in self.window_receivers.values()
            ]
            unique_window_normals = [np.frombuffer(arr) for arr in set(window_normals)]
            self.sensor_sun_receiver = SunReceiver(
                self.config.settings.sun_basis,
                sun_matrix=self.direct_sun_matrix,
                full_mod=True,
            )
            self.view_sun_receiver = SunReceiver(
                self.config.settings.sun_basis,
                sun_matrix=self.direct_sun_matrix,
                window_normals=unique_window_normals,
                full_mod=False,
            )
        else:
            self.sensor_sun_receiver = SunReceiver(
                self.config.settings.sun_basis, full_mod=True
            )
            self.view_sun_receiver = SunReceiver(
                self.config.settings.sun_basis, full_mod=False
            )

    def _prepare_mapping_octrees(self):
        blacked_out_windows = []
        glowing_windows = []
        for _, sender in self.window_senders.items():
            for window in sender.surfaces:
                blacked_out_windows.append(
                    str(
                        pr.Primitive(
                            "black",
                            window.ptype,
                            window.identifier,
                            window.sargs,
                            window.fargs,
                        )
                    )
                )
                glowing_windows.append(
                    str(
                        pr.Primitive(
                            "glowing",
                            window.ptype,
                            window.identifier,
                            window.sargs,
                            window.fargs,
                        )
                    )
                )
        black = pr.Primitive("void", "plastic", "black", [], [0, 0, 0, 0, 0])
        glow = pr.Primitive("void", "glow", "glowing", [], [1, 1, 1, 0])
        blacked_out_windows = str(black) + " ".join(blacked_out_windows)
        glowing_windows = str(glow) + " ".join(glowing_windows)
        with open(self.vmap_oct, "wb") as wtr:
            wtr.write(pr.oconv(stdin=glowing_windows.encode(), octree=self.octree))
        logger.info("Generating view matrix material map octree")
        with open(self.cdmap_oct, "wb") as wtr:
            wtr.write(pr.oconv(stdin=blacked_out_windows.encode(), octree=self.octree))

    def generate_matrices(self):
        if self.mfile.exists():
            if not self.config.settings.overwrite:
                self.load_matrices()
                return
        logger.info("Generating matrices...")
        logger.info("Step 1/5: Generating view matrices...")
        for mtx in self.view_window_matrices.values():
            mtx.generate(self.config.settings.view_window_matrix, memmap=True)
        for mtx in self.sensor_window_matrices.values():
            mtx.generate(self.config.settings.sensor_window_matrix)
        logger.info("Step 2/5: Generating daylight matrices...")
        for mtx in self.daylight_matrices.values():
            mtx.generate(self.config.settings.daylight_matrix)
        logger.info("Step 3/5: Generating direct view matrices...")
        for _, mtx in self.view_window_direct_matrices.items():
            mtx.generate(["-ab", "1"], sparse=True)
        for _, mtx in self.sensor_window_direct_matrices.items():
            mtx.generate(["-ab", "1"], sparse=True)
        logger.info("Step 4/5: Generating direct daylight matrices...")
        for _, mtx in self.daylight_direct_matrices.items():
            mtx.generate(["-ab", "0"], sparse=True)
        logger.info("Step 5/5: Generating direct sun matrices...")
        for _, mtx in self.sensor_sun_direct_matrices.items():
            mtx.generate(["-ab", "0"])
        for _, mtx in self.view_sun_direct_matrices.items():
            mtx.generate(["-ab", "0"])
        for _, mtx in self.view_sun_direct_illuminance_matrices.items():
            mtx.generate(["-ab", "0", "-i+"])
        logger.info("Done!")
        if self.config.settings.save_matrices:
            self.save_matrices()

    def load_matrices(self):
        """
        """
        logger.info(f"Loading matrices from {self.mfile}")
        mdata = np.load(self.mfile, allow_pickle=True)
        for view, mtx in self.view_window_matrices.items():
            mtx.array = mdata[f"{view}_window_matrix"] 
        for sensor, mtx in self.sensor_window_matrices.items():
            mtx.array = mdata[f"{sensor}_window_matrix"] 
        for window, mtx in self.daylight_matrices.items():
            mtx.array = mdata[f"{window}_daylight_matrix"] 
        for view, mtx in self.view_window_direct_matrices.items():
            mtx.array = mdata[f"{view}_window_direct_matrix"] 
        for sensor, mtx in self.sensor_window_direct_matrices.items():
            mtx.array = mdata[f"{sensor}_window_direct_matrix"] 
        for window, mtx in self.daylight_direct_matrices.items():
            mtx.array = mdata[f"{window}_daylight_direct_matrix"] 
        for sensor, mtx in self.sensor_sun_direct_matrices.items():
            mtx.array = mdata[f"{sensor}_sun_direct_matrix"] 
        for view, mtx in self.view_sun_direct_matrices.items():
            mtx.array = mdata[f"{view}_sun_direct_matrix"] 
        for view, mtx in self.view_sun_direct_illuminance_matrices.items():
            mtx.array = mdata[f"{view}_sun_direct_illuminance_matrix"] 


    def calculate_view_from_wea(self, view: str):
        logger.info("Step 1/2: Generating sky matrix from wea")
        sky_matrix = self.get_sky_matrix_from_wea(
            int(self.config.settings.sky_basis[-1])
        )
        direct_sky_matrix = self.get_sky_matrix_from_wea(
            int(self.config.settings.sky_basis[-1]), sun_only=True
        )
        direct_sky_matrix = to_sparse_matrix3(direct_sky_matrix)
        logger.info("Step 2/2: Multiplying matrices...")
        chunksize = 300
        shape = (
            self.view_window_matrices[view].nrows,
            sky_matrix.shape[1],
            3,
        )
        res = np.memmap(
            f"{view}_5ph.dat", shape=shape, dtype=np.float64, mode="w+", order="F"
        )
        for idx in range(0, sky_matrix.shape[1], chunksize):
            end = min(idx + chunksize, sky_matrix.shape[1])
            _res = [[], [], []]
            for widx, _name in enumerate(self.config.model.windows):
                for c in range(3):
                    tdmx = np.dot(
                        self.window_bsdfs[_name][:, :, c],
                        self.daylight_matrices[_name].array[:, :, c],
                    )
                    tdsmx = np.dot(tdmx, sky_matrix[:, idx:end, c])
                    vtdsmx = np.dot(
                        self.view_window_matrices[view].array[widx][:, :, c], tdsmx
                    )
                    tdmx = np.dot(
                        csr_matrix(self.window_bsdfs[_name][:, :, c]),
                        self.daylight_direct_matrices[_name].array[c],
                    )
                    tdsmx = np.dot(tdmx, direct_sky_matrix[c][:, idx:end])
                    vtdsmx_d = np.dot(
                        self.view_window_direct_matrices[view].array[widx][c], tdsmx
                    )
                    _res[c].append(vtdsmx - vtdsmx_d.toarray())
            for c in range(3):
                cdr = np.dot(
                    self.view_sun_direct_matrices[view].array[c],
                    self.direct_sun_matrix[c][:, idx:end],
                )
                cdf = np.dot(
                    self.view_sun_direct_illuminance_matrices[view].array[c],
                    self.direct_sun_matrix[c][:, idx:end],
                ).multiply(csr_matrix(self.cdmap[view][:, :, c]))
                res[:, idx:end, c] = (
                    np.sum(_res[c], axis=0) + cdr.toarray() + cdf.toarray()
                )
            res.flush()
        return res

    def calculate_sensor_from_wea(self, sensor):
        sky_matrix = self.get_sky_matrix_from_wea(
            int(self.config.settings.sky_basis[-1])
        )
        direct_sky_matrix = self.get_sky_matrix_from_wea(
            int(self.config.settings.sky_basis[-1]), sun_only=True
        )
        direct_sky_matrix = to_sparse_matrix3(direct_sky_matrix)
        res3 = np.zeros((self.sensor_senders[sensor].yres, sky_matrix.shape[1]))
        res3d = np.zeros((self.sensor_senders[sensor].yres, sky_matrix.shape[1]))
        for idx, _name in enumerate(self.config.model.windows):
            res3 += matrix_multiply_rgb(
                self.sensor_window_matrices[sensor].array[idx],
                self.window_bsdfs[_name],
                self.daylight_matrices[_name].array,
                sky_matrix,
                weights=[47.4, 119.9, 11.6],
            )
            res3d += sparse_matrix_multiply_rgb_vtds(
                self.sensor_window_direct_matrices[sensor].array[idx],
                self.window_bsdfs[_name],
                self.daylight_direct_matrices[_name].array,
                direct_sky_matrix,
                weights=[47.4, 119.9, 11.6],
            )
        rescd = np.zeros((self.sensor_senders[sensor].yres, sky_matrix.shape[1]))
        for c, w in enumerate([47.4, 119.9, 11.6]):
            rescd += w * np.dot(
                self.sensor_sun_direct_matrices[sensor].array[c],
                self.direct_sun_matrix[c],
            )
        return res3 - res3d + rescd

    def save_matrices(self):
        matrices = {}
        for view, mtx in self.view_window_matrices.items():
            matrices[f"{view}_window_matrix"] = mtx.array
        for sensor, mtx in self.sensor_window_matrices.items():
            matrices[f"{sensor}_window_matrix"] = mtx.array
        for window, mtx in self.daylight_matrices.items():
            matrices[f"{window}_daylight_matrix"] = mtx.array
        for view, mtx in self.view_window_direct_matrices.items():
            matrices[f"{view}_window_direct_matrix"] = mtx.array
        for sensor, mtx in self.sensor_window_direct_matrices.items():
            matrices[f"{sensor}_window_direct_matrix"] = mtx.array
        for window, mtx in self.daylight_direct_matrices.items():
            matrices[f"{window}_daylight_direct_matrix"] = mtx.array
        for sensor, mtx in self.sensor_sun_direct_matrices.items():
            matrices[f"{sensor}_sun_direct_matrix"] = mtx.array
        for view, mtx in self.view_sun_direct_matrices.items():
            matrices[f"{view}_sun_direct_matrix"] = mtx.array
        for view, mtx in self.view_sun_direct_illuminance_matrices.items():
            matrices[f"{view}_sun_direct_illuminance_matrix"] = mtx.array
        np.savez_compressed(self.mfile, **matrices)


def get_workflow(config):
    workflow = None
    if config.settings.method.lower().startswith(("2", "two")):
        workflow = TwoPhaseMethod(config)
    elif config.settings.method.lower().startswith(("3", "three")):
        workflow = ThreePhaseMethod(config)
    elif config.settings.method.lower().startswith(("5", "five")):
        workflow = FivePhaseMethod(config)
    else:
        raise NotImplementedError("Method not implemented")
    return workflow
