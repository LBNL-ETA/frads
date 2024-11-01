"""Typical Radiance matrix-based simulation workflows"""

import hashlib
import logging
import os
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from shutil import rmtree
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pyradiance as pr
from pyradiance.util import parse_view
from scipy.sparse import csr_matrix

from frads.matrix import (
    BASIS_DIMENSION,
    Matrix,
    SensorSender,
    SkyReceiver,
    SunMatrix,
    SunReceiver,
    SurfaceReceiver,
    SurfaceSender,
    ViewSender,
    load_binary_matrix,
    load_matrix,
    matrix_multiply_rgb,
    sparse_matrix_multiply_rgb_vtds,
    to_sparse_matrix3,
)
from frads.sky import WeaData, WeaMetaData, gen_perez_sky, parse_epw, parse_wea
from frads.utils import (
    minutes_to_datetime,
    parse_polygon,
    parse_rad_header,
    polygon_primitive,
    random_string,
)

logger: logging.Logger = logging.getLogger("frads.methods")


@dataclass
class SceneConfig:
    """Radiance scene configuration object.

    It can be initialized with either a raw data bytes or a list
    of files. If a list of files is provided, they will be concatenated in
    the order they are provided.

    Attributes:
        files: A list of files to be concatenated to form the scene.
        for name, material in self.model.material.items():
            materials[name] = parse_material(name, material)
        for name, material in  self.model.material_no_mass.items():
            materials[name] = parse_material_no_mass(name, material)
        for name, material in self.model.window_material_simple_glazing_system.items():
            materials[name] = parse_window_material_simple_glazing_system(name, material)
        for name, material in self.model.window_material_glazing.items():
            materials[name] = parse_window_material_glazing(name, material)
        for name, material in self.model.window_material_blind.items():
            materials.update(parse_window_material_blind(material))
        bytes: A raw data string to be used as the scene.
        files_mtime: Files last modification time.
    """

    files: List[Path] = field(default_factory=list)
    bytes: bytes = b""
    files_mtime: List[float] = field(init=False, default_factory=list)

    def __post_init__(self):
        if len(self.files) > 0:
            for fpath in self.files:
                self.files_mtime.append(os.path.getmtime(fpath))


@dataclass
class MatrixConfig:
    matrix_file: Union[str, Path] = ""
    matrix_data: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.matrix_data is None:
            self.matrix_data = load_matrix(self.matrix_file)
        elif isinstance(self.matrix_data, list):
            self.matrix_data = np.array(self.matrix_data)


@dataclass
class ShadingGeometryConfig:
    shading_geometry_file: Union[str, Path] = ""
    shading_geometry_bytes: Optional[np.ndarray] = None

    def __post_init__(self): ...


@dataclass
class MaterialConfig:
    """Material file/data configuration object.

    It can be initialized with either a raw data bytes or a list
    of files. If a list of files is provided, they will be concatenated in
    the order they are provided.

    Attributes:
        file: A file to be used as the material.
        bytes: A raw data string to be used as the material.
        matrices: A dictionary of matrix files/data.
        glazing_materials: A dictionary of glazing materials used for edgps calculations.
        file_mtime: File last modification time.

    Raises:
        ValueError: If no file, bytes, or matrices are provided.
    """

    files: List[Path] = field(default_factory=list)
    bytes: bytes = b""
    matrices: Dict[str, MatrixConfig] = field(default_factory=dict)
    glazing_materials: Dict[str, pr.Primitive] = field(default_factory=dict)
    files_mtime: List[float] = field(init=False, default_factory=list)

    def __post_init__(self):
        if len(self.files) > 0:
            for fpath in self.files:
                self.files_mtime.append(os.path.getmtime(fpath))
        for k, v in self.matrices.items():
            if isinstance(v, dict):
                self.matrices[k] = MatrixConfig(**v)
        if self.bytes == b"" and len(self.files) == 0 and len(self.matrices) == 0:
            raise ValueError("MaterialConfig must have either file, bytes or matrices")


@dataclass
class WindowConfig:
    """Window file/data configuration object.

    Each WindowConfig instance coresponds to a window group, which
    can be initialized with either a file path or byte strings.
    In addition, the BSDF matrix files/data, high resolution tensor
    tree files, and shading geometry files or bytestring data can
    be initialized as well.

    Attributes:
        file: A file to be used as the window group.
        bytes: A raw data string to be used as the window group.
        matrix_name: A matrix name to be used for the window group.
        proxy_geometry: A raw data string to be used as the shading geometry.
        files_mtime: Files last modification time.

    Raises:
        ValueError: If neither file nor bytes are provided.
    """

    file: Union[str, Path] = ""
    bytes: bytes = b""
    matrix_name: str = ""
    proxy_geometry: Dict[str, List[pr.Primitive]] = field(default_factory=dict)
    files_mtime: List[float] = field(init=False, default_factory=list)

    def __post_init__(self):
        if os.path.exists(self.file):
            self.files_mtime.append(os.path.getmtime(self.file))
            if not isinstance(self.file, Path):
                self.file = Path(self.file)
        if self.bytes == b"":
            if self.file != "":
                with open(self.file, "rb") as f:
                    self.bytes = f.read()
            else:
                raise ValueError("WindowConfig must have either file or bytes")


@dataclass
class SensorConfig:
    """
    A configuration class for sensors that includes information on the file,
    data, and file modification time.

    Attributes:
        file: Path to the file containing sensor data. Default is an empty string.
        data: List of lists containing float data.
            Default is an empty list.
        file_mtime: Modification time of the file. This attribute is
            automatically initialized based on the 'file' attribute.

    Raises:
        ValueError: If neither file nor data are provided.
    """

    file: str = ""
    data: List[List[float]] = field(default_factory=list)
    file_mtime: float = field(init=False, default=0.0)

    def __post_init__(self):
        """
        Post-initialization method to set the file modification time and load data
        from the file if necessary.
        """
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
    """
    A configuration class for views that includes information on the file,
    data, x/y resoluation, and file modification time.

    Attributes:
        file: Path to the file containing view data. Default is an empty string.
        view: A View object. Default is an empty string.
        xres: X resolution of the view. Default is 512.
        yres: Y resolution of the view. Default is 512.
        file_mtime: Modification time of the file. This attribute is
            automatically initialized based on the 'file' attribute.

    Raises:
        ValueError: If neither file nor view are provided.
    """

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
        if os.path.exists(self.file) and self.view == "":
            self.view = pr.load_views(self.file)[0]
        elif self.view != "":
            if not isinstance(self.view, pr.View):
                self.view = parse_view(self.view)
        else:
            raise ValueError("ViewConfig must have either file or view")


@dataclass
class SurfaceConfig:
    """
    A configuration class for surfaces that includes information on the file,
    data, basis, and file modification time.

    Attributes:
        file: Path to the file containing surface data. Default is an empty string.
        primitives: A list of primitives. Default is an empty list.
        basis: A string representing the basis. Default is 'u'.
        file_mtime: Modification time of the file. This attribute is
            automatically initialized based on the 'file' attribute.

    Raises:
        ValueError: If neither file nor primitives are provided.
    """

    file: Union[str, Path] = ""
    primitives: List[pr.Primitive] = field(default_factory=list)
    basis: str = "u"
    file_mtime: float = field(init=False, default=0.0)

    def __post_init__(self):
        if self.file != "":
            self.file_mtime = os.path.getmtime(self.file)
        if not isinstance(self.file, Path):
            self.file = Path(self.file)
        if self.file.exists() and len(self.primitives) == 0:
            self.primitives = pr.parse_primitive(self.file.read_text())
        elif len(self.primitives) == 0:
            raise ValueError("SurfaceConfig must have either file or primitives")


@dataclass
class Settings:
    """Settings is a dataclass that holds the settings for a Radiance simulation.

    Attributes:
        name: The name of the simulation.
        num_processors: The number of processors to use for the simulation.
        method: The Radiance method to use for the simulation.
        overwrite: Whether to overwrite existing files.
        save_matrices: Whether to save the matrices generated by the simulation.
        sky_basis: The sky basis to use for the simulation.
        window_basis: The window basis to use for the simulation.
        non_coplanar_basis: The non-coplanar basis to use for the simulation.
        sun_basis: The sun basis to use for the simulation.
        sun_culling: Whether to cull suns.
        separate_direct: Whether to separate direct and indirect contributions.
        epw_file: The path to the EPW file to use for the simulation.
        wea_file: The path to the WEA file to use for the simulation.
        start_hour: The start hour for the simulation.
        end_hour: The end hour for the simulation.
        daylight_hours_only: Whether to simulate only daylight hours.
        latitude: The latitude for the simulation.
        longitude: The longitude for the simulation.
        timezone: The timezone for the simulation.
        orientation: sky rotation.
        site_elevation: The elevation for the simulation.
        sensor_sky_matrix: The sky matrix sampling parameters
        view_sky_matrix: View sky matrix sampling parameters
        sensor_sun_matrix: Sensor sun matrix sampling parameters
        view_sun_matrix: View sun matrix sampling parameters
        sensor_window_matrix: Sensor window matrix sampling parameters
        view_window_matrix: View window matrix sampling parameters
        daylight_matrix: Daylight matrix sampling parameters
    """

    name: str = field(default="")
    num_processors: int = 4
    method: str = field(default="3phase")
    overwrite: bool = False
    save_matrices: bool = False
    matrix_dir: str = field(default="Matrices")
    sky_basis: str = field(default="r1")
    window_basis: str = field(default="kf")
    non_coplanar_basis: str = field(default="kf")
    sun_basis: str = field(default="r6")
    sun_culling: bool = field(default=True)
    separate_direct: bool = field(default=False)
    epw_file: str = field(default="")
    wea_file: str = field(default="")
    start_hour: float = field(default=8)
    end_hour: float = field(default=18)
    daylight_hours_only: bool = True
    latitude: float = field(default=37)
    longitude: float = field(default=122)
    time_zone: int = field(default=120)
    orientation: float = field(default=0)
    site_elevation: float = field(default=100)
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
    surface_window_matrix: List[str] = field(
        default_factory=lambda: [
            "-ab",
            "5",
            "-ad",
            "8192",
            "-lw",
            "5e-5",
            "-c",
            "10000",
        ]
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
    """Model dataclass.

    Attributes:
        scene: SceneConfig object
        windows: A dictionary of WindowConfig
        materials: MaterialConfig object
        sensors: A dictionary of SensorConfig
        views: A dictionary of ViewConfig
    """

    materials: "MaterialConfig"
    scene: "SceneConfig" = field(default_factory=SceneConfig)
    windows: Dict[str, "WindowConfig"] = field(default_factory=dict)
    sensors: Dict[str, "SensorConfig"] = field(default_factory=dict)
    views: Dict[str, "ViewConfig"] = field(default_factory=dict)
    surfaces: Dict[str, "SurfaceConfig"] = field(default_factory=dict)

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
        for k, v in self.surfaces.items():
            if isinstance(v, dict):
                self.surfaces[k] = SurfaceConfig(**v)

        self.scene_cfg = True
        self.windows_cfg = True
        self.sensors_cfg = True
        self.views_cfg = True
        self.surfaces_cfg = True

        if self.scene == SceneConfig() or self.scene == {}:
            self.scene_cfg = False
        if self.windows == {}:
            self.windows_cfg = False
        if self.sensors == {}:
            self.sensors_cfg = False
        if self.views == {}:
            self.views_cfg = False
        if self.surfaces == {}:
            self.surfaces_cfg = False

        # add view to sensors if not already there
        for k, v in self.views.items():
            if k in self.sensors:
                if self.sensors[k].data == [
                    self.views[k].view.position + self.views[k].view.direction
                ]:
                    continue
                else:
                    raise ValueError(f"Sensor {k} data does not match view {k} data")
            else:
                self.sensors[k] = SensorConfig(
                    data=[self.views[k].view.position + self.views[k].view.direction]
                )

        for k, v in self.windows.items():
            if v.matrix_name != "":
                if v.matrix_name not in self.materials.matrices:
                    raise ValueError(
                        f"{k} matrix name {v.matrix_name} not found in materials"
                    )


@dataclass
class WorkflowConfig:
    """Workflow configuration dataclass.

    Workflow configuration is initialized with the Settings
    and Model dataclasses. A hash string is generated from
    the config content.

    Attributes:
        settings: A Settings object.
        model: A Model object.
        hash_str: A hash string of the config content.
    """

    settings: "Settings"
    model: "Model"
    hash_str: str = field(init=False)

    def __post_init__(self):
        if (
            not self.model.sensors_cfg
            and not self.model.views_cfg
            and not self.model.surfaces_cfg
        ):
            raise ValueError(
                f"Sensors, views, or surfaces must be specified for {self.settings.method} method"
            )
        if (
            self.settings.method == "3phase" or self.settings.method == "5phase"
        ) and not self.model.windows_cfg:
            raise ValueError(
                f"Windows must be specified in Model for the {self.settings.method} method"
            )
        if isinstance(self.settings, dict):
            self.settings = Settings(**self.settings)
        if isinstance(self.model, dict):
            self.model = Model(**self.model)
        self.hash_str = hashlib.md5(str(self.__dict__).encode()).hexdigest()[:16]

    @staticmethod
    def from_dict(obj: Dict[str, Any]) -> "WorkflowConfig":
        """Generate a WorkflowConfig object from a dictionary.
        Args:
            obj: A dictionary of workflow configuration.
        Returns:
            A WorkflowConfig object.
        """
        settings = Settings(**obj["settings"])
        model = Model(**obj["model"])
        return WorkflowConfig(settings, model)


class PhaseMethod:
    """Base class for phase methods.

    This class is not meant to be used by itself.
    Use one of the subclasses instead.
    The base class instantiate a set of common attributes,
    along with a host of methods that are shared by all phase methods.

    Attributes:
        config: A WorkflowConfig object.
        view_senders: A dictionary of ViewSender objects.
        sensor_senders: A dictionary of SensorSender objects.
        sky_receiver: A SkyReceiver object.
        wea_header: Weather file header string
        wea_metadata: Weather file metadata object
        wea_str: Weather data string
        tmpdir: A temporary directory for storing intermediate files.
        octdir: A directory for storing octree files.
        mtxdir: A directory for storing matrix files.
        mfile: A matrix file path.
    """

    def __init__(self, config: WorkflowConfig):
        """
        Initialize a phase method.

        Args:
            config: A WorkflowConfig object.
        """
        self.config = config

        # Setup the view and sensor senders
        self.view_senders = {}
        self.sensor_senders = {}
        self.surface_senders = {}
        for name, sensors in self.config.model.sensors.items():
            self.sensor_senders[name] = SensorSender(sensors.data)
        for name, view in self.config.model.views.items():
            self.view_senders[name] = ViewSender(
                view.view, xres=view.xres, yres=view.yres
            )
        for name, surface in self.config.model.surfaces.items():
            self.surface_senders[name] = SurfaceSender(
                surfaces=surface.primitives,
                basis=surface.basis,
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
            self.wea_data = None

        # Setup Temp and Octrees directory
        self.tmpdir = Path("Temp")
        self.tmpdir.mkdir(exist_ok=True)
        self.octdir = Path("Octrees")
        self.octdir.mkdir(exist_ok=True)
        self.mtxdir = Path(self.config.settings.matrix_dir)
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

    def get_sky_matrix(
        self,
        time: Union[datetime, List[datetime]],
        dni: Union[float, List[float]],
        dhi: Union[float, List[float]],
        solar_spectrum: bool = False,
    ) -> np.ndarray:
        """Generates a sky matrix based on the time, Direct Normal Irradiance (DNI), and
        Diffuse Horizontal Irradiance (DHI).

        Args:
            time: The specific time for the matrix.
            dni: The Direct Normal Irradiance value.
            dhi: The Diffuse Horizontal Irradiance value.

        Returns:
            numpy.ndarray: The generated sky matrix, with dimensions based on the
                BASIS_DIMENSION setting for the current sky_basis configuration.
        """
        _wea = self.wea_header
        _ncols = 1
        if (
            isinstance(time, datetime)
            and isinstance(dni, (float, int))
            and isinstance(dhi, (float, int))
        ):
            _wea += str(WeaData(time, dni, dhi))
        elif isinstance(time, list) and isinstance(dni, list) and isinstance(dhi, list):
            rows = [str(WeaData(t, n, d)) for t, n, d in zip(time, dni, dhi)]
            _wea += "\n".join(rows)
            _ncols = len(time)
        else:
            raise ValueError(
                "Time, DNI, and DHI must be either single values or lists of values"
            )
        smx = pr.gendaymtx(
            _wea.encode(),
            outform="d",
            mfactor=int(self.config.settings.sky_basis[-1]),
            header=False,
            solar_radiance=solar_spectrum,
        )
        return load_binary_matrix(
            smx,
            nrows=BASIS_DIMENSION[self.config.settings.sky_basis] + 1,
            ncols=_ncols,
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
    """Implements two phase method."""

    def __init__(self, config: WorkflowConfig):
        """Initializes the two phase method

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
        """Generate matrices for all view and sensor points."""
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

    def calculate_view(
        self, view: str, time: datetime, dni: float, dhi: float
    ) -> np.ndarray:
        """Calculate (render) a view.
        Args:
            view: A view name, must bed loaded during configuration.
            time: A datetime object
            dni: Direct normal irradiance
            dhi: Diffuse horizontal irradiance
        Returns:
            A image as a numpy array
        """
        sky_matrix = self.get_sky_matrix(time, dni, dhi)
        return matrix_multiply_rgb(self.view_sky_matrices[view].array, sky_matrix)

    def calculate_sensor(
        self, sensor: str, time: datetime, dni: float, dhi: float
    ) -> np.ndarray:
        """Calculate a sensor view.

        Args:
            sensor: A sensor name, must be loaded during configuration.
            time: A datetime object
            dni: Direct normal irradiance
            dhi: Diffuse horizontal irradiance
        Returns:
            Sensor illuminance value
        """
        sky_matrix = self.get_sky_matrix(time, dni, dhi)
        return matrix_multiply_rgb(
            self.sensor_sky_matrices[sensor].array,
            sky_matrix,
            weights=[47.4, 119.9, 11.6],
        )

    def calculate_view_from_wea(self, view: str) -> np.ndarray:
        """Render a series of images for a view. Rendering using
        the weather file loaded during configuration.

        Args:
            view: View name, must be loaded during configuration.
        Returns:
            A numpy array containing a series of images
        """
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
            f"{view}_2ph.dat",
            shape=shape,
            dtype=np.float64,
            mode="w+",
            order="F",
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

    def calculate_sensor_from_wea(self, sensor: str) -> np.ndarray:
        """Calculate a sensor for tue duration of the weather file
        that's loaded during configuration.

        Args:
            sensor: sensor name, must be loaded during configuration.
        Returns:
            A numpy array containing a series of illuminance values
        """
        if self.wea_data is None:
            raise ValueError("No wea data available")
        return matrix_multiply_rgb(
            self.sensor_sky_matrices[sensor].array,
            self.get_sky_matrix_from_wea(int(self.config.settings.sky_basis[-1])),
            weights=[47.4, 119.9, 11.6],
        )

    def save_matrices(self):
        """Save matrices to a .npz file in the Matrices directory.
        File name is the hash string of the configuration.
        """
        matrices = {}
        for view, mtx in self.view_sky_matrices.items():
            matrices[f"{view}_sky_matrix"] = mtx.array
        for sensor, mtx in self.sensor_sky_matrices.items():
            matrices[f"{sensor}_sky_matrix"] = mtx.array
        np.savez(self.mtxdir / self.config.hash_str, **matrices)


class ThreePhaseMethod(PhaseMethod):
    """Three phase method implementation.

    Attributes:
        config: A WorkflowConfig object
        octree: A path to the octree file
        window_senders: A dictionary of window sender matrices
        window_receivers: A dictionary of window receiver matrices
        window_bsdfs: A dictionary of window BSDF matrices
        daylight_matrices: A dictionary of daylight matrices
        view_window_matrices: A dictionary of view window matrices
        sensor_window_matrices: A dictionary of sensor window matrices
    """

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
        self.window_senders: Dict[str, SurfaceSender] = {}
        self.window_receivers = {}
        self.window_bsdfs = {}
        self.daylight_matrices = {}
        for _name, window in self.config.model.windows.items():
            _prims = pr.parse_primitive(window.bytes.decode())
            if window.matrix_name != "":
                self.window_bsdfs[_name] = self.config.model.materials.matrices[
                    window.matrix_name
                ].matrix_data
                window_basis = [
                    k
                    for k, v in BASIS_DIMENSION.items()
                    if v == self.window_bsdfs[_name].shape[0]
                ][0]
            else:
                # raise ValueError("No matrix data or file available", _name)
                logger.info(f"No matrix data or file available: {_name}")
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
        self.surface_window_matrices = {}
        for _v, sender in self.view_senders.items():
            self.view_window_matrices[_v] = Matrix(
                sender, list(self.window_receivers.values()), self.octree
            )
        for _s, sender in self.sensor_senders.items():
            self.sensor_window_matrices[_s] = Matrix(
                sender, list(self.window_receivers.values()), self.octree
            )
        for _s, sender in self.surface_senders.items():
            self.surface_window_matrices[_s] = Matrix(
                sender, list(self.window_receivers.values()), self.octree
            )

    def generate_matrices(self, view_matrices: bool = True):
        """Generate all required matrices

        Args:
            view_matrices: Toggle to generate view matrices. Toggle it off can be
                useful for not needing the view matrices but still need the view data
                for things like edgps calculation.
        """
        if self.mfile.exists() and (not self.config.settings.overwrite):
            self.load_matrices()
            return
        if view_matrices:
            for _, mtx in self.view_window_matrices.items():
                mtx.generate(
                    self.config.settings.view_window_matrix,
                    nproc=self.config.settings.num_processors,
                )
        for _, mtx in self.sensor_window_matrices.items():
            mtx.generate(
                self.config.settings.sensor_window_matrix,
                nproc=self.config.settings.num_processors,
            )
        for _, mtx in self.surface_window_matrices.items():
            mtx.generate(
                self.config.settings.surface_window_matrix,
                nproc=self.config.settings.num_processors,
            )
        for _, mtx in self.daylight_matrices.items():
            mtx.generate(
                self.config.settings.daylight_matrix,
                nproc=self.config.settings.num_processors,
            )
        if self.config.settings.save_matrices:
            self.save_matrices()

    def load_matrices(self):
        """Load matrices from a .npz file in the Matrices directory."""
        logger.info(f"Loading matrices from {self.mfile}")
        mdata = np.load(self.mfile)
        for view, mtx in self.view_window_matrices.items():
            if (key := f"{view}_window_matrix") in mdata:
                mtx.array = mdata[key]
        for sensor, mtx in self.sensor_window_matrices.items():
            mtx.array = mdata[f"{sensor}_window_matrix"]
        for surface, mtx in self.surface_window_matrices.items():
            mtx.array = mdata[f"{surface}_window_matrix"]
        for name, mtx in self.daylight_matrices.items():
            mtx.array = mdata[f"{name}_daylight_matrix"]

    def calculate_view(
        self,
        view: str,
        bsdf: np.ndarray,
        time: datetime,
        dni: float,
        dhi: float,
    ) -> np.ndarray:
        """Calculate (render) a view.

        Args:
            view: The view name
            bsdf: The BSDF matrix
            time: The datetime object
            dni: The direct normal irradiance
            dhi: The diffuse horizontal irradiance
        Returns:
            A image as numpy array
        """
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
        bsdf: Dict[str, str],
        time: datetime,
        dni: float,
        dhi: float,
    ) -> np.ndarray:
        """Calculate illuminance for a sensor.

        Args:
            sensor: The sensor name
            bsdf: A dictionary of window name as key and bsdf matrix or matrix name as value
            time: The datetime object
            dni: The direct normal irradiance
            dhi: The diffuse horizontal irradiance
        Returns:
            A float value of illuminance
        """
        sky_matrix = self.get_sky_matrix(time, dni, dhi)
        res = []
        if isinstance(bsdf, list):
            if len(bsdf) != len(self.config.model.windows):
                raise ValueError("Number of BSDF should match number of windows.")
        for idx, _name in enumerate(self.config.model.windows):
            _bsdf = self.config.model.materials.matrices[bsdf[_name]].matrix_data
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

    def calculate_view_from_wea(self, view: str) -> np.ndarray:
        """Calculate(render) view from wea data.

        Args:
            view: The view name
        Returns:
            A series of HDR images as a numpy array
        """
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
            f"{view}_3ph.dat",
            shape=shape,
            dtype=np.float64,
            mode="w+",
            order="F",
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

    def calculate_sensor_from_wea(self, sensor: str) -> np.ndarray:
        """Calculates the sensor values from wea data.

        Args:
            sensor: The specific sensor for which the calculation is to be
                performed.

        Returns:
            numpy.ndarray: A matrix containing the calculated sensor values based on
                the Weather Attribute data, sensor configuration, and various matrices
                related to windows, daylight, and sky.

        Raises:
            ValueError: If no wea data is available.

        Examples:
            sensor_values = sensor_config.calculate_sensor_from_wea("sensor_name")
        """
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

    def calculate_surface(
        self,
        surface: str,
        bsdf: Dict[str, str],
        time: datetime,
        dni: float,
        dhi: float,
        solar_spectrum: bool = False,
        sky_matrix: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        weights = [47.4, 119.9, 11.6]
        if solar_spectrum:
            weights = [1.0, 1.0, 1.0]
        if sky_matrix is None:
            sky_matrix = self.get_sky_matrix(
                time, dni, dhi, solar_spectrum=solar_spectrum
            )
        res = np.zeros((self.surface_senders[surface].yres, sky_matrix.shape[1]))
        for idx, _name in enumerate(self.config.model.windows):
            _bsdf = self.config.model.materials.matrices[bsdf[_name]].matrix_data
            res += matrix_multiply_rgb(
                self.surface_window_matrices[surface].array[idx],
                _bsdf,
                self.daylight_matrices[_name].array,
                sky_matrix,
                weights=weights,
            )
        return res

    def calculate_edgps(
        self,
        view: str,
        bsdf: Dict[str, str],
        time: datetime,
        dni: float,
        dhi: float,
        ambient_bounce: int = 0,
        save_hdr: Optional[Union[str, Path]] = None,
    ) -> tuple[float,float]:
        """Calculate enhanced simplified daylight glare probability (EDGPs) for a view.

        Args:
            view: view name, must be in config.model.views
            bsdf: a dictionary of window name as key and bsdf matrix or matrix name as value
            time: datetime object
            dni: direct normal irradiance
            dhi: diffuse horizontal irradiance
            ambient_bounce: ambient bounce, default to 1. Could be set to zero for
                macroscopic non-scattering systems.
        Returns:
            EDGPs
        """
        # generate octree with bsdf
        stdins = []
        stdins.append(
            gen_perez_sky(
                time,
                self.wea_metadata.latitude,
                self.wea_metadata.longitude,
                self.wea_metadata.timezone,
                dirnorm=dni,
                diffhor=dhi,
            )
        )
        for wname, sname in bsdf.items():
            if (_gms := self.config.model.materials.glazing_materials) != {}:
                gmaterial = _gms[sname]
                stdins.append(gmaterial.bytes)
                for prim in self.window_senders[wname].surfaces:
                    stdins.append(replace(prim, modifier=gmaterial.identifier).bytes)
            if (_pgs := self.config.model.windows[wname].proxy_geometry) != {}:
                for prim in _pgs[sname]:
                    stdins.append(prim.bytes)

        octree = f"{random_string(5)}.oct"
        with open(octree, "wb") as f:
            f.write(pr.oconv(stdin=b"".join(stdins), octree=self.octree))

        # render image with -ab 1
        params = ["-ab", str(ambient_bounce)]
        hdr = pr.rpict(
            self.view_senders[view].view.args(),
            octree,
            xres=800,
            yres=800,
            params=params,
        )
        ev = self.calculate_sensor(
            view,
            bsdf,
            time,
            dni,
            dhi,
        )
        if save_hdr is not None:
            with open(save_hdr, "wb") as f:
                f.write(hdr)
        res = pr.evalglare(hdr, fast=1, correction_mode="l-", ev=ev.item())
        edgps = float(res)
        os.remove(octree)
        return edgps, ev.item()

    def save_matrices(self):
        """Saves the view window matrices, sensor window matrices, and daylight matrices
        to a NumPy `.npz` file.

        The matrices are saved with keys formed by concatenating the corresponding
        view, sensor, or window name with '_window_matrix' or '_daylight_matrix'.
        """
        matrices = {}
        for view, mtx in self.view_window_matrices.items():
            matrices[f"{view}_window_matrix"] = mtx.array
        for sensor, mtx in self.sensor_window_matrices.items():
            matrices[f"{sensor}_window_matrix"] = mtx.array
        for surface, mtx in self.surface_window_matrices.items():
            matrices[f"{surface}_window_matrix"] = mtx.array
        for window, mtx in self.daylight_matrices.items():
            matrices[f"{window}_daylight_matrix"] = mtx.array
        np.savez(self.mfile, **matrices)


class FivePhaseMethod(PhaseMethod):
    """
    A class representing the Five-Phase Method, which is an extension of the
    three-phase method, allowing for more complex simulations. It includes
    various matrices, octrees, and other attributes used in the simulation.

    Attributes:
        blacked_out_octree: Path to the octree with blacked-out surfaces.
        vmap_oct: Path to the vmap octree.
        cdmap_oct: Path to the cdmap octree.
        window_senders: Dictionary of window sender objects.
        window_receivers: Dictionary of window receiver objects.
        window_bsdfs: Dictionary of window BSDFs.
        view_window_matrices: Dictionary of view window matrices.
        sensor_window_matrices: Dictionary of sensor window matrices.
        daylight_matrices: Dictionary of daylight matrices.
        direct_sun_matrix: Direct sun matrix.
    """

    def __init__(self, config: WorkflowConfig):
        """
        Initializes the FivePhaseMethod object by setting up octrees, matrices,
        and other necessary attributes.

        Reads materials and scene files, constructs necessary octrees, and
        prepares window objects, sun receivers, and various mapping matrices
        based on the provided configuration.

        Args:
            config: WorkflowConfig object containing all necessary information for
                initializing the five-phase method.
        """
        super().__init__(config)
        with open(self.octree, "wb") as f:
            f.write(
                pr.oconv(
                    *config.model.materials.files,
                    *config.model.scene.files,
                    stdin=(config.model.materials.bytes + config.model.scene.bytes),
                )
            )
        self.blacked_out_octree: Path = self.octdir / f"{random_string(5)}.oct"
        self.vmap_oct: Path = self.octdir / f"vmap_{random_string(5)}.oct"
        self.cdmap_oct: Path = self.octdir / f"cdmap_{random_string(5)}.oct"
        self.window_senders: Dict[str, SurfaceSender] = {}
        self.window_receivers: Dict[str, SurfaceReceiver] = {}
        self.window_bsdfs: Dict[str, np.ndarray] = {}
        self.view_window_matrices: Dict[str, Matrix] = {}
        self.sensor_window_matrices: Dict[str, Matrix] = {}
        self.daylight_matrices: Dict[str, Matrix] = {}
        self.view_window_direct_matrices: Dict[str, Matrix] = {}
        self.sensor_window_direct_matrices: Dict[str, Matrix] = {}
        self.daylight_direct_matrices: Dict[str, Matrix] = {}
        self.sensor_sun_direct_matrices: Dict[str, SunMatrix] = {}
        self.view_sun_direct_matrices: Dict[str, SunMatrix] = {}
        self.view_sun_direct_illuminance_matrices: Dict[str, SunMatrix] = {}
        self.vmap: Dict[str, np.ndarray] = {}
        self.cdmap: Dict[str, np.ndarray] = {}
        self.direct_sun_matrix: np.ndarray = self.get_sky_matrix_from_wea(
            mfactor=int(self.config.settings.sun_basis[-1]),
            onesun=True,
            sun_only=True,
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
            black_scene += pr.xform(self.config.model.scene.bytes, modifier="black")
        black = pr.Primitive("void", "plastic", "black", [], [0, 0, 0, 0, 0])
        glow = pr.Primitive("void", "glow", "glowing", [], [1, 1, 1, 0])
        with open(self.blacked_out_octree, "wb") as f:
            f.write(
                pr.oconv(
                    *self.config.model.materials.files,
                    # *self.config.model.windows,
                    stdin=self.config.model.materials.bytes
                    + glow.bytes
                    + black.bytes
                    + black_scene,
                )
            )

    def _prepare_window_objects(self):
        for _name, window in self.config.model.windows.items():
            _prims = pr.parse_primitive(window.bytes)
            self.window_receivers[_name] = SurfaceReceiver(
                _prims, self.config.settings.window_basis
            )
            self.window_senders[_name] = SurfaceSender(
                _prims, self.config.settings.window_basis
            )
            if window.matrix_name != "":
                self.window_bsdfs[_name] = self.config.model.materials.matrices[
                    window.matrix_name
                ].matrix_data
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
                sender,
                list(self.window_receivers.values()),
                self.blacked_out_octree,
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
                sender,
                list(self.window_receivers.values()),
                self.blacked_out_octree,
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
            mtx.generate(
                self.config.settings.view_window_matrix,
                memmap=True,
                nproc=self.config.settings.num_processors,
            )
        for mtx in self.sensor_window_matrices.values():
            mtx.generate(
                self.config.settings.sensor_window_matrix,
                nproc=self.config.settings.num_processors,
            )
        logger.info("Step 2/5: Generating daylight matrices...")
        for mtx in self.daylight_matrices.values():
            mtx.generate(
                self.config.settings.daylight_matrix,
                nproc=self.config.settings.num_processors,
            )
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
        """ """
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
            f"{view}_5ph.dat",
            shape=shape,
            dtype=np.float64,
            mode="w+",
            order="F",
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
                        self.view_window_matrices[view].array[widx][:, :, c],
                        tdsmx,
                    )
                    tdmx = np.dot(
                        csr_matrix(self.window_bsdfs[_name][:, :, c]),
                        self.daylight_direct_matrices[_name].array[c],
                    )
                    tdsmx = np.dot(tdmx, direct_sky_matrix[c][:, idx:end])
                    vtdsmx_d = np.dot(
                        self.view_window_direct_matrices[view].array[widx][c],
                        tdsmx,
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
