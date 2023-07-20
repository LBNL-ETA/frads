"""Typical Radiance matrix-based simulation workflows
"""

from dataclasses import dataclass, field
from datetime import datetime
import logging
from pathlib import Path
from typing import Any, Dict, List
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
)
from frads.sky import (
    WeaData,
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
from scipy.sparse import csr_matrix


logger: logging.Logger = logging.getLogger("frads.methods")


@dataclass
class SceneConfig:
    files: List[Path] = field(default_factory=list)
    data: str = ""


@dataclass
class MaterialConfig:
    files: List[Path] = field(default_factory=list)
    data: str = ""


@dataclass
class WindowConfig:
    file: Path = Path()
    data: str = ""
    matrix_file: str = ""
    matrix_data: List[List[float]] = field(default_factory=list)
    shading_geometry_file: Path = Path()
    shading_geometry_data: str = ""
    tensor_tree_file: Path = Path()

    def __post_init__(self):
        if not isinstance(self.file, Path):
            self.file = Path(self.file)
        if not isinstance(self.shading_geometry_file, Path):
            self.shading_geometry_file = Path(self.shading_geometry_file)
        if not isinstance(self.tensor_tree_file, Path):
            self.tensor_tree_file = Path(self.tensor_tree_file)
        if self.data == "":
            with open(self.file) as f:
                self.data = f.read()


@dataclass
class SensorConfig:
    file: str = ""
    data: List[List[float]] = field(default_factory=list)

    def __post_init__(self):
        if len(self.data) == 0:
            if self.file != "":
                with open(self.file) as f:
                    self.data = [[float(i) for i in l.split()] for l in f.readlines()]
            else:
                raise ValueError("SensorConfig must have either file or data")


@dataclass
class ViewConfig:
    file: Path = Path()
    data: str = ""
    xres: int = 512
    yres: int = 512

    def __post_init__(self):
        if not isinstance(self.file, Path):
            self.file = Path(self.file)


@dataclass
class Settings:
    method: str = field(default="3phase")
    sky_basis: str = field(default="r1")
    window_basis: str = field(default="kf")
    sun_basis: str = field(default="r6")
    sun_culling: bool = field(default=True)
    epw_file: str = field(default="")
    wea_file: str = field(default="")
    latitude: int = field(default=37)
    longitude: int = field(default=122)
    time_zone: int = field(default=120)
    site_elevation: int = field(default=100)
    sensor_sky_matrix: List[str] = field(default_factory=list)
    view_sky_matrix: List[str] = field(default_factory=list)
    daylight_matrix: List[str] = field(
        default_factory=lambda: ["-ab", "2", "-c", "5000"]
    )
    sensor_window_matrix: List[str] = field(
        default_factory=lambda: ["-ab", "5", "-ad", "8192", "-lw", "5e-5"]
    )
    view_window_matrix: List[str] = field(
        default_factory=lambda: ["-ab", "5", "-ad", "8192", "-lw", "5e-5"]
    )


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

    def __post_init__(self):
        if isinstance(self.settings, dict):
            self.settings = Settings(**self.settings)
        if isinstance(self.model, dict):
            self.model = Model(**self.model)

    @staticmethod
    def from_dict(obj: Dict[str, Any]) -> "WorkflowConfig":
        settings = Settings(**obj["settings"])
        model = Model(**obj["model"])
        return WorkflowConfig(settings, model)


class PhaseMethod:
    def __init__(self, config):
        self.config = config
        self.view_senders = {}
        self.sensor_senders = {}
        for name, sensors in self.config.model.sensors.items():
            self.sensor_senders[name] = SensorSender(sensors.data)
        for name, view in self.config.model.views.items():
            self.view_senders[name] = ViewSender(
                pr.load_views(view.file)[0], xres=view.xres, yres=view.yres
            )
        self.sky_receiver = SkyReceiver(self.config.settings.sky_basis)
        self.wea_metadata, self.wea_data, self.wea_str = None, None, None
        if self.config.settings.epw_file != "":
            with open(self.config.settings.epw_file) as f:
                wea_metadata, self.wea_data = parse_epw(f.read())
            self.wea_header = wea_metadata.wea_header()
            self.wea_str = self.wea_header + "\n".join(str(d) for d in self.wea_data)
        elif self.config.settings.wea_file != "":
            with open(self.config.settings.wea_file) as f:
                wea_metadata, self.wea_data = parse_wea(f.read())
            self.wea_header = wea_metadata.wea_header()
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
                f"place {'_'.join(str(i) for i in [self.config.settings.latitude, self.config.settings.longitude])}\n"
                f"latitude {self.config.settings.latitude}\n"
                f"longitude {self.config.settings.longitude}\n"
                f"time_zone {self.config.settings.time_zone}\n"
                f"site_elevation {self.config.settings.site_elevation}\n"
                f"weather_data_file_units 1\n"
            )
        self.tmpdir = Path("Temp")
        self.tmpdir.mkdir(exist_ok=True)
        self.octdir = Path("Octrees")
        self.octdir.mkdir(exist_ok=True)
        self.octree = self.octdir / f"{random_string(5)}.oct"
        with open(self.octree, "wb") as f:
            f.write(
                pr.oconv(
                    *config.model.materials.files,
                    *config.model.scene.files,
                    # *config.model.windows,  # need to include windows only in two phase
                    stdin=(
                        config.model.materials.data + config.model.scene.data
                    ).encode(),
                )
            )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        rmtree("Octrees")
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
    def __init__(self, config):
        super().__init__(config)
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

    def generate_matrices(self):
        for _, mtx in self.view_sky_matrices.items():
            mtx.generate(self.config.settings.view_sky_matrix)
        for _, mtx in self.sensor_sky_matrices.items():
            mtx.generate(self.config.settings.sensor_sky_matrix)

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

    def save_matrices(self, file):
        matrices = {}
        for view, mtx in self.view_sky_matrices.items():
            matrices[f"{view}_sky_matrix"] = mtx.array
        for sensor, mtx in self.sensor_sky_matrices.items():
            matrices[f"{sensor}_sky_matrix"] = mtx.array
        np.savez(file, **matrices)


class ThreePhaseMethod(PhaseMethod):
    def __init__(self, config):
        super().__init__(config)
        self.window_senders = {}
        self.window_receivers = {}
        self.window_bsdfs = {}
        self.daylight_matrices = {}
        for _name, window in self.config.model.windows.items():
            _prims = pr.parse_primitive(window.data)
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
                print("No matrix data or file available", _name)
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

    def generate_matrices(self):
        for _, mtx in self.view_window_matrices.items():
            mtx.generate(self.config.settings.view_window_matrix)
        for _, mtx in self.sensor_window_matrices.items():
            mtx.generate(self.config.settings.sensor_window_matrix)
        for _, mtx in self.daylight_matrices.items():
            mtx.generate(self.config.settings.daylight_matrix)

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
        res = np.zeros((self.sensor_senders[sensor].yres, sky_matrix.shape[1], 3))
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
        self, view: str, shades: List[pr.Primitive], bsdf, date_time, dni, dhi
    ):
        # generate octree with bsdf
        sky = gen_perez_sky(date_time, dni, dhi)
        octree = "test.oct"
        with open(octree, "wb") as f:
            f.write(pr.oconv(*shades, stdin=sky, octree=self.octree))
        # render image with -ab 1
        params = ["-ab", "1"]
        hdr = pr.rpict(
            self.view_senders[view].view,
            octree,
            xres=self.view_senders[view].xres,
            yres=self.view_senders[view].yres,
            params=params,
        )
        ev = self.calculate_sensor(
            view,
            bsdf,
            date_time,
            dni,
            dhi,
        )
        dgp = pr.evalglare(hdr, ev)
        return dgp

    def save_matrices(self, file):
        matrices = {}
        for view, mtx in self.view_window_matrices.items():
            matrices[f"{view}_window_matrix"] = mtx.array
        for sensor, mtx in self.sensor_window_matrices.items():
            matrices[f"{sensor}_window_matrix"] = mtx.array
        for window, mtx in self.daylight_matrices.items():
            matrices[f"{window}_daylight_matrix"] = mtx.array
        np.savez(file, **matrices)


class FivePhaseMethod(PhaseMethod):
    def __init__(self, config):
        super().__init__(config)
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
        self.direct_sun_matrix = np.array(
            (
                csr_matrix(self.direct_sun_matrix[:, :, 0]),
                csr_matrix(self.direct_sun_matrix[:, :, 1]),
                csr_matrix(self.direct_sun_matrix[:, :, 2]),
            )
        )
        self._gen_blacked_out_octree()
        self._prepare_mapping_octrees()
        self._prepare_view_sender_objects()
        self._prepare_sensor_sender_objects()

    def _gen_blacked_out_octree(self):
        black_scene = b"\n".join(
            pr.xform(s, modifier="black") for s in self.config.model.scene.files
        )
        if self.config.model.scene.data != "":
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
                    stdin=self.config.model.materials.data.encode()
                    + str(glow).encode()
                    + str(black).encode()
                    + black_scene,
                )
            )

    def _prepare_window_objects(self):
        for _name, window in self.config.model.windows.items():
            _prims = pr.parse_primitive(window.data)
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
        print("Generating matrices...")
        print("Step 1/5: Generating view matrices...")
        for mtx in self.view_window_matrices.values():
            mtx.generate(self.config.settings.view_window_matrix, memmap=True)
        for mtx in self.sensor_window_matrices.values():
            mtx.generate(self.config.settings.sensor_window_matrix)
        print("Step 2/5: Generating daylight matrices...")
        for mtx in self.daylight_matrices.values():
            mtx.generate(self.config.settings.daylight_matrix)
        print("Step 3/5: Generating direct view matrices...")
        for _, mtx in self.view_window_direct_matrices.items():
            mtx.generate(["-ab", "1"], sparse=True)
        for _, mtx in self.sensor_window_direct_matrices.items():
            mtx.generate(["-ab", "1"], sparse=True)
        print("Step 4/5: Generating direct daylight matrices...")
        for _, mtx in self.daylight_direct_matrices.items():
            mtx.generate(["-ab", "0"], sparse=True)
        print("Step 5/5: Generating direct sun matrices...")
        for _, mtx in self.sensor_sun_direct_matrices.items():
            mtx.generate(["-ab", "0"])
        for _, mtx in self.view_sun_direct_matrices.items():
            mtx.generate(["-ab", "0"])
        for _, mtx in self.view_sun_direct_illuminance_matrices.items():
            mtx.generate(["-ab", "0", "-i+"])
        print("Done!")

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)

    def calculate_view_from_wea(self, view: str):
        print("Step 1/2: Generating sky matrix from wea")
        sky_matrix = self.get_sky_matrix_from_wea(
            int(self.config.settings.sky_basis[-1])
        )
        direct_sky_matrix = self.get_sky_matrix_from_wea(
            int(self.config.settings.sky_basis[-1]), sun_only=True
        )
        direct_sky_matrix = [
            csr_matrix(direct_sky_matrix[:, :, 0]),
            csr_matrix(direct_sky_matrix[:, :, 1]),
            csr_matrix(direct_sky_matrix[:, :, 2]),
        ]
        print("Step 2/2: Multiplying matrices...")
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
            print(idx)
            end = min(idx + chunksize, sky_matrix.shape[1])
            _chunksize = end - idx
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
        res3 = []
        res3d = []
        for idx, _name in enumerate(self.config.model.windows):
            res3.append(
                matrix_multiply_rgb(
                    self.sensor_window_matrices[sensor].array[idx],
                    self.window_bsdfs[_name],
                    self.daylight_matrices[_name].array,
                    sky_matrix,
                    weights=[47.4, 119.9, 11.6],
                )
            )
            res3d.append(
                matrix_multiply_rgb(
                    self.sensor_window_direct_matrices[sensor].array[idx],
                    self.window_bsdfs[_name],
                    self.daylight_direct_matrices[_name].array,
                    direct_sky_matrix,
                    weights=[47.4, 119.9, 11.6],
                )
            )
        rescd = matrix_multiply_rgb(
            self.sensor_sun_direct_matrices[sensor].array,
            self.direct_sun_matrix,
            weights=[47.4, 119.9, 11.6],
        )
        return np.sum(res3, axis=0) - np.sum(res3d, axis=0) + rescd

    def save_matrices(self, file):
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
        np.savez_compressed(file, **matrices)


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
