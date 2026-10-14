"""Regression tests for the matrix cache key collision (save/load_matrices).

Legacy cache files keyed view, sensor, and surface matrices by bare entity
name (e.g. '{name}_window_matrix' for views AND sensors). When a view and a
sensor shared a name -- e.g. both named after the zone -- save_matrices
silently overwrote one with the other and load_matrices restored a corrupted
matrix without raising. These tests exercise the namespaced schema (v2)
roundtrip for all three phase methods and the legacy-file handling.

The tests bypass __init__ (which requires Radiance octree generation) and
drive save_matrices/load_matrices directly on minimal instances.
"""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from frads.methods import (
    FivePhaseMethod,
    ThreePhaseMethod,
    TwoPhaseMethod,
    _matrix_cache_prefixes,
)


def _mtx(shape, fill):
    return SimpleNamespace(array=np.full(shape, fill, dtype=np.float64))


def _empty_like(matrices):
    return {name: SimpleNamespace(array=None) for name in matrices}


class TestMatrixCacheRoundtrip(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_two_phase_view_sensor_name_collision(self):
        """View and sensor named identically survive a save/load roundtrip."""
        m = TwoPhaseMethod.__new__(TwoPhaseMethod)
        m.mtxdir = self.tmpdir
        m.config = SimpleNamespace(hash_str="cache")
        m.mfile = self.tmpdir / "cache.npz"
        m.view_sky_matrices = {"zone": _mtx((16, 146, 3), 1.0)}
        m.sensor_sky_matrices = {"zone": _mtx((4, 146, 3), 2.0)}
        m.save_matrices()

        m.view_sky_matrices = _empty_like(m.view_sky_matrices)
        m.sensor_sky_matrices = _empty_like(m.sensor_sky_matrices)
        m.load_matrices()
        self.assertEqual(m.view_sky_matrices["zone"].array.shape, (16, 146, 3))
        self.assertTrue((m.view_sky_matrices["zone"].array == 1.0).all())
        self.assertEqual(m.sensor_sky_matrices["zone"].array.shape, (4, 146, 3))
        self.assertTrue((m.sensor_sky_matrices["zone"].array == 2.0).all())

    def test_three_phase_view_sensor_surface_name_collision(self):
        """View, sensor, and surface with one shared name stay distinct."""
        m = ThreePhaseMethod.__new__(ThreePhaseMethod)
        m.mfile = self.tmpdir / "cache3.npz"
        m.view_window_matrices = {"zone": _mtx((16, 145, 3), 1.0)}
        m.sensor_window_matrices = {"zone": _mtx((4, 145, 3), 2.0)}
        m.surface_window_matrices = {"zone": _mtx((9, 145, 3), 3.0)}
        m.daylight_matrices = {"win1": _mtx((145, 146, 3), 4.0)}
        m.save_matrices()

        for attr in ("view_window_matrices", "sensor_window_matrices",
                     "surface_window_matrices", "daylight_matrices"):
            setattr(m, attr, _empty_like(getattr(m, attr)))
        m.load_matrices()
        self.assertEqual(m.view_window_matrices["zone"].array.shape, (16, 145, 3))
        self.assertEqual(m.sensor_window_matrices["zone"].array.shape, (4, 145, 3))
        self.assertEqual(m.surface_window_matrices["zone"].array.shape, (9, 145, 3))
        self.assertEqual(m.daylight_matrices["win1"].array.shape, (145, 146, 3))

    def test_five_phase_view_sensor_name_collision(self):
        """FivePhase roundtrip with colliding names incl. direct/sun keys."""
        m = FivePhaseMethod.__new__(FivePhaseMethod)
        m.mfile = self.tmpdir / "cache5.npz"
        m.view_window_matrices = {"zone": _mtx((16, 145, 3), 1.0)}
        m.sensor_window_matrices = {"zone": _mtx((4, 145, 3), 2.0)}
        m.daylight_matrices = {"win1": _mtx((145, 146, 3), 3.0)}
        m.view_window_direct_matrices = {"zone": _mtx((16, 145, 3), 4.0)}
        m.sensor_window_direct_matrices = {"zone": _mtx((4, 145, 3), 5.0)}
        m.daylight_direct_matrices = {"win1": _mtx((145, 146, 3), 6.0)}
        m.sensor_sun_direct_matrices = {"zone": _mtx((4, 5185, 3), 7.0)}
        m.view_sun_direct_matrices = {"zone": _mtx((16, 5185, 3), 8.0)}
        m.view_sun_direct_illuminance_matrices = {"zone": _mtx((16, 5185, 3), 9.0)}
        m.save_matrices()

        for attr in ("view_window_matrices", "sensor_window_matrices",
                     "daylight_matrices", "view_window_direct_matrices",
                     "sensor_window_direct_matrices", "daylight_direct_matrices",
                     "sensor_sun_direct_matrices", "view_sun_direct_matrices",
                     "view_sun_direct_illuminance_matrices"):
            setattr(m, attr, _empty_like(getattr(m, attr)))
        m.load_matrices()
        self.assertEqual(m.view_window_matrices["zone"].array.shape, (16, 145, 3))
        self.assertTrue((m.view_window_matrices["zone"].array == 1.0).all())
        self.assertEqual(m.sensor_window_matrices["zone"].array.shape, (4, 145, 3))
        self.assertEqual(m.sensor_sun_direct_matrices["zone"].array.shape, (4, 5185, 3))
        self.assertTrue((m.view_sun_direct_matrices["zone"].array == 8.0).all())
        self.assertTrue(
            (m.view_sun_direct_illuminance_matrices["zone"].array == 9.0).all()
        )

    def test_legacy_cache_without_collision_loads(self):
        """A v1 (un-namespaced) file with distinct names still loads."""
        mfile = self.tmpdir / "legacy.npz"
        np.savez(
            mfile,
            **{
                "v1_sky_matrix": np.ones((16, 146, 3)),
                "s1_sky_matrix": np.full((4, 146, 3), 2.0),
            },
        )
        m = TwoPhaseMethod.__new__(TwoPhaseMethod)
        m.mfile = mfile
        m.view_sky_matrices = {"v1": SimpleNamespace(array=None)}
        m.sensor_sky_matrices = {"s1": SimpleNamespace(array=None)}
        m.load_matrices()
        self.assertEqual(m.view_sky_matrices["v1"].array.shape, (16, 146, 3))
        self.assertEqual(m.sensor_sky_matrices["s1"].array.shape, (4, 146, 3))

    def test_legacy_cache_with_collision_raises(self):
        """A v1 file whose view/sensor names collide is corrupted -> error."""
        mfile = self.tmpdir / "legacy_bad.npz"
        np.savez(mfile, **{"zone_sky_matrix": np.full((4, 146, 3), 2.0)})
        m = TwoPhaseMethod.__new__(TwoPhaseMethod)
        m.mfile = mfile
        m.view_sky_matrices = {"zone": SimpleNamespace(array=None)}
        m.sensor_sky_matrices = {"zone": SimpleNamespace(array=None)}
        with self.assertRaises(ValueError):
            m.load_matrices()

    def test_prefix_helper_v2(self):
        mdata = {"_schema_version": np.int64(2)}
        self.assertEqual(
            _matrix_cache_prefixes(mdata, Path("x.npz"), {"a"}, {"a"}),
            ("view_", "sensor_", "surface_"),
        )


if __name__ == "__main__":
    unittest.main()
