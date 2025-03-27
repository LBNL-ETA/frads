from pathlib import Path
import tempfile

from frads.eplus import EnergyPlusSetup, load_energyplus_model
from frads.window import create_glazing_system, LayerInput
from pyenergyplus.dataset import ref_models
import pytest


@pytest.fixture
def medium_office():
    return load_energyplus_model(ref_models["medium_office"])


@pytest.fixture
def glass_path(resources_dir):
    return resources_dir / "igsdb_product_7406.json"


@pytest.fixture
def blinds_path(resources_dir):
    return resources_dir / "igsdb_product_19732.json"


def test_add_glazingsystem(medium_office, glass_path):
    layer1 = LayerInput(
        input=glass_path,
    )
    gs = create_glazing_system(
        name="test",
        layers=[layer1],
    )
    medium_office.add_glazing_system(gs)
    assert medium_office.construction_complex_fenestration_state != {}
    assert isinstance(medium_office.construction_complex_fenestration_state, dict)
    assert isinstance(medium_office.matrix_two_dimension, dict)
    assert isinstance(medium_office.window_material_glazing, dict)
    assert isinstance(medium_office.window_thermal_model_params, dict)


def test_add_glazingsystem_with_blinds(medium_office, glass_path, blinds_path):
    layer1 = LayerInput(
        input=glass_path,
    )
    layer2 = LayerInput(
        input=blinds_path,
        slat_angle=45,
    )
    gs = create_glazing_system(
        name="gs3",
        layer_inputs=[layer1, layer2],
    )
    medium_office.add_glazing_system(gs)

    epsetup = EnergyPlusSetup(medium_office)
    epsetup.run(design_day=True)
    # assert medium_office.construction_complex_fenestration_state != {}
    # assert isinstance(
    #     medium_office.construction_complex_fenestration_state, dict
    # )
    # assert isinstance(medium_office.matrix_two_dimension, dict)
    # assert isinstance(medium_office.window_material_glazing, dict)
    # assert isinstance(medium_office.window_thermal_model_params, dict)


def test_add_lighting(medium_office):
    try:
        medium_office.add_lighting("z1", 100)  # zone does not exist
        assert False
    except ValueError:
        pass


def test_add_lighting1(medium_office):
    try:
        medium_office.add_lighting(
            "Perimeter_bot_ZN_1", 100
        )  # zone already has lighting
        assert False
    except ValueError:
        pass


def test_add_lighting2(medium_office):
    medium_office.add_lighting("Perimeter_bot_ZN_1", 100, replace=True)

    assert isinstance(medium_office.lights, dict)
    assert isinstance(medium_office.schedule_constant, dict)
    assert isinstance(medium_office.schedule_type_limits, dict)


def test_output_variable(medium_office):
    """Test adding output variable to an EnergyPlusModel."""
    medium_office.add_output(
        output_name="Zone Mean Air Temperature", output_type="variable"
    )

    assert "Zone Mean Air Temperature" in [
        i.variable_name for i in medium_office.output_variable.values()
    ]


def test_output_meter(medium_office):
    """Test adding output meter to an EnergyPlusModel."""
    medium_office.add_output(
        output_name="CO2:Facility",
        output_type="meter",
        reporting_frequency="Hourly",
    )

    assert "CO2:Facility" in [i.key_name for i in medium_office.output_meter.values()]
    assert "Hourly" in [
        i.reporting_frequency.value for i in medium_office.output_meter.values()
    ]


def test_energyplussetup(medium_office):
    """Test running EnergyPlusSetup."""

    tmpdir = tempfile.mkdtemp()
    ep = EnergyPlusSetup(medium_office)
    ep.run(output_directory=tmpdir, design_day=True)
    assert (Path(tmpdir) / "eplusout.csv").exists()


def test_add_proxy_geometry(medium_office, glass_path):
    gs = create_glazing_system(
        name="test",
        layers=[glass_path],
    )
    tmpdir = tempfile.mkdtemp()
    ep = EnergyPlusSetup(medium_office, enable_radiance=True)
    ep.add_proxy_geometry(gs)
