import os
from pathlib import Path

from frads.eplus import EnergyPlusModel, EnergyPlusSetup, load_energyplus_model
from frads.window import GlazingSystem, create_glazing_system
from pyenergyplus.dataset import ref_models
import pytest

test_dir = Path(__file__).resolve().parent
resource_dir = test_dir / "Resources"

idf_path = resource_dir / "RefBldgMediumOfficeNew2004_southzone.idf"
glazing_path = resource_dir / "igsdb_product_7406.json"

@pytest.fixture
def medium_office():
    return load_energyplus_model(ref_models["medium_office"])


def test_add_glazingsystem(medium_office):
    gs = create_glazing_system(
        name="test",
        layers=[glazing_path],
    )
    medium_office.add_glazing_system(gs)
    assert medium_office.construction_complex_fenestration_state != {}
    assert isinstance(medium_office.construction_complex_fenestration_state, dict)
    assert isinstance(medium_office.matrix_two_dimension, dict)
    assert isinstance(medium_office.window_material_glazing, dict)
    assert isinstance(medium_office.window_thermal_model_params, dict)


def test_add_lighting(medium_office):
    try:
        medium_office.add_lighting("z1")  # zone does not exist
        assert False
    except ValueError:
        pass


def test_add_lighting1(medium_office):
    try:
        medium_office.add_lighting("Perimeter_bot_ZN_1")  # zone already has lighting
        assert False
    except ValueError:
        pass


def test_add_lighting2(medium_office):
    medium_office.add_lighting("Perimeter_bot_ZN_1", replace=True)

    assert isinstance(medium_office.lights, dict)
    assert isinstance(medium_office.schedule_constant, dict)
    assert isinstance(medium_office.schedule_type_limits, dict)


def test_output_variable(medium_office):
    """Test adding output variable to an EnergyPlusModel."""
    medium_office.add_output(output_name="Zone Mean Air Temperature", output_type="variable")

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

    assert "CO2:Facility" in [
        i.key_name for i in medium_office.output_meter.values()
    ]
    assert "Hourly" in [
        i.reporting_frequency.value for i in medium_office.output_meter.values()
    ]


def test_energyplussetup(medium_office):
    """Test running EnergyPlusSetup."""

    ep = EnergyPlusSetup(medium_office)
    ep.run(design_day=True)
    assert Path("eplusout.csv").exists()
    os.remove("eplus.json")
    os.remove("eplusout.csv")
    os.remove("eplusout.eso")
    os.remove("eplusout.mtr")
    os.remove("eplusout.rdd")
    os.remove("eplusout.bnd")
    os.remove("eplusout.eio")
    os.remove("eplusout.mtd")
    os.remove("eplusout.mdd")
    os.remove("eplusssz.csv")
    os.remove("eplustbl.htm")
    os.remove("epluszsz.csv")
    os.remove("epmodel.json")
    os.remove("eplusmtr.csv")
    os.remove("eplusout.audit")
    os.remove("eplusout.dxf")
    os.remove("eplusout.end")
    os.remove("eplusout.shd")
    os.remove("eplusout.sql")
    os.remove("sqlite.err")

