from pathlib import Path

from frads.eplus import EnergyPlusModel, EnergyPlusSetup, load_energyplus_model
from frads.window import GlazingSystem

test_dir = Path(__file__).resolve().parent
resource_dir = test_dir / "Resources"

idf_path = resource_dir / "RefBldgMediumOfficeNew2004_southzone.idf"
glazing_path = resource_dir / "igsdb_product_7406.json"



def test_add_glazingsystem():
    epmodel = load_energyplus_model(idf_path)
    gs = GlazingSystem()
    gs.add_glazing_layer(glazing_path)
    epmodel.add_glazing_system(gs)
    assert epmodel.construction_complex_fenestration_state != {}
    assert isinstance(epmodel.construction_complex_fenestration_state, dict)
    assert isinstance(epmodel.matrix_two_dimension, dict)
    assert isinstance(epmodel.window_material_glazing, dict)
    assert isinstance(epmodel.window_thermal_model_params, dict)


def test_add_lighting():
    epmodel = load_energyplus_model(idf_path)
    try:
        epmodel.add_lighting("z1")  # zone does not exist
        assert False
    except ValueError:
        pass


def test_add_lighting1():
    epmodel = load_energyplus_model(idf_path)
    try:
        epmodel.add_lighting("Perimeter_bot_ZN_1")  # zone already has lighting
        assert False
    except ValueError:
        pass


def test_add_lighting2():
    epmodel = load_energyplus_model(idf_path)
    epmodel.add_lighting("Perimeter_bot_ZN_1", replace=True)

    assert isinstance(epmodel.lights, dict)
    assert isinstance(epmodel.schedule_constant, dict)
    assert isinstance(epmodel.schedule_type_limits, dict)


def test_output_variable():
    """Test adding output variable to an EnergyPlusModel."""
    epmodel = load_energyplus_model(idf_path)
    epmodel.add_output(output_name="Zone Mean Air Temperature", output_type="variable")

    assert "Zone Mean Air Temperature" in [
        i.variable_name for i in epmodel.output_variable.values()
    ]


def test_output_meter():
    """Test adding output meter to an EnergyPlusModel."""
    epmodel = load_energyplus_model(idf_path)
    epmodel.add_output(
        output_name="CO2:Facility",
        output_type="meter",
        reporting_frequency="Hourly",
    )

    assert "CO2:Facility" in [
        i.key_name for i in epmodel.output_meter.values()
    ]
    assert "Hourly" in [
        i.reporting_frequency.value for i in epmodel.output_meter.values()
    ]


def test_energyplussetup():
    """Test running EnergyPlusSetup."""
    epmodel = load_energyplus_model(idf_path)  # file with Design Day

    ep = EnergyPlusSetup(epmodel)
    ep.run(design_day=True)
