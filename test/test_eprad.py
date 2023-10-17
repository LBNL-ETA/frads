from frads.eprad import EnergyPlusModel, EnergyPlusSetup
from frads.window import GlazingSystem
import pytest


@pytest.fixture
def idf_path(resources_dir):
    return resources_dir / "RefBldgMediumOfficeNew2004_southzone.idf"


@pytest.fixture
def glazing_path(resources_dir):
    return resources_dir / "igsdb_product_7406.json"


@pytest.fixture
def epmodel(idf_path):
    return EnergyPlusModel(idf_path)


def test_energyplusmodel(epmodel):
    assert isinstance(epmodel, EnergyPlusModel)
    assert isinstance(epmodel.floors, list)
    assert isinstance(epmodel.window_walls, list)


def test_add_glazingsystem(epmodel, glazing_path):
    gs = GlazingSystem()
    gs.add_glazing_layer(glazing_path)
    epmodel.add_glazing_system(gs)
    assert epmodel.complex_fenestration_states != []
    # assert isinstance(epmodel.epjs["construction_complex_fenestration_state"], dict)
    # assert isinstance(epmodel.epjs["matrix_two_dimension"], dict)
    # assert isinstance(epmodel.epjs["window_material_glazing"], dict)
    # assert isinstance(epmodel.epjs["window_material_gas"], dict)
    # assert isinstance(epmodel.epjs["window_material_gap"], dict)
    # assert isinstance(epmodel.epjs["window_material_complex_shade"], dict)
    # assert isinstance(epmodel.epjs["window_thermal_model_params"], dict)


def test_add_lighting(epmodel):
    with pytest.raises(ValueError):
        epmodel.add_lighting("z1")


def test_add_lighting1(epmodel):
    with pytest.raises(ValueError):
        epmodel.add_lighting("Perimeter_bot_ZN_1", replace=False)


def test_add_lighting2(epmodel):
    epmodel.add_lighting("Perimeter_bot_ZN_1", replace=True)

    # assert isinstance(epmodel.lights, dict)
    # assert isinstance(epmodel.schedule_constant, dict)
    # assert isinstance(epmodel.schedule_type_limits, dict)


def test_output_variable(epmodel):
    """Test adding output variable to an EnergyPlusModel."""
    epmodel.add_output(output_name="Zone Mean Air Temperature", output_type="variable")

    # assert "Zone Mean Air Temperature" in [
    #     i.variable_name for i in epmodel.output_variable.values()
    # ]


def test_output_meter(epmodel):
    """Test adding output meter to an EnergyPlusModel."""
    epmodel.add_output(
        output_name="CO2:Facility",
        output_type="meter",
        reporting_frequency="Hourly",
    )

    assert "CO2:Facility" in [
        i["key_name"] for i in epmodel.epjs["Output:Meter"].values()
    ]
    assert "Hourly" in [
        i["reporting_frequency"] for i in epmodel.epjs["Output:Meter"].values()
    ]


def test_energyplussetup(epmodel):
    """Test running EnergyPlusSetup."""
    ep = EnergyPlusSetup(epmodel)
    ep.run(design_day=True)
