from pathlib import Path

from frads.eprad import EnergyPlusModel, EnergyPlusSetup
from frads.window import GlazingSystem

test_dir = Path(__file__).resolve().parent
resource_dir = test_dir / "Resources"

idf_path = resource_dir / "RefBldgMediumOfficeNew2004_southzone.idf"
glazing_path = resource_dir / "igsdb_product_7406.json"


def test_energyplusmodel():
    epmodel = EnergyPlusModel(idf_path)
    assert isinstance(epmodel.epjs, dict)
    assert isinstance(epmodel.floors, list)
    assert isinstance(epmodel.zones, list)
    assert isinstance(epmodel.window_walls, list)
    assert isinstance(epmodel.windows, list)
    assert isinstance(epmodel.lights, list)
    assert isinstance(epmodel.complex_fenestration_states, list)


def test_add_glazingsystem():
    epmodel = EnergyPlusModel(idf_path)
    gs = GlazingSystem()
    gs.add_glazing_layer(glazing_path)
    epmodel.add_glazing_system(gs)
    assert isinstance(epmodel.complex_fenestration_states, list)
    assert epmodel.complex_fenestration_states != []
    assert isinstance(epmodel.epjs["Construction:ComplexFenestrationState"], dict)
    assert isinstance(epmodel.epjs["Matrix:TwoDimension"], dict)
    assert isinstance(epmodel.epjs["WindowMaterial:Glazing"], dict)
    assert isinstance(epmodel.epjs["WindowMaterial:Gas"], dict)
    assert isinstance(epmodel.epjs["WindowMaterial:Gap"], dict)
    assert isinstance(epmodel.epjs["WindowMaterial:ComplexShade"], dict)
    assert isinstance(epmodel.epjs["WindowThermalModel:Params"], dict)


def test_add_lighting():
    epmodel = EnergyPlusModel(idf_path)
    try:
        epmodel.add_lighting("z1")  # zone does not exist
        assert False
    except ValueError:
        pass


def test_add_lighting1():
    epmodel = EnergyPlusModel(idf_path)
    try:
        epmodel.add_lighting("Perimeter_bot_ZN_1")  # zone already has lighting
        assert False
    except ValueError:
        pass


def test_add_lighting2():
    epmodel = EnergyPlusModel(idf_path)
    epmodel.add_lighting("Perimeter_bot_ZN_1", replace=True)

    assert isinstance(epmodel.epjs["Lights"], dict)
    assert isinstance(epmodel.epjs["Schedule:Constant"], dict)
    assert isinstance(epmodel.epjs["ScheduleTypeLimits"], dict)


def test_output_variable():
    """Test adding output variable to an EnergyPlusModel."""
    epmodel = EnergyPlusModel(idf_path)
    epmodel.add_output(output_name="Zone Mean Air Temperature", output_type="variable")

    assert "Zone Mean Air Temperature" in [
        i["variable_name"] for i in epmodel.epjs["Output:Variable"].values()
    ]


def test_output_meter():
    """Test adding output meter to an EnergyPlusModel."""
    epmodel = EnergyPlusModel(idf_path)
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


def test_energyplussetup():
    """Test running EnergyPlusSetup."""
    epmodel = EnergyPlusModel(idf_path)  # file with Design Day

    ep = EnergyPlusSetup(epmodel)
    ep.run(design_day=True)
