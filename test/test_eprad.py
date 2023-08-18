from pathlib import Path

from frads import eprad

test_dir = Path(__file__).resolve().parent


def test_output():
    """Test adding output to an EnergyPlusModel."""
    epmodel = eprad.EnergyPlusModel(
        Path("Resources/RefBldgMediumOfficeNew2004_Chicago.idf")
    )
    epmodel.add_output(opt_name="Zone Mean Air Temperature", opt_type="variable")

    assert "Zone Mean Air Temperature" in [
        i["variable_name"] for i in epmodel.epjs["Output:Variable"].values()
    ]
