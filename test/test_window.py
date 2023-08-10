from pathlib import Path
from frads.window import GlazingSystem, AIR, ARGON
from frads.eprad import EPModel


test_dir = Path(__file__).parent.resolve()
resource_dir = test_dir / "Resources"

glass_path = resource_dir / "CLEAR_3.DAT"
shading_path = resource_dir / "2011-SA1.xml"


def test_simple_glazingsystem():
    """
    Test the GlazingSystem class.
    Build a GlazingSystem object consisting of two layer of clear glass
    """
    gs = GlazingSystem()
    gs.add_glazing_layer(glass_path)
    gs.add_glazing_layer(glass_path)

    assert gs.layers[0].product_name == "Generic Clear Glass"
    assert gs.layers[1].product_name == "Generic Clear Glass"
    assert gs.name == f"{gs.layers[0].product_name}_{gs.layers[1].product_name}"
    assert gs.gaps[0][0][0] == AIR
    assert gs.gaps[0][0][1] == 1
    assert gs.gaps[0][1] == 0.0127
    assert round(gs._thickness, 6) == round(
        sum(
            [
                gs.layers[0].thickness / 1e3,
                gs.gaps[0][1],
                gs.layers[1].thickness / 1e3,
            ]
        ),
        6,
    )


def test_customized_gap():
    """
    Test the building of a customized gap.
    A 0.03 m thick gap between the two glass layers. The gap is filled with 90% argon and 10% air.
    """
    gs = GlazingSystem()
    gs.add_glazing_layer(glass_path)
    gs.add_glazing_layer(glass_path)
    gs.gaps = [((AIR, 0.1), (ARGON, 0.9), 0.03)]

    assert gs.gaps[0][0][0] == AIR
    assert gs.gaps[0][0][1] == 0.1
    assert gs.gaps[0][1][0] == ARGON
    assert gs.gaps[0][1][1] == 0.9
    assert gs.gaps[0][2] == 0.03
    assert round(gs._thickness, 6) == round(
        sum(
            [
                gs.layers[0].thickness / 1e3,
                gs.gaps[0][2],
                gs.layers[1].thickness / 1e3,
            ]
        ),
        6,
    )


def test_window():
    # Create a window
    gs = GlazingSystem()
    gs.add_glazing_layer(test_dir / "Resources" / "igsdb_product_7968.json")
    gs.compute_solar_photopic_results()
