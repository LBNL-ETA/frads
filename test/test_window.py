from pathlib import Path
from frads.window import GlazingSystem, AIR, ARGON


test_dir = Path(__file__).parent.resolve()
resource_dir = test_dir / "Resources"

glass_path = resource_dir / "CLEAR_3.DAT"
shade_path = resource_dir / "2011-SA1.xml"


def test_simple_glazingsystem():
    """
    Test the GlazingSystem class.
    Build a GlazingSystem object consisting of two layer of clear glass.

    Check the thickness of the glazing system.
    Check the order and name of the layers.
    Check the composition of the default gap.
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

    Check the thickness of the glazing system.
    Check the order and composition of the gap.
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


def test_multilayer_glazing_shading():
    """
    Test GlazingSystem object with multiple layers of glazing and shading and more than one customized gap.

    Check the thickness of the glazing system.
    Check the order of the layers.
    Check the order and composition of the gaps.
    """
    gs = GlazingSystem()
    gs.add_glazing_layer(glass_path)
    gs.add_glazing_layer(glass_path)
    gs.add_shading_layer(shade_path)
    gs.gaps = [((AIR, 0.1), (ARGON, 0.9), 0.03), ((AIR, 1), 0.01)]

    assert gs.layers[0].product_name == "Generic Clear Glass"
    assert gs.layers[1].product_name == "Generic Clear Glass"
    assert gs.layers[2].product_name == "Satine 5500 5%, White Pearl"

    assert (
        gs.name
        == f"{gs.layers[0].product_name}_{gs.layers[1].product_name}_{gs.layers[2].product_name}"
    )
    assert gs.gaps[0][0][0] == AIR
    assert gs.gaps[0][0][1] == 0.1
    assert gs.gaps[0][1][0] == ARGON
    assert gs.gaps[0][1][1] == 0.9
    assert gs.gaps[0][2] == 0.03
    assert gs.gaps[1][0][0] == AIR
    assert gs.gaps[1][0][1] == 1
    assert gs.gaps[1][1] == 0.01
    assert round(gs._thickness, 6) == round(
        sum(
            [
                gs.layers[0].thickness / 1e3,
                gs.gaps[0][2],
                gs.layers[1].thickness / 1e3,
                gs.gaps[1][1],
                gs.layers[2].thickness / 1e3,
            ]
        ),
        6,
    )

    assert gs.photopic_results is None
    assert gs.solar_results is None


def test_compute_results():
    """
    Test the computation of the solar and photopic results.

    Check the results are not None.
    """
    gs = GlazingSystem()
    gs.add_glazing_layer(glass_path)
    gs.compute_solar_photopic_results()

    assert gs.photopic_results is not None
    assert gs.solar_results is not None
