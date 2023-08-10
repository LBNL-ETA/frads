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
    Build a GlazingSystem object consisting of the following:
        - a single layer of clear glass
        - a single layer of shading product
    """
    gs_ec60 = GlazingSystem()
    gs_ec60.add_glazing_layer(glass_path)
    gs_ec60.add_shading_layer(shading_path)

    assert gs_ec60.layers[0].product_name == "Generic Clear Glass"
    assert gs_ec60.layers[1].product_name == "Satine 5500 5%, White Pearl"
    assert (
        gs_ec60.name
        == f"{gs_ec60.layers[0].product_name}_{gs_ec60.layers[1].product_name}"
    )
    assert gs_ec60.gaps[0][0][0] == AIR
    assert gs_ec60.gaps[0][0][1] == 1
    assert gs_ec60.gaps[0][1] == 0.0127
    assert gs_ec60._thickness == sum(
        [
            gs_ec60.layers[0].thickness / 1e3,
            gs_ec60.gaps[0][1],
            gs_ec60.layers[1].thickness / 1e3,
        ]
    )


# def test_glazingsystem():
#     """
#     Test the GlazingSystem class.
#     Build a GlazingSystem object consisting of the following:
#         - a single layer of clear glass
#         - a single layer of shading product
#         - a 0.0127 m thick gap between the two layer. The gap is filled with 90% argon and 10% air.
#     """
#     gs_ec60 = GlazingSystem()
#     gs_ec60.add_glazing_layer(glass_path)
#     gs_ec60.add_shading_layer(shading_path)
#     gs_ec60.gaps = [((AIR, 0.1), (ARGON, 0.9), 0.0127)]
#     gs_ec60.add_shading_layer(shading_path)
#     gs_ec60.gaps = [((AIR, 0.1), (ARGON, 0.9), 0.0127)]

#     assert gs_ec60.layers[0].name == "CLEAR_3.DAT"
#     assert gs_ec60.layers[1].name == "klems_aniso_high.xml"
#     assert gs_ec60.gaps[0][0][0] == AIR
#     assert gs_ec60.gaps[0][1][0] == ARGON
#     assert gs_ec60.gaps[0][0][1] == 0.1
#     assert gs_ec60.gaps[0][1][1] == 0.9
#     assert gs_ec60.gaps[0][2] == 0.0127
# assert gs_ec60._thickness ==


def test_window():
    # Create a window
    gs = GlazingSystem()
    gs.add_glazing_layer(test_dir / "Resources" / "igsdb_product_7968.json")
    gs.compute_solar_photopic_results()
