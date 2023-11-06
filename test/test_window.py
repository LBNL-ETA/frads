from pathlib import Path
from frads.window import (
    GasType,
    Gas,
    Gap,
    GlazingSystemBSDF,
    create_glazing_system_from_files,
)
import pytest


@pytest.fixture
def glass_path(resources_dir):
    return resources_dir / "igsdb_product_7406.json"


@pytest.fixture
def shade_path(resources_dir):
    return resources_dir / "2011-SA1.xml"


@pytest.fixture
def glazing_system(glass_path):
    gs = create_glazing_system_from_files(
        name="gs1",
        layers=[glass_path, glass_path],
    )
    return gs


@pytest.fixture
def output_json():
    opath = Path("tmp.json")
    yield opath
    if opath.exists():
        opath.unlink()


def test_save_and_load(glazing_system, output_json):
    """
    Test the save method of the GlazingSystem class.
    """
    glazing_system.save(output_json)
    assert output_json.exists()
    new_glzsys = GlazingSystemBSDF.from_json(output_json)
    assert new_glzsys.name == glazing_system.name
    assert (
        new_glzsys.visible_back_reflectance == glazing_system.visible_back_reflectance
    )


def test_simple_glazingsystem(glazing_system):
    """
    Test the GlazingSystem class.
    Build a GlazingSystem object consisting of two layer of clear glass.

    Check the thickness of the glazing system.
    Check the order and name of the layers.
    Check the composition of the default gap.
    """
    assert glazing_system.layers[0].name.startswith("SageGlass")
    assert glazing_system.layers[1].name.startswith("SageGlass")
    assert glazing_system.name == "gs1"
    assert glazing_system.gaps[0].gases[0].gas == GasType.air
    assert glazing_system.gaps[0].gases[0].ratio == 1
    assert glazing_system.gaps[0].thickness == 0.0127


def test_customized_gap(glass_path):
    """
    Test the building of a customized gap.
    A 0.03 m thick gap between the two glass layers. The gap is filled with 90% argon and 10% air.

    Check the thickness of the glazing system.
    Check the order and composition of the gap.
    """
    gs = create_glazing_system_from_files(
        name="gs2",
        layers=[glass_path, glass_path],
        gaps=[
            Gap(
                gases=[
                    Gas(gas=GasType.air, ratio=0.1),
                    Gas(gas=GasType.argon, ratio=0.9),
                ],
                thickness=0.03,
            )
        ],
    )

    assert gs.gaps[0].gases[0].gas == "air"
    assert gs.gaps[0].gases[0].ratio == 0.1
    assert gs.gaps[0].gases[1].gas == "argon"
    assert gs.gaps[0].gases[1].ratio == 0.9
    assert gs.gaps[0].thickness == 0.03


def test_multilayer_glazing_shading(glass_path, shade_path):
    """
    Test GlazingSystem object with multiple layers of glazing and shading and more than one customized gap.

    Check the thickness of the glazing system.
    Check the order of the layers.
    Check the order and composition of the gaps.
    """
    gs = create_glazing_system_from_files(
        name="gs3",
        layers=[glass_path, glass_path, shade_path],
        gaps=[
            Gap(
                gases=[
                    Gas(gas=GasType.air, ratio=0.1),
                    Gas(gas=GasType.argon, ratio=0.9),
                ],
                thickness=0.03,
            ),
            Gap(gases=[Gas(gas=GasType.air, ratio=1)], thickness=0.01),
        ],
    )

    assert gs.layers[0].name.startswith("SageGlass")
    assert gs.layers[1].name.startswith("SageGlass")
    assert gs.layers[2].name == "Satine 5500 5%, White Pearl"

    assert gs.name == "gs3"
    assert gs.gaps[0].gases[0].gas == GasType.air
    assert gs.gaps[0].gases[0].ratio == 0.1
    assert gs.gaps[0].gases[1].gas == GasType.argon
    assert gs.gaps[0].gases[1].ratio == 0.9
    assert gs.gaps[0].thickness == 0.03
    assert gs.gaps[1].gases[0].gas == GasType.air
    assert gs.gaps[1].gases[0].ratio == 1
    assert gs.gaps[1].thickness == 0.01

    assert gs.visible_back_reflectance is not None
    assert gs.solar_back_absorptance is not None
