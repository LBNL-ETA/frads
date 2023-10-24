import os
from pathlib import Path
from frads.window import create_glazing_system, Gas, Gap, GlazingSystem, AIR, ARGON
import pytest


@pytest.fixture
def glass_path(resources_dir):
    return resources_dir / "CLEAR_3.DAT"

@pytest.fixture
def shade_path(resources_dir):
    return resources_dir / "2011-SA1.xml"

@pytest.fixture
def glazing_system(glass_path):
    gs = create_glazing_system(
        name="gs1",
        layers=[glass_path, glass_path],
    )
    return gs

def test_save_and_load(glazing_system):
    """
    Test the save method of the GlazingSystem class.
    """
    glazing_system.save("test.json")
    assert Path("test.json").exists()
    gs2 = GlazingSystem.from_json("test.json")
    os.remove("test.json")
    assert gs2.name == glazing_system.name
    assert gs2.visible_back_reflectance == glazing_system.visible_back_reflectance


def test_simple_glazingsystem(glazing_system):
    """
    Test the GlazingSystem class.
    Build a GlazingSystem object consisting of two layer of clear glass.

    Check the thickness of the glazing system.
    Check the order and name of the layers.
    Check the composition of the default gap.
    """

    assert glazing_system.layers[0].product_name == "Generic Clear Glass"
    assert glazing_system.layers[1].product_name == "Generic Clear Glass"
    assert glazing_system.name == "gs1"
    assert glazing_system.gaps[0].gas[0].gas == "air"
    assert glazing_system.gaps[0].gas[0].ratio == 1
    assert glazing_system.gaps[0].thickness == 0.0127


def test_customized_gap(glass_path):
    """
    Test the building of a customized gap.
    A 0.03 m thick gap between the two glass layers. The gap is filled with 90% argon and 10% air.

    Check the thickness of the glazing system.
    Check the order and composition of the gap.
    """
    gs = create_glazing_system(
        name="gs2",
        layers=[glass_path, glass_path],
        gaps=[Gap([Gas("air", 0.1), Gas("argon", 0.9)], 0.03)],
    )

    assert gs.gaps[0].gas[0].gas == "air"
    assert gs.gaps[0].gas[0].ratio == 0.1
    assert gs.gaps[0].gas[1].gas == "argon"
    assert gs.gaps[0].gas[1].ratio == 0.9
    assert gs.gaps[0].thickness == 0.03


def test_multilayer_glazing_shading(glass_path, shade_path):
    """
    Test GlazingSystem object with multiple layers of glazing and shading and more than one customized gap.

    Check the thickness of the glazing system.
    Check the order of the layers.
    Check the order and composition of the gaps.
    """
    gs = create_glazing_system(
        name="gs3",
        layers=[glass_path, glass_path, shade_path],
        gaps=[
            Gap([Gas("air", 0.1), Gas("argon", 0.9)], 0.03),
            Gap([Gas("air", 1)], 0.01),
        ],
    )

    assert gs.layers[0].product_name == "Generic Clear Glass"
    assert gs.layers[1].product_name == "Generic Clear Glass"
    assert gs.layers[2].product_name == "Satine 5500 5%, White Pearl"

    assert gs.name == "gs3"
    assert gs.gaps[0].gas[0].gas == "air"
    assert gs.gaps[0].gas[0].ratio == 0.1
    assert gs.gaps[0].gas[1].gas == "argon"
    assert gs.gaps[0].gas[1].ratio == 0.9
    assert gs.gaps[0].thickness == 0.03
    assert gs.gaps[1].gas[0].gas == "air"
    assert gs.gaps[1].gas[0].ratio == 1
    assert gs.gaps[1].thickness == 0.01

    assert gs.visible_back_reflectance is not None
    assert gs.solar_back_absorptance is not None
