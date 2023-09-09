from pathlib import Path
import pytest


@pytest.fixture(scope="session")
def resources_dir():
    yield Path(__file__).parent / "Resources"


@pytest.fixture(scope="session")
def objects_dir():
    yield Path(__file__).parent / "Objects"
