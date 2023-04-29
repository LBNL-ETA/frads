import os
from pathlib import Path
import shutil
import sys
sys.path.append(".")

from frads import window
import pywincalc as pwc

test_dir = Path(__file__).resolve().parent

def test_window():
    # Create a window
    gs = window.GlazingSystem()
    gs.add_glazing_layer(test_dir / "Resources" / "igsdb_product_7968.json")
    gs.compute_solar_photopic_results()


