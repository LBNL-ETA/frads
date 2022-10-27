import argparse
import os
from pathlib import Path
import sys
sys.path.append('.')
import shutil

from frads import cli


def test_init():
    os.chdir("test")
    args = argparse.Namespace()
    args.name = "test"
    args.wea_path = Path("Resources/test.wea")
    args.object = [Path("Objects/wall.rad")]
    args.material = [Path("Objects/materials.mat")]
    args.grid = None
    args.bsdf = None
    args.window = None
    cli.mrad_init(args)
    assert os.path.isfile("test.cfg")
    os.remove("test.cfg")
    args.wea_path = None
    args.name = "default"
    args.epw_path = Path("Resources/USA_CA_Oakland.Intl.AP.724930_TMY3.epw")
    cli.mrad_init(args)
    assert os.path.isfile("default.cfg")
    os.remove("default.cfg")
    os.chdir("..")


def test_two_phase():
    os.chdir("test")
    args = argparse.Namespace()
    args.verbose = 4
    args.cfg = Path("two_phase.cfg")
    cli.mrad_run(args)
    assert os.path.isfile("Matrices/pdsmx_two_phase_floor.mtx")
    dsmx_size = os.stat("Matrices/pdsmx_two_phase_floor.mtx").st_size
    assert dsmx_size >= 2.5e6
    assert os.path.isdir("Matrices/vdsmx_two_phase_view_00")
    vsmx_size = len(os.listdir("Matrices/vdsmx_two_phase_view_00"))
    assert vsmx_size == 2306
    view_results = Path("Results/view_two_phase_view_00").glob("*.hdr")
    assert len(list(view_results)) == 2
    shutil.rmtree("Matrices")
    shutil.rmtree("Results")
    os.chdir("..")


def test_three_phase():
    os.chdir("test")
    args = argparse.Namespace()
    args.verbose = 4
    args.cfg = Path("three_phase.cfg")
    cli.mrad_run(args)
    view_results = Path("Results/view_three_phase_view_00").glob("*.hdr")
    assert len(list(view_results)) == 1
    shutil.rmtree("Matrices")
    shutil.rmtree("Results")
    os.chdir("..")

def test_five_phase():
    os.chdir("test")
    args = argparse.Namespace()
    args.verbose = 4
    args.cfg = Path("five_phase.cfg")
    cli.mrad_run(args)
    view_results = Path("Results/view_five_phase_view_00").glob("*.hdr")
    assert len(list(view_results)) == 4387
    shutil.rmtree("Matrices")
    shutil.rmtree("Results")
    os.chdir("..")

def test_five_phase2():
    os.chdir("test")
    args = argparse.Namespace()
    args.verbose = 4
    args.cfg = Path("five_phase2.cfg")
    cli.mrad_run(args)
    view_results = Path("Results/view_five_phase2_view_00").glob("*.hdr")
    assert len(list(view_results)) == 4387
    shutil.rmtree("Matrices")
    shutil.rmtree("Results")
    os.chdir("..")
