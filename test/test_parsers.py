"""Unit tests for frads.parsers module."""
import os
import sys
from pathlib import Path
sys.path.append(".")

from frads import parsers
from frads.geom import Vector


testdir = Path(os.path.dirname(__file__))
resdir = testdir / "Resources"
objdir = testdir / "Objects"
epw_path = resdir / "USA_CA_Oakland.Intl.AP.724930_TMY3.epw"
wea_path = resdir / "oak.wea"


def test_parse_mrad_config():
    window_paths = [
        Path("test", "Objects", "lower_glass.rad"),
        Path("test", "Objects", "upper_glass.rad"),
    ]
    vdict = {"vf": "test/v1a.vf", "x": 4, "y": 4}
    vmx_opt = {"ab": 2, "ad": 64, "lw": 1e-4}
    epw_path = Path("Resources", "USA_CA_Oakland.Intl.AP.724930_TMY3.epw")
    cfg_path = Path("test", "Resources", "test.cfg")
    config = parsers.parse_mrad_config(cfg_path)
    view = config["RaySender"].getview("view")
    assert config["Model"].getpaths("windows") == window_paths
    assert config["Model"].getpaths("ncps") == []
    assert view.vtype == "a"
    assert view.position == Vector(17, 5, 4)
    assert view.hori == 180
    assert config["SimControl"].getoptions("vmx_opt") == vmx_opt
    assert config["Site"].getpaths("epw_path") == [epw_path]
    assert config["Site"].getpaths("wea_path") is None


def test_parse_vu():
    inp_str = "-vta -vv 180 -vh 180 -vp 0 0 0 -vd 0 -1 0"
    res = parsers.parse_vu(inp_str)
    answer = {"vt": "a", "vv": 180, "vh": 180,
              "vp": [0, 0, 0], "vd": [0, -1, 0]}
    assert res.position == Vector(0, 0, 0)
    assert res.direction == Vector(0, -1, 0)
    assert res.vtype == "a"
    assert res.hori == 180
    res2 = parsers.parse_vu("")
    assert res2 is None


def test_parse_opt():
    inp_str = "-ab 8 -ad 1024 -I+ -u- -c 8 -aa .1 -lw 1e-8"
    res = parsers.parse_opt(inp_str)
    answer = {
        "ab": 8,
        "ad": 1024,
        "I": True,
        "u": False,
        "c": 8,
        "aa": 0.1,
        "lw": 1e-8,
    }
    assert res == answer


def test_parser_epw():
    with open(epw_path) as rdr:
        wea_metadata, wea_data = parsers.parse_epw(rdr.read())
    assert wea_data[0].time.month == 1
    assert wea_data[-1].time.month == 12
    assert wea_data[-1].time.day == 31
    assert wea_metadata.latitude == 37.72

def test_parser_wea():
    with open(wea_path) as rdr:
        wea_metadata, wea_data = parsers.parse_wea(rdr.read())
    assert wea_data[0].time.month == 1
    assert wea_data[-1].time.month == 12
    assert wea_data[-1].time.day == 31
    assert wea_data[-1].dni == 0
    assert wea_metadata.latitude == 37.72

def test_parse_idf():
    pass

def test_parse_optics():
    pass

def test_parse_igsdb_json():
    pass

def test_parse_branch():
    pass

def test_parse_ttree():
    pass

def get_nested_list_levels():
    pass
