"""Multiply Radiance matrix files.
"""

import logging
import os
from pathlib import Path
import subprocess as sp
from typing import Tuple, List
from typing import Optional

from frads import parsers
from frads import utils

try:
    import numpy as np

    NUMPY_FOUND = True
except ModuleNotFoundError:
    NUMPY_FOUND = False


logger: logging.Logger = logging.getLogger("frads.mtxmult")


def batch_dctimestep(
    mtx: List[Path], sky_dir: Path, out_dir: Path, nproc: Optional[int] = None
) -> None:
    """Run dctimestep in batches."""
    out_dir.mkdir(parents=True, exist_ok=True)
    nproc = 1 if nproc is None else nproc
    if len(mtx) == 0:
        raise ValueError("Input matrices files empty")
    if not sky_dir.is_dir():
        raise ValueError("Sky directory not exist")
    cmds: List[List[str]] = []
    outs: List[str] = []
    for sky in sorted(sky_dir.glob("*")):
        out_path = out_dir / sky.with_suffix(".hdr")
        cmd = ["dctimestep"] + [str(f) for f in mtx]
        cmd.append(str(sky))
        cmds.append(cmd)
        outs.append(str(out_path))
    utils.batch_process(cmds, opaths=outs, nproc=nproc)


def batch_pcomb(
    inp: List[Path], ops: List[str], out_dir: Path, nproc: Optional[int] = None
) -> None:
    """
    inputs: e.g.[inpdir1, inpdir2, inpdir3.hdr]
    ops: e.g.['+', '-']
    out_dir: output directory
    Resulting files will be named after first input item.
    """
    out_dir.mkdir(exist_ok=True)
    nproc = 1 if nproc is None else nproc
    if (len(inp) < 2) or (len(ops) < 1):
        raise ValueError("Invalid # of inputs")
    if inp[0].is_file():
        raise ValueError("First input cannot be a file, please reformulate")
    expanded_inp: List[List[Path]] = []
    max_len = 0
    for i in inp:
        if i.is_dir():
            hdrs = sorted(i.glob("*.hdr"))
            max_len = max(max_len, len(hdrs))
            expanded_inp.append(list(hdrs))
        elif i.is_file():
            expanded_inp.append([i])
    if max_len > 1:
        for idx, ip in enumerate(expanded_inp):
            if len(ip) == 1:
                expanded_inp[idx] = ip * max_len
    equal_len = all(len(i) == len(expanded_inp[0]) for i in expanded_inp)
    if not equal_len:
        logger.warning("Input directories don't the same number of files")
    grouped = [list(i) for i in zip(*expanded_inp)]
    rstr_list = [f"ri({i+1})" for i, _ in enumerate(expanded_inp)]
    frstr_list = rstr_list + ops
    frstr_list[::2] = rstr_list
    frstr_list[1::2] = ops
    frstr_list.insert(0, "ro=")
    frstr = "".join(frstr_list)
    fgstr = frstr.replace("r", "g")
    fbstr = frstr.replace("r", "b")
    pcomb_expr = frstr + ";" + fgstr + ";" + fbstr
    cmds = []
    opaths = []
    for group in grouped:
        cmd = ["pcomb", "-e", pcomb_expr]
        for cp in group:
            cmd += ["-o", str(cp)]
        cmds.append(cmd)
        opaths.append(out_dir / group[0].name)
    utils.batch_process(cmds, opaths=opaths, nproc=nproc)


def mtxstr2nparray(data_str: bytes):
    """Convert Radiance 3-channel matrix file to numpy array.

    Args:
        data_str: file data string

    Returns:
        RGB numpy arrays
    """
    if not data_str.startswith(b"#?RADIANCE"):
        raise ValueError("No header found")
    if b"\r\n\r\n" in data_str:
        linesep2 = b"\r\n\r\n"
    else:
        linesep2 = b"\n\n"
    chunks = data_str.split(linesep2, 1)
    nrow, ncol, ncomp, dtype = parsers.parse_rad_header(chunks[0].decode())
    if dtype == "ascii":
        data = np.array([line.split() for line in chunks[1].splitlines()], dtype=float)
    else:
        if dtype == "float":
            data = np.frombuffer(chunks[1], np.single).reshape(nrow, ncol * ncomp)
        elif dtype == "double":
            data = np.frombuffer(chunks[1], np.double).reshape(nrow, ncol * ncomp)
        else:
            raise ValueError(f"Unsupported data type {dtype}")
    rdata = data[:, ::ncomp]
    gdata = data[:, 1::ncomp]
    bdata = data[:, 2::ncomp]
    if (len(bdata) != nrow) or (len(bdata[0]) != ncol):
        raise ValueError("Parsing matrix file failed")
    return rdata, gdata, bdata


def smx2nparray(data_str):
    """Convert Radiance sky matrix file to numpy array."""
    if not data_str.startswith(b"#?RADIANCE"):
        raise ValueError("No header found")
    if b"\r\n\r\n" in data_str:
        linesep2 = b"\r\n\r\n"
    else:
        linesep2 = b"\n\n"
    chunks = data_str.split(linesep2)
    header_str = chunks[0].decode()
    content = chunks[1:]
    nrow, ncol, ncomp, dtype = parsers.parse_rad_header(header_str)
    if dtype == "ascii":
        data = [i.splitlines() for i in content if i != b""]
        rdata = np.array(
            [[i.split()[::ncomp][0] for i in row] for row in data], dtype=float
        )
        gdata = np.array(
            [[i.split()[1::ncomp][0] for i in row] for row in data], dtype=float
        )
        bdata = np.array(
            [[i.split()[2::ncomp][0] for i in row] for row in data], dtype=float
        )
    else:
        if dtype == "float":
            data = np.frombuffer(b"\n\n".join(content), np.single).reshape(
                nrow, ncol * ncomp
            )
        elif dtype == "double":
            data = np.frombuffer(b"\n\n".join(content), np.double).reshape(
                nrow, ncol * ncomp
            )
        else:
            raise ValueError(f"Unsupported data type {dtype}")
        rdata = data[:, 0::ncomp]
        gdata = data[:, 1::ncomp]
        bdata = data[:, 2::ncomp]
    if ncol == 1:
        assert np.size(bdata, 1) == nrow
        assert np.size(bdata, 0) == ncol
        rdata = rdata.T
        gdata = gdata.T
        bdata = bdata.T
    else:
        assert np.size(bdata, 0) == nrow
        assert np.size(bdata, 1) == ncol
    return rdata, gdata, bdata


def numpy_mtxmult(mtxs, weight: Optional[Tuple[float, ...]] = None):
    """Matrix multiplication with Numpy."""
    weight = (47.4, 119.9, 11.6) if weight is None else weight
    resr = np.linalg.multi_dot([mat[0] for mat in mtxs]) * weight[0]
    resg = np.linalg.multi_dot([mat[1] for mat in mtxs]) * weight[1]
    resb = np.linalg.multi_dot([mat[2] for mat in mtxs]) * weight[2]
    return resr + resg + resb


def rad_mtxmult3(*mtxs, weights: tuple = (), no_header: bool = True):
    """Multiply matrices using dctimstep,
    Applying weights using rmtxop and remove header if needed.

    Args:
        mtx: Radiance matrices file paths
        weights: weight for the three channels
        no_header: whether to remove header from output

    Returns:
        output as byte strings

    Raises:
        Only works with four or two matrices.
    """
    if len(mtxs) not in (2, 4):
        raise ValueError("Only works with two or four matrices")
    if os.path.isfile(mtxs[-1]):
        cmd1 = ["dctimestep", "-od"] + list(mtxs)
        inp1 = None
    else:
        cmd1 = ["dctimestep", "-od"] + list(mtxs)[:-1]
        inp1 = mtxs[-1].encode()
    if weights == ():
        weights = (47.4, 119.9, 11.6)
        logger.warning("Using default photopic RGB weights")
    cmd2 = [
        "rmtxop",
        "-fa",
        "-c",
        str(weights[0]),
        str(weights[1]),
        str(weights[2]),
        "-",
        "-t",
    ]
    out1 = sp.Popen(cmd1, stdin=inp1, stdout=sp.PIPE)
    out2 = sp.Popen(cmd2, stdin=out1.stdout, stdout=sp.PIPE)
    if out1.stdout is not None:
        out1.stdout.close()
    if no_header:
        cmd3 = ["getinfo", "-"]
        out3 = sp.Popen(cmd3, stdin=out2.stdout, stdout=sp.PIPE)
        if out2.stdout is not None:
            out2.stdout.close()
        out = out3.communicate()[0]
    else:
        out = out2.communicate()[0]
    return out


def mtxmult(*mtxs):
    """Generic matrix multiplications."""
    if NUMPY_FOUND:

        def mtx_parser(fpath):
            if fpath.suffix == ".xml":
                proc = sp.run(
                    ["rmtxop", fpath],
                    check=True,
                    stdout=sp.PIPE,
                )
                raw = proc.stdout
            else:
                with open(fpath, "rb") as rdr:
                    raw = rdr.read()
            return mtxstr2nparray(raw)

        npmtx = [mtx_parser(mtx) for mtx in mtxs[:-1]]
        if os.path.isfile(mtxs[-1]):
            with open(mtxs[-1], "rb") as rdr:
                smx_str = rdr.read()
        else:
            smx_str = mtxs[-1]
        npmtx.append(smx2nparray(smx_str))
        return numpy_mtxmult(npmtx)
    return rad_mtxmult3(*mtxs)


def get_imgmult_cmd(*mtx: Path, odir: Path) -> List[str]:
    """Image-based matrix multiplication using dctimestep."""
    odir.mkdir(exist_ok=True)
    cmd = ["dctimestep", "-oc", "-o", str(odir / "%04d.hdr")]
    cmd += [str(m) for m in mtx]
    return cmd
