"""Multiply Radiance matrix files.
"""

import argparse
import multiprocessing as mp
import glob
import logging
import os
import re
import subprocess as sp
try:
    import numpy as np
    NUMPY_FOUND = True
except ModuleNotFoundError:
    NUMPY_FOUND = False

from frads import util


logger = logging.getLogger("frads.radutil")


def parse_rad_header(header_str: str) -> tuple:
    """Parse a Radiance matrix file header."""
    compiled = re.compile(
        r' NROWS=(.*) | NCOLS=(.*) | NCOMP=(.*) | FORMAT=(.*) ', flags=re.X)
    matches = compiled.findall(header_str)
    if len(matches) != 4:
        raise ValueError("Can't find one of the header entries.")
    nrow = int([mat[0] for mat in matches if mat[0] != ''][0])
    ncol = int([mat[1] for mat in matches if mat[1] != ''][0])
    ncomp = int([mat[2] for mat in matches if mat[2] != ''][0])
    dtype = [mat[3] for mat in matches if mat[3] != ''][0].strip()
    return nrow, ncol, ncomp, dtype


def pcomb(inputs):
    """Image operations with pcomb.
    Parameter: inputs,
        e.g: ['img1.hdr', '+', img2.hdr', '-', 'img3.hdr', 'output.hdr']
    """
    input_list = inputs[:-1]
    out_dir = inputs[-1]
    component_idx = range(0, len(input_list), 2)
    components = [input_list[i] for i in component_idx]
    color_op_list = []
    for c in 'rgb':
        color_op = input_list[:]
        for i in component_idx:
            color_op[i] = '%si(%d)' % (c, i/2+1)
        cstr = '%so=%s' % (c, ''.join(color_op))
        color_op_list.append(cstr)
    rgb_str = ';'.join(color_op_list)
    cmd = ['pcomb', '-e', '%s' % rgb_str]
    img_name = util.basename(input_list[0], keep_ext=True)
    for c in components:
        cmd.append('-o')
        cmd.append(c)
    res = util.spcheckout(cmd)
    with open(os.path.join(out_dir, img_name), 'wb') as wtr:
        wtr.write(res)


def dctimestep(input_list):
    """Image operations in forms of Vs, VDs, VTDs, VDFs, VTDFs."""
    inputs = input_list[:-1]
    out_dir = input_list[-1]
    inp_dir_count = len(inputs)
    sky = input_list[-2]
    img_name = util.basename(sky)
    out_path = os.path.join(out_dir, img_name)

    if inputs[1].endswith('.xml') is False\
            and inp_dir_count > 2 and os.path.isdir(inputs[0]):
        combined = "'!rmtxop %s" % (' '.join(inputs[1:-1]))
        img = [i for i in os.listdir(inputs[0]) if i.endswith('.hdr')][0]
        str_count = len(img.split('.hdr')[0])  # figure out unix %0d string
        appendi = r"%0"+"%sd.hdr" % (str_count)
        new_inp_dir = [os.path.join(inputs[0], appendi), combined]
        cmd = "dctimestep -oc %s %s' > %s.hdr" \
            % (' '.join(new_inp_dir), sky, out_path)

    else:
        if not os.path.isdir(inputs[0]) and not inputs[1].endswith('.xml'):
            combined = os.path.join(os.path.dirname(inputs[0]), "tmp.vfdmtx")
            stderr = combine_mtx(inputs[:-1], combined)
            if stderr != "":
                print(stderr)
                return
            inputs_ = [combined]

        elif inp_dir_count == 5:
            combined = os.path.join(os.path.dirname(inputs[2]), "tmp.fdmtx")
            stderr = combine_mtx(inputs[2:4], combined)
            if stderr != "":
                print(stderr)
                return
            inputs_ = [inputs[0], inputs[1], combined]

        else:
            inputs_ = inputs[:-1]

        if os.path.isdir(inputs[0]):
            img = [i for i in os.listdir(inputs[0])
                   if i.endswith('.hdr')][0]
            str_count = len(img.split('.hdr')[0])
            appendi = r"%0"+"%sd.hdr" % (str_count)
            inputs_[0] = os.path.join(inputs[0], appendi)
            out_ext = ".hdr"
        else:
            out_ext = ".dat"

        input_string = ' '.join(inputs_)
        out_path = out_path + out_ext
        cmd = "dctimestep %s %s > %s" % (input_string, sky, out_path)
    sp.call(cmd, shell=True)


def dctsop(inputs, out_dir, nproc=1):
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    nproc = mp.cpu_count() if nproc is None else nproc
    process = mp.Pool(nproc)
    assert len(inputs) > 1
    mtx_inp = inputs[:-1]
    sky_dir = inputs[-1]
    sky_files = (os.path.join(sky_dir, i) for i in os.listdir(sky_dir))
    grouped = [mtx_inp+[skv] for skv in sky_files]
    [sub.append(out_dir) for sub in grouped]
    process.map(dctimestep, grouped)


def pcombop(inputs, out_dir, nproc=1):
    """
    inputs(list): e.g.[inpdir1,'+',inpdir2,'-',inpdir3]
    out_dir: output directory
    """
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    nproc = mp.cpu_count() if nproc is None else nproc
    process = mp.Pool(nproc)
    assert len(inputs) > 2
    assert len(inputs) % 2 == 1
    inp_dir = [inputs[i] for i in range(0, len(inputs), 2)]
    inp_cnt = len(inp_dir)
    ops = [inputs[i] for i in range(1, len(inputs), 2)]
    inp_lists = []
    for i in inp_dir:
        if os.path.isdir(i):
            inp_lists.append(glob.glob(os.path.join(i, '*.hdr')))
        else:
            inp_lists.append(i)
    inp_lists_full = []
    for i in range(inp_cnt):
        if os.path.isdir(inp_dir[i]):
            inp_lists_full.append(inp_lists[i])
        else:
            inp_lists_full.append(inp_dir[i])
    max_len = sorted([len(i) for i in inp_lists_full if type(i) == list])[0]
    for i in range(len(inp_lists_full)):
        if type(inp_lists_full[i]) is not list:
            inp_lists_full[i] = [inp_lists_full[i]] * max_len
    equal_len = all(len(i) == len(inp_lists_full[0]) for i in inp_lists_full)
    if not equal_len:
        logger.warning("Input directories don't the same number of files")
    grouped = [list(i) for i in zip(*inp_lists_full)]
    [sub.insert(i, ops[int((i-1)/2)])
     for sub in grouped for i in range(1, len(sub)+1, 2)]
    [sub.append(out_dir) for sub in grouped]
    process.map(pcomb, grouped)


def rpxop():
    """Operate on input directories given a operation type."""
    PROGRAM_SCRIPTION = "Image operations with parallel processing"
    parser = argparse.ArgumentParser(
        prog='imgop', description=PROGRAM_SCRIPTION)
    parser.add_argument('-t', type=str, required=True,
                        choices=['dcts', 'pcomb'],
                        help='operation types: {pcomb|dcts}')
    parser.add_argument('-i', type=str, required=True,
                        nargs="+", help='list of inputs')
    parser.add_argument('-o', type=str, required=True, help="output directory")
    parser.add_argument('-n', type=int, help='number of processors to use')
    args = parser.parse_args()
    if args.t == "pcomb":
        pcombop(args.i, args.o, nproc=args.n)
    elif args.t == 'dcts':
        dctsop(args.i, args.o, nproc=args.n)


def combine_mtx(mtxs, out_dir):
    """."""
    cmd = "rmtxop"
    args = " -ff %s > %s" % (" ".join(mtxs), out_dir)
    process = sp.Popen(cmd + args, stdout=sp.PIPE, stderr=sp.PIPE, shell=True)
    _, stderr = process.communicate()
    return stderr


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
    chunks = data_str.split(linesep2)
    nrow, ncol, ncomp, dtype = parse_rad_header(chunks[0].decode())
    if dtype == 'ascii':
        data = np.array(
            [line.split() for line in chunks[1].splitlines()], dtype=float)
    else:
        if dtype == 'float':
            data = np.frombuffer(chunks[1], np.single).reshape(nrow, ncol * ncomp)
        elif dtype == 'double':
            data = np.frombuffer(chunks[1], np.double).reshape(nrow, ncol * ncomp)
        else:
            raise ValueError("Unsupported data type %s" % dtype)
    rdata = data[:, ::ncomp]
    gdata = data[:, 1::ncomp]
    bdata = data[:, 2::ncomp]
    if (len(bdata) != nrow) or (len(bdata[0]) != ncol):
        raise ValueError("Parsing matrix file failed")
    return rdata, gdata, bdata


def smx2nparray(data_str):
    """Convert Radiance sky matrix file to numpy array.
    """
    if not data_str.startswith(b"#?RADIANCE"):
        raise ValueError("No header found")
    if b"\r\n\r\n" in data_str:
        linesep2 = b"\r\n\r\n"
    else:
        linesep2 = b"\n\n"
    chunks = data_str.split(linesep2)
    header_str = chunks[0].decode()
    content = chunks[1:]
    nrow, ncol, ncomp, dtype = parse_rad_header(header_str)
    if dtype == 'ascii':
        data = [i.splitlines() for i in content if i != b'']
        rdata = np.array([[i.split()[::ncomp][0] for i in row] for row in data],
                         dtype=float)
        gdata = np.array([[i.split()[1::ncomp][0] for i in row] for row in data],
                         dtype=float)
        bdata = np.array([[i.split()[2::ncomp][0] for i in row] for row in data],
                         dtype=float)
    else:
        if dtype == 'float':
            data = np.frombuffer(b"\n\n".join(content), np.single).reshape(
                nrow, ncol * ncomp)
        elif dtype == 'double':
            data = np.frombuffer(b"\n\n".join(content), np.double).reshape(
                nrow, ncol * ncomp)
        else:
            raise ValueError("Unsupported data type %s" % dtype)
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


def numpy_mtxmult(mtxs, weight=None):
    """Matrix multiplication with Numpy."""
    weight = (47.4, 119.9, 11.6) if weight is None else weight
    resr = np.linalg.multi_dot([mat[0] for mat in mtxs]) * weight[0]
    resg = np.linalg.multi_dot([mat[1] for mat in mtxs]) * weight[1]
    resb = np.linalg.multi_dot([mat[2] for mat in mtxs]) * weight[2]
    return resr + resg + resb


def rad_mtxmult3(*mtxs, weights: tuple = (), no_header=True):
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
        cmd1 = ['dctimestep', '-od'] + list(mtxs)
        inp1 = None
    else:
        cmd1 = ['dctimestep', '-od'] + list(mtxs)[:-1]
        inp1 = mtxs[-1].encode()
    if weights == ():
        weights = (47.4, 119.9, 11.6)
        logger.warning("Using default photopic RGB weights")
    cmd2 = ['rmtxop', '-fa', '-c', str(weights[0]),
            str(weights[1]), str(weights[2]), '-', '-t']
    out1 = sp.Popen(cmd1, stdin=inp1, stdout=sp.PIPE)
    out2 = sp.Popen(cmd2, stdin=out1.stdout, stdout=sp.PIPE)
    out1.stdout.close()
    if no_header:
        cmd3 = ['getinfo', '-']
        out3 = sp.Popen(cmd3, stdin=out2.stdout, stdout=sp.PIPE)
        out2.stdout.close()
        out = out3.communicate()[0]
    else:
        out = out2.communicate()[0]
    return out


def mtxmult(*mtxs):
    if NUMPY_FOUND:
        def mtx_parser(fpath):
            if fpath.endswith('.xml'):
                raw = util.spcheckout(['rmtxop', fpath])
            else:
                with open(fpath, 'rb') as rdr:
                    raw = rdr.read()
            return mtxstr2nparray(raw)
        npmtx = [mtx_parser(mtx) for mtx in mtxs[:-1]]
        if os.path.isfile(mtxs[-1]):
            with open(mtxs[-1], 'rb') as rdr:
                smx_str = rdr.read()
        else:
            smx_str = mtxs[-1]
        npmtx.append(smx2nparray(smx_str))
        return numpy_mtxmult(npmtx)
    else:
        return rad_mtxmult3(*mtxs)


def imgmult(*mtx, odir):
    """Image-based matrix multiplication using dctimestep."""
    util.mkdir_p(odir)
    cmd = ['dctimestep', '-oc', '-o', os.path.join(odir, '%04d.hdr')]
    cmd += list(mtx)
    return cmd


def dctsnp():
    """Commandline program that performs matrix multiplication using numpy."""
    if not NUMPY_FOUND:
        print("Numpy not found")
        return
    aparser = argparse.ArgumentParser(
        prog='dctsnp',
        description='dctimestep but using numpy (non-image)')
    aparser.add_argument('-m', '--mtx', required=True,
                         nargs='+', help='scene matrix')
    aparser.add_argument('-s', '--smx', required=True, help='sky matrix')
    aparser.add_argument('-w', '--weight', type=float, default=None,
                         nargs=3, help='RGB weights')
    aparser.add_argument('-o', '--output', required=True, help='output path')
    args = aparser.parse_args()

    def mtx_parser(fpath):
        if fpath.endswith('.xml'):
            raw = util.spcheckout(['rmtxop', fpath])
        else:
            with open(fpath, 'rb') as rdr:
                raw = rdr.read()
        return mtxstr2nparray(raw)
    npmtx = [mtx_parser(mtx) for mtx in args.mtx]
    with open(args.smx, 'rb') as rdr:
        npmtx.append(smx2nparray(rdr.read()))
    result = numpy_mtxmult(npmtx, weight=args.weight)
    np.savetxt(args.output, result)
