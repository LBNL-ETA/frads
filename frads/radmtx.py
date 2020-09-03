"""
Support matrices generation.

T.Wang
"""

import os
import copy
import subprocess as sp
import tempfile as tf
import logging
from frads import makesky
from frads import radgeom
from frads import radutil

import pdb
logger = logging.getLogger('frads.radmtx')


class Sender:
    """Sender object for matrix generation."""

    def __init__(self, *, form, sender, xres, yres):
        """Instantiate the instance.

        Parameters:
            form(str): Sender as (s, v, p) for surface, view, and points;
            path(str): sender file path;
            sender(str):  content of the sender file;
            xres(int): x resolution of the image;
            yres(int): y resoluation or line count if form is pts;
        """
        self.form = form
        self.sender = sender
        self.xres = xres
        self.yres = yres
        logger.debug("Sender: %s", sender)

    @classmethod
    def as_surface(cls, *, prim_list, basis, offset=None, left=None):
        """
        Construct a sender from a surface.
        Parameters:
            prim_list(list): a list of primitives(dictionary)
            basis(str): sender sampling basis;
            offset(float): move the sender surface in its normal direction;
        """
        prim_str = prepare_surface(prims=prim_list, basis=basis, offset=offset,
                                   left=left, source=None, out=None)
        return cls(form='s', sender=prim_str, xres=None, yres=None)

    @classmethod
    def as_view(cls, *, vu_dict, ray_cnt, xres, yres):
        """
        Construct a sender from a view.
        Parameters:
            vu_dict(dict): a dictionary containing view parameters;
            ray_cnt(int): ray count;
            xres, yres(int): image resolution
            c2c(bool): Set to True to trim the fisheye corner rays.
        """
        assert None not in (xres, yres), "Need to specify resolution"
        vcmd = f"vwrays {radutil.opt2str(vu_dict)} -x {xres} -y {yres} -d"
        res_eval = radutil.spcheckout(vcmd.split()).decode().split()
        xres, yres = res_eval[1], res_eval[3]
        logger.info("Changed resolution to %s %s", xres, yres)
        cmd = f"vwrays -ff -x {xres} -y {yres} "
        if ray_cnt > 1:
            vu_dict['c'] = ray_cnt
            vu_dict['pj'] = 0.7  # placeholder
        logger.debug("Ray count is %s", ray_cnt)
        cmd += radutil.opt2str(vu_dict)
        if vu_dict['vt'] == 'a':
            cmd += "|" + Sender.crop2circle(ray_cnt, xres)
        vrays = sp.run(cmd, shell=True, check=True, stdout=sp.PIPE).stdout
        return cls(form='v', sender=vrays, xres=xres, yres=yres)

    @classmethod
    def as_pts(cls, *, pts_list, ray_cnt=1):
        """Construct a sender from a list of points.
        Parameters:
            pts_list(list): a list of list of float
            ray_cnt(int): sender ray count
        """
        assert pts_list is not None, "pts_list is None"
        pts_list = [i for i in pts_list for _ in range(ray_cnt)]
        grid_str = os.linesep.join(
            [' '.join(map(str, li)) for li in pts_list]) + os.linesep
        return cls(form='p', sender=grid_str, xres=None, yres=len(pts_list))

    @staticmethod
    def crop2circle(ray_cnt, xres):
        """Flush the corner rays from a fisheye view
        Parameters:
            ray_cnt(int): ray count;
            xres(int): resolution of the square image;
        """
        cmd = "rcalc -if6 -of "
        cmd += f'-e "DIM:{xres};CNT:{ray_cnt}" '
        cmd += '-e "pn=(recno-1)/CNT+.5" '
        cmd += '-e "frac(x):x-floor(x)" '
        cmd += '-e "xpos=frac(pn/DIM);ypos=pn/(DIM*DIM)" '
        cmd += '-e "incir=if(.25-(xpos-.5)*(xpos-.5)-(ypos-.5)*(ypos-.5),1,0)" '
        cmd += ' -e "$1=$1;$2=$2;$3=$3;$4=$4*incir;$5=$5*incir;$6=$6*incir"'
        if os.name == "posix":
            cmd = cmd.replace('"', "'")
        return cmd


class Receiver:
    """Receiver object for matrix generation."""

    def __init__(self, receiver, basis, modifier=None):
        """Instantiate the receiver object.
        Parameters:
            receiver (str): filepath {sky | sun | file_path}
        """
        self.receiver = receiver
        self.basis = basis
        self.modifier = modifier
        logger.debug("Receivers: %s", receiver)

    def __add__(self, other):
        self.receiver += other.receiver
        return self

    @classmethod
    def as_sun(cls, *, basis, smx_path, window_paths):
        """
        basis: receiver sampling basis {kf | r1 | sc25...}
        """
        gensun = makesky.Gensun(int(basis[-1]))
        if (smx_path is None) and (window_paths is None):
            str_repr = gensun.gen_full()
        else:
            str_repr = gensun.gen_cull(smx_path=smx_path, window_paths=window_paths)
        return cls(receiver=str_repr, basis=basis, modifier=gensun.mod_str)

    @classmethod
    def as_sky(cls, basis):
        """
        basis: receiver sampling basis {kf | r1 | sc25...}
        """
        assert basis.startswith('r'), 'Sky basis need to be Treganza/Reinhart'
        sky_str = makesky.basis_glow(basis)
        logger.debug(sky_str)
        return cls(receiver=sky_str, basis=basis)

    @classmethod
    def as_surface(cls, prim_list, basis, out,
                   offset=None, left=False, source='glow'):
        """
        basis: receiver sampling basis {kf | r1 | sc25...}
        """
        rcvr_str = prepare_surface(prims=prim_list, basis=basis, offset=offset,
                                   left=left, source=source, out=out)
        return cls(receiver=rcvr_str, basis=basis)


def prepare_surface(*, prims, basis, left, offset, source, out):
    """Prepare the sender or receiver surface, adding appropriate tags."""
    assert basis is not None, 'Sampling basis cannot be None'
    primscopy = copy.deepcopy(prims)
    upvector = radutil.up_vector(prims)
    # basis = "-" + basis if left else basis
    upvector = "-" + upvector if left else upvector
    modifier_set = {p['modifier'] for p in prims}
    if len(modifier_set) != 1:
        logger.warning("Primitives don't share modifier")
    src_mod = f"rflx{prims[0]['modifier']}"
    header = f'#@rfluxmtx h={basis} u={upvector}\n'
    if out is not None:
        header += f"#@rfluxmtx o={out}\n\n"
    if source is not None:
        source_line = f"void {source} {src_mod}\n0\n0\n4 1 1 1 0\n\n"
        header += source_line
    modifiers = [p['modifier'] for p in primscopy]
    # identifiers = [p['identifier'] for p in primscopy]
    for prim in primscopy:
        if prim['identifier'] in modifiers:
            prim['identifier'] = 'discarded'
    for prim in primscopy:
        prim['modifier'] = src_mod
    content = ''
    if offset is not None:
        for prim in primscopy:
            poly = prim['polygon']
            offset_vec = poly.normal().scale(offset)
            moved_pts = [pt + offset_vec for pt in poly.vertices]
            prim['real_args'] = radgeom.Polygon(moved_pts).to_real()
            content += radutil.put_primitive(prim)
    else:
        for prim in primscopy:
            content += radutil.put_primitive(prim)
    return header + content


def rfluxmtx(*, sender, receiver, env, opt=None, out=None):
    """Calling rfluxmtx to generate the matrices."""
    assert None not in (sender, receiver), "Sender/Receiver object is None"
    opt = '' if opt is None else opt
    with tf.TemporaryDirectory() as tempd:
        receiver_path = os.path.join(tempd, 'receiver')
        with open(receiver_path, 'w') as wtr:
            wtr.write(receiver.receiver)
        cmd = ['rfluxmtx'] + opt.split()
        if sender.form == 's':
            sender_path = os.path.join(tempd, 'sender')
            with open(sender_path, 'w') as wtr:
                wtr.write(sender.sender)
            cmd.extend([sender_path, receiver_path])
            stdin = None
        elif sender.form == 'p':
            cmd.extend(['-I+', '-faa', '-y', str(sender.yres), '-', receiver_path])
            stdin = sender.sender.encode()
        elif sender.form == 'v':
            cmd.extend(["-ffc", "-x", sender.xres, "-y", sender.yres, "-ld-"])
            if out is not None:
                radutil.mkdir_p(out)
                out = os.path.join(out, '%04d.hdr')
                cmd.extend(["-o", out])
            cmd.extend(['-', receiver_path])
            stdin = sender.sender
        cmd.extend(env)
        return radutil.spcheckout(cmd, input=stdin)


def rcvr_oct(receiver, env, oct_path):
    """Generate an octree of the environment and the receiver."""
    with tf.TemporaryDirectory() as tempd:
        receiver_path = os.path.join(tempd, 'rcvr_path')
        with open(receiver_path, 'w') as wtr:
            wtr.write(receiver.receiver)
        ocmd = ['oconv', '-f'] + env + [receiver_path]
        octree = radutil.spcheckout(ocmd)
        with open(oct_path, 'wb') as wtr:
            wtr.write(octree)


def rcontrib(*, sender, modifier, octree, out, opt):
    """Calling rcontrib to generate the matrices."""
    lopt = opt.split()
    lopt.append('-fo+')
    with tf.TemporaryDirectory() as tempd:
        modifier_path = os.path.join(tempd, 'modifier')
        with open(modifier_path, 'w') as wtr:
            wtr.write(modifier)
        cmd = ['rcontrib'] + lopt
        if sender.form == 'p':
            cmd += ['-I+', '-faf', '-y', sender.yres]
            stdin = sender.sender.encode()
        elif sender.form == 'v':
            radutil.mkdir_p(out)
            out = os.path.join(out, '%04d.hdr')
            cmd += ['-ffc', '-x', sender.xres, '-y', sender.yres]
            stdin = sender.sender
        cmd += ['-o', out, '-M', modifier_path, octree]
        radutil.spcheckout(cmd, input=stdin)


# if __name__ == '__main__':
#     with open('test.vf') as rdr:
#         vuline = rdr.read()
#     vu_dict = radutil.parse_vu(vuline)
#     vu_sndr = Sender.as_view(vu_dict=vu_dict,xres=1000, yres=1000)
#     pts_list = [[0,0,0,0,-1,0],[1,2,3,0,0,1]]
#     pt_sndr = Sender.as_pts(pts_list)
#     with open('test.rad') as rdr:
#         rline = rdr.readlines()
#     prim_list = radutil.parse_primitive(rline)
#     srf_sndr = Sender.as_surface(prim_list=prim_list, basis='kf', offset=None)
#     rsky = Receiver.as_sky(basis='r4')
#     rsun = Receiver.as_sun(basis='r6', smx_path=None, window_paths=None)
#     with open('test.rad') as rdr:
#         rline = rdr.readlines()
#     prim_list = radutil.parse_primitive(rline)
#     rsrf = Receiver.as_surface(prim_list=prim_list, basis='kf', offset=1, left=False,
#                         source='glow',out=None)
