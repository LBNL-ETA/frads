#!/usr/bin/env python
"""
Generate F matrix.
Window zone 0-9

"""

import logging
import math
import os
import shutil
import subprocess as sp
import tempfile as tf
from typing import List

from frads import radmtx as rm
from frads import radgeom as rg
from frads import radutil, util

logger = logging.getLogger('frads.mfacade')

Primitive = radutil.Primitive


class Genfmtx(object):
    """Generate facade matrix."""

    def __init__(self, *, win_polygons, port_prim, out, env, sbasis, rbasis, opt, refl, forw, wrap):
        """Generate the appropriate aperture for F matrix generation.
        Parameters:
            win_prim: window geometry primitive;
            ncs_prim: shading geometry primitive;
            out_path: output file path
            depth: distance between the window and the farther point of the shade
            geometry;
            scale: scale factor to window geometry so that it encapsulate the
            projected shadeing geometry onto the window surface;
            FN: Bool, whether to generate four aperture separately
            rs: receiver sampling basis;
            ss: sender sampling basis;
            refl: Do reflection?;
            forw: Do forward tracing calculation?;
            wrap: wrap to .XML file?;
            **kwargs: other arguments that will be passed to Genmtx;
        """
        self.win_polygon = win_polygons
        self.port_prim = port_prim

        self.out = out
        self.outdir: str = os.path.dirname(out)
        self.out_name = util.basename(out)
        self.env = env

        self.rbasis = rbasis
        self.sbasis = sbasis
        self.opt = opt
        self.refl = refl
        if wrap == True and rbasis.startswith('sc') and sbasis.startswith('sc'):
            sc = int(rbasis[2:])
            ttlog2 = math.log(sc, 2)
            assert ttlog2 % int(ttlog2) == 0
            self.ttrank = 4  # only anisotropic
            self.pctcull = 90
            self.ttlog2 = int(ttlog2)
            self.opt += ' -hd -ff'
        self.td = tf.mkdtemp()
        src_dict = {}
        fwrap_dict = {}
        for idx in range(len(self.win_polygon)):
            _tf = f'tf{idx}'
            _rf = f'rf{idx}'
            _tb = f'tb{idx}'
            _rb = f'rb{idx}'
            src_dict[_tb] = os.path.join(self.td, f'{_tb}.dat')
            fwrap_dict[_tb] = os.path.join(self.td, f'{_tb}p.dat')
            if forw:
                src_dict[_tf] = os.path.join(self.td, f'{_tf}.dat')
                fwrap_dict[_tf] = os.path.join(self.td, f'{_tf}p.dat')
            if refl:
                src_dict[_rb] = os.path.join(self.td, f'{_rb}.dat')
                fwrap_dict[_rb] = os.path.join(self.td, f'{_rb}p.dat')
                if forw:
                    src_dict[_rf] = os.path.join(self.td, f'{_rf}.dat')
                    fwrap_dict[_rf] = os.path.join(self.td, f'{_rf}p.dat')
        self.compute_back(src_dict)
        if forw:
            self.compute_front(src_dict)
        self.src_dict = src_dict
        self.fwrap_dict = fwrap_dict
        if wrap:
            self.wrap()
        else:
            for key in src_dict:
                out_name = f"{self.out_name}_{key}.mtx"
                shutil.move(src_dict[key], os.path.join(self.outdir, out_name))
        shutil.rmtree(self.td, ignore_errors=True)

    def compute_back(self, src_dict):
        """compute front side calculation(backwards)."""
        logger.info('Computing for front side')
        for idx, win_polygon in enumerate(self.win_polygon):
            logger.info(f'Front transmission for window {idx}')
            front_rcvr = rm.Receiver.as_surface(
                prim_list=self.port_prim, basis=self.rbasis,
                left=True, offset=None, source='glow', out=src_dict[f'tb{idx}'])
            sndr_prim = radutil.polygon2prim(win_polygon, 'fsender', f'window{idx}')
            sndr = rm.Sender.as_surface(
                prim_list=[sndr_prim], basis=self.sbasis, left=True, offset=None)
            if self.refl:
                logger.info(f'Front reflection for window {idx}')
                back_window = win_polygon.flip()
                back_window_prim = radutil.polygon2prim(
                    back_window, 'breceiver', f'window{idx}')
                back_rcvr = rm.Receiver.as_surface(
                    prim_list=[back_window_prim], basis="-"+self.rbasis,
                    left=False, offset=None, source='glow', out=src_dict[f'rb{idx}'])
                front_rcvr += back_rcvr
            rm.rfluxmtx(sender=sndr, receiver=front_rcvr, env=self.env,
                        out=None, opt=self.opt)

    def compute_front(self, src_dict):
        """compute back side calculation."""
        sndr_prim = []
        for p in self.port_prim:
            np = p.copy()
            np['real_args'] = np['polygon'].flip().to_real()
            sndr_prim.append(np)
        sndr = rm.Sender.as_surface(
            prim_list=sndr_prim, basis="-"+self.rbasis, offset=None, left=False)
        logger.info('Computing for back side')
        for idx in range(len(self.win_polygon)):
            logger.info(f'Back transmission for window {idx}')
            win_polygon = self.win_polygon[idx].flip()
            rcvr_prim = radutil.polygon2prim(win_polygon, 'breceiver', f'window{idx}')
            rcvr = rm.Receiver.as_surface(
                prim_list=[rcvr_prim], basis="-"+self.sbasis,
                left=False, offset=None, source='glow', out=src_dict[f'tf{idx}'])
            if self.refl:
                logger.info(f'Back reflection for window {idx}')
                brcvr_prim = [
                    radutil.polygon2prim(self.port_prim[i]['polygon'], 'freceiver', 'window' + str(i))
                    for i in range(len(self.port_prim))]
                brcvr = rm.Receiver.as_surface(
                    prim_list=brcvr_prim, basis=self.rbasis,
                    left=False, offset=None, source='glow',
                    out=src_dict[f'rf{idx}'])
                rcvr += brcvr
            rm.rfluxmtx(sender=sndr, receiver=rcvr, env=self.env, out=None,
                       opt=self.opt)

    def klems_wrap(self):
        """prepare wrapping for Klems basis."""
        for key in self.src_dict:
            for _ in range(len(self.win_polygon)):
                inp = self.src_dict[key]
                out = self.fwrap_dict[key]
                cmd = f"rmtxop -fa -t -c .265 .67 .065 {inp} | getinfo - > {out}"
                sp.call(cmd, shell=True)

    def wrap(self, **kwargs):
        """call wrapBSDF to wrap a XML file."""
        if self.sbasis.startswith('sc') and self.rbasis.startswith('sc'):
            for i in range(len(self.win_polygon)):
                sub_key = [k for k in self.src_dict if k.endswith(str(i))]
                sub_dict = {k: self.fwrap_dict[k] for k in sub_key}
                for key in sub_key:
                    self.rttree_reduce(self.src_dict[key], self.fwrap_dict[key])
                cmd = 'wrapBSDF -a t4 -s Visible {} '.format(' '.join(
                    [" ".join(('-' + i[:2], j)) for i, j in sub_dict.items()]))
                cmd += f"> {self.out_name}.xml"
                sp.call(cmd, shell=True)
        else:
            self.klems_wrap()
            for i, _ in enumerate(self.win_polygon):
                out_name = os.path.join(self.outdir, f"{self.out_name}_{i}.xml")
                sub_dict = {
                    k: self.fwrap_dict[k]
                    for k in self.fwrap_dict if k.endswith(str(i))
                }
                cmd = 'wrapBSDF -a {} -c {} '.format(self.rbasis, ' '.join(
                    [" ".join(('-' + i[:2], j)) for i, j in sub_dict.items()]))
                cmd += f'> {out_name}'
                sp.call(cmd, shell=True)

    def rttree_reduce(self, src, dest, spec='Visible'):
        """call rttree_reduce to reduce shirley-chiu to tensor tree.
        copied from genBSDF."""
        CIEuv = 'Xi=.5141*Ri+.3239*Gi+.1620*Bi;'
        CIEuv += 'Yi=.2651*Ri+.6701*Gi+.0648*Bi;'
        CIEuv += 'Zi=.0241*Ri+.1229*Gi+.8530*Bi;'
        CIEuv += 'den=Xi+15*Yi+3*Zi;'
        CIEuv += 'uprime=if(Yi,4*Xi/den,4/19);'
        CIEuv += 'vprime=if(Yi,9*Yi/den,9/19);'

        ns2 = int((2**self.ttlog2)**2)
        if spec == 'Visible':
            cmd = f'rcalc -e "Omega:PI/{ns2}" '
            cmd += '-e "Ri=$1;Gi=$2;Bi=$3" '
            cmd += f'-e "{CIEuv}" -e "$1=Yi/Omega" '
        elif spec == 'CIE-u':
            cmd = 'rcalc -e "Ri=$1;Gi=$2;Bi=$3" '
            cmd += f'-e "{CIEuv}" -e "$1=uprime"'
        elif spec == 'CIE-v':
            cmd = 'rcalc -e "Ri=$1;Gi=$2;Bi=$3" '
            cmd += f'-e "{CIEuv}" -e "$1=vprime"'

        if os.name == 'posix':
            cmd = cmd[:5] + ' -if3' + cmd[5:]
            cmd = cmd.replace('"', "'")
        if self.pctcull >= 0:
            avg = "-a" if self.refl else ""
            pcull = self.pctcull if spec == 'Visible' else (
                100 - (100 - self.pctcull) * .25)
            cmd2 = f"rcollate -ho -oc 1 {src} | "
            cmd2 += cmd
            if os.name == 'posix':
                cmd2 = cmd + f" -of {src} "
            cmd2 += f"| rttree_reduce {avg} -h -ff -t {pcull} -r {self.ttrank} -g {self.ttlog2} "
            cmd2 += f"> {dest}"
        else:
            if os.name == 'posix':
                cmd2 = cmd + f" {src}"
            else:
                cmd2 = f"rcollate -ho -oc 1 {src} | " + cmd
        logger.info(cmd2)
        sp.call(cmd2, shell=True)

def genport(*, wpolys, npolys, depth, scale):
    """Generate the appropriate aperture for F matrix generation."""
    if len(wpolys) > 1:
        wpoly = merge_windows(wpolys)
    else:
        wpoly = wpolys[0]
    wpoly = radutil.parse_polygon(wpoly.real_arg)
    wnorm = wpoly.normal()
    if npolys is not None:
        all_ports = get_port(wpoly, wnorm, npolys)
    elif depth is None:
        raise ValueError('Need to specify (depth and scale) or ncp path')
    else:  # user direct input
        extrude_vector = wpoly.normal().reverse().scale(depth)
        scale_vector = rg.Vector(scale, scale, scale)
        scaled_window = wpoly.scale(scale_vector, wpoly.centroid())
        all_ports = scaled_window.extrude(extrude_vector)[1:]
    port_prims = []
    for pi in range(len(all_ports)):
        new_prim = radutil.polygon2prim(all_ports[pi], 'port',
                                     'portf%s' % str(pi + 1))
        logger.debug(str(new_prim))
        port_prims.append(new_prim)
    return port_prims

def get_port(win_polygon, win_norm, ncs_prims):
    """
    Generate ports polygons that encapsulate the window and NCP geometries.

    window and NCP geometries are rotated around +Z axis until
    the area projected onto XY plane is the smallest, thus the systems are facing
    orthogonal direction. A boundary box is then generated with a slight
    outward offset. This boundary box is then rotated back the same amount
    to encapsulate the original window and NCP geomteries.
    """
    ncs_polygon = [radutil.parse_polygon(p.real_arg)
                   for p in ncs_prims if p.ptype=='polygon']
    if 1 in [int(abs(i)) for i in win_norm.to_list()]:
        ncs_polygon.append(win_polygon)
        bbox = rg.getbbox(ncs_polygon, offset=0.00)
        bbox.remove([b for b in bbox if b.normal().reverse()==win_norm][0])
        return [b.move(win_norm.scale(-.1)) for b in bbox]
    xax = [1, 0, 0]
    _xax = [-1, 0, 0]
    yax = [0, 1, 0]
    _yax = [0, -1, 0]
    zaxis = rg.Vector(0, 0, 1)
    rm_pg = [xax, _yax, _xax, yax]
    area_list = []
    win_normals = []
    # Find axiel aligned rotation angle
    bboxes = []
    for deg in range(90):
        rad = math.radians(deg)
        win_polygon_r = win_polygon.rotate(zaxis, rad)
        win_normals.append(win_polygon_r.normal())
        ncs_polygon_r = [p.rotate(zaxis, rad) for p in ncs_polygon]
        ncs_polygon_r.append(win_polygon_r)
        _bbox = rg.getbbox(ncs_polygon_r, offset=0.001)
        bboxes.append(_bbox)
        area_list.append(_bbox[0].area())
    # Rotate to position
    deg = area_list.index(min(area_list))
    rrad = math.radians(deg)
    bbox = bboxes[deg]
    _win_normal = [round(i, 1) for i in win_normals[deg].to_list()]
    del bbox[rm_pg.index(_win_normal) + 2]
    rotate_back = [pg.rotate(zaxis, rrad * -1) for pg in bbox]
    return rotate_back

def merge_windows(primitive_list: List[Primitive]):
    """Merge rectangles if coplanar."""
    polygons = [radutil.parse_polygon(p.real_arg) for p in primitive_list]
    normals = [p.normal() for p in polygons]
    if len(set(normals)) > 1:
        logger.warning("Windows Oriented Differently")
    points = [i for p in polygons for i in p.vertices]
    hull_polygon = rg.convexhull(points, normals[0])
    modifier = primitive_list[0].modifier
    identifier = primitive_list[0].identifier
    new_prim = radutil.polygon2prim(hull_polygon, modifier, identifier)
    return new_prim


