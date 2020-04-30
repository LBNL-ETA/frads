#!/usr/bin/env python
"""Generate F matrix.
Window zone 0-9

T.Wang
"""

import argparse
from frads import radmtx as rm
from frads import radgeom as rg
import math
import os
from frads import radutil
import shutil
import subprocess as sp
import tempfile as tf
import logging

logger = logging.getLogger('frads.mfacade')

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
        self.outdir = os.path.dirname(out)
        self.out_name = radutil.basename(out)
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
            src_dict[_tf] = os.path.join(self.td, f'{_tf}.dat')
            fwrap_dict[_tf] = os.path.join(self.td, f'{_tf}p.dat')
            if forw:
                src_dict[_tb] = os.path.join(self.td, f'{_tb}.dat')
                fwrap_dict[_tb] = os.path.join(self.td, f'{_tb}p.dat')
            if refl:
                src_dict[_rf] = os.path.join(self.td, f'{_rf}.dat')
                fwrap_dict[_rf] = os.path.join(self.td, f'{_rf}p.dat')
                if forw:
                    src_dict[_rb] = os.path.join(self.td, f'{_rb}.dat')
                    fwrap_dict[_rb] = os.path.join(self.td, f'{_rb}p.dat')
        self.compute_front(src_dict)
        if forw:
            self.compute_back(src_dict)
        self.src_dict = src_dict
        self.fwrap_dict = fwrap_dict
        if wrap:
            self.wrap()
        else:
            for key in src_dict:
                out_name = f"{self.out_name}_{key}.mtx"
                shutil.move(src_dict[key], os.path.join(self.outdir, out_name))
        shutil.rmtree(self.td, ignore_errors=True)

    def compute_front(self, src_dict):
        """compute front side calculation(backwards)."""
        logger.info('Computing for front side')
        for i in range(len(self.win_polygon)):
            logger.info(f'Front transmission for window {i}')
            front_rcvr = rm.Receiver.as_surface(tmpdir=self.td,
                prim_list=self.port_prim, basis=self.rbasis,
                left=None, offset=None, source='glow', out=src_dict[f'tf{i}'])
            win_polygon = self.win_polygon[i]
            sndr_prim = polygon_prim(win_polygon, 'fsender', f'window{i}')
            sndr = rm.Sender.as_surface(
                tmpdir=self.td, prim_list=[sndr_prim], basis=self.sbasis, offset=None)
            if self.refl:
                logger.info(f'Front reflection for window {i}')
                back_window = win_polygon.flip()
                back_window_prim = polygon_prim(
                    back_window, 'breceiver', f'window{i}')
                back_rcvr = rm.Receiver.as_surface(tmpdir=self.td,
                    prim_list=[back_window_prim], basis=self.rbasis,
                    left=True, offset=None, source='glow', out=src_dict[f'rf{i}'])
                front_rcvr += back_rcvr
            rm.rfluxmtx(sender=sndr, receiver=front_rcvr, env=self.env,
                        out=None, opt=self.opt)

    def compute_back(self, src_dict):
        """compute back side calculation."""
        sndr_prim = []
        for p in self.port_prim:
            np = p.copy()
            np['real_args'] = np['polygon'].flip().to_real()
            sndr_prim.append(np)
        sndr = rm.Sender.as_surface(tmpdir=self.td,
            prim_list=sndr_prim, basis=self.rbasis, offset=None)
        logger.info('Computing for back side')
        for idx in range(len(self.win_polygon)):
            logger.info(f'Back transmission for window {idx}')
            win_polygon = self.win_polygon[idx].flip()
            rcvr_prim = polygon_prim(win_polygon, 'breceiver', f'window{idx}')
            rcvr = rm.Receiver.as_surface(tmpdir=self.td,
                prim_list=[rcvr_prim], basis=self.sbasis,
                left=None, offset=None, source='glow', out=src_dict[f'tb{idx}'])
            if self.refl:
                logger.info(f'Back reflection for window {idx}')
                brcvr_prim = [
                    polygon_prim(self.port_prim[i]['polygon'], 'freceiver', 'window' + str(i))
                    for i in range(len(self.port_prim))]
                brcvr = rm.Receiver.as_surface(tmpdir=self.td,
                    prim_list=brcvr_prim, basis=self.rbasis,
                    left=True, offset=None, source='glow',
                    out=src_dict[f'rb{idx}'])
                rcvr += brcvr
            rm.rfluxmtx(sender=sndr, receiver=rcvr, env=self.env, out=None,
                       opt=self.opt)

    def klems_wrap(self):
        """prepare wrapping for Klems basis."""
        for key in self.src_dict:
            for i in range(len(self.win_polygon)):
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
                    self.rttree_reduce(key[:-1], self.src_dict[key],
                                       self.fwrap_dict[key])
                cmd = 'wrapBSDF -a t4 -s Visible {} '.format(' '.join(
                    [" ".join(('-' + i[:2], j)) for i, j in sub_dict.items()]))
                cmd += f"> {self.out_name}.xml"
                sp.call(cmd, shell=True)
        else:
            self.klems_wrap()
            for i in range(len(self.win_polygon)):
                sub_dict = {
                    k: self.fwrap_dict[k]
                    for k in self.fwrap_dict if k.endswith(str(i))
                }
                cmd = 'wrapBSDF -a {} -c {} '.format(self.rbasis, ' '.join(
                    [" ".join(('-' + i[:2], j)) for i, j in sub_dict.items()]))
                cmd += f'> {self.out_name}.xml'
                sp.call(cmd, shell=True)

    def rttree_reduce(self, typ, src, dest, spec='Visible'):
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
            cmd2 += f"| rttree_reduce {avg} -h -ff -t {pcull} -r 4 -g {self.ttlog2} "
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
    wpoly = wpoly['polygon']
    wnorm = wpoly.normal()
    wcntr = wpoly.centroid()
    if npolys is not None:
        all_ports = get_port(wpoly, wnorm, npolys)
    elif self.depth is None:
        raise 'Missing param: need to specify (depth and scale) or ncs file path'
    else:  # user direct input
        extrude_vector = wpoly.normal().reverse().scale(depth)
        scale_vector = rg.Vector(scale, scale, scale)
        scaled_window = wpoly.scale(scale_vector, wpoly.centroid())
        all_ports = scaled_window.extrude(extrude_vector)[1:]
    ports = all_ports
    port_prims = []
    for pi in range(len(all_ports)):
        new_prim = polygon_prim(all_ports[pi], 'port',
                                     'portf%s' % str(pi + 1))
        logger.debug(radutil.put_primitive(new_prim))
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
    ncs_polygon = [p['polygon'] for p in ncs_prims if p['type']=='polygon']
    if 1 in [int(abs(i)) for i in win_norm.to_list()]:
        ncs_polygon.append(win_polygon)
        bbox = rg.getbbox(ncs_polygon, offset=0.001)
        bbox.remove([b for b in bbox if b.normal().reverse()==win_norm][0])
        return bbox
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

def merge_window(primitive_list):
    """Merge rectangles if coplanar."""
    polygons = [p['polygon'] for p in primitive_list]
    normals = [p.normal() for p in polygons]
    norm_set = set([n.to_list() for n in normals])
    if len(norm_set) > 1:
        warn_msg = "windows oriented differently"
    points = [i for p in polygon for i in p.vertices]
    chull = rg.Convexhull(points, normals[0])
    hull_polygon = chull.hull
    real_args = hull_polygon.toreal()
    modifier = primitive_list[0]['modifier']
    identifier = primitive_list[0]['identifier']
    new_prim = self.polygon_prim(hull_polygon, modifier, identifier)
    return new_prim

def polygon_prim(polygon, mod, ident):
    """prepare a polygon primitive."""
    new_prim = {'str_args': '0', 'int_arg': '0', 'type': 'polygon'}
    new_prim['modifier'] = mod
    new_prim['identifier'] = ident
    new_prim['polygon'] = polygon
    new_prim['real_args'] = polygon.to_real()
    return new_prim

