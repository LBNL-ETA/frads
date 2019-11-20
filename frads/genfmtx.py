#!/usr/bin/env python
"""Generate F matrix.
Window zone 0-9

T.Wang
"""

import argparse
from frads import genmtx
from frads import radgeom
import math
import os
from frads import radutil
import shutil
import subprocess as sp
import tempfile as tf


class Genfmtx(object):
    """Generate facade matrix."""

    def __init__(self,
                 win_prim=None,
                 out_path=None,
                 ncs_prim=None,
                 depth=None,
                 scale=None,
                 FN=False,
                 merge=True,
                 ss='kf',
                 rs='kf',
                 refl=False,
                 forw=False,
                 wrap=False,
                 **kwargs):
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
        self.kwargs = kwargs
        self.refl = refl

        self.win_prims = win_prim
        self.ncs_prim = ncs_prim

        self.out_path = out_path
        self.outdir = os.path.dirname(out_path)
        self.out_name = radutil.basename(out_path)

        self.depth = depth
        self.scale = scale
        self.FN = FN
        self.merge = merge
        self.rbasis = rs
        self.sbasis = ss
        if wrap == True and rs.startswith('sc') and ss.startswith('sc'):
            sc = int(rs[2:])
            ttlog2 = math.log(sc, 2)
            assert ttlog2 % int(ttlog2) == 0
            self.ttrank = 4  # only anisotropic
            self.pctcull = 90
            self.ttlog2 = int(ttlog2)
            self.kwargs['opt'] += ' -hd'
            self.kwargs['opt'] += ' -ff'
        self.genport()
        td = tf.mkdtemp()
        src_dict = {}
        fwrap_dict = {}
        for idx in range(len(self.win_polygon)):
            _tf = 'tf{}'.format(idx)
            _rf = 'rf{}'.format(idx)
            _tb = 'tb{}'.format(idx)
            _rb = 'rb{}'.format(idx)
            src_dict[_tf] = os.path.join(td, '{}.dat'.format(_tf))
            fwrap_dict[_tf] = os.path.join(td, '{}p.dat'.format(_tf))
            if forw:
                src_dict[_tb] = os.path.join(td, '{}.dat'.format(_tb))
                fwrap_dict[_tb] = os.path.join(td, '{}p.dat'.format(_tb))
            if refl:
                src_dict[_rf] = os.path.join(td, '{}.dat'.format(_rf))
                fwrap_dict[_rf] = os.path.join(td, '{}p.dat'.format(_rf))
                if forw:
                    src_dict[_rb] = os.path.join(td, '{}.dat'.format(_rb))
                    fwrap_dict[_rb] = os.path.join(td, '{}p.dat'.format(_rb))
        self.compute_front(src_dict)
        if forw:
            self.compute_back(src_dict)
        self.src_dict = src_dict
        self.fwrap_dict = fwrap_dict
        if wrap:
            self.wrap()
        else:
            for key in src_dict:
                out_name = "{}_{}.mtx".format(self.out_name, key)
                shutil.move(src_dict[key], out_name)
        shutil.rmtree(td, ignore_errors=True)

    def compute_front(self, src_dict):
        """compute front side calculation(backwards)."""
        for i in range(len(self.win_polygon)):
            front_rcvr = genmtx.Receiver(self.port_prims,
                                         self.rbasis,
                                         out=src_dict['tf{}'.format(i)])
            win_polygon = self.win_polygon[i]
            sndr_prim = self.polygon_prim(win_polygon, 'fsender',
                                          'window{}'.format(i))
            sndr = genmtx.Sender(sndr_prim, basis=self.sbasis)
            if self.refl:
                back_window = win_polygon.flip()
                back_window_prim = self.polygon_prim(back_window, 'breceiver',
                                                     'window{}'.format(i))
                back_rcvr = genmtx.Receiver(back_window_prim,
                                            self.rbasis,
                                            left=True,
                                            out=src_dict['rf{}'.format(i)])
                front_rcvr += back_rcvr
            genmtx.Genmtx(sender=sndr, receiver=front_rcvr, **self.kwargs)

    def compute_back(self, src_dict):
        """compute back side calculation."""
        sndr_prim = []
        for p in self.port_prims:
            np = p.copy()
            np['real_args'] = np['polygon'].flip().to_real()
            sndr_prim.append(np)
        sndr = genmtx.Sender(sndr_prim, basis=self.rbasis)
        for idx in range(len(self.win_polygon)):
            win_polygon = self.win_polygon[idx].flip()
            rcvr_prim = self.polygon_prim(win_polygon, 'breceiver',
                                          'window{}'.format(idx))
            rcvr = genmtx.Receiver(rcvr_prim,
                                   self.sbasis,
                                   out=src_dict['tb{}'.format(idx)])
            if self.refl:
                brcvr_prim = [
                    self.polygon_prim(self.ports[i], 'freceiver',
                                      'window' + str(i))
                    for i in range(len(self.ports))
                ]
                brcvr = genmtx.Receiver(self.port_prims,
                                        self.rbasis,
                                        left=True,
                                        out=src_dict['rb{}'.format(idx)])
                rcvr += brcvr
            genmtx.Genmtx(sender=sndr, receiver=rcvr, **self.kwargs)

    def klems_wrap(self):
        """prepare wrapping for Klems basis."""
        for key in self.src_dict:
            for i in range(len(self.win_polygon)):
                inp = self.src_dict[key]
                out = self.fwrap_dict[key]
                cmd = "rmtxop -fa -t -c .265 .67 .065 {} | getinfo - > {}".format(
                    inp, out)
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
                cmd += "> {}.xml".format(self.out_name)
                print(cmd)
                sp.call(cmd, shell=True)
        else:
            self.klems_wrap()
            for i in range(len(self.win_polygon)):
                sub_dict = {
                    k: self.fwrap_dict[k]
                    for k in self.fwrap_dict if k.endswith(str(i))
                }
                cmd = 'wrapBSDF -a kf -c {} '.format(' '.join(
                    [" ".join(('-' + i[:2], j)) for i, j in sub_dict.items()]))
                cmd += '> {}.xml'.format(self.out_name)
                print(cmd)
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
            cmd = 'rcalc -e "Omega:PI/{}" '.format(ns2)
            cmd += '-e "Ri=$1;Gi=$2;Bi=$3" '
            cmd += '-e "{}" '.format(CIEuv)
            cmd += '-e "$1=Yi/Omega" '
        elif spec == 'CIE-u':
            cmd = 'rcalc -e "Ri=$1;Gi=$2;Bi=$3" '
            cmd += '-e "{}" '.format(CIEuv)
            cmd += '-e "$1=uprime"'
        elif spec == 'CIE-v':
            cmd = 'rcalc -e "Ri=$1;Gi=$2;Bi=$3" '
            cmd += '-e "{}" '.format(CIEuv)
            cmd += '-e "$1=vprime"'

        if os.name == 'posix':
            cmd = cmd[:5] + ' -if3' + cmd[5:]
            cmd = cmd.replace('"', "'")
        if self.pctcull >= 0:
            avg = "-a" if self.refl else ""
            pcull = self.pctcull if spec == 'Visible' else (
                100 - (100 - self.pctcull) * .25)
            cmd2 = "rcollate -ho -oc 1 {} | ".format(src)
            cmd2 += cmd
            if os.name == 'posix':
                cmd2 = cmd + " -of {} ".format(src)
            cmd2 += "| rttree_reduce {} -h -ff -t {} -r 4 -g {} ".format(
                avg, pcull, self.ttlog2)
            cmd2 += "> {}".format(dest)
        else:
            if os.name == 'posix':
                cmd2 = cmd + " {}".format(src)
            else:
                cmd2 = "rcollate -ho -oc 1 {} | ".format(src) + cmd
        print(cmd2)
        sp.call(cmd2, shell=True)

    def genport(self):
        """Generate the appropriate aperture for F matrix generation."""
        polygon_prims = [p for p in self.win_prims if p['type'] == 'polygon']
        self.win_polygon = [p['polygon'] for p in polygon_prims]
        if len(polygon_prims) > 1:
            win_prim = merge_windows(polygon_prims)
        else:
            win_prim = polygon_prims[0]
        win_polygon = win_prim['polygon']
        if self.merge:
            self.win_polygon = [win_polygon]
        win_norm = win_polygon.normal()
        win_cntr = win_polygon.centroid()

        if self.ncs_prim is not None:
            ncs_prims = [p for p in self.ncs_prim if p['type'] == 'polygon']
            all_ports = self.get_port(win_polygon, ncs_prims)
        elif self.depth is None:
            raise 'Missing param: need to specify (depth and scale) or ncs file path'
        else:  # user direct input
            extrude_vector = win_norm.reverse().scale(self.depth)
            scale_vector = radgeom.Vector(self.scale, self.scale, self.scale)
            scaled_window = win_polygon.scale(scale_vector, win_cntr)
            all_ports = scaled_window.extrude(extrude_vector)[1:]
        self.ports = all_ports
        self.port_prims = []
        for pi in range(len(all_ports)):
            new_prim = self.polygon_prim(all_ports[pi], 'port',
                                         'portf%s' % str(pi + 1))
            self.port_prims.append(new_prim)

    def get_port(self, win_polygon, ncs_prims):
        """
        Generate ports polygons that encapsulate the window and NCP geometries.

        Method: window and NCP geometries are rotated around +Z axis until
        the area projected onto XY plane is the smallest. A boundary box is
        then generated with a slight outward offset. This boundary box is
        then rotated back the same amount to encapsulate the original window
        and NCP geomteries.
        """
        xax = [1, 0, 0]
        _xax = [-1, 0, 0]
        yax = [0, 1, 0]
        _yax = [0, -1, 0]
        zaxis = radgeom.Vector(0, 0, 1)
        rm_pg = [xax, _yax, _xax, yax]
        area_list = []
        ncs_polygon = [p['polygon'] for p in ncs_prims]
        win_normals = []
        # Find axiel aligned rotation angle
        bboxes = []
        for deg in range(90):
            rad = math.radians(deg)
            win_polygon_r = win_polygon.rotate(zaxis, rad)
            win_normals.append(win_polygon_r.normal())
            ncs_polygon_r = [p.rotate(zaxis, rad) for p in ncs_polygon]
            ncs_polygon_r.append(win_polygon_r)
            _bbox = radutil.getbbox(ncs_polygon_r, offset=0.001)
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

    def merge_window(self, primitive_list):
        """Merge rectangles if coplanar."""
        polygons = [p['polygon'] for p in primitive_list]
        normals = [p.normal() for p in polygons]
        norm_set = set([n.to_list() for n in normals])
        if len(norm_set) > 1:
            warn_msg = "windows oriented differently"
        points = [i for p in polygon for i in p.vertices]
        chull = radgeom.Convexhull(points, normals[0])
        hull_polygon = chull.hull
        real_args = hull_polygon.toreal()
        modifier = primitive_list[0]['modifier']
        identifier = primitive_list[0]['identifier']
        new_prim = self.polygon_prim(hull_polygon, modifier, identifier)
        return new_prim

    def polygon_prim(self, polygon, mod, ident):
        """prepare a polygon primitive."""
        new_prim = {'str_args': '0', 'int_arg': '0', 'type': 'polygon'}
        new_prim['modifier'] = mod
        new_prim['identifier'] = ident
        new_prim['polygon'] = polygon
        new_prim['real_args'] = polygon.to_real()
        return new_prim


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-wf', help='window rad file path')
    parser.add_argument('-sf', help='shade rad file path')
    parser.add_argument('-refl', action='store_true', help='Do reflection?')
    parser.add_argument('-forw', help='Do forward calculation?')
    parser.add_argument('-merge', help='merge window polygons')
    parser.add_argument('-FN', action='store_true', help='FN matrix type')
    parser.add_argument('-depth', type=float, help='specify system depth')
    parser.add_argument('-scale',
                        type=float,
                        default=1.0,
                        help='fmtx port scale')
    parser.add_argument('-wrap',
                        action='store_true',
                        help='generate xml file instead')
    parser = genmtx.genmtx_args(parser)
    args = parser.parse_args()
    win_prim = radutil.parse_primitive(args.wf)
    ncs_prim = radutil.parse_primitive(args.sf)
    Genfmtx(win_prim=win_prim,
            ncs_prim=ncs_prim,
            out_path=args.o,
            **vars(args))
