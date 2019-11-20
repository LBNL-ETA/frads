#!/usr/bin/env python3
import subprocess as sp
import os
import argparse

if __name__ == "__main__":
    aparser = argparse.ArgumentParser()
    aparser.add_argument('-x', required=True)
    aparser.add_argument('-y', required=True)
    aparser.add_argument('-c', default='1')
    aparser.add_argument('-vf', required=True)
    args = aparser.parse_args()
    cmd = "vwrays -ff -vf {} -x {} -y {} ".format(args.vf, args.x, args.y)
    cmd += '-c {} -pj 0.7 '.format(args.c)
    cmd += "| rcalc -if6 -of "
    cmd += '-e "DIM:{};CNT:{}" '.format(args.x, args.c)
    cmd += '-e "pn=(recno-1)/CNT+.5" '
    cmd += '-e "frac(x):x-floor(x)" -e "xpos=frac(pn/DIM);ypos=pn/(DIM*DIM)"'
    cmd += ' -e "incir=if(.25-(xpos-.5)*(xpos-.5)-(ypos-.5)*(ypos-.5),1,0)"'
    cmd += ' -e "$1=$1;$2=$2;$3=$3;$4=$4*incir;$5=$5*incir;$6=$6*incir"'
    if os.name == 'posix':
        cmd = cmd.replace('"', "'")
    sp.run(cmd, shell=True)
