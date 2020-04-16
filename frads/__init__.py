import shutil
import subprocess as sp

# Check if Radiance is installed more or less
rad_progs = [
    'rfluxmtx',
    'total',
    'getinfo',
    'pcomb',
    'dctimestep',
    'rmtxop',
    'gendaymtx',
]
for prog in rad_progs:
    ppath = shutil.which(prog)
    if ppath is None:
        raise Exception(f"{prog} not found; check Radiance installation")

# Check Radiance version, need to be at least 5.X
rad_version = sp.run("rtrace -version", check=True,
                     shell=True, stdout=sp.PIPE).stdout.decode().split()[1]
if int(rad_version[0]) < 5:
    raise Exception("Old Radiance version detected, please upgrade to the latest")

