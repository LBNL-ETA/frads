import shutil
import subprocess as sp
import sys

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
        print(f"{prog} not found; check Radiance installation")

# Check Radiance version, need to be at least 5.X
rad_version = sp.run(["rtrace", "-version"], check=True, stdout=sp.PIPE).stdout.decode().split()[1]
if int(rad_version[0]) < 5:
    print(f"Radiance version {rad_version} detected, please upgrade")

if not sys.platform.startswith('win')
    import resource
    slimit, hlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    if slimit < 10000 or hlimit < 10000:
        resource.setrlimit(resource.RLIMIT_NOFILE, (131072, 131072))

