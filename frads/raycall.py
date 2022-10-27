"""Call Radiance trace related stuff."""
import logging
from pathlib import Path
import subprocess as sp
from typing import Tuple
from typing import Iterable
from typing import List
from typing import Optional
from typing import Sequence


from frads import sky
from frads import utils
from frads.types import Primitive
from frads.types import Options
from frads.types import View

logger = logging.getLogger("frads.raycall")


def get_oconv_command(
    *paths, has_stdin: bool = False, frozen: bool = False, octree=None
) -> List[str]:
    """Get oconv command.

    Args:
        paths: input file paths. (remember material definitions go first)
        has_stdin: whether we have stdin for this oconv call?
        frozen: Whether we want to have a frozen octree.
        octree: Existing octree to bind with.
    Returns:
        Oconv call command as a list of strings.
    """
    command = ["oconv"]
    if octree:
        command.append("-i")
        command.append("octree")
    if frozen:
        command.append("-f")
    if has_stdin:
        command.append("-")
    command.extend(paths)
    return command


def get_rtrace_command(
    option: List[str], octree: Path, header: bool = False
) -> List[str]:
    """Get rtrace command.

    Args:
        option: rtrace options.
        octree: octree path.
        header: Whether we want header info in the output.
    Returns:
        Rtrace call command as a list of strings.
    """
    command = ["rtrace"]
    if not header:
        command.append("-h")
    command += [*option, str(octree)]
    return command


def get_rfluxmtx_command(
    receiver: Path,
    option: Optional[List[str]] = None,
    sys_paths: Optional[Iterable[Path]] = None,
    octree: Optional[Path] = None,
    sender: Optional[Path] = None,
) -> List[str]:
    """
    Sender: stdin, polygon
    Receiver: surface with -o
    """
    command = ["rfluxmtx"]
    if option:
        command.extend(option)
    if isinstance(sender, Path):
        command.append(str(sender))
    else:
        command.append("-")
    command.append(str(receiver))
    if octree:
        command.extend(["-i", str(octree)])
    if sys_paths:
        for path in sys_paths:
            command.append(str(path))
    return command


def get_rcontrib_command(
    option: List[str], yres, outpath: Path, modifier: Path, octree: Path, xres=None
) -> List[str]:
    """
    Needs a wrapper for options.
    grouped by modifiers
    -n -V -c [ -fo | -r ]
    -e  -f
    -x -y -o -p -b -bn { -m | -M }
    [rtrace option]
    octree
    """
    command = ["rcontrib", *option]
    if xres:
        command += ["-ffc", "-x", xres]
    else:
        command.append("-faf")
    command += ["-y", yres]
    command += ["-o", str(outpath), "-M", str(modifier), str(octree)]
    command.append(str(octree))
    return command


def oconv(*paths, outpath, stdin=None, frozen: bool = False, octree=None) -> None:
    has_stdin = bool(stdin)
    oconv_command = get_oconv_command(
        *paths, has_stdin=has_stdin, octree=octree, frozen=frozen
    )
    logger.info(oconv_command)
    utils.run_write(oconv_command, outpath, stdin=stdin)


def rtrace(
    rays, option: List[str], octree: Path, out=None, header: bool = False
) -> Tuple[str, str]:
    rtrace_command = get_rtrace_command(option, octree, header=header)
    proc = sp.run(
        rtrace_command,
        check=True,
        encoding="ascii",
        input=rays,
        stderr=sp.PIPE,
        stdout=sp.PIPE,
    )
    return proc.stdout, proc.stderr


def pcond(
    hdr: Path,
    human: bool = False,
    acuity: bool = False,
    veiling: bool = False,
    sense: bool = False,
    closs: bool = False,
    linear: bool = False,
) -> str:
    """Condition a HDR image.

    This one has reduced functionality.

    Args:
        hdr: input hdr
        human: Mimic human visual response. This is the same as turning on all acuity,
            veiling, sensitivity, and color loss.
        acuity: Defocus darker region.
        veiling: Add veiling glare.
        sense: Use human contrast sensitivity, simulating eye internal scattering.
        linear: Use a linear reponse function instead of the standard dynamic range
            compression. This preseves the extremas.

    Returns:
        output hdr
    """
    command = ["pcond"]
    if human:
        command.append("-h")
    else:
        if acuity:
            command.append("-a")
        if veiling:
            command.append("-v")
        if sense:
            command.append("-s")
        if closs:
            command.append("-c")
    if linear:
        command.append("-l")
    command.append(str(hdr))
    proc = sp.run(command, check=True, stdout=sp.PIPE)
    return proc.stdout


def get_rpict_command(
    view: View,
    options: Optional[Options] = None,
    octree: Optional[Path] = None,
) -> List[str]:
    command = ["rpict"]
    command.extend(view.args())
    if options is not None:
        command.extend(options.args())
    if octree:
        command.append(str(octree))
    return command


def render(
    view: View,
    room: Sequence[Primitive],
    sky: Primitive,
    options: Optional[Options] = None,
):
    """Render a image."""
    oconv_command = get_oconv_command(has_stdin=True)
    input_prim = "\n".join(map(str, room))
    input_prim += "\n" + sky
    oconv_proc = sp.run(
        oconv_command,
        check=True,
        input=input_prim.encode(),
        stdout=sp.PIPE,
    )
    rpict_command = get_rpict_command(view, options)
    logger.info(rpict_command)
    rpict_proc = sp.run(
        rpict_command, check=True, input=oconv_proc.stdout, stdout=sp.PIPE
    )
    return rpict_proc


def renderf(*paths, view, wea, meta, options):
    """Render a image."""
    ssky = sky.gen_perez_sky(wea, meta)
    oconv_command = get_oconv_command(*paths, has_stdin=True)
    oconv_proc = sp.run(
        oconv_command, check=True, input=ssky.encode("ascii"), stdout=sp.PIPE
    )
    rpict_command = get_rpict_command(view, options)
    rpict_proc = sp.run(
        rpict_command, check=True, input=oconv_proc.stdout, stdout=sp.PIPE
    )
    return rpict_proc


def get_pixel_values(path=None, stdin=None, gamma: Optional[float] = None):
    """Get HDR pixel values.

    Use pvalue to get hdr pixel values as double precision floats.

    Args:
        path: hdr file path. Either path or stdin is used, path takes
            precedence.
        stdin: hdr data as bytes. Either path or stdin is used, path
            takes precedence.
    Returns:
        xres, yres, buf
    """
    cmd = ["pvalue", "-o", "-h", "-dd"]
    if gamma:
        cmd.append("-g")
        cmd.append(str(gamma))
    if path is not None:
        cmd.append(path)
        proc = sp.run(cmd, check=True, input=None, stdout=sp.PIPE)
    elif stdin:
        proc = sp.run(cmd, check=True, input=stdin, stdout=sp.PIPE)
    resstr, buf = proc.stdout.split(b"\n", 1)
    c1, yres, c2, xres = resstr.split()
    if c1 != b"-Y" and c2 != b"+X":
        raise ValueError("Non-standard Radiance image orientation.")
    return xres, yres, buf
