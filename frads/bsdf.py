import math
from dataclasses import dataclass
from typing import List


BASIS_DICT = {
    "145": "Klems Full",
    "73": "Klems Half",
    "41": "Klems Quarter",
}

ABASE_LIST = {
    "Klems Full": [
        (0.0, 1),
        (5.0, 8),
        (15.0, 16),
        (25.0, 20),
        (35.0, 24),
        (45.0, 24),
        (55.0, 24),
        (65.0, 16),
        (75.0, 12),
        (90.0, 0),
    ],
    "Klems Half": [
        (0.0, 1),
        (6.5, 8),
        (19.5, 12),
        (32.5, 16),
        (46.5, 20),
        (61.5, 12),
        (76.5, 4),
        (90.0, 0),
    ],
    "Klems Quarter": [(0.0, 1), (9.0, 8), (27.0, 12), (46.0, 12), (66.0, 8), (90.0, 0)],
}


@dataclass
class ScatteringData:
    """Scattering data object.

    Attributes:
        sdata: scattering data in nested lists.
        ncolumn: number of columns
        nrow: number of rows
    """

    sdata: List[float]
    ncolumn: int
    nrow: int

    def __str__(self) -> str:
        out = "#?RADIANCE\nNCOMP=3\n"
        out += f"NROWS={self.nrow}\nNCOLS={self.ncolumn}\n"
        out += "FORMAT=ascii\n\n"
        for col in range(self.ncolumn):
            for row in range(self.nrow):
                val = self.sdata[row + col * self.ncolumn]
                string = "\t".join([f"{val:7.5f}"] * 3)
                out += string + "\t"
            out += "\n"
        return out


@dataclass
class BSDFData:
    """BSDF data object.

    Attributes:
        bsdf: BSDF data.
        ncolumn: number of columns
        nrow: number of rows
    """

    bsdf: List[float]
    ncolumn: int
    nrow: int

    def to_sdata(self) -> ScatteringData:
        """Covert a bsdf object into a sdata object."""
        basis = BASIS_DICT[str(self.ncolumn)]
        lambdas = angle_basis_coeff(basis)
        sdata = []
        for irow in range(self.nrow):
            for icol, lam in zip(range(self.ncolumn), lambdas):
                sdata.append(self.bsdf[icol + irow * self.ncolumn] * lam)
        return ScatteringData(sdata, self.ncolumn, self.nrow)


@dataclass(frozen=True)
class RadMatrix:
    """Radiance matrix object.

    Attributes:
        tf: front-side transmission
        tb: back-side transmission
    """

    tf: BSDFData
    tb: BSDFData


def lambda_calc(theta_lr: float, theta_up: float, nphi: float) -> float:
    """."""
    return (
        (
            math.cos(math.pi / 180 * theta_lr) ** 2
            - math.cos(math.pi / 180 * theta_up) ** 2
        )
        * math.pi
        / nphi
    )


def angle_basis_coeff(basis: str) -> List[float]:
    """Calculate klems basis coefficient"""
    ablist = ABASE_LIST[basis]
    lambdas = []
    for i in range(len(ablist) - 1):
        tu = ablist[i + 1][0]
        tl = ablist[i][0]
        np = ablist[i][1]
        lambdas.extend([lambda_calc(tl, tu, np) for _ in range(np)])
    return lambdas


def sdata2bsdf(sdata: ScatteringData) -> BSDFData:
    """Convert sdata object to bsdf object."""
    basis = BASIS_DICT[str(sdata.ncolumn)]
    lambdas = angle_basis_coeff(basis)
    bsdf = []
    for irow in range(sdata.nrow):
        _row = []
        for icol, lam in zip(range(sdata.ncolumn), lambdas):
            _row.append(sdata.sdata[irow][icol] / lam)
        bsdf.append(_row)
    return BSDFData(bsdf, sdata.ncolumn, sdata.nrow)


# def bsdf2sdata(bsdf: BSDFData) -> ScatteringData:
#     """Covert a bsdf object into a sdata object."""
#     basis = BASIS_DICT[str(bsdf.ncolumn)]
#     lambdas = angle_basis_coeff(basis)
#     sdata = []
#     for irow in range(bsdf.nrow):
#         _row = []
#         for icol, lam in zip(range(bsdf.ncolumn), lambdas):
#             _row.append(bsdf.bsdf[irow][icol] * lam) 
#         sdata.append(_row)
#     return ScatteringData(sdata, bsdf.ncolumn, bsdf.nrow)
