#!/usr/bin/env python3
"""Demonstration of parsing an anisotropic tensor tree BSDF for transmission
plotting to a square instead of a disk

.xml parsing not included, testing on a file containing a set of data only.
Parsing through a generation 6 tensor tree single-sided transmission takes ~5 sec.
A full data set, including front and back, trans and refl, would take ~20 sec to parse.
Plotting is done through nested grids in matplotlib, which can be very slow for large trees

Usage:
    After exiting the first default plot
    Generate a plot by:
        loaded.plot_grid(loaded.lookup(x, y))
        x, y: incident grid position

T.Wang"""

import re
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import pdb
import sys


class ParseTre(object):
    def __init__(self, data_string):
        tree = self.tokenize(data_string)
        self.parsed = self.parse_root(tree)

    def tokenize(self, string):
        """Tokenize the string."""
        toks = re.compile(
            ' +|[-+]?(\d+([.,]\d*)?|[.,]\d+)([eE][-+]?\d+)+|[\d*\.\d+]+|[{}]')
        for match in toks.finditer(string):
            s = match.group(0)
            if s[0] in ' ,':
                continue
            else:
                yield s

    def parse_inner(self, tokens):
        """Parse the branches until the end."""
        children = []
        while True:
            val = next(tokens)
            if val == '{':
                children.append(self.parse_inner(tokens))
            elif val == '}':
                return children
            else:
                children.append(float(val))

    def parse_root(self, tokens):
        """Parse a tokenized tree."""
        name = next(tokens)
        assert name == '{'
        return self.parse_inner(tokens)


class LoadTre(object):
    """lookup and plot the parsed tree (nested list)"""

    def __init__(self, parsed):
        self.parsed = parsed
        level = lambda L: isinstance(L, list) and max(map(level, L)) + 1
        self.depth = level(self.parsed)
        self.cmap_jet = plt.get_cmap('jet')

    def lookup(self, xs, ys):
        """Get the exiting distribution given an incident grid position."""
        br_ord = self.get_border(xs, ys)
        quads = [self.parsed[i] for i in br_ord]
        data = [self.traverse(quad, xs, ys) for quad in quads]
        return data

    def get_lorder(self, x, y):
        """Get the leaf order by x and y signage."""
        if x < 0:
            if y < 0:
                lorder = range(0, 4)
            else:
                lorder = range(4, 8)
        else:
            if y < 0:
                lorder = range(8, 12)
            else:
                lorder = range(12, 16)
        return lorder

    def get_border(self, x, y):
        """Get the branch order by x and y signage."""
        if x < 0:
            if y < 0:
                border = range(0, 16, 4)
            else:
                border = range(2, 16, 4)
        else:
            if y < 0:
                border = range(1, 16, 4)
            else:
                border = range(3, 16, 4)
        return border

    def traverse(self, quad, x, y, n=1):
        """Traverse a quadrant."""
        if len(quad) == 1:  #single leaf
            res = quad
        else:
            res = []
            # get x, y signage in relation to branches
            _x = x + 1 / (2**n) if x < 0 else x - 1 / (2**n)
            _y = y + 1 / (2**n) if y < 0 else y - 1 / (2**n)
            n += 1
            # which four branches to get? get index for them
            if n < self.depth:
                ochild = self.get_border(_x, _y)
            else:
                ochild = self.get_lorder(_x, _y)
            sub_quad = [quad[i] for i in ochild]
            if all(isinstance(i, float) for i in sub_quad):
                res = sub_quad  #single leaf for each branch
            else:  #branches within this quadrant
                for sq in sub_quad:
                    if len(sq) > 4:  # there is another branch
                        res.append(self.traverse(sq, _x, _y, n=n))
                    else:  # just a leaf
                        res.append(sq)
        return res

    def plot_square(self, grid, data):
        """Plot a single square."""
        ax = self.fig.add_subplot(grid)
        ax.add_patch(
            plt.Rectangle((0, 0),
                          1,
                          1,
                          facecolor=self.cmap_jet(self.norm(data))))
        ax.set_xticks([])
        ax.set_yticks([])
        self.fig.add_subplot(ax)

    def plot_quad(self, grid, data, depth):
        """Recursively plotting the squares in nested grids."""
        if isinstance(data, float):
            self.plot_square(grid, data)
        elif len(data) == 1:
            self.plot_square(grid, data[0])
        else:
            depth += 1
            inner_grid = gridspec.GridSpecFromSubplotSpec(2,
                                                          2,
                                                          subplot_spec=grid,
                                                          wspace=0.0,
                                                          hspace=0.0)
            if depth < self.depth:
                self.plot_quad(inner_grid[0, 1], data[0], depth)
                self.plot_quad(inner_grid[0, 0], data[1], depth)
                self.plot_quad(inner_grid[1, 1], data[2], depth)
                self.plot_quad(inner_grid[1, 0], data[3], depth)
            else:
                self.plot_quad(inner_grid[0, 1], data[0], depth)
                self.plot_quad(inner_grid[1, 1], data[1], depth)
                self.plot_quad(inner_grid[0, 0], data[2], depth)
                self.plot_quad(inner_grid[1, 0], data[3], depth)

    def plot_grid(self, data):
        """Plot all four quadrant."""
        plt.close()
        self.norm = mpl.colors.LogNorm(vmin=1e-2, vmax=100)
        self.fig = plt.figure(figsize=(4, 4))
        grid = gridspec.GridSpec(2, 2, wspace=0.0, hspace=0.0)
        self.plot_quad(grid[0, 1], data[0], 1)
        self.plot_quad(grid[0, 0], data[1], 1)
        self.plot_quad(grid[1, 1], data[2], 1)
        self.plot_quad(grid[1, 0], data[3], 1)
        plt.show()


if __name__ == "__main__":
    fname = sys.argv[1]
    with open(fname) as rdr:
        example = rdr.read().strip()
    tree = ParseTre(example)
    loaded = LoadTre(tree.parsed)
    test = loaded.lookup(0, 0)
    loaded.plot_grid(test)
    pdb.set_trace()
