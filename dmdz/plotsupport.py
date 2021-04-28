import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
from .tools import timestamp, timeprint

# plt.rcdefaults()  # Reset the rcParams defaults

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# SET FONT SIZES

rcparams_global = {
    "figure.dpi": 120,
    "figure.facecolor": "white",
    "figure.figsize": (7, 3),  # inches
    # "figure.autolayout": True,
    "savefig.dpi": 450,
    "savefig.facecolor": "white",
    "savefig.edgecolor": "white",
    "savefig.transparent": False,
    "savefig.format": "svg",
    # "savefig.pad_inches": 0.1,
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsfonts,amsmath,amssymb}"
}

SML_SIZE = 10
MED_SIZE = 12
BIG_SIZE = 14

rcparams_font ={
    'font.family':'sans-serif',
    'font.size': SML_SIZE,
    'axes.titlesize': SML_SIZE,
    'axes.labelsize': MED_SIZE,
    'xtick.labelsize': SML_SIZE,
    'ytick.labelsize': SML_SIZE,
    'legend.fontsize': SML_SIZE,
    'figure.titlesize': BIG_SIZE
}
rcparams_global.update(rcparams_font)
plt.rcParams.update(rcparams_global)


# TODO implement darkmode. find a way to quickly switch or print both plots
# rcparams_darkmode = {
#     "lines.color": "white",
#     "patch.edgecolor": "white",
#     "text.color": "white",
#     "axes.facecolor": "white",
#     "axes.edgecolor": "lightgray",
#     "axes.labelcolor": "white",
#     "xtick.color": "white",
#     "ytick.color": "white",
#     "grid.color": "lightgray",
#     "figure.facecolor": "black",
#     "figure.edgecolor": "black",
#     "savefig.facecolor": "black",
#     "savefig.edgecolor": "black"
# }



# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# SAVING FIGS

# SAVEFIG_KW = dict(facecolor='w', edgecolor='w', transparent=False, format="svg",
#                              bbox_inches="tight", pad_inches=0.0, metadata=None)

def set_figdir(dir=None, verbose=True):
    # sets a directory for figures.
    if dir is None:
        # if current directory is src, then figs directory is set one level above
        # but otherwise make the figs directory at the same level.
        if os.getcwd().split('/')[-1] == 'src':
            dir = "../figs"
        else:
            dir = "./figs"
    if not os.path.isdir(dir):
        os.makedirs(dir)
    if not os.path.isdir(dir+"/transparent"):
        os.makedirs(dir+"/transparent")
    if verbose:
        print("Fig directory set to %s" % dir)
    return dir


def savefigz(fname="Untitled", dir=None, time=True, verbose=True, transparent=False):
    if dir is None:
        dir = set_figdir(verbose=False)

    if time:
        figtime = timestamp()
        fname = figtime + '-' + fname

    fig_filename = "%s/%s"%(dir, fname)
    plt.tight_layout()

    if transparent:
        plt.savefig(fig_filename, bbox_inches="tight", transparent=transparent)
    else:
        plt.savefig(fig_filename, bbox_inches="tight")

    if verbose:
        timeprint("Saved fig %s"%fig_filename)

    return fig_filename




# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# PLOT ADJUSTMENTS

def xtick_locator(base, ax):
    """ Places x-ticks at regular intervals in the given base."""
    loc = plticker.MultipleLocator(base=base)
    ax.xaxis.set_major_locator(loc)


def ytick_locator(base, ax):
    """ Places y-ticks at regular intervals in the given base."""
    loc = plticker.MultipleLocator(base=base)
    ax.yaxis.set_major_locator(loc)


def set_aspect_equal_ratio(ax):
    """
    Sets equal paper-length axes to create a square shaped plot.
    Useful where ax.set_aspect('equal') doesn't create square plots.
    """
    ax.set_aspect(1./ax.get_data_ratio())


def spline(x, y, points=500, k=3):
    """Interpolates a k-spline for x,y data. Can then use plt.plot(*spline(x,y))."""
    spl = make_interp_spline(x, y, k=k)
    x_new = np.linspace(min(x), max(x), points)
    return x_new, spl(x_new)


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# COLORMAP STUFF

def colormap_create(r,g,b,end=1):
    """
    Creates a matplotlib linear colormap.
    Specify starting r,g,b in [0,1] range.
    Default is to end at 1 = white. (0 = black or 0.5 = gray)
    """
    cmap = np.ones((256,4))
    for j,x in enumerate([r,g,b]):
        cmap[:, j] = np.linspace(x, end, 256)
    return ListedColormap(cmap)

def colormap_join(cmap1, cmap2, d1=1, d2=1):
    """
    Joins two colormaps for, say, a diverging spectrum.
    d1 and d2 should be 1 or -1 depending on if you need to reverse the
    order of cmap1 or cmap2 respectively.
    """
    cmap = np.vstack((cmap1(np.linspace(0, 1, 128)[::d1]),
               cmap2(np.linspace(0, 1, 128)[::d2])))
    return ListedColormap(cmap)

# for example, we can create these colormaps
cm_yellows = colormap_create(250/256, 235/256, 15/256)
cm_purples = colormap_create(128/256, 0/256, 128/256)
cm_YPu = colormap_join(cm_purples, cm_yellows, d2=-1)
cm_diverge = colormap_join(cm.binary, cm.binary, d1=-1)
cm_diverge_r = colormap_join(cm.binary, cm.binary, d2=-1)

def get_line_color(label, ax):
    """Retrieves line color when given a line label and axis. If no label, use 'Line <number>'"""
    return [l._color for l in ax.lines if l._label == label][0]