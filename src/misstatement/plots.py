"""Figure style shared by the notebooks: grey for context, one blue accent for the point being made."""
import matplotlib as mpl
import matplotlib.pyplot as plt
from .config import FIGURES

ACCENT = '#2f6fb3'
GREY = '#9a9a9a'
LIGHT = '#d0d0d0'
GOOD, BAD = '#3a8a5c', '#b8473d'
SERIES = ['#2f6fb3', '#e08a2e', '#3a8a5c', '#8a5cb8', '#b8473d', '#5a5a5a']


def style():
    mpl.rcParams.update({
        'figure.dpi': 110, 'savefig.dpi': 300, 'font.size': 10, 'axes.titlesize': 11, 'axes.titleweight': 'bold',
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': '#e6e6e6',
        'grid.linewidth': 0.6, 'axes.edgecolor': '#555555', 'legend.frameon': False, 'font.family': 'DejaVu Sans',
    })


def save(fig, name):
    fig.savefig(FIGURES / f'{name}.png', bbox_inches='tight')
    fig.savefig(FIGURES / f'{name}.pdf', bbox_inches='tight')
    return FIGURES / f'{name}.png'


def years(ax):
    """Integer year ticks on the x axis."""
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    return ax
