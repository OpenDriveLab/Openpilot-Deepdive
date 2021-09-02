import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes


def draw_trajectory_on_ax(ax: Axes, trajectories, confs, line_type='o-', transparent=True, xlim=(-30, 30), ylim=(0, 100)):
    '''
    ax: matplotlib.axes.Axes, the axis to draw trajectories on
    trajectories: List of numpy arrays of shape (num_points, 2 or 3)
    confs: List of numbers, 1 means gt
    '''

    for idx, (trajectory, conf) in enumerate(zip(trajectories, confs)):
        label = 'gt' if conf == 1 else 'pred%d (%.3f)' % (idx, conf)
        alpha = np.clip(conf, 0.1, None) if transparent else 1.0
        ax.plot(-trajectory[:, 1], trajectory[:, 0], line_type, label=label, alpha=alpha)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend()

    return ax
