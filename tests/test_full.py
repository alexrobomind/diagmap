import scipy.interpolate
import scipy.spatial
import multiprocessing.pool
import networkx as nx
import numpy as np

from tqdm.auto import tqdm, trange

import diagmap as dm

import matplotlib

matplotlib.use("tkAgg")
import matplotlib.pyplot as plt


def test_full():
    r0 = 5

    r = np.linspace(0, 1, 100)
    theta = np.linspace(0, 2 * np.pi, 300, endpoint=False)
    phi = np.linspace(0, 2 * np.pi, 3, endpoint=False)

    r = r[:, None, None]
    theta = theta[None, :, None]
    phi = phi[None, None, :]

    x = np.cos(phi) * (r0 + r * np.cos(theta))
    y = np.sin(phi) * (r0 + r * np.cos(theta))
    z = r * np.sin(theta) + 0 * phi
    points = np.stack([x, y, z], axis=0)

    ax_x = np.cos(phi) * r0
    ax_y = np.sin(phi) * r0
    ax_z = 0 * ax_x
    ax = np.stack([ax_x, ax_y, ax_z], axis=0)[:, 0, 0, :]

    mapping = dm.Mapping(points, ax)

    x = np.linspace(4, 6, 100)
    z = np.linspace(-1, 1, 100)

    gx, gz = np.meshgrid(x, z, indexing="xy")

    gy = 0 * gx

    gv = mapping(gx, gy, gz)

    rm = np.sqrt((gx - r0) ** 2 + gz ** 2)

    delta = rm - gv

    # Make sure we have some non-NaN values
    assert np.sum(np.isfinite(delta)) > 0

    # Set NaN deltas to 0
    delta[~np.isfinite(delta)] = 0

    assert np.max(np.abs(delta)) < 1e-3
