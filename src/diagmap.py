import scipy.interpolate
import scipy.spatial
import multiprocessing.pool
import networkx as nx
import numpy as np

from tqdm.auto import tqdm, trange


def _calculate_distances(points, ax, single_section=False):
    """Actual function implementation"""
    n_surfs = points.shape[1]
    n_turns = points.shape[2]
    n_phi = points.shape[3]

    n_axis = ax.shape[1]

    # Build index tree for axis
    ax_tree = scipy.spatial.cKDTree(ax.transpose([1, 0]))

    surf_ds = np.full(dtype=np.float32, shape=[n_surfs, n_surfs], fill_value=np.inf)
    ax_ds = np.full(dtype=np.float32, shape=[n_surfs], fill_value=np.inf)

    x, y, z = points
    r = np.sqrt(x ** 2 + y ** 2)

    points_rz = np.stack([r, z], axis=0)

    with multiprocessing.pool.ThreadPool(6) as pool:
        for i_phi in trange(n_phi):
            # print('\tBuilding slice trees')

            surf_points = [
                points[:, i_surf, np.isfinite(points[0, i_surf, :, i_phi]), i_phi]
                for i_surf in range(n_surfs)
            ]

            surf_points_rz = [
                points_rz[:, i_surf, np.isfinite(points[0, i_surf, :, i_phi]), i_phi]
                for i_surf in range(n_surfs)
            ]

            surf_points = [np.transpose(s) for s in surf_points]
            surf_points_rz = [np.transpose(s) for s in surf_points_rz]

            # Build index tree for every surface in this phi plane
            surf_trees = [
                scipy.spatial.cKDTree(surf, leafsize=10) if len(surf) > 0 else None
                for surf in surf_points_rz
            ]

            # Cross-query all surfaces
            # print('\tComputing slice <-> slice distances')
            for i_surf1 in range(n_surfs):
                if surf_trees[i_surf1] is None:
                    continue

                # if(i_surf1 % 10 == 0):
                #    print("\t\t", i_surf1)

                def apply_s2(i_surf2):
                    if surf_trees[i_surf2] is None:
                        return

                    d_pre = surf_ds[i_surf1, i_surf2]

                    # d, _ = surf_trees[i_surf1].query(surf_points_[i_surf2][0], distance_upper_bound = d_pre)
                    # d = min(d_pre, d)

                    d_nearest, _ = surf_trees[i_surf1].query(
                        surf_points_rz[i_surf2], distance_upper_bound=d_pre
                    )
                    d = np.amin(d_nearest)

                    surf_ds[i_surf1, i_surf2] = min(d, d_pre)

                pool.map(apply_s2, range(n_surfs))

            # Query surface against axis
            # print('\tComputing slice <-> axis distances')
            for i_surf in range(n_surfs):
                if surf_trees[i_surf] is None:
                    continue

                d_pre = ax_ds[i_surf]
                d_nearest, _ = ax_tree.query(
                    surf_points[i_surf], distance_upper_bound=d_pre
                )
                d = np.amin(d_nearest)

                ax_ds[i_surf] = min(d, d_pre)

            if single_section:
                break

    # Build distance graph
    graph = nx.Graph()

    graph.add_node("ax")
    graph.add_nodes_from(range(n_surfs))

    for i_surf in range(n_surfs):
        if np.isfinite(ax_ds[i_surf]):
            graph.add_edge("ax", i_surf, weight=ax_ds[i_surf])

    for i_surf1 in range(n_surfs):
        for i_surf2 in range(n_surfs):
            if np.isfinite(surf_ds[i_surf1, i_surf2]):
                graph.add_edge(i_surf1, i_surf2, weight=surf_ds[i_surf1, i_surf2])

    distances = nx.single_source_dijkstra_path_length(graph, "ax")

    dist_tot = np.asarray(
        [distances.get(i, np.inf) for i in range(n_surfs)], dtype=np.float32
    )

    return surf_ds, ax_ds, dist_tot


class Mapping:
    """Class that builds and stores a toroidally interpolated mapping function"""

    def __init__(self, points, ax, period=1):
        """
        Creates mapping function

        Args:
                points: A [3, n_surfaces, n_turns, n_phi] shaped array, where the first dimension
                  is the coordinate dimension (cartesian x, y, z), n_surfaces is the number of
                  fieldlines / flux surfaces, n_turns is the number of points per surface &
                  cross-section and n_phi is the number of toroidal corss-sections.
                param ax: A [3, n_point] shaped array, holding points of the magnetic axis in
                  cartesian coordinates. The points do not have to be aligned with the toroidal
                  cross sections presented above.
        """
        self.points = points
        self.ax = ax

        surf_ds, ax_ds, dist_tot = _calculate_distances(points, ax)

        self.distances = dist_tot
        self.period = period

        self._calc_phi()
        self._sort()
        self._build_phi_interpolator()
        self._build_interpolators()

    def _calc_phi(self):
        x, y, z = self.points
        phis = np.arctan2(y, x)
        phis = np.mean(phis, axis=(0, 1))

        span = 2 * np.pi / self.period
        phis %= span

        self.phis = phis

    def _sort(self):
        idx = np.argsort(self.phis)

        self.phis = self.phis[idx]
        self.points = self.points[:, :, :, idx]

    def _build_phi_interpolator(self):
        """Builds an interpolator mapping the phi coordinate to interpolation weights for cross-sections"""

        p = self.phis
        span = 2 * np.pi / self.period

        x = np.concatenate([[p[-1] - span], p, [p[0] + span]])

        eye = np.eye(len(self.phis))

        y = np.concatenate([[eye[-1]], eye, [eye[0]]])

        interpolator = scipy.interpolate.interp1d(x, y, axis=0)
        self._phi_interpolator = interpolator

    def _build_interpolators(self):
        n_surfs = self.points.shape[1]

        assert self.distances.shape == (n_surfs,)

        # Build Delaunay interpolators
        def single_interpolator(points):
            # Convert points from [x, y, z] to [r, z]
            x, y, z = points
            r = np.sqrt(x ** 2 + y ** 2)
            points = np.asarray([r, z])

            # Transpose points shape from [rz, n_surf, n_turn] to [n_surf, n_turn, rz]
            points = np.transpose(points, [1, 2, 0])

            # Produce length array with matching shape
            distances = self.distances
            distances = np.broadcast_to(distances[:, None], points.shape[:-1])

            # Collapse surface and turn dimension together
            points = points.reshape([-1, 2])
            distances = distances.reshape([-1])

            # Filter nonsense values
            idx = np.isfinite(points[:, 0])
            points = points[idx]
            distances = distances[idx]

            # Build Delaunay interpolator
            interpolator = scipy.interpolate.LinearNDInterpolator(points, distances)

            return interpolator

        self.interpolators = [
            single_interpolator(self.points[:, :, :, i])
            for i in trange(self.points.shape[3])
        ]

    def __call__(self, x, y, z):
        """
        Evaluates the mapping coordinate at given points in 3D space. Parameters
          are three numpy-array likes for x, y, and z coordinates, which must be
          broadcastable to a common shape.

        Returns:
                An array of the common broadcast shape of x, y, and z, holding the
                scalar mapping coordinate values for all points.
        """

        def inner(x, y, z):
            phi = np.arctan2(y, x)
            r = np.sqrt(x ** 2 + y ** 2)

            span = 2 * np.pi / self.period
            ws = self._phi_interpolator(phi % span)

            result = 0

            for i, w in enumerate(ws):
                if w == 0:
                    continue

                val = self.interpolators[i]([r, z])
                result += w * val

            return val

        return np.vectorize(inner)(x, y, z)
