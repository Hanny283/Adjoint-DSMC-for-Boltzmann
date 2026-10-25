"""
Free-form deformation (FFD) body and polyline wall queries.

The design variables are the Cartesian coordinates Q of a tensor-product
B-spline control lattice that encloses the body (Li & Wu, adjoint shape
optimization of oscillatory rarefied flows, eq. 53):

    r_s(Q) = sum_{i,j} B_{i,p}(u_s) B_{j,p}(v_s) Q_{ij}

(u_s, v_s) are fixed by inverting the initial lattice. With control points on
the Greville grid the basis reproduces linears, so Q = Q0 is the identity map
and every embedded sample stays where it was.

The wall used by the DSMC forward pass is the closed polyline through those
samples. Particles hit it by an exact segment-segment test (2x2), not by the
polar Fourier solver in ``adjoint.boundary_geometry``.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Clamped uniform B-splines (Piegl & Tiller)
# ---------------------------------------------------------------------------

def open_uniform_knot_vector(n_ctrl: int, degree: int) -> np.ndarray:
    """Clamped uniform knots on [0, 1]. Length ``n_ctrl + degree + 1``."""
    if degree < 1:
        raise ValueError("degree must be >= 1")
    if n_ctrl < degree + 1:
        raise ValueError(f"need at least {degree + 1} control points, got {n_ctrl}")
    n_interior = n_ctrl - degree - 1
    if n_interior > 0:
        interior = np.linspace(0.0, 1.0, n_interior + 2)[1:-1]
    else:
        interior = np.zeros(0)
    return np.concatenate([
        np.zeros(degree + 1),
        np.asarray(interior, float),
        np.ones(degree + 1),
    ])


def greville_abscissae(knots: np.ndarray, degree: int) -> np.ndarray:
    """Greville sites ξ_i = (u_{i+1} + ... + u_{i+degree}) / degree.

    These are the control-point parameters for which the spline reproduces
    the identity: sum_i B_i(u) ξ_i = u.
    """
    n = len(knots) - degree - 1
    g = np.empty(n)
    for i in range(n):
        g[i] = np.mean(knots[i + 1:i + 1 + degree])
    return g


def basis_funs(span: int, u: float, knots: np.ndarray, degree: int) -> np.ndarray:
    """Nonzero basis values at ``u``, for indices ``span-degree .. span``."""
    left = np.zeros(degree + 1)
    right = np.zeros(degree + 1)
    N = np.zeros(degree + 1)
    N[0] = 1.0
    for j in range(1, degree + 1):
        left[j] = u - knots[span + 1 - j]
        right[j] = knots[span + j] - u
        saved = 0.0
        for r in range(j):
            denom = right[r + 1] + left[j - r]
            temp = 0.0 if denom == 0.0 else N[r] / denom
            N[r] = saved + right[r + 1] * temp
            saved = left[j - r] * temp
        N[j] = saved
    return N


def basis_matrix(u, knots: np.ndarray, degree: int) -> np.ndarray:
    """Every basis function at each parameter. Shape ``(len(u), n_ctrl)``."""
    u = np.clip(np.asarray(u, dtype=float).ravel(), 0.0, 1.0)
    n = len(knots) - degree - 1
    # searchsorted(side="right")-1 is the knot span; clamp into the open range.
    spans = np.searchsorted(knots, u, side="right") - 1
    spans = np.clip(spans, degree, n - 1)
    B = np.zeros((len(u), n))
    for s, uu in enumerate(u):
        span = int(spans[s])
        B[s, span - degree:span + 1] = basis_funs(span, float(uu), knots, degree)
    return B


# ---------------------------------------------------------------------------
# Lattice and embedded wall
# ---------------------------------------------------------------------------

class FFDLattice:
    """Tensor-product FFD on an axis-aligned rectangle.

    Parameters live in ``[0, 1]^2``. Physical coordinates of the initial
    (identity) control net are the Greville sites scaled onto
    ``[origin, origin + extent]``.
    """

    def __init__(self, origin, extent, nu: int, nv: int, degree: int = 3):
        self.origin = np.asarray(origin, dtype=float).reshape(2)
        self.extent = np.asarray(extent, dtype=float).reshape(2)
        if np.any(self.extent <= 0):
            raise ValueError("lattice extent must be positive")
        self.nu = int(nu)
        self.nv = int(nv)
        self.degree = int(degree)
        self.knots_u = open_uniform_knot_vector(self.nu, self.degree)
        self.knots_v = open_uniform_knot_vector(self.nv, self.degree)
        self.greville_u = greville_abscissae(self.knots_u, self.degree)
        self.greville_v = greville_abscissae(self.knots_v, self.degree)
        gx = self.origin[0] + self.greville_u * self.extent[0]
        gy = self.origin[1] + self.greville_v * self.extent[1]
        # indexing='ij' so Q[i, j] corresponds to (greville_u[i], greville_v[j])
        self.Q0 = np.stack(np.meshgrid(gx, gy, indexing="ij"), axis=-1)

    def parameters_of(self, points):
        """Invert the identity map. ``points`` is ``(S, 2)`` in physical space."""
        points = np.asarray(points, dtype=float)
        u = (points[:, 0] - self.origin[0]) / self.extent[0]
        v = (points[:, 1] - self.origin[1]) / self.extent[1]
        if np.any(u < -1e-8) or np.any(u > 1.0 + 1e-8) or np.any(v < -1e-8) or np.any(v > 1.0 + 1e-8):
            raise ValueError(
                "a wall sample lies outside the FFD rectangle; enlarge the lattice "
                "(it has to enclose the initial body)"
            )
        return np.clip(u, 0.0, 1.0), np.clip(v, 0.0, 1.0)

    def weights(self, u, v) -> np.ndarray:
        """Tensor-product basis, shape ``(S, nu, nv)``. Constant in Q (eq. 54)."""
        Bu = basis_matrix(u, self.knots_u, self.degree)
        Bv = basis_matrix(v, self.knots_v, self.degree)
        return Bu[:, :, None] * Bv[:, None, :]

    def deform(self, weights: np.ndarray, Q) -> np.ndarray:
        """Evaluate ``r_s(Q)``. ``weights`` is ``(S, nu, nv)``, ``Q`` is ``(nu, nv, 2)``."""
        Q = np.asarray(Q, dtype=float)
        if Q.shape != (self.nu, self.nv, 2):
            raise ValueError(f"Q must have shape {(self.nu, self.nv, 2)}, got {Q.shape}")
        return np.einsum("sij,ijc->sc", weights, Q, optimize=True)


class EmbeddedBody:
    """Initial wall samples with frozen ``(u_s, v_s)`` inside a lattice."""

    def __init__(self, lattice: FFDLattice, wall_points):
        self.lattice = lattice
        self.wall0 = np.asarray(wall_points, dtype=float)
        if self.wall0.ndim != 2 or self.wall0.shape[1] != 2 or len(self.wall0) < 3:
            raise ValueError("wall_points must be (M, 2) with M >= 3")
        self.u, self.v = lattice.parameters_of(self.wall0)
        self.weights = lattice.weights(self.u, self.v)

    def boundary(self, Q) -> np.ndarray:
        """Deformed sample nodes, shape ``(M, 2)``, not closed by repetition."""
        return self.lattice.deform(self.weights, Q)


# 6x6 cubic Greville lattice on [-1.6, 1.6]^2, unit circle embedded. These four
# moves fold the right-hand side so some rays from the origin cross the wall
# three times. The polygon stays simple (no self-intersection) and the origin
# stays inside. Indices are (i along x, j along y).
DENT_PERTURBATION = np.array([
    [4, 2, -1.5, -1.2],
    [4, 3, -1.5, 1.2],
    [3, 1, 0.8, -0.8],
    [3, 4, 0.8, 0.8],
], dtype=float)


def apply_perturbation(Q, perturb) -> np.ndarray:
    """Return a copy of ``Q`` shifted by ``perturb``.

    ``perturb`` is either an array shaped like ``Q`` (a full delta lattice) or
    rows ``(i, j, dx, dy)`` added onto those control points.
    """
    Q = np.array(Q, dtype=float, copy=True)
    if perturb is None:
        return Q
    perturb = np.asarray(perturb, dtype=float)
    if perturb.size == 0:
        return Q
    if perturb.shape == Q.shape:
        return Q + perturb
    rows = np.atleast_2d(perturb)
    if rows.shape[1] != 4:
        raise ValueError("perturbation must be (i, j, dx, dy) rows or an array shaped like Q")
    for i, j, dx, dy in rows:
        Q[int(i), int(j), 0] += dx
        Q[int(i), int(j), 1] += dy
    return Q


def sample_circle(radius: float, n_samples: int, center=(0.0, 0.0)) -> np.ndarray:
    """CCW samples of a circle. Vertices lie on the circle (the chords do not)."""
    if n_samples < 8:
        raise ValueError("n_samples must be >= 8")
    theta = np.linspace(0.0, 2.0 * np.pi, int(n_samples), endpoint=False)
    c = np.asarray(center, dtype=float)
    return np.column_stack([c[0] + radius * np.cos(theta), c[1] + radius * np.sin(theta)])


def sample_polar_wall(C, n_samples: int, center=(0.0, 0.0)) -> np.ndarray:
    """CCW samples of the star-shaped Fourier wall used by ``runInflow``.

    ``C = [c0, a1, ..., aN, b1, ..., bN]`` and ``r(θ) = c0 + Σ a_k cos kθ + b_k sin kθ``
    with ``θ`` running over a full turn. Sample nodes lie on that curve. The
    polyline between them is the body the FFD forward pass actually reflects on.
    """
    from adjoint.boundary_geometry import _radius_r_vec

    if n_samples < 8:
        raise ValueError("n_samples must be >= 8")
    C = np.asarray(C, dtype=float).ravel()
    theta = np.linspace(0.0, 1.0, int(n_samples), endpoint=False)
    radius = _radius_r_vec(theta, C)
    ang = 2.0 * np.pi * theta
    c = np.asarray(center, dtype=float)
    return np.column_stack([c[0] + radius * np.cos(ang), c[1] + radius * np.sin(ang)])


# ---------------------------------------------------------------------------
# Polygon measures
# ---------------------------------------------------------------------------

def signed_area(points) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def polygon_perimeter(points) -> float:
    edge = np.roll(points, -1, axis=0) - points
    return float(np.linalg.norm(edge, axis=1).sum())


def ensure_ccw(points) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if signed_area(points) < 0.0:
        return points[::-1].copy()
    return np.array(points, dtype=float, copy=True)


def points_in_polygon(points, poly, chunk: int = 20000) -> np.ndarray:
    """Winding-number point-in-polygon. True where the winding number is nonzero.

    ``poly`` is ``(M, 2)`` without a repeated closing vertex. Boundary points
    follow the usual half-open edge rule and are not a reliable "inside".
    """
    points = np.asarray(points, dtype=float)
    single = points.ndim == 1
    if single:
        points = points.reshape(1, 2)
    poly = np.asarray(poly, dtype=float)
    out = np.empty(len(points), dtype=bool)
    x1 = poly[:, 0]
    y1 = poly[:, 1]
    x2 = np.roll(poly[:, 0], -1)
    y2 = np.roll(poly[:, 1], -1)
    for i0 in range(0, len(points), chunk):
        sl = points[i0:i0 + chunk]
        out[i0:i0 + chunk] = _winding_chunk(sl[:, 0], sl[:, 1], x1, y1, x2, y2)
    return bool(out[0]) if single else out


def _winding_chunk(px, py, x1, y1, x2, y2) -> np.ndarray:
    # Hormann-Agathos / Sunday winding number. Edges are (M,), points (N,).
    y1b = y1[None, :]
    y2b = y2[None, :]
    x1b = x1[None, :]
    x2b = x2[None, :]
    y = py[:, None]
    x = px[:, None]
    upward = (y1b <= y) & (y2b > y)
    downward = (y1b > y) & (y2b <= y)
    cross = (x2b - x1b) * (y - y1b) - (x - x1b) * (y2b - y1b)
    wn = np.sum(upward & (cross > 0.0), axis=1) - np.sum(downward & (cross < 0.0), axis=1)
    return wn != 0


def contains_points(points, poly, bbox_min=None, bbox_max=None) -> np.ndarray:
    """``points_in_polygon`` with a bounding-box reject for the channel-scale sets."""
    points = np.asarray(points, dtype=float)
    single = points.ndim == 1
    if single:
        points = points.reshape(1, 2)
    if bbox_min is None:
        bbox_min = poly.min(axis=0)
        bbox_max = poly.max(axis=0)
    inside_bb = (
        (points[:, 0] >= bbox_min[0]) & (points[:, 0] <= bbox_max[0])
        & (points[:, 1] >= bbox_min[1]) & (points[:, 1] <= bbox_max[1])
    )
    out = np.zeros(len(points), dtype=bool)
    if np.any(inside_bb):
        out[inside_bb] = points_in_polygon(points[inside_bb], poly)
    return bool(out[0]) if single else out


def self_intersection_count(points, tol: float = 1e-12) -> int:
    """Proper intersections of non-adjacent edges. Neighbours that share a vertex do not count."""
    pts = np.asarray(points, dtype=float)
    m = len(pts)
    a = pts
    b = np.roll(pts, -1, axis=0)
    count = 0
    for i in range(m):
        for j in range(i + 1, m):
            if j == i + 1 or (i == 0 and j == m - 1):
                continue
            t, u, denom = _intersect_paired(a[i:i + 1], b[i:i + 1], a[j:j + 1], b[j:j + 1])
            if abs(float(denom[0])) <= tol:
                continue
            if tol < float(t[0]) < 1.0 - tol and tol < float(u[0]) < 1.0 - tol:
                count += 1
    return count


def radial_crossing_counts(points, origin=(0.0, 0.0), n_angles: int = 720) -> np.ndarray:
    """How many times each ray from ``origin`` meets the polyline.

    A vertex is counted once (``u`` in ``[0, 1)``). A radial graph about
    ``origin`` — the star-shaped case the Fourier solver assumes — has count 1
    on every ray.
    """
    origin = np.asarray(origin, dtype=float)
    ang = np.linspace(0.0, 2.0 * np.pi, int(n_angles), endpoint=False)
    p0 = np.repeat(origin.reshape(1, 2), len(ang), axis=0)
    p1 = origin + 1.0e6 * np.column_stack([np.cos(ang), np.sin(ang)])
    a = np.asarray(points, dtype=float)
    b = np.roll(a, -1, axis=0)
    r = p1 - p0
    e = b - a
    denom = r[:, 0, None] * e[None, :, 1] - r[:, 1, None] * e[None, :, 0]
    qp = a[None, :, :] - p0[:, None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (qp[:, :, 0] * e[None, :, 1] - qp[:, :, 1] * e[None, :, 0]) / denom
        u = (qp[:, :, 0] * r[:, None, 1] - qp[:, :, 1] * r[:, None, 0]) / denom
    valid = (np.abs(denom) > 1e-14) & (t > 1e-12) & (t <= 1.0) & (u >= -1e-9) & (u <= 1.0 + 1e-9)
    # A ray through a vertex meets two edges at the same point. Count that once,
    # otherwise a circle's sample nodes look like a double crossing.
    raw = valid.sum(axis=1)
    counts = raw.copy()
    for i in np.nonzero(raw > 1)[0]:
        js = np.nonzero(valid[i])[0]
        pts_hit = p0[i] + t[i, js, None] * r[i]
        used = np.zeros(len(js), dtype=bool)
        clusters = 0
        for a in range(len(js)):
            if used[a]:
                continue
            clusters += 1
            used |= np.linalg.norm(pts_hit - pts_hit[a], axis=1) < 1e-6
        counts[i] = clusters
    return counts.astype(int)


def is_radial_graph(points, origin=(0.0, 0.0), n_angles: int = 720) -> bool:
    return bool(np.all(radial_crossing_counts(points, origin, n_angles) == 1))


def shape_report(points, origin=(0.0, 0.0)) -> dict:
    pts = np.asarray(points, dtype=float)
    crossings = radial_crossing_counts(pts, origin)
    return {
        "n": int(len(pts)),
        "area": abs(signed_area(pts)),
        "signed_area": signed_area(pts),
        "perimeter": polygon_perimeter(pts),
        "self_intersections": self_intersection_count(pts),
        "radial_graph": bool(np.all(crossings == 1)),
        "radial_crossings_min": int(crossings.min()),
        "radial_crossings_max": int(crossings.max()),
        "bbox_min": pts.min(axis=0),
        "bbox_max": pts.max(axis=0),
    }


# ---------------------------------------------------------------------------
# Segment-segment hits and specular reflection
# ---------------------------------------------------------------------------

_VERTEX_U = 1e-6
_VERTEX_DT = 1e-6
_DENOM_EPS = 1e-14
_ENTER_EPS = 1e-14


def _intersect_paired(p0, p1, a, b):
    """One particle segment per row against one wall segment per row.

    Returns ``t, u, denom`` with ``p0 + t (p1-p0) = a + u (b-a)``.
    """
    r = p1 - p0
    e = b - a
    denom = r[:, 0] * e[:, 1] - r[:, 1] * e[:, 0]
    qp = a - p0
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (qp[:, 0] * e[:, 1] - qp[:, 1] * e[:, 0]) / denom
        u = (qp[:, 0] * r[:, 1] - qp[:, 1] * r[:, 0]) / denom
    return t, u, denom


class BoundaryQuery:
    """Closed polyline plus a uniform grid of its edges.

    The grid is a broad-phase: a particle segment is tested only against edges
    whose expanded bounding boxes share a bin with it. The narrow phase is the
    exact 2x2 segment intersection. Outward normals assume a CCW boundary
    (interior on the left); ``ensure_ccw`` enforces that when the polygon is
    not flipped inside-out.
    """

    def __init__(self, points, n_bins: int = 16):
        raw = np.asarray(points, dtype=float)
        if signed_area(raw) < 0.0 and self_intersection_count(raw) == 0:
            raw = raw[::-1].copy()
        else:
            raw = np.array(raw, dtype=float, copy=True)
        self.pts = raw
        self.m = len(raw)
        self.a = raw
        self.b = np.roll(raw, -1, axis=0)
        edge = self.b - self.a
        length = np.linalg.norm(edge, axis=1)
        self.edge = edge
        self.length = length
        # Right-hand normal of a CCW traversal points out of the interior.
        outward = np.stack([edge[:, 1], -edge[:, 0]], axis=1)
        self.normal = np.zeros_like(outward)
        ok = length > 1e-14
        self.normal[ok] = outward[ok] / length[ok, None]
        pad = 1e-8
        self.bbox_min = self.pts.min(axis=0) - pad
        self.bbox_max = self.pts.max(axis=0) + pad
        self.n_bins = max(1, int(n_bins))
        extent = np.maximum(self.bbox_max - self.bbox_min, 1e-6)
        self.cell = extent / self.n_bins
        self._build_bins()

    def _build_bins(self):
        n = self.n_bins
        buckets = [[] for _ in range(n * n)]
        pad = 1e-8
        for i in range(self.m):
            if self.length[i] <= 1e-14:
                continue
            lo = np.minimum(self.a[i], self.b[i]) - pad
            hi = np.maximum(self.a[i], self.b[i]) + pad
            ix0, iy0 = self._clamp_cell(lo)
            ix1, iy1 = self._clamp_cell(hi)
            for ix in range(ix0, ix1 + 1):
                for iy in range(iy0, iy1 + 1):
                    buckets[ix * n + iy].append(i)
        self.buckets = [np.asarray(b, dtype=np.int32) for b in buckets]

    def _clamp_cell(self, xy):
        ij = np.floor((xy - self.bbox_min) / self.cell).astype(int)
        return int(np.clip(ij[0], 0, self.n_bins - 1)), int(np.clip(ij[1], 0, self.n_bins - 1))

    def contains(self, points) -> np.ndarray:
        return contains_points(points, self.pts, self.bbox_min, self.bbox_max)

    def earliest(self, p0, p1, mask=None, skip_seg=None, t_floor: float = 1e-9):
        """Earliest entering crossing of each particle segment with the wall.

        Entering means the particle moves against the outward normal, which is
        the gas-to-solid crossing. Returns ``hit, t, normal, point, seg``.
        ``t`` is the fraction along ``p0 -> p1``. A hit on a vertex averages
        the outward normals of the two edges that meet there.
        """
        p0 = np.asarray(p0, dtype=float)
        p1 = np.asarray(p1, dtype=float)
        n = len(p0)
        if mask is None:
            mask = np.ones(n, dtype=bool)
        else:
            mask = np.asarray(mask, dtype=bool)
        best_t = np.full(n, np.inf)
        best_seg = np.full(n, -1, dtype=np.int32)
        smin = np.minimum(p0, p1)
        smax = np.maximum(p0, p1)
        overlap = (
            mask
            & np.all(smax >= self.bbox_min, axis=1)
            & np.all(smin <= self.bbox_max, axis=1)
        )
        cand = np.nonzero(overlap)[0]
        if len(cand) == 0:
            return self._pack(best_t, best_seg, p0, p1)

        pad = 1e-8
        c0 = np.floor((smin[cand] - pad - self.bbox_min) / self.cell).astype(int)
        c1 = np.floor((smax[cand] + pad - self.bbox_min) / self.cell).astype(int)
        np.clip(c0, 0, self.n_bins - 1, out=c0)
        np.clip(c1, 0, self.n_bins - 1, out=c1)
        if skip_seg is None:
            skip_seg = np.full(n, -1, dtype=np.int32)
        else:
            skip_seg = np.asarray(skip_seg, dtype=np.int32)

        nb = self.n_bins
        for ix in range(nb):
            col = (c0[:, 0] <= ix) & (c1[:, 0] >= ix)
            if not np.any(col):
                continue
            for iy in range(nb):
                segs = self.buckets[ix * nb + iy]
                if len(segs) == 0:
                    continue
                local = col & (c0[:, 1] <= iy) & (c1[:, 1] >= iy)
                if not np.any(local):
                    continue
                ids = cand[local]
                self._accumulate(ids, p0, p1, segs, best_t, best_seg, skip_seg, t_floor)
        return self._pack(best_t, best_seg, p0, p1, skip_seg, t_floor)

    def _accumulate(self, ids, p0, p1, segs, best_t, best_seg, skip_seg, t_floor):
        # Chunk so a hot bin cannot allocate an (N_particles, M) pair matrix.
        step = max(1, 8192 // max(1, len(segs)))
        for i0 in range(0, len(ids), step):
            sl = ids[i0:i0 + step]
            t, u, denom, entering = self._pair_matrix(p0[sl], p1[sl], segs)
            valid = (
                (np.abs(denom) > _DENOM_EPS)
                & entering
                & (t > t_floor)
                & (t <= 1.0)
                & (u >= -1e-9)
                & (u <= 1.0 + 1e-9)
            )
            # Drop an immediate re-hit of the facet we just reflected off.
            # A later return to that same facet (real second bounce) has larger t.
            own = segs[None, :] == skip_seg[sl, None]
            valid &= ~(own & (t < 1e-5))
            t_use = np.where(valid, t, np.inf)
            local_min = t_use.min(axis=1)
            better = local_min < best_t[sl]
            if not np.any(better):
                continue
            local_j = np.argmin(t_use, axis=1)
            take = np.nonzero(better)[0]
            best_t[sl[take]] = local_min[take]
            best_seg[sl[take]] = segs[local_j[take]]

    def _pair_matrix(self, p0, p1, segs):
        a = self.a[segs]
        b = self.b[segs]
        nrm = self.normal[segs]
        r = p1 - p0
        e = b - a
        denom = r[:, 0, None] * e[None, :, 1] - r[:, 1, None] * e[None, :, 0]
        qp0 = a[None, :, 0] - p0[:, None, 0]
        qp1 = a[None, :, 1] - p0[:, None, 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (qp0 * e[None, :, 1] - qp1 * e[None, :, 0]) / denom
            u = (qp0 * r[:, None, 1] - qp1 * r[:, None, 0]) / denom
        ndr = nrm[None, :, 0] * r[:, None, 0] + nrm[None, :, 1] * r[:, None, 1]
        entering = ndr < -_ENTER_EPS
        return t, u, denom, entering

    def _pack(self, best_t, best_seg, p0, p1, skip_seg=None, t_floor=1e-9):
        n = len(p0)
        hit = np.isfinite(best_t) & (best_seg >= 0)
        t_out = np.where(hit, best_t, np.nan)
        normal = np.zeros((n, 2))
        point = np.zeros((n, 2))
        seg = best_seg.copy()
        if not np.any(hit):
            return hit, t_out, normal, point, seg
        ids = np.nonzero(hit)[0]
        r = p1[ids] - p0[ids]
        tt = best_t[ids]
        point[ids] = p0[ids] + tt[:, None] * r
        seg_i = best_seg[ids]
        a = self.a[seg_i]
        e = self.edge[seg_i]
        u = np.sum((point[ids] - a) * e, axis=1) / np.maximum(self.length[seg_i] ** 2, 1e-30)
        normal[ids] = self.normal[seg_i]
        # Vertex: the winning edge ends (or starts) on a shared node. Fold in
        # the neighbour when the same particle segment enters that edge too.
        near_end = u >= 1.0 - _VERTEX_U
        near_start = u <= _VERTEX_U
        self._merge_vertex(ids[near_end], (seg_i[near_end] + 1) % self.m, p0, p1, t_out, normal, seg, skip_seg, t_floor)
        self._merge_vertex(ids[near_start], (seg_i[near_start] - 1) % self.m, p0, p1, t_out, normal, seg, skip_seg, t_floor)
        return hit, t_out, normal, point, seg

    def _merge_vertex(self, ids, neigh, p0, p1, t_out, normal, seg, skip_seg, t_floor):
        if len(ids) == 0:
            return
        a = self.a[neigh]
        b = self.b[neigh]
        t, u, denom = _intersect_paired(p0[ids], p1[ids], a, b)
        r = p1[ids] - p0[ids]
        nrm = self.normal[neigh]
        ndr = nrm[:, 0] * r[:, 0] + nrm[:, 1] * r[:, 1]
        ok = (
            (np.abs(denom) > _DENOM_EPS)
            & (ndr < -_ENTER_EPS)
            & (t > t_floor)
            & (t <= 1.0)
            & (u >= -1e-8)
            & (u <= 1.0 + 1e-8)
            & (np.abs(t - t_out[ids]) <= _VERTEX_DT)
        )
        if skip_seg is not None:
            ok &= ~((neigh == skip_seg[ids]) & (t < 1e-5))
        if not np.any(ok):
            return
        take = ids[ok]
        summed = normal[take] + nrm[ok]
        norm = np.linalg.norm(summed, axis=1, keepdims=True)
        good = norm.ravel() > 1e-8
        if not np.any(good):
            return
        normal[take[good]] = summed[good] / norm[good]


def fly_and_reflect(p0, velocity, dt, query: BoundaryQuery, max_bounces: int = 12, nudge: float = 1e-8):
    """Free-flight plus specular bounces against ``query``.

    Each particle flies ``pos + t * velocity`` for a duration ``dt``. At the
    earliest entering crossing the velocity is reflected about that edge's
    outward unit normal,

        v <- v - 2 (n · v) n,

    and the remaining flight time is flown in the new direction. The streamwise
    momentum given to the wall, in the same units as ``runInflow.flight_step``,
    is ``2 (n · v_in) n_x`` summed over bounces (no particle mass).

    Returns ``pos, vel, impulse_x, bounces, resample, frozen, stats``.
    ``resample`` marks particles that are still inside after the bounce budget
    (the caller redraws them, as ``flight_step`` does). ``frozen`` marks a
    missed crossing that the bisection safety net also could not repair: the
    particle is put back at ``p0`` and contributes no impulse, matching the
    invalid-theta freeze in the polar solver.
    """
    pos = np.array(p0, dtype=float, copy=True)
    vel = np.array(velocity, dtype=float, copy=True)
    origin = pos.copy()
    vel0 = vel.copy()
    n = len(pos)
    time_left = np.full(n, float(dt))
    impulse = np.zeros(n)
    bounces = np.zeros(n, dtype=np.int32)
    active = np.ones(n, dtype=bool)
    skip_seg = np.full(n, -1, dtype=np.int32)
    resample = np.zeros(n, dtype=bool)
    frozen = np.zeros(n, dtype=bool)

    for _ in range(max(1, int(max_bounces))):
        flying = active & (time_left > 1e-14)
        if not np.any(flying):
            break
        p1 = pos + time_left[:, None] * vel
        hit, t, nrm, point, seg = query.earliest(pos, p1, mask=flying, skip_seg=skip_seg)
        miss = flying & ~hit
        if np.any(miss):
            pos[miss] = p1[miss]
            time_left[miss] = 0.0
            active[miss] = False
        if not np.any(hit):
            break
        # A vertex hit stored the bisector; count it when the normal is not
        # exactly one of the two edge normals. Diagnostic only.
        ndv = np.einsum("ij,ij->i", nrm[hit], vel[hit])
        impulse[hit] += 2.0 * ndv * nrm[hit, 0]
        vel[hit] = vel[hit] - 2.0 * ndv[:, None] * nrm[hit]
        landed = _nudge_outside(point[hit], nrm[hit], query, nudge)
        pos[hit] = landed
        time_left[hit] *= (1.0 - t[hit])
        skip_seg[hit] = seg[hit]
        bounces[hit] += 1
        done = hit & (time_left <= 1e-14)
        active[done] = False

    # Bounce budget exhausted while a crossing is still ahead: finish the step
    # only if the straight continuation stays in the gas, otherwise redraw.
    still = active & (time_left > 1e-14)
    if np.any(still):
        end = pos[still] + time_left[still, None] * vel[still]
        sunk = query.contains(end)
        end_ids = np.nonzero(still)[0]
        free = end_ids[~sunk]
        trapped = end_ids[sunk]
        if len(free):
            pos[free] = end[~sunk]
            time_left[free] = 0.0
        if len(trapped):
            resample[trapped] = True
        active[still] = False

    inside = query.contains(pos)
    # No facet was found but the endpoint is in the solid: the crossing test
    # missed (vertex / grazing). Bisect on the inside predicate and reflect
    # once about the nearest edge. If that also fails, freeze at the start.
    missed = inside & (bounces == 0) & ~resample
    n_fallback = 0
    n_fallback_ok = 0
    if np.any(missed):
        for i in np.nonzero(missed)[0]:
            n_fallback += 1
            repaired = _bisect_reflect(origin[i], vel0[i], float(dt), query, nudge)
            if repaired is None:
                pos[i] = origin[i]
                vel[i] = vel0[i]
                impulse[i] = 0.0
                frozen[i] = True
            else:
                pos[i], vel[i], impulse[i] = repaired
                bounces[i] = 1
                n_fallback_ok += 1
        inside = query.contains(pos)
    # Reflected, but the image is still inside the solid (the polar code's
    # "still inside" branch). Caller redraws position and direction.
    resample |= inside & ~frozen

    stats = {
        "n_hit_particles": int(np.sum(bounces > 0)),
        "n_bounce_events": int(bounces.sum()),
        "n_fallback": n_fallback,
        "n_fallback_ok": n_fallback_ok,
        "n_frozen": int(frozen.sum()),
        "n_resample": int(resample.sum()),
    }
    return pos, vel, impulse, bounces, resample, frozen, stats


def _nudge_outside(point, normal, query: BoundaryQuery, nudge: float) -> np.ndarray:
    """Step off the wall along the outward normal until the sample is in the gas."""
    landed = point + nudge * normal
    inside = query.contains(landed)
    if not np.any(inside):
        return landed
    for scale in (10.0, 100.0, 1.0e3, 1.0e4):
        trial = point + (nudge * scale) * normal
        still = query.contains(trial)
        landed = np.where(inside[:, None] & ~still[:, None], trial, landed)
        inside = inside & still
        if not np.any(inside):
            break
    return landed


def _nearest_segment(query: BoundaryQuery, point) -> int:
    a = query.a
    e = query.edge
    el2 = np.maximum(query.length ** 2, 1e-30)
    u = np.clip(np.sum((point - a) * e, axis=1) / el2, 0.0, 1.0)
    proj = a + u[:, None] * e
    d2 = np.sum((proj - point) ** 2, axis=1)
    # Prefer a real edge over a collapsed one.
    d2 = np.where(query.length > 1e-14, d2, np.inf)
    return int(np.argmin(d2))


def _bisect_reflect(p0, velocity, dt, query: BoundaryQuery, nudge: float):
    """Find a gas/solid transition along the free-flight segment and reflect once."""
    p0 = np.asarray(p0, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    end = p0 + dt * velocity
    if bool(query.contains(p0.reshape(1, 2))[0]):
        return None
    if not bool(query.contains(end.reshape(1, 2))[0]):
        return None
    lo, hi = 0.0, 1.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        sample = p0 + mid * dt * velocity
        if bool(query.contains(sample.reshape(1, 2))[0]):
            hi = mid
        else:
            lo = mid
    pt = p0 + hi * dt * velocity
    seg = _nearest_segment(query, pt)
    nrm = query.normal[seg].copy()
    if float(np.dot(nrm, velocity)) > 0.0:
        nrm = -nrm
    ndv = float(np.dot(nrm, velocity))
    if ndv >= -1e-14:
        return None
    vel = velocity - 2.0 * ndv * nrm
    pos = pt + nudge * nrm + (1.0 - hi) * dt * vel
    # One extra facet if the reflected ray immediately re-enters. Enough for
    # the safety net; the main loop already handles the generic case.
    if bool(query.contains(pos.reshape(1, 2))[0]):
        pos = pt + max(nudge * 100.0, 1e-6) * nrm
        if bool(query.contains(pos.reshape(1, 2))[0]):
            return None
    return pos, vel, 2.0 * ndv * float(nrm[0])
