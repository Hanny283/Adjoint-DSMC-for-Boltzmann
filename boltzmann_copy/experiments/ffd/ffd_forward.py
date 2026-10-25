"""
Forward pass of the open-channel DSMC drag problem on an FFD polyline body.

Inflow, outflow, labelled randomness and Maxwell-molecule collisions are the
routines in ``experiments/simple/runInflow.py``. Two pieces of that module are
hard-wired to the polar Fourier wall (``inside_body`` and ``flight_step``), so
they are not called as-is:

* particle-in-body tests (prefill, the initial drop, the collision-cell gas
  fraction, the rare post-reflection redraw) use a winding-number test on the
  polyline;
* free flight reflects by segment-segment intersection (see ``ffd_geometry``).

``collide_open`` itself is called unchanged. For the duration of the call its
gas-fraction helper is pointed at the polygon; the original function is put
back before we return, so a later Fourier run in the same process is the
stock solver.

Drag is the same estimator as ``runInflow``: the window sum of
``2 (n · v_in) n_x`` over specular hits, divided by the window length. Particle
mass is not folded in. One bounce on a straight edge deposits the same impulse
and the same end state as the tangent-line reflection in ``flight_step``.
"""

from __future__ import annotations

import numpy as np

import runInflow as ri
from ffd_geometry import BoundaryQuery, fly_and_reflect


def collision_gas_fraction(poly, n_sub: int = 16) -> np.ndarray:
    """Gas area fraction of every collision cell, same sub-grid as ``runInflow``.

    Indexing is ``ix * ny + iy``, matching ``collide_open``.
    """
    nx, ny = ri.N_COLL_CELLS
    u = (np.arange(n_sub) + 0.5) / n_sub
    cx = -ri.LX / 2 + (np.arange(nx)[:, None] + u[None, :]) * (ri.LX / nx)
    cy = -ri.LY / 2 + (np.arange(ny)[:, None] + u[None, :]) * (ri.LY / ny)
    x, y = np.broadcast_arrays(cx[:, None, :, None], cy[None, :, None, :])
    pts = np.stack([np.reshape(x, -1), np.reshape(y, -1)], axis=1)
    query = poly if isinstance(poly, BoundaryQuery) else BoundaryQuery(poly)
    inside = query.contains(pts).reshape(nx, ny, n_sub * n_sub)
    return (1.0 - inside.mean(axis=2)).reshape(-1)


def _with_inside(query, fn):
    saved = ri.inside_body
    ri.inside_body = lambda xs, C, query=query: query.contains(np.asarray(xs, float))
    try:
        return fn()
    finally:
        ri.inside_body = saved


def _prefill(x, v, alive, pid, seed, query):
    return _with_inside(query, lambda: ri.prefill(x, v, alive, pid, seed, None))


def _resample_outside(n, rng, query):
    return _with_inside(query, lambda: ri._resample_outside(None, n, rng))


def _collide(x, v, alive, pid, seed, k, gas_frac):
    saved = ri._cell_gas_fraction
    ri._cell_gas_fraction = lambda C, n_sub=16, gas_frac=gas_frac: gas_frac
    try:
        return ri.collide_open(x, v, alive, pid, seed, k, None)
    finally:
        ri._cell_gas_fraction = saved


def flight_ffd(x, v, alive, query: BoundaryQuery, refl_rng, weight: float):
    """One free-flight step. Returns ``(Ix_step, exited_slots, stats)``."""
    idx = np.nonzero(alive)[0]
    empty = np.empty(0, dtype=int)
    if len(idx) == 0:
        return 0.0, empty, {"n_hit_particles": 0, "n_bounce_events": 0,
                            "n_fallback": 0, "n_fallback_ok": 0, "n_frozen": 0, "n_resample": 0}
    pos, vel, impulse, _bounces, resample, _frozen, stats = fly_and_reflect(
        x[idx, :2], v[idx, :2], ri.DT, query
    )
    if np.any(resample):
        ns = int(resample.sum())
        pos[resample] = _resample_outside(ns, refl_rng, query)
        speed = np.linalg.norm(vel[resample], axis=1)
        ang = refl_rng.uniform(0.0, 2.0 * np.pi, ns)
        vel[resample, 0] = speed * np.cos(ang)
        vel[resample, 1] = speed * np.sin(ang)
    x[idx, 0] = pos[:, 0]
    x[idx, 1] = pos[:, 1]
    v[idx, 0] = vel[:, 0]
    v[idx, 1] = vel[:, 1]
    gone = (x[idx, 0] > ri.LX / 2) | (x[idx, 0] < -ri.LX / 2)
    if ri.SIDE_BC == "periodic":
        x[idx, 1] = (x[idx, 1] + ri.LY / 2) % ri.LY - ri.LY / 2
    else:
        gone |= np.abs(x[idx, 1]) > ri.LY / 2
    exited = idx[gone]
    alive[exited] = False
    return float(weight) * float(impulse.sum()), exited, stats


def _drop_inside(x, alive, query):
    inside = np.zeros(len(alive), dtype=bool)
    on = np.nonzero(alive)[0]
    if len(on):
        inside[on] = query.contains(x[on])
    alive &= ~inside


def simulate_ffd(query: BoundaryQuery, seed: int, t_warm: float = 0.0, gas_frac=None) -> dict:
    """Counted drag window at ``query``, using the current ``runInflow`` globals.

    ``t_warm`` (if positive) is flown first with negative step labels, the same
    convention as ``runInflow.warm_state``, so those collision/inflow draws do
    not reuse the counted window's labels. The Fourier on-disk warm cache is
    not read: it was produced by the polar hit solver.
    """
    if gas_frac is None and ri.COLLISIONS:
        gas_frac = collision_gas_fraction(query)
    cap = int(ri.CAP)
    x = np.zeros((cap, 2))
    v = np.zeros((cap, ri.VEL_DIM))
    alive = np.zeros(cap, dtype=bool)
    pid = np.zeros(cap, dtype=np.int64)
    counter = _prefill(x, v, alive, pid, seed, query) if ri.PREFILL else 0
    _drop_inside(x, alive, query)

    def advance(kk, weight, refl_key):
        if ri.COLLISIONS:
            _collide(x, v, alive, pid, seed, kk, gas_frac)
        ix, _exited, stats = flight_ffd(x, v, alive, query, ri._reflection_rng(seed, refl_key), weight)
        return ix, stats

    k_warm = int(round(float(t_warm) / ri.DT))
    for i in range(k_warm):
        kk = -k_warm + i
        advance(kk, 0.0, 2_000_000 - kk)
        counter = ri._advance_inject(x, v, alive, pid, seed, kk, counter)

    m = ri.n_steps()
    # +1 unweighted step, matching _simulate, so the counted steps see the same
    # number of collision/inflow draws as the Fourier driver. The extra step
    # does not add to the impulse.
    n_alive = np.zeros(m + 1, dtype=int)
    n_hits = np.zeros(m + 1, dtype=int)
    n_events = np.zeros(m + 1, dtype=int)
    n_fallback = np.zeros(m + 1, dtype=int)
    n_frozen = np.zeros(m + 1, dtype=int)
    n_resample = np.zeros(m + 1, dtype=int)
    ix_total = 0.0
    for k in range(m + 1):
        weight = ri._step_weight(k) if k < m else 0.0
        ix_step, stats = advance(k, weight, k)
        counter = ri._advance_inject(x, v, alive, pid, seed, k, counter)
        ix_total += ix_step
        n_alive[k] = int(alive.sum())
        n_hits[k] = stats["n_hit_particles"]
        n_events[k] = stats["n_bounce_events"]
        n_fallback[k] = stats["n_fallback"]
        n_frozen[k] = stats["n_frozen"]
        n_resample[k] = stats["n_resample"]
    window = m * ri.DT - ri.T_START
    drag = ix_total / window if window != 0.0 else float("nan")
    on = np.nonzero(alive)[0]
    return {
        "Ix": ix_total,
        "drag": drag,
        "n_alive": n_alive,
        "n_hits": n_hits,
        "n_bounce_events": n_events,
        "n_fallback": n_fallback,
        "n_frozen": n_frozen,
        "n_resample": n_resample,
        "x": x[on].copy(),
        "v": v[on].copy(),
        "seed": int(seed),
        "t_warm": float(t_warm),
    }
