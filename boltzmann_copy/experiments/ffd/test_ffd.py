"""
Unit tests for the FFD wall: identity map, point-in-polygon, segment hits.

Run from anywhere:

    python boltzmann_copy/experiments/ffd/test_ffd.py
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "experiments", "simple"))
sys.path.insert(0, HERE)

from adjoint.boundary_geometry import _radius_r_vec  # noqa: E402
import runInflow as ri  # noqa: E402
from ffd_geometry import (  # noqa: E402
    BoundaryQuery,
    EmbeddedBody,
    FFDLattice,
    apply_perturbation,
    basis_matrix,
    fly_and_reflect,
    is_radial_graph,
    points_in_polygon,
    radial_crossing_counts,
    sample_circle,
    sample_polar_wall,
    self_intersection_count,
    shape_report,
)
from ffd_forward import simulate_ffd  # noqa: E402


def _lattice(nu=6, nv=6, degree=3, half=1.6):
    return FFDLattice(origin=(-half, -half), extent=(2 * half, 2 * half), nu=nu, nv=nv, degree=degree)


class TestFFDIdentity(unittest.TestCase):
    def test_partition_of_unity_and_linear_precision(self):
        lat = _lattice()
        u = np.linspace(0.0, 1.0, 21)
        Bu = basis_matrix(u, lat.knots_u, lat.degree)
        self.assertTrue(np.allclose(Bu.sum(axis=1), 1.0, atol=1e-12))
        # sum_i B_i(u) ξ_i = u  (Greville sites)
        reproduced = Bu @ lat.greville_u
        self.assertTrue(np.allclose(reproduced, u, atol=1e-12))

    def test_random_interior_points_are_fixed_at_q0(self):
        for degree, n in ((1, 2), (2, 5), (3, 6), (3, 8)):
            lat = _lattice(nu=n, nv=n, degree=degree)
            rng = np.random.default_rng(degree * 10 + n)
            uv = rng.uniform(0.0, 1.0, size=(400, 2))
            physical = lat.origin + uv * lat.extent
            w = lat.weights(uv[:, 0], uv[:, 1])
            out = lat.deform(w, lat.Q0)
            self.assertTrue(np.allclose(out, physical, atol=1e-12), msg=f"degree {degree} n {n}")

    def test_circle_samples_stay_on_the_circle(self):
        lat = _lattice()
        wall = sample_circle(1.0, 256)
        body = EmbeddedBody(lat, wall)
        out = body.boundary(lat.Q0)
        self.assertTrue(np.allclose(out, wall, atol=1e-12))
        self.assertTrue(np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-12))
        self.assertTrue(is_radial_graph(out))

    def test_fourier_samples_match_the_polar_curve(self):
        C = np.array([1.0, 0.15, -0.05, 0.02, 0.08, -0.04, 0.01])
        wall = sample_polar_wall(C, 128)
        theta = np.linspace(0.0, 1.0, 128, endpoint=False)
        radius = _radius_r_vec(theta, C)
        ang = 2.0 * np.pi * theta
        expect = np.column_stack([radius * np.cos(ang), radius * np.sin(ang)])
        self.assertTrue(np.allclose(wall, expect, atol=1e-12))
        lat = _lattice(half=2.0)
        body = EmbeddedBody(lat, wall)
        self.assertTrue(np.allclose(body.boundary(lat.Q0), wall, atol=1e-12))

    def test_moving_one_control_point_matches_eq54(self):
        lat = _lattice()
        wall = sample_circle(1.0, 64)
        body = EmbeddedBody(lat, wall)
        delta = np.array([0.02, -0.015])
        Q = lat.Q0.copy()
        Q[2, 3] += delta
        got = body.boundary(Q)
        expect = wall + body.weights[:, 2, 3][:, None] * delta
        self.assertTrue(np.allclose(got, expect, atol=1e-12))
        # A full-lattice delta array and the (i, j, dx, dy) rows agree.
        rows = apply_perturbation(lat.Q0, [(2, 3, delta[0], delta[1])])
        self.assertTrue(np.allclose(rows, Q))

    def test_sample_outside_the_rectangle_is_rejected(self):
        lat = _lattice(half=1.6)
        with self.assertRaises(ValueError):
            EmbeddedBody(lat, sample_circle(2.0, 32))


class TestDentPreset(unittest.TestCase):
    def test_dent_is_a_simple_nonstar_body(self):
        from ffd_geometry import DENT_PERTURBATION
        lat = _lattice()
        body = EmbeddedBody(lat, sample_circle(1.0, 256))
        Q = apply_perturbation(lat.Q0, DENT_PERTURBATION)
        pts = body.boundary(Q)
        report = shape_report(pts)
        self.assertEqual(report["self_intersections"], 0)
        self.assertFalse(report["radial_graph"])
        self.assertGreater(report["radial_crossings_max"], 1)
        self.assertGreater(report["signed_area"], 0.0)
        self.assertTrue(bool(points_in_polygon(np.array([[0.0, 0.0]]), pts)))


class TestPointInPolygon(unittest.TestCase):
    def test_circle_polyline_matches_polar_inside_away_from_the_wall(self):
        wall = sample_circle(1.0, 256)
        rng = np.random.default_rng(1)
        pts = rng.uniform(-2.0, 2.0, size=(20000, 2))
        radius = np.linalg.norm(pts, axis=1)
        keep = np.abs(radius - 1.0) > 1e-3
        polar = ri.inside_body(pts[keep], ri.C_init)
        poly = points_in_polygon(pts[keep], wall)
        self.assertTrue(np.array_equal(polar, poly))

    def test_c_shape_is_not_a_radial_graph(self):
        # Solid "C" opening to +x. The notch is gas; the stem and arms are solid.
        poly = np.array([
            [0.0, 0.0], [3.0, 0.0], [3.0, 1.0], [1.0, 1.0],
            [1.0, 2.0], [3.0, 2.0], [3.0, 3.0], [0.0, 3.0],
        ])
        self.assertGreater(shape_report(poly)["signed_area"], 0.0)
        probes = np.array([
            [0.5, 1.5],   # stem, solid
            [2.0, 0.5],   # lower arm, solid
            [2.0, 1.5],   # notch, gas
            [4.0, 1.5],   # outside
            [1.5, -0.2],  # outside
        ])
        inside = points_in_polygon(probes, poly)
        self.assertTrue(np.array_equal(inside, [True, True, False, False, False]))
        # From a point in the notch a radial graph would see exactly one hit per ray.
        counts = radial_crossing_counts(poly, origin=(2.0, 1.5), n_angles=360)
        self.assertFalse(is_radial_graph(poly, origin=(2.0, 1.5), n_angles=360))
        self.assertGreater(int(counts.max()), 1)
        self.assertGreater(self_intersection_count(np.array([
            [0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0],
        ])), 0)
        self.assertEqual(self_intersection_count(poly), 0)


class TestSegmentReflection(unittest.TestCase):
    def _square(self):
        # CCW square, x in [0, 5], y in [-5, 5]. Left face x=0, outward normal -x.
        return BoundaryQuery(np.array([
            [0.0, -5.0], [5.0, -5.0], [5.0, 5.0], [0.0, 5.0],
        ]), n_bins=8)

    def test_flat_wall_matches_householder_and_keeps_flying(self):
        query = self._square()
        p0 = np.array([[-1.0, 0.2]])
        v = np.array([[1.0, 0.3]])
        dt = 3.0
        pos, vel, impulse, bounces, resample, frozen, _stats = fly_and_reflect(p0, v, dt, query, nudge=1e-10)
        # Hit x=0 at t=1/3, reflect, fly the remaining 2 time units.
        self.assertEqual(int(bounces[0]), 1)
        self.assertFalse(bool(resample[0]))
        self.assertFalse(bool(frozen[0]))
        self.assertAlmostEqual(float(impulse[0]), 2.0, places=6)
        self.assertTrue(np.allclose(vel[0], [-1.0, 0.3], atol=1e-8))
        self.assertTrue(np.allclose(pos[0], [-2.0, 1.1], atol=1e-6))
        # Same end state as reflecting the free-flight endpoint about the line x=0.
        end = p0[0] + dt * v[0]
        n = np.array([-1.0, 0.0])
        c = 0.0
        mirrored = end - 2.0 * (np.dot(n, end) - c) * n
        self.assertTrue(np.allclose(pos[0], mirrored, atol=1e-6))

    def test_earliest_crossing_not_the_exit(self):
        query = self._square()
        # Segment from x=-2 to x=6 crosses the entry face long before the exit face.
        p0 = np.array([[-2.0, 0.0]])
        v = np.array([[1.0, 0.0]])
        _hit, t, _n, point, seg = query.earliest(p0, p0 + 8.0 * v)
        self.assertTrue(bool(_hit[0]))
        self.assertAlmostEqual(float(t[0]), 0.25, places=8)
        self.assertAlmostEqual(float(point[0, 0]), 0.0, places=8)
        self.assertLess(float(point[0, 0]), 1.0)

    def test_miss_and_parallel_do_not_hit(self):
        query = self._square()
        p0 = np.array([[-1.0, 0.0], [1.0, 6.0], [6.0, 0.0]])
        v = np.array([[-1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        pos, vel, impulse, bounces, resample, frozen, _stats = fly_and_reflect(p0, v, 1.0, query)
        self.assertTrue(np.all(bounces == 0))
        self.assertTrue(np.allclose(impulse, 0.0))
        self.assertTrue(np.allclose(pos, p0 + v))
        self.assertFalse(np.any(resample))
        self.assertFalse(np.any(frozen))
        self.assertTrue(np.all(np.isfinite(pos)) and np.all(np.isfinite(vel)))

    def test_corner_hit_is_finite_and_preserves_speed(self):
        # CCW square [-ish] with a corner at (1, 1). Ray from (2, 2) through that corner.
        query = BoundaryQuery(np.array([
            [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0],
        ]), n_bins=4)
        p0 = np.array([[2.0, 2.0]])
        v = np.array([[-1.0, -1.0]])
        pos, vel, impulse, bounces, resample, frozen, _stats = fly_and_reflect(p0, v, 2.0, query, nudge=1e-9)
        self.assertGreaterEqual(int(bounces[0]), 1)
        self.assertFalse(bool(frozen[0]))
        self.assertFalse(bool(query.contains(pos)[0]))
        self.assertAlmostEqual(float(np.linalg.norm(vel[0])), float(np.linalg.norm(v[0])), places=8)
        self.assertTrue(np.all(np.isfinite(pos)) and np.all(np.isfinite(vel)))
        # Symmetric convex corner: one bisector reflection sends the particle back out.
        self.assertEqual(int(bounces[0]), 1)
        self.assertAlmostEqual(float(impulse[0]), -2.0, places=5)
        self.assertTrue(np.allclose(vel[0], [1.0, 1.0], atol=1e-6))

    def test_two_bounces_in_a_nonstar_notch(self):
        poly = np.array([
            [0.0, 0.0], [3.0, 0.0], [3.0, 1.0], [1.0, 1.0],
            [1.0, 2.0], [3.0, 2.0], [3.0, 3.0], [0.0, 3.0],
        ])
        self.assertFalse(is_radial_graph(poly, origin=(2.0, 1.5)))
        query = BoundaryQuery(poly, n_bins=6)
        # See the derivation in the module tests: bottom of the notch, then the back wall.
        p0 = np.array([[2.0, 1.25]])
        v = np.array([[-1.0, -0.5]])
        dt = 1.2
        pos, vel, impulse, bounces, resample, frozen, stats = fly_and_reflect(p0, v, dt, query, nudge=1e-9)
        self.assertEqual(int(bounces[0]), 2, msg=str(stats))
        self.assertFalse(bool(resample[0]))
        self.assertFalse(bool(frozen[0]))
        self.assertAlmostEqual(float(impulse[0]), -2.0, places=5)
        self.assertTrue(np.allclose(vel[0], [1.0, 0.5], atol=1e-6))
        self.assertFalse(bool(query.contains(pos)[0]))
        self.assertGreater(float(pos[0, 0]), 1.0)
        self.assertTrue(1.0 < float(pos[0, 1]) < 2.0)
        self.assertAlmostEqual(float(np.linalg.norm(vel[0])), float(np.linalg.norm(v[0])), places=8)

    def test_broadphase_finds_a_long_ray_against_the_circle(self):
        query = BoundaryQuery(sample_circle(1.0, 256), n_bins=16)
        p0 = np.array([[-10.0, 0.2]])
        v = np.array([[1.0, 0.0]])
        hit, t, _n, point, _seg = query.earliest(p0, p0 + 20.0 * v)
        self.assertTrue(bool(hit[0]))
        # Analytic circle hit: x = -sqrt(1 - 0.2^2), the polyline is within one sagitta.
        x_exact = -np.sqrt(1.0 - 0.2 ** 2)
        self.assertAlmostEqual(float(point[0, 0]), x_exact, delta=0.002)
        self.assertAlmostEqual(float(point[0, 1]), 0.2, delta=0.002)
        self.assertLess(float(t[0]), 0.5)

    def test_batch_particles_do_not_interfere(self):
        query = self._square()
        p0 = np.array([[-1.0, 0.0], [1.0, 6.0], [-1.0, 1.0]])
        v = np.array([[1.0, 0.0], [0.2, 0.0], [1.0, 0.0]])
        _pos, _vel, impulse, bounces, _r, _f, _s = fly_and_reflect(p0, v, 3.0, query, nudge=1e-10)
        self.assertEqual(int(bounces[0]), 1)
        self.assertEqual(int(bounces[1]), 0)
        self.assertEqual(int(bounces[2]), 1)
        self.assertAlmostEqual(float(impulse[1]), 0.0, places=12)
        self.assertAlmostEqual(float(impulse[0]), 2.0, places=6)
        self.assertAlmostEqual(float(impulse[2]), 2.0, places=6)


class TestForwardSmoke(unittest.TestCase):
    def test_identity_forward_runs_and_restores_the_polar_hooks(self):
        snap = {k: getattr(ri, k) for k in
                ("N_REF", "CAP", "M_P", "N0", "N_STEPS", "TOBS", "T_START",
                 "WARM_START", "COLLISIONS", "PREFILL")}
        try:
            ri.WARM_START = False
            ri.COLLISIONS = False
            ri.PREFILL = True
            ri.T_START = 0.0
            ri.TOBS = 0.2
            ri.N_STEPS = 2
            ri.set_particle_count(300, cap=800)
            lat = _lattice()
            wall = sample_circle(1.0, 64)
            body = EmbeddedBody(lat, wall)
            query = BoundaryQuery(body.boundary(lat.Q0), n_bins=8)
            out = simulate_ffd(query, seed=0, t_warm=0.0)
            self.assertTrue(np.isfinite(out["drag"]))
            self.assertGreater(out["drag"], 0.0)
            self.assertGreater(int(out["n_hits"].sum()), 0)
        finally:
            for k, v in snap.items():
                setattr(ri, k, v)
            ri.set_particle_count(snap["N_REF"], cap=snap["CAP"])
        # The polar hooks are the originals again: origin is inside the unit circle.
        self.assertTrue(bool(ri.inside_body(np.array([[0.0, 0.0]]), ri.C_init)[0]))
        self.assertFalse(bool(ri.inside_body(np.array([[3.0, 0.0]]), ri.C_init)[0]))
        ri.selftest_geometry()

    def test_collision_free_circle_matches_fourier_drag(self):
        """Same seed, collisions off: the polyline impulse matches the polar solver."""
        snap = {k: getattr(ri, k) for k in
                ("N_REF", "CAP", "M_P", "N0", "N_STEPS", "TOBS", "T_START",
                 "WARM_START", "COLLISIONS", "PREFILL")}
        try:
            ri.WARM_START = False
            ri.COLLISIONS = False
            ri.PREFILL = True
            ri.T_START = 0.0
            ri.TOBS = 0.5
            ri.N_STEPS = int(round(ri.TOBS / ri.DT))
            ri.set_particle_count(2500, cap=4000)
            lat = _lattice()
            body = EmbeddedBody(lat, sample_circle(1.0, 128))
            query = BoundaryQuery(body.boundary(lat.Q0), n_bins=8)
            ffd = simulate_ffd(query, seed=0, t_warm=0.0)
            ix, _steps, _edge = ri.simulate_and_record(ri.C_init, 0)
            fourier = float(ri.drag_force(ix))
            rel = abs(ffd["drag"] - fourier) / abs(fourier)
            self.assertLess(rel, 2e-3)
            self.assertEqual(int(ffd["n_fallback"].sum()), 0)
            self.assertGreater(ffd["drag"], 0.0)
        finally:
            for k, v in snap.items():
                setattr(ri, k, v)
            ri.set_particle_count(snap["N_REF"], cap=snap["CAP"])


if __name__ == "__main__":
    unittest.main()
