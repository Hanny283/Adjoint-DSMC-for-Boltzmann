# FFD forward pass (specular channel drag)

A separate forward solver for the open-channel DSMC problem in
`experiments/simple/runInflow.py`. The body is no longer the star-shaped
Fourier radius

    r(θ) = c0 + Σ a_k cos(kθ) + b_k sin(kθ).

It is a closed polyline whose vertices are embedded in a tensor-product
B-spline control lattice (Li & Wu, eq. 53). Specular reflection only. There is
no adjoint, no shape gradient, and no change to the optimiser.

`experiments/simple/runInflow.py`, `shape_optimizer.py`, and
`src/adjoint/boundary_geometry.py` are not modified. Inflow, labelled
randomness, and Maxwell-molecule collisions are imported from `runInflow` and
called as they stand. The two pieces that assume a polar wall — the inside
test and the free-flight reflection — have FFD versions in this folder.

## Files

| File | Role |
|---|---|
| `ffd_geometry.py` | Clamped B-spline lattice, Greville identity map, winding-number inside test, uniform-grid broad-phase, segment–segment intersection, specular reflection with the remaining flight time. |
| `ffd_forward.py` | One DSMC window on that polyline. Prefill, gas-fraction, and the rare redraw call `runInflow` with the inside test swapped for the duration of the call, then restored. `collide_open` is the original function. |
| `run_ffd_forward.py` | Config block plus CLI. Prints drag, writes `summary_*.json`, a plot of the boundary and the control lattice, and an `.npz` of `Q` and the polyline. |
| `test_ffd.py` | Identity map, point-in-polygon (circle and a non-star C), segment hits, a vertex, two bounces in the notch, and a collision-free drag check against the Fourier circle. |

## Geometry

Control points `Q[i, j]` sit on the Greville grid of a clamped uniform
B-spline (default cubic, `6×6`, rectangle `[-1.6, 1.6]²`). At `Q0` the map is
the identity, so a sample of the baseline wall is reproduced exactly. Default
baseline: the unit circle, the same body as `runInflow.C_init`. Pass Fourier
coefficients to sample the current star-shaped family instead.

`(u_s, v_s)` is stored once per sample and never updated. Moving `Q` moves the
samples by the tensor-product weights (eq. 54). The DSMC wall is the polyline
through those samples, oriented counterclockwise, with an outward normal on
each edge.

Design variables for a quick non-circle test:

- `--preset dent` applies the four shifts in `DENT_PERTURBATION` (only the
  default `6×6` lattice on `[-1.6, 1.6]²`). The image is a simple polygon, the
  origin is inside, and some rays from the origin cross the wall three times,
  so it is not a radial graph.
- `--perturb i,j,dx,dy` (repeatable) adds a shift to one control point.
- `--control-points file.npy` replaces `Q` with an array of shape `(nu, nv, 2)`.
- `--fourier 1,0.15,0,0,0.05,0` samples that Fourier star as the initial wall
  (the lattice has to contain it; enlarge `--lattice-half` if a sample falls
  outside).

## What a particle does

1. Broad-phase: reject segments whose box misses the body box, then test only
   the edges in the uniform bins the segment overlaps.
2. Narrow-phase: solve the 2×2 system for each candidate edge. Keep crossings
   that enter the solid (`n · v < 0`) with the smallest flight fraction.
   A hit on a vertex averages the two outward normals.
3. Specular reflection `v ← v − 2(n·v)n`, then fly the remaining time. Repeat
   until the step is over or the bounce budget (12) is spent.
4. Drag contribution of each bounce is `2 (n·v_in) n_x`, the same expression
   as `runInflow.flight_step`, summed and divided by the window length. No
   particle mass. On a single straight edge this is the same end state as
   reflecting the free-flight endpoint through the tangent line.
5. A particle that is still inside after reflection is redrawn, as in
   `flight_step`. If the segment test misses a crossing that the inside test
   still sees, a bisection along the segment reflects once; if that also
   fails the particle freezes at the start of the step and adds no impulse
   (the polar solver's invalid-θ branch).

The counted window follows `runInflow`: step `k` covers `(k dt, (k+1) dt]`,
`T_START = 0`, and one extra unweighted step is taken so the collision and
inflow labels line up with the Fourier driver. The Fourier on-disk warm cache
is not read. `--t-warm` flies that many time units first, with negative step
labels, using this hit solver.

## How to run

From the repository root (the directory that contains `boltzmann_copy/`):

```bash
python boltzmann_copy/experiments/ffd/test_ffd.py
```

The script does not care about the working directory. It needs the same
imports `runInflow.py` needs (`numpy`, `scipy`, `matplotlib`, and the adjoint
package's import-time dependencies).

### Quick test on fibonacci

A few seconds. Cold start, collisions on, identity circle.

```bash
python boltzmann_copy/experiments/ffd/test_ffd.py
python boltzmann_copy/experiments/ffd/run_ffd_forward.py \
    --preset identity --n-ref 4000 --t-obs 1 --seed 0
```

Plot and `summary_*.json` land in `boltzmann_copy/experiments/ffd/output/`.

### Comparison against the star-shaped solver

Collisions off, this is a geometry check: same prefill, same inflow, no
collision re-pairing. Collisions on, one particle that the chord and the arc
classify differently changes partners for the rest of the window, so the
same-seed gap is another draw of the seed noise.

```bash
python boltzmann_copy/experiments/ffd/run_ffd_forward.py \
    --compare --no-collisions --n-ref 30000 --t-obs 2 --seed 0

python boltzmann_copy/experiments/ffd/run_ffd_forward.py \
    --compare --n-ref 30000 --t-obs 3 --seed 0
```

`--compare` forces the identity lattice and a cold start, and also runs seed
`seed+1` so the Fourier seed-to-seed scatter is printed next to the gap.
`--no-noise-seed` skips that second seed.

### Non-star body

```bash
python boltzmann_copy/experiments/ffd/run_ffd_forward.py \
    --preset dent --n-ref 20000 --t-obs 2 --seed 0
```

### Longer run

Production count and the usual window, with a 60-time-unit warm-up at this
wall (not the Fourier cache). At `N_REF = 10^6` the warm-up is the expensive
part; the counted window is a small fraction of it.

```bash
python boltzmann_copy/experiments/ffd/run_ffd_forward.py \
    --preset identity --n-ref 1000000 --t-obs 5 --t-warm 60 --seed 0
```

Other useful flags: `--nu`, `--nv`, `--degree`, `--lattice-half`,
`--n-boundary`, `--n-bins`, `--fourier`, `--perturb`, `--control-points`,
`--no-collisions`, `--no-plot`, `--out-dir`.

## Validation numbers

Measured in this tree, cold start, `M = 256` samples unless noted, `dt = 0.1`,
unit circle at `Q0`. Drag is `F_x` in the same raw units as `runInflow`
(window sum of `2(n·v)n_x`, divided by the window length).

**Collisions off, `N_REF = 30000`, window length 2.** Hit rate matches.
Fallback / freeze / resample counts were 0.

| seed | FFD | Fourier | relative gap |
|---|---:|---:|---:|
| 0 | 1055.37 | 1055.33 | 3.3×10⁻⁵ |
| 1 | 1068.91 | 1068.81 | 9.9×10⁻⁵ |

The same pair at `M = 128` is still under 10⁻³ (8.7×10⁻⁴ and 3.8×10⁻⁴).
The unit test `test_collision_free_circle_matches_fourier_drag` locks a
smaller version of this check (relative gap below 2×10⁻³).

**Collisions on, `N_REF = 30000`, window length 3, `M = 256`.**

| seed | FFD | Fourier | hits/step FFD | hits/step Fourier | relative gap |
|---|---:|---:|---:|---:|---:|
| 0 | 963.95 | 952.48 | 56.23 | 56.17 | 1.2×10⁻² |
| 1 | 966.74 | 957.84 | 57.73 | 57.27 | 9.3×10⁻³ |

Fourier seed-to-seed scatter on that pair is 5.6×10⁻³. A second window
(length 2, same `N_REF`) gave same-seed gaps of 2×10⁻⁴ and 3×10⁻² against a
Fourier seed scatter of 2.6×10⁻². The sign is not stable, and the collision-free
gap is four orders of magnitude smaller, so this is collision re-pairing from
the few particles in the chord-to-arc gap, not a biased normal.

**Dent, collisions on, `N_REF = 20000`, window length 2, seed 0.**
`F_x = 761.83`. Area 2.46 (the circle is π), perimeter 7.07, no
self-intersections, ray crossings from the origin run from 1 to 3. Fallback,
freeze, and resample counts were 0. The drag is not a drop relative to the
circle: the area is not held fixed.

Alive counts in the collision-free runs agreed to a fraction of a particle, so
the winding test and the 16×16 gas-fraction subgrid agree with `inside_body`
on this circle.

## Where this disagrees with `ffd_proposal.md`

The proposal is a plan for the adjoint and the optimiser. This directory is
only the forward pass, in new files, so a Fourier run of `runInflow.py` is
byte-for-byte the old code.

- No `Shape` protocol, no edit to `flight_step` / `backward_pass_impulse` /
  `project_C`. The proposal's step 1 would have been the highest-conflict
  edit and is not needed to measure drag.
- The wall the particles hit is the polyline, not a periodic spline through
  the samples. The proposal wanted the spline so the normal is `C¹` before
  any adjoint. On the circle, with collisions off, the facet normal already
  reproduces the Fourier drag to about 10⁻⁴ at `M = 256`, because one bounce
  off a straight edge matches the existing tangent reflection and the chord
  gap is far below a mean free path. A spline is the right next step when the
  normal has to be differentiated, not for this drag check.
- Remaining flight time and multiple bounces are handled. `runInflow` reflects
  once, about the tangent, and resamples if the image is still inside. On a
  convex body those are the same update; the notch test is the case the polar
  solver cannot represent.
- No area/perimeter rescaling, no curvature penalty, no `η` projection, no
  `y`-mirror reduction, no warm-cache key change. `Q` is used as given.

## Limitations and open questions

- **Is a crossing test enough without an inside test?** Not for initialization:
  prefill, the drop at `t = 0`, and the collision-cell gas fraction are
  inside tests. During a step the crossing test is the event detector. Every
  validation run here had zero bisection fallbacks, so on these bodies the
  segment test did not miss an entry. The fallback is still there for a
  vertex or a grazing overlap, where the 2×2 system is degenerate. A particle
  that starts the step inside the solid is not un-stuck by a crossing test;
  those are dropped up front or redrawn after a failed reflection.
- **Lattice size.** `6×6` cubic on `[-1.6, 1.6]²` is the accelerometer
  resolution from the proposal, and `Q0` fixes the unit circle. The four
  corner controls barely move that circle. Breaking the radial-graph property
  took O(1) shifts of the controls that actually support the wall (see
  `DENT_PERTURBATION`), not a 10⁻³ nudge. `8×8` is a flag (`--nu 8 --nv 8`),
  not the default. Re-fitting `(u_s, v_s)` when the body grows would be a
  different design map; it is not done.
- **Sample count.** `M = 256` (chord gap ~10⁻⁴ on the unit circle, cell size
  0.125). Collision-free drag is already inside 10⁻³ at `M = 128`. Raising
  `M` does not remove the collision-on seed noise. Self-intersection and
  radial-graph checks are O(M²) Python and become the slow part of startup
  past a few thousand samples.
- **Facet normals.** Drag on a coarse polygon depends on which edge is hit.
  The adjoint, when it is written, should not differentiate a jump in `n`
  across a vertex without a decision about that jump. The chain rule in the
  proposal (eq. 55) is unchanged either way: it only needs `∂L/∂r_s`.
- **Constraints.** Nothing here holds area or perimeter. A dent and the circle
  are different sizes; their drag values are not an optimisation result.
- **Self-intersection and folds** are reported, not rejected. A crossed
  polyline still runs; the winding number is nonzero in the overlap and the
  drag should not be trusted.
- **One segment per step in the Fourier tape.** We can bounce more than once
  inside `dt`. The circle does not (bounce count equals hit count in the
  collision-free check). A concave dent can. The future adjoint's single-hit
  tape would be the same approximation the polar code already makes.
- **Warm starts** are recomputed every launch. There is no FFD cache file.
  Do not point this solver at `experiments/simple/warm_cache/`.
