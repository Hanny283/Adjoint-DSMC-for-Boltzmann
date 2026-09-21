# `shape_gradient.py` — gradient of L with respect to Fourier coefficients C

This module takes the adjoint arrays β and α (from `forward_pass.backward_pass`)
and produces **dL/dC** — the gradient of the objective with respect to the boundary
shape coefficients.  This is the output the optimiser actually uses.

---

## Why a separate module?

The coefficients **C** appear only at wall-reflection events.  Particles in free flight
or mid-collision do not "feel" C at all.  So `shape_gradient` only needs to visit the
boundary-hit records in the tape and, at each hit, contract β and α with the
C-Jacobians of the reflection map.

    dL/dC = Σ_{(k,i): wall hit} [ β_{k+1,i}ᵀ ∂ṽ/∂C + α_{k+1,i}ᵀ ∂x̃/∂C ]

The adjoints β, α carry sensitivity **through time**; this formula converts that
time-domain sensitivity into **coefficient-space** sensitivity at each wall event.

---

## How ∂ṽ/∂C and ∂x̃/∂C are computed

The reflected velocity and position depend on C through two paths:

1. **θ_inter depends on C** — the wall curve r(θ; C) changes when C changes, so the
   intersection angle θ̃ moves (implicit differentiation of F(θ; C) = 0).
2. **The normal and offset depend on C at fixed θ** — even if θ̃ did not move, the
   normal **n̂**(θ; C) and offset c̃(θ; C) change with C.

`dtheta_inter_dC` computes the implicit derivative ∂θ̃/∂C via:

    ∂θ̃/∂Cⱼ = −(∂F/∂Cⱼ) / (∂F/∂θ)  =  −dr_j · factor / denominator

where denominator = ∂F/∂θ is the same quantity that appears in `adjoint_jacobians.py`
for ∂θ̃/∂v and ∂θ̃/∂x.

The full C-Jacobians then chain both paths together:
- `dv_reflected_dC` = ∂ṽ/∂C (shape 2 × (2N+1))
- `dx_reflected_dC` = ∂x̃/∂C (shape 2 × (2N+1))

---

## Near-tangential stability guard

When a particle hits the wall nearly tangentially, the denominator ∂F/∂θ is very
small and the gradient contribution diverges.  Three guards are applied:

1. **`dtheta_inter_dC`** — if |denominator| < 1e-2, return a zero vector (no
   contribution to dL/dC from this event).
2. **`adjoint_jacobians.dtheta_dv_prime` and `dtheta_dx`** — same guard, preventing
   β and α from blowing up before they reach `shape_gradient`.
3. **Per-particle clipping in `shape_gradient`** — each particle's contribution
   `c = β_{k+1}ᵀ ∂ṽ/∂C + α_{k+1}ᵀ ∂x̃/∂C` is clipped to unit norm before being
   added to the gradient.  This prevents any single event from dominating the
   gradient direction even if the guards above missed it.

---

## Constraint functions

In addition to the shape gradient itself, this module contains functions used by the
experiment scripts to enforce domain constraints:

### Perimeter (numerical quadrature)

    perimeter(C, n_quad=400) = ∫₀¹ ‖γ′(θ; C)‖ dθ  ≈ mean over 400 quadrature points
    perimeter_gradient(C)    = dP/dC  (same quadrature)

These are used by `project_step_perimeter_cap` to enforce a hard upper bound on
perimeter (a material budget).

### Area (exact closed form)

    area(C) = π c₀² + (π/2) Σₖ (aₖ² + bₖ²)
    area_gradient(C): grad[0] = 2π c₀,  grad[1:] = π · C[1:]

Derived from Parseval's theorem: Area = π ∫₀¹ r(θ)² dθ, and ∫₀¹ r² dθ = c₀² +
(1/2) Σ(aₖ²+bₖ²).  No quadrature needed — the gradient is exact and costs O(N)
to evaluate (N = number of Fourier modes).

**Why prefer area over perimeter for the convergence proof:**  under a fixed-area
constraint, the circle is the unique global minimum of mean |x|² for any number
of Fourier modes (Jensen's inequality).  Under a perimeter constraint this fails
for modes k ≥ 3.  See `experiments/writeup.md` for the full argument.

---

## Functions summary

| Function | Shape | Description |
|----------|-------|-------------|
| `dr_dC(theta, C)` | (2N+1,) | ∂r/∂C at fixed θ — the basis vector for r's C-gradient. |
| `drtheta_dC(theta, C)` | (2N+1,) | ∂(∂r/∂θ)/∂C at fixed θ. |
| `dtheta_inter_dC(theta, v, C)` | (2N+1,) | ∂θ̃/∂C via implicit differentiation; zeroed if near-tangential. |
| `dv_reflected_dC(theta, v, C)` | (2, 2N+1) | ∂ṽ/∂C — full chain through θ̃ and C. |
| `dx_reflected_dC(theta, x, v, C)` | (2, 2N+1) | ∂x̃/∂C — full chain. |
| `shape_gradient(history, betas, alphas)` | (2N+1,) | Main entry point — sums over all wall-hit records. |
| `perimeter(C)` | scalar | Numerical approximation ∫‖γ′‖dθ. |
| `perimeter_gradient(C)` | (2N+1,) | dP/dC via same quadrature. |
| `area(C)` | scalar | **Exact** enclosed area — no quadrature. |
| `area_gradient(C)` | (2N+1,) | **Exact** dA/dC — O(N), no quadrature. |
| `project_step_perimeter_cap(...)` | (2N+1,) | Apply gradient step with hard perimeter cap. |

---

## End-to-end summary

```
forward tape
    ↓
backward_pass  →  β (vel. adjoint), α (pos. adjoint)  at every time step
    ↓
shape_gradient:  at each wall hit  →  β_{k+1}ᵀ ∂ṽ/∂C + α_{k+1}ᵀ ∂x̃/∂C
    ↓
dL/dC  →  optimizer updates C
```
