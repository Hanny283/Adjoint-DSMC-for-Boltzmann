# `adjoint_jacobians.py` — closed-form Jacobians for reflection and collision

This module answers: **if I perturb x or v by a small amount before a sub-step,
how does the output (ṽ, x̃) change?**  These derivatives are the building blocks
of the backward (adjoint) pass.  They are derived in closed form from the paper
(Lemmas 2.2–2.10, Props. 2.1–2.2) so the backward pass never needs finite differences.

---

## Why closed-form Jacobians?

Each simulation step applies a nonlinear map (collision then flight and reflection):

    (x_new, v_new) = F(x_old, v_old)

To propagate sensitivity backward in time (adjoint = reverse-mode autodiff), we need
the **transpose** of the Jacobian of F.  Deriving these by hand from the geometry and
the Nanbu-Babovsky collision rule gives exact formulas that are fast to evaluate.

---

## Convention on transposes

In the backward pass, sensitivities are propagated as:

    β_k  ←  Mᵀ β_{k+1} + Nᵀ α_{k+1}
    α_k  ←  Gᵀ α_{k+1}

The matrices M, N, G are **not symmetric**, so the transpose matters.  All Jacobians
here are implemented so that the backward pass correctly uses the transpose of the
forward linearisation.

---

## Near-tangential stability guard

Several functions compute expressions of the form `numerator / denominator` where
the denominator is ∂F/∂θ — the derivative of the wall-intersection equation with
respect to θ.  When a particle travels nearly parallel to (tangent to) the wall,
this denominator approaches zero, and the gradient contribution from that particle
diverges to infinity.

Every affected function checks:

    if abs(denominator) < 1e-2:
        return zeros    # discard this near-tangential event

This threshold corresponds to approximately 0.7° from tangential incidence.  The
contribution of that particle to the gradient is discarded rather than allowed to
corrupt the gradient direction.  The same guard appears in `shape_gradient.py` for
the C-space gradient, and in `dtheta_inter_dC`.

---

## Functions

### Normal and intersection sensitivity

| Function | Paper | Plain meaning |
|----------|-------|---------------|
| `dn_dtheta(theta, C)` | Lemma 2.6 | The unit normal **n̂** rotates as the hit angle θ̃ shifts — this is that rotation rate. |
| `dtheta_dv_prime(theta_inter, x, v, C)` | Lemma 2.7 | If the pre-reflection velocity v′ changes, the hit point θ̃ moves; this is ∂θ̃/∂v′. |
| `dtheta_dx(theta_inter, v, C)` | Lemma 2.10 | If the pre-step position x changes, the hit point moves; this is ∂θ̃/∂x. |
| `dc_dv_prime(...)` | Lemma 2.8 | How the reflection offset c̃ moves when v′ changes (chains through θ̃). |
| `dc_dtheta_scalar(theta, C)` | Used in Lemma 2.9 | Scalar ∂c̃/∂θ̃ — building block for G. |

### Reflection Jacobians (for the β update)

| Function | Paper | Plain meaning |
|----------|-------|---------------|
| `dv_reflected_dv(theta_inter, x, v, C)` | Lemma 2.2 (J=I) | ∂ṽ/∂v′: how the post-bounce velocity changes if the pre-bounce velocity changes. |
| `compute_M_ki(..., J)` | Lemma 2.2 | Full ∂ṽ/∂v_pre = (∂ṽ/∂v′) · J, where J = ∂v′/∂v is the collision Jacobian. |
| `compute_N_ki(...)` | Lemma 2.3 | **N**_{k,i} = ∂x̃/∂v_pre — how the post-bounce position depends on the pre-step velocity (through flight time Δt and the wall hit). |

### Position Jacobian (for the α update)

| Function | Paper | Plain meaning |
|----------|-------|---------------|
| `compute_G_ki(...)` | Lemma 2.9 | **G**_{k,i} = ∂x̃/∂x at a wall hit — how the post-bounce position changes when the pre-step position changes. Used in Prop. 2.2 for the α update. |
| `compute_H_ki(...)` | Lemma dv'dx | **H**_{k,i} = ∂ṽ/∂x at a wall hit — how the post-bounce *velocity* changes when the pre-step *position* changes (the normal at the hit point depends on where the particle started). Also feeds Prop. 2.2, coupling β back into α whenever the particle reflects. |

### Collision pair

| Function | Paper | Plain meaning |
|----------|-------|---------------|
| `collision_jacobian(v_i, v_j, omega)` | Prop. 2.1 | 4×4 Jacobian ∂(v_i′, v_j′)/∂(v_i, v_j) for one Nanbu-Babovsky collision with random unit vector ω. |
| `collision_jacobian_transpose(...)` | Prop. 2.1 | Its transpose, used in the backward pass to update β for both colliding particles simultaneously. |

### Packaged backward update steps

| Function | Paper |
|----------|-------|
| `apply_proposition_21(...)` | Prop. 2.1 — one β update for a colliding pair. |
| `apply_proposition_22(...)` | Prop. 2.2 — α_{k} = Gᵀ α_{k+1} + Hᵀ β_{k+1} if wall hit; α_{k} = α_{k+1} otherwise. |

`forward_pass.SimulationHistory.backward_pass` does **not** call `apply_proposition_21`/
`apply_proposition_22` — it re-implements the same Propositions inline (as
`vectorized_boundary_backward`, using `dv_reflected_dv`, `compute_N_ki`,
`collision_jacobian_transpose`, and its own fused G/H computation) for
performance. `apply_proposition_21`/`apply_proposition_22` are kept as
readable, per-particle reference implementations of the same math and are
exercised by the test suite, not by the production simulation loop.

---

## Why each piece is necessary

- **Without** `dn_dtheta`, `dtheta_dv_prime`, `dtheta_dx`: the reflection Jacobians
  (M, N, G) are incomplete — you cannot differentiate the wall normal and offset as
  the particle position and velocity change.
- **Without** `collision_jacobian_transpose`: you cannot propagate β backward through
  the randomised pairwise collision while keeping the two particles **coupled** correctly.
  The 4×4 structure captures the off-diagonal coupling between the two particles' adjoints.
