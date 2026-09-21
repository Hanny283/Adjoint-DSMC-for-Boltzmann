# `forward_pass.py` — simulation, records, and the backward adjoint pass

This module has two jobs:

1. **Forward pass** — move particles step by step (collision, flight, reflection),
   recording a complete tape of everything that happened.
2. **Backward pass** — starting from the terminal gradient of the objective, walk
   the tape in reverse and compute β (velocity adjoint) and α (position adjoint)
   at every time step.

It implements the discrete trajectory and Propositions 2.1–2.2 of the paper.

---

## Forward pass: what gets recorded and why

The forward pass stores a `SimulationHistory` — a list of `StepRecord` objects, one
per time step.  Each `StepRecord` contains:

| Record type | What is stored | Why the backward pass needs it |
|-------------|----------------|-------------------------------|
| `CollisionRecord` | particle indices i, j; pre-collision velocities; drawn ω | The collision Jacobian depends on the pre-collision velocities and ω; the backward pass must replay the **same** ω to get the correct transpose. |
| `BoundaryRecord` | pre-reflection position x_k and velocity v′; intersection angle θ_inter; whether the particle left the domain | The reflection Jacobians M, N, G depend on θ_inter, v′, and x_k; in/out flag determines whether to apply the reflection update or the free-flight update. |
| `StepRecord` | start state, collision list, boundary list | One complete time slice. |

Nothing is re-computed in the backward pass — it only reads from the tape.

---

## Backward pass: what β and α mean

`backward_pass(terminal_beta_fn, terminal_alpha_fn)` returns two arrays of shape
(M+1, N, 2): `betas[k, i]` = β_{k,i} and `alphas[k, i]` = α_{k,i}.

**α_{k,i}** answers: *"If I nudge particle i's position at time k by a small vector δ,
how much does the objective L change?"*  Formally, α_{k,i} = ∂L/∂x_{k,i}.

**β_{k,i}** answers: *"If I nudge particle i's velocity at time k by δ, how much does
L change?"*  Formally, β_{k,i} = ∂L/∂v_{k,i}.

### Terminal conditions (step M)

The user supplies two functions:

- `terminal_alpha_fn(v_M, x_M)` — returns ∂L/∂x_{M,i} for each particle i.
- `terminal_beta_fn(v_M, x_M)` — returns ∂L/∂v_{M,i} for each particle i.

**Example for L = (1/N) Σᵢ |x_{M,i}|²:**
```
terminal_beta  = 0            (L does not depend on velocities)
terminal_alpha = 2 x_M / N   (gradient of mean squared distance)
```

**Example for L = −(1/N) Σᵢ φ(x_{M,i})** (Gaussian overlap):
```
terminal_beta  = 0
terminal_alpha = (1/N) Σⱼ exp(−|x−tⱼ|²/2σ²) · (x−tⱼ)/σ²
```

### α update (Prop. 2.2)

Walk backward from k = M−1 to k = 0:

- **If particle i did not hit the wall at step k:** the map from x_{k,i} to
  x_{k+1,i} is a simple linear shift (free flight), so α_{k,i} = α_{k+1,i}.
- **If particle i hit the wall:** the map includes reflection, so
  α_{k,i} = G_{k,i}ᵀ α_{k+1,i}, where G_{k,i} = ∂x̃/∂x is computed from
  `compute_G_ki` in `adjoint_jacobians.py`.

### β update (Prop. 2.1)

At each step:

1. **Free flight contribution:** velocity v drives future positions x = x + Δt v,
   so β_{k,i} picks up a Δt · α_{k+1,i} term.
2. **Wall reflection (if outside):** apply the reflection linearisation:
   β_{k,i} ← Mᵀ β_{k+1,i} + Nᵀ α_{k+1,i}
   using `compute_M_ki` and `compute_N_ki` from `adjoint_jacobians.py`.
3. **Collision:** for each colliding pair (i, j), multiply the stacked 4-vector
   [β_{k,i}; β_{k,j}] by the 4×4 collision Jacobian transpose to correctly update
   both particles' β values while keeping them coupled (Prop. 2.1).

---

## Key classes

### `ForwardSimulation`

- **`__init__(C, dt, ...)`** — stores C, Δt, RNG; builds the collision mesh (Bird
  parameter e controls collision rate: higher e → fewer collisions per step).
- **`run(positions, velocities, n_steps)`** — returns a `SimulationHistory`.

### `SimulationHistory`

- **`backward_pass(terminal_beta_fn, terminal_alpha_fn)`** — returns `betas` and
  `alphas` arrays, shape (M+1, N, 2).
- **`final_positions`** / **`final_velocities`** — particle state at the last step.

### Helper functions

- `_is_inside(x, C)` / `_is_inside_batch(x, C)` — star-shape interior test using
  `radius_r` from `boundary_geometry`: a particle is inside if its distance from
  the origin is less than r(θ) in the direction of x.
- `_reflect(x, v, theta_inter, C)` — applies specular reflection using `normal_n`
  and `compute_c_inter` from `boundary_geometry`.

---

## Relation to the paper

- State updates **ṽ, x̃** match Section 2 (reflection when x′ ∉ Ω).
- `backward_pass` is the algorithmic form of Propositions 2.1 and 2.2.
- `shape_gradient.py` uses the output (β, α) together with the stored tape to
  form dL/dC; it does not need to be called here.
