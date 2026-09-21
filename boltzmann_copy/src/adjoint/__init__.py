"""
Adjoint module for DSMC shape optimisation with Fourier-parameterised boundary.

Public API
----------
boundary_geometry
    radius_r, radius_r_theta, radius_r_theta_theta
    gamma, gamma_prime
    f_unnormalized, f_unnormalized_prime
    normal_n          -- eq (7)
    solve_theta_inter -- eq (10)
    compute_c_inter   -- eq (11)

adjoint_jacobians
    dn_dtheta              -- Lemma 2.6
    dtheta_dv_prime        -- Lemma 2.7
    dtheta_dx              -- Lemma 2.10
    dc_dv_prime            -- Lemma 2.8
    dc_dtheta_scalar       -- ∂c̃/∂θ̃  (needed for G_{k,i})
    dv_reflected_dv        -- Lemma 2.2  (∂ṽ/∂v with J=I)
    compute_M_ki           -- full M_{k,i} = (∂ṽ/∂v') J
    compute_N_ki           -- N_{k,i} from Lemma 2.3 / eq (24)
    compute_G_ki           -- G_{k,i} = ∂x̃/∂x  (Lemma 2.9)
    compute_H_ki           -- H_{k,i} = ∂ṽ/∂x  (Lemma dv'dx)
    collision_jacobian     -- 4×4 J_coll  (Proposition 2.1)
    collision_jacobian_transpose
    apply_proposition_21   -- one backward step for β
    apply_proposition_22   -- one backward step for α

forward_pass
    CollisionRecord        -- recorded collision event
    BoundaryRecord         -- recorded boundary reflection event
    StepRecord             -- all events at one time step
    SimulationHistory      -- full forward trace + backward_pass()
    ForwardSimulation      -- driver: run() → SimulationHistory
"""

from .boundary_geometry import (
    radius_r,
    radius_r_theta,
    radius_r_theta_theta,
    gamma,
    gamma_prime,
    f_unnormalized,
    f_unnormalized_prime,
    normal_n,
    solve_theta_inter,
    compute_c_inter,
)

from .adjoint_jacobians import (
    dn_dtheta,
    dtheta_dv_prime,
    dtheta_dx,
    dc_dv_prime,
    dc_dtheta_scalar,
    dv_reflected_dv,
    compute_M_ki,
    compute_N_ki,
    compute_G_ki,
    compute_H_ki,
    collision_jacobian,
    collision_jacobian_transpose,
    apply_proposition_21,
    apply_proposition_22,
)

from .forward_pass import (
    CollisionRecord,
    BoundaryRecord,
    CollisionBatch,
    BoundaryBatch,
    StepRecord,
    SimulationHistory,
    ForwardSimulation,
)

from .shape_gradient import (
    dr_dC,
    drtheta_dC,
    dtheta_inter_dC,
    dv_reflected_dC,
    dx_reflected_dC,
    shape_gradient,
    area,
    area_gradient,
    perimeter,
    perimeter_gradient,
    perimeter_scale_factor,
    perimeter_rescale,
    rescaled_shape_gradient,
    project_step_perimeter_cap,
)

__all__ = [
    # boundary geometry
    "radius_r", "radius_r_theta", "radius_r_theta_theta",
    "gamma", "gamma_prime",
    "f_unnormalized", "f_unnormalized_prime",
    "normal_n", "solve_theta_inter", "compute_c_inter",
    # adjoint jacobians
    "dn_dtheta", "dtheta_dv_prime", "dtheta_dx", "dc_dv_prime",
    "dc_dtheta_scalar",
    "dv_reflected_dv", "compute_M_ki", "compute_N_ki", "compute_G_ki", "compute_H_ki",
    "collision_jacobian", "collision_jacobian_transpose",
    "apply_proposition_21", "apply_proposition_22",
    # forward pass
    "CollisionRecord", "BoundaryRecord",
    "CollisionBatch", "BoundaryBatch", "StepRecord",
    "SimulationHistory", "ForwardSimulation",
    # shape gradient
    "dr_dC", "drtheta_dC", "dtheta_inter_dC",
    "dv_reflected_dC", "dx_reflected_dC",
    "shape_gradient", "area", "area_gradient",
    "perimeter", "perimeter_gradient",
    "perimeter_scale_factor", "perimeter_rescale", "rescaled_shape_gradient",
    "project_step_perimeter_cap",
]
