from __future__ import annotations

import numpy as np


# def _hcost(instance: Model):
#     return instance.output.cost


# def _hcost_prime(
#     y: np.ndarray,
#     control_vector: np.ndarray,
#     mask: np.ndarray,
#     instance: Model,
#     parameters_bgd: ParametersDT,
#     states_bgd: StatesDT,
# ):
#     # % Set parameters or states
#     for i, name in enumerate(control_vector):
#         if name in instance.setup._parameters_name:
#             getattr(instance.parameters, name)[mask] = y[:, i]

#         else:
#             getattr(instance.states, name)[mask] = y[:, i]

#     parameters_b = instance.parameters.copy()
#     parameters_bgd_b = instance.parameters.copy()

#     states_b = instance.states.copy()
#     states_bgd_b = instance.states.copy()

#     output_b = instance.output.copy()

#     cost = np.float32(0)
#     cost_b = np.float32(1)

#     forward_b(
#         instance.setup,
#         instance.mesh,
#         instance.input_data,
#         instance.parameters,
#         parameters_b,
#         parameters_bgd,
#         parameters_bgd_b,
#         instance.states,
#         states_b,
#         states_bgd,
#         states_bgd_b,
#         instance.output,
#         output_b,
#         cost,
#         cost_b,
#     )

#     grad = np.transpose(
#         [
#             getattr(parameters_b, name)[mask]
#             if name in instance.setup._parameters_name
#             else getattr(states_b, name)[mask]
#             for name in control_vector
#         ]
#     )

#     return grad


def _inf_norm(grad: np.ndarray):
    return np.amax(np.abs(grad))
