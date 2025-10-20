from typing import Optional

import numpy as np
import cvxpy as cp

from .controller_base import ControllerBase
from ..env.system_model import SystemModel

class MPC_RocketLander(ControllerBase):
    """Short linear MPC for RocketLander using cvxpy."""

    def __init__(self, env, horizon: int = 10, sample_time: float = 0.1,
                 Q: Optional[np.ndarray] = None, R: Optional[np.ndarray] = None) -> None:
        self.env = env
        self.horizon = int(horizon)
        self.Q = (np.diag([3.0, 0.1, 2.0, 1.0, 120.0, 30.0]) if Q is None else np.asarray(Q, float))
        self.R = (np.diag([0.01, 0.01, 0.01]) if R is None else np.asarray(R, float))

        self.u_size = 3
        self.y_size = 6
        cfg = self.env.unwrapped.cfg

        self.u_min = self.env.action_space.low.astype(np.float64)
        self.u_max = self.env.action_space.high.astype(np.float64)
        self.y_min = np.array([0, 0, -np.inf, -np.inf, -cfg.theta_limit, -np.inf], dtype=np.float64)
        self.y_max = np.array([cfg.width, cfg.height, np.inf, np.inf, cfg.theta_limit, np.inf], dtype=np.float64)

        model = SystemModel(self.env)
        model.discretize_system_matrices(sample_time)
        self.A, self.B = model.get_discrete_linear_system_matrices()

    def reset(self, initial_state: Optional[np.ndarray] = None) -> None:
        return

    def update(self, measurement, target) -> np.ndarray:
        # Define variables
        u = cp.Variable((self.u_size, self.horizon))
        y = cp.Variable((self.y_size, self.horizon + 1))

        # Define objective and constraints
        cost = 0
        constraints = [y[:, 0] == measurement]
        for t in range(self.horizon):
            cost += cp.quad_form(y[:, t] - target, self.Q) + cp.quad_form(u[:, t], self.R)
            constraints += [y[:, t + 1] == self.A @ y[:, t] + self.B @ u[:, t],
                            u[:, t] >= self.u_min, u[:, t] <= self.u_max,
                            y[:, t] >= self.y_min, y[:, t] <= self.y_max]
        constraints += [y[:, -1] >= self.y_min, y[:, -1] <= self.y_max]

        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve(solver=cp.OSQP, verbose=False, max_iter=20000)

        if u.value is None:
            return np.zeros(3, dtype=np.float64)
        return np.clip(u[:, 0].value, self.u_min, self.u_max)
