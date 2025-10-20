from typing import Optional, Tuple
import math
import numpy as np
from .controller_base import ControllerBase

class PID_Controller(ControllerBase):
    def __init__(self, kp: float, ki: float, kd: float,
        min_output: float = -math.inf, max_output: float = math.inf,
        anti_windup_gain: float = 1.0) -> None:

        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.min_output = min_output
        self.max_output = max_output
        self.anti_windup_gain = anti_windup_gain

        self._integrator = 0.0
        self._prev_measurement = None

    def reset(self, initial_state: Optional[float] = None) -> None:
        self._integrator = 0.0
        self._prev_measurement = initial_state

    def update(self, measurement: float, target: float, dt: float = 1.0) -> float:
        error = target - measurement
        # Proportional
        p = self.kp * error
        # Derivative on measurement to avoid derivative kick
        if self._prev_measurement is None or dt <= 0:
            d = 0.0
        else:
            d_measurement = (measurement - self._prev_measurement) / dt
            d = self.kd * d_measurement
        self._prev_measurement = measurement
        # Integral
        i = self._integrator
        #  Use anti-windup back-calculation to avoid further saturation
        unlimited_output = p + i + d
        limited_output = min(self.max_output, max(self.min_output, unlimited_output))
        windup_error = limited_output - unlimited_output
        self._integrator += (self.ki * error * dt) + (self.anti_windup_gain * windup_error * dt)

        return limited_output
    
    def update_by_error(self, error, dt_error, dt: float = 1.0):
        # Proportional
        p = self.kp * error
        # Derivative
        d = self.kd * dt_error
        # Integral
        i = self._integrator
        #  Use anti-windup back-calculation to avoid further saturation
        unlimited_output = p + i + d
        limited_output = min(self.max_output, max(self.min_output, unlimited_output))
        windup_error = limited_output - unlimited_output
        self._integrator += (self.ki * error * dt) + (self.anti_windup_gain * windup_error * dt)

        return limited_output

class PID_RocketLander(ControllerBase):
    """ Tuned PID Benchmark against which all other algorithms are compared. """

    def __init__(self, Fe_PID_params, psi_PID_params, FsTheta_PID_params,
        min_output: Tuple = None, max_output: Tuple = None):

        self.Fe_PID = PID_Controller(*Fe_PID_params)
        self.psi_PID = PID_Controller(*psi_PID_params)
        self.Fs_theta_PID = PID_Controller(*FsTheta_PID_params)
        self.min_output = np.array(min_output) if min_output is not None else np.full(3, -math.inf)
        self.max_output = np.array(max_output) if max_output is not None else np.full(3, math.inf)

    def reset(self):
        self.Fe_PID.reset()
        self.psi_PID.reset()
        self.Fs_theta_PID.reset()

    def update(self, measurement, target):
        x, y, vel_x, vel_y, theta, omega, legContact_left, legContact_right = measurement
        x_target, y_target = target[0], target[1]
        dx = x - x_target
        dy = y - y_target
        # ------------------------------------------
        y_ref = -0.1  # Adjust speed
        y_error = y_ref - dy + 0.1 * dx
        y_dterror = -vel_y + 0.1 * vel_x
        Fe = self.Fe_PID.update_by_error(y_error, y_dterror) * (abs(dx) * 50 + 1)
        # ------------------------------------------
        theta_ref = 0
        theta_error = theta_ref - theta + 0.2 * dx  # theta is negative when slanted to the north east
        theta_dterror = -omega + 0.2 * vel_x
        Fs_theta = self.Fs_theta_PID.update_by_error(theta_error, theta_dterror)
        Fs = -Fs_theta  # + Fs_x
        # ------------------------------------------
        theta_ref = 0
        theta_error = -theta_ref + theta
        theta_dterror = omega
        if (abs(dx) > 0.01 and dy < 0.5):
            theta_error = theta_error - 0.06 * dx  # theta is negative when slanted to the right
            theta_dterror = theta_dterror - 0.06 * vel_x
        psi = self.psi_PID.update_by_error(theta_error, theta_dterror)

        if legContact_left and legContact_right:  # legs have contact
            Fe = 0
            Fs = 0
        
        # Clip to output limits
        output = np.array([Fe, Fs, psi], dtype=np.float64)
        output = np.clip(output, self.min_output, self.max_output)
        return output


class PID_LunarLander(ControllerBase):
    """Simple PID for LunarLander continuous: vertical thrust + attitude control.

    - vertical_pid: controls vertical speed/altitude using main engine
    - angle_pid: stabilizes angle and tracks an angle target derived from lateral error
    """

    def __init__(self,
        vertical_PID_params: Tuple[float, float, float],
        angle_PID_params: Tuple[float, float, float],
        min_output: Tuple,
        max_output: Tuple,
    ) -> None:
        self.vertical_pid = PID_Controller(*vertical_PID_params)
        self.angle_pid = PID_Controller(*angle_PID_params)
        self.min_output = np.array(min_output, dtype=np.float64)
        self.max_output = np.array(max_output, dtype=np.float64)

    def reset(self) -> None:
        self.vertical_pid.reset()
        self.angle_pid.reset()

    def update(self, measurement, target):
        # measurement: [x, y, x_dot, y_dot, theta, theta_dot, left_contact, right_contact]
        x, y, x_dot, y_dot, theta, theta_dot = measurement[:6]
        x_target, y_target = target[0], target[1]
        dx = x - x_target
        dy = y - y_target

        # Vertical control: aim for target height with small vertical velocity
        y_error = y_target - y
        y_dterror = -y_dot
        main_thrust = self.vertical_pid.update_by_error(y_error, y_dterror)

        # Lateral/attitude: compute desired angle to reduce lateral error and velocity
        angle_target = -0.4 * (x - x_target) - 0.2 * x_dot
        angle_error = angle_target - theta
        angle_dterror = -theta_dot
        torque_cmd = self.angle_pid.update_by_error(angle_error, angle_dterror)

        # ------------------------------------------
        y_ref = -0.1  # Adjust speed
        y_error = y_ref - dy + 0.1 * dx
        y_dterror = -vel_y + 0.1 * vel_x
        Fe = self.Fe_PID.update_by_error(y_error, y_dterror) * (abs(dx) * 50 + 1)
        # ------------------------------------------
        theta_ref = 0
        theta_error = theta_ref - theta + 0.2 * dx  # theta is negative when slanted to the north east
        theta_dterror = -omega + 0.2 * vel_x
        Fs_theta = self.Fs_theta_PID.update_by_error(theta_error, theta_dterror)
        Fs = -Fs_theta  # + Fs_x

        # Clip and output [main, lateral]
        output = np.array([main_thrust, torque_cmd], dtype=np.float64)
        output = np.clip(output, self.min_output, self.max_output)
        return output