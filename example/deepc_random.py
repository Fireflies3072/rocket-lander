import random
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import numpy as np
from coco_rocket_lander.algs.pid import PID_RocketLander
from coco_rocket_lander.algs.deepc import DeePC_Controller, DeePC_Analyzer

args = {
    "initial_position": (0.4, 0.8, 0.0)
}

# --- Configuration ---
# DeePC Parameters
T_ini = 1
T_f = 10

# --- Environment Setup ---
env = gym.make("coco_rocket_lander/RocketLander-v0", render_mode="rgb_array", args=args)
env = gym.wrappers.RecordVideo(env, 'video', episode_trigger=lambda x: True, name_prefix="rocket_lander")

# Get system dimensions from the environment
# u_size: dimension of action space (main engine, side engine, nozzle angle)
# y_size: dimension of observation space (x,y pos, x,y vel, angle, ang. vel)
u_size = 3
y_size = 6

# PID controller
engine_pid_params = [10, 0, 10]
engine_vector_pid_params = [0.085, 0.001, 10.55]
side_engine_pid_params = [5, 0, 6]
pid = PID_RocketLander(engine_pid_params, engine_vector_pid_params, side_engine_pid_params,
                        env.action_space.low, env.action_space.high)

# DeePC controller
deepc = DeePC_Controller(
    u_size=u_size,
    y_size=y_size,
    T_ini=T_ini,
    T_f=T_f,
    Q=[10, 1, 1, 1, 1e3, 1],
    R=[1.5, 0.01, 0.01],
    lambda_g=1e-2,
    lambda_y=1e5,
    min_output=env.action_space.low,
    max_output=env.action_space.high
)

# DeePC analyzer
analyzer = DeePC_Analyzer(labels=['x', 'y', 'vx', 'vy', 'angle', 'ang_vel'])

# ===================================================================
#  PHASE 1: DATA COLLECTION
# ===================================================================
print("--- Starting Phase 1: Data Collection ---")
collected = 0

state, _ = env.reset()
while collected < 200:
    # Run only a few steps for each episode
    for _ in range(200):
        action = np.array([
            random.uniform(env.action_space.low[0],  env.action_space.high[0]),  # main thrust in [0,1]
            random.uniform(env.action_space.low[1],  env.action_space.high[1]),  # side in [-1,1]
            random.uniform(env.action_space.low[2],  env.action_space.high[2]),  # nozzle in [-1,1]
        ], dtype=np.float64)
        next_state, _, done, _, _ = env.step(action)
        deepc.add_sample(action, next_state[:6])
        state = next_state
        collected += 1
        if done:
            break
    # Reset after a few steps
    state, _ = env.reset()

# Build the Hankel matrices from the collected data
deepc.build_hankel_from_buffer()
print("DeePC Hankel matrices built successfully.")
print(f"Collected {deepc.U_p.shape[1]} samples")


# ===================================================================
#  PHASE 2: PREDICTIVE CONTROL
# ===================================================================
print("\n--- Starting Phase 2: DeePC Control ---")

# Reset the environment and the controller's history for a fresh start
deepc.reset(initial_y=state[:6])
# Define the reference (target) state for the rocket
# Target: land at the landing position with zero velocity and angle
landing_position = env.unwrapped.get_landing_position()
reference = np.zeros(y_size)
reference[0] = landing_position[0]
reference[1] = landing_position[1]
hover_state = state[:6]

for i in range(2000):
    # Let DeePC calculate the optimal action
    action, y_next, g, sigma_y = deepc._update_complete(state[:6], reference)
    # If the legs are in contact, set both main and side engine thrusts to 0
    if state[6] and state[7]:
        action[:2] = 0
    
    # Apply the calculated action to the environment
    next_state, rewards, done, _, info = env.step(action)

    # Add the (action, measurement, target) pair to the DeePC analyzer
    analyzer.add_sample(action, y_next, state[:6], reference, g, sigma_y)

    # Update observation
    state = next_state
    
    # Check if simulation ended
    if done:
        break

print("Control phase finished.")
env.close()

# Analyze the DeePC results
analyzer.analyze()
