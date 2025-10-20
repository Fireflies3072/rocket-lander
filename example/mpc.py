import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import numpy as np
from coco_rocket_lander.algs.mpc import MPC_RocketLander

args = {
    "initial_position": (31/33.333, 20/26.666, 0)
}

# --- Configuration ---
# MPC Parameters
u_size = 3
y_size = 6
horizon = 10
sample_time = 0.1
Q = np.diag([3.0, 0.1, 2.0, 1.0, 120.0, 30.0])
R = np.diag([0.01, 0.01, 0.01])

# --- Environment Setup ---
env = gym.make("coco_rocket_lander/RocketLander-v0", render_mode="rgb_array", args=args)
env = gym.wrappers.RecordVideo(env, 'video', episode_trigger=lambda x: True, name_prefix="rocket_lander")

# MPC controller
mpc = MPC_RocketLander(env, horizon, sample_time, Q, R)

# Define the reference (target) state for the rocket
# Target: land at the landing position with zero velocity and angle
landing_position = env.unwrapped.get_landing_position()
reference = np.zeros(y_size, dtype=np.float64)
reference[0] = landing_position[0]
reference[1] = landing_position[1]

# Reset the environment
state, _ = env.reset(seed=0)

for i in range(2000):
    # Let MPC calculate the optimal action
    action = mpc.update(state[:6], reference)
    # If the legs are in contact, set both main and side engine thrusts to 0
    if state[6] and state[7]:
        action[:2] = 0
    
    # Apply the calculated action to the environment
    next_state, rewards, done, _, info = env.step(action)

    # Update observation
    state = next_state
    
    # Check if simulation ended
    if done:
        break

print("Control phase finished.")
env.close()
