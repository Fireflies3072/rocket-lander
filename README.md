# Computational Control 2023 Rocket Lander Project

A Box2D Gymnasium environment which simulates a Falcon 9 ocean barge landing

Modified by Dylan Vogel and Gerasimos Maltezos for the 2023 Computation Control course at ETH Zurich  
Original environment created by Reuben Ferrante (https://github.com/arex18/rocket-lander)

## Usage


### Dependency Installation
Clone the environment using `git`. Call the following commands from the project folder:

```bash
python3 -m pip install --upgrade pip
python3 -m pip install poetry
python3 -m pip install .
```

This will install the environment and all necessary dependencies.

### Running
By default, a `run_simulation.py` script is included in the root of the project folder. This script will initialize an
environment and step through it using the basic MPC controller. The controller should be able to land the rocket under
gentle initialization conditions. 

Various arguments can be specified by adding them to the `args` dictionary which is passed to the environment on 
initialization. For a complete list of arguments, please refer to the `UserArgs` class in
[env_cfg.py](coco_rocket_lander/env/env_cfg.py).