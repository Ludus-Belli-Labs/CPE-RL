# CPE-RL

A reinforcement learning project for Command PE / Command: Modern Operations that uses a custom Gymnasium environment and a DQN agent.

## Overview

- `Deep_Q-Network/` contains the current DQN implementation, environment wrapper, and training loop.
- `Q-learning/` contains an older / alternative tabular Q-learning implementation.
- `RandomScen.py` picks a random scenario XML from built-in scenario templates.
- `CMO_SocketClient.py` provides a TCP socket client to communicate with the simulation.
- `LuaHandler.py` defines Lua commands used by the environment to query and control units.
- `requirements.txt` lists the Python dependencies.
- `Scene/` contains scenario files used by the environment and random scenario generator.

## Components

### `Deep_Q-Network/Env.py`

- Defines `CPEEnv`, a Gymnasium-style environment.
- Uses `CMO_SocketClient` to send Lua commands to the simulator.
- Builds a 10-dimensional observation vector:
  - contact posture codes for up to 4 contacts
  - movement state
  - radar state
  - ally distance bucket
  - target distance bucket
  - mission type
  - time remaining bucket
- Exposes a discrete action space with 10 actions.

### `Deep_Q-Network/DQN.py`

- Implements a DQN agent with an MLP policy network and target network.
- Uses experience replay and epsilon-greedy exploration.
- Saves and loads checkpoints with PyTorch.
- Default hyperparameters are defined in `Deep_Q-Network/Config.py`.

### `Deep_Q-Network/Train.py`

- Runs episodes until the scenario ends.
- Logs training metrics via `TrainingGraphs`.
- Saves checkpoints every `checkpoint` episodes and final model to `model.pt`.

## Dependencies

Install dependencies in a Python virtual environment:

python -m venv .venv
.\.venv\Scripts\Activate
python -m pip install -r requirements.txt


`requirements.txt` currently includes:

- `torch`
- `matplotlib`
- `tqdm`
- `gymnasium`
- `numpy`

## Running training

From Command Professional Edition folder with the CommandCLI present:

```CMD
./CommandCLI.exe -mode I -scenfile "\CPE-RL\Scene\Escort\Escort1.scen" -port 7777
```
In the "-scenfile" put the path to the scen that you want to start in. This scen will be run manually and after it is done then the actual training will begin with the scenes from the Scene folder.

From the repository root:

```CMD
python Deep_Q-Network\Train.py
```

This will start training for the default 100 episodes and save:

- `model.pt` in the root directory
- checkpoint files in `Deep_Q-Network/checkpoints/`
- training graph data via `TrainingGraph`

## Notes

- The environment connects to the simulator via TCP at `127.0.0.1:7777` by default.
- The scenario is selected randomly from `RandomScen.py`.
- Mission types include `Patrol` and `Strike`, with mission-specific move and attack behavior.

## Troubleshooting

Check that the port you set in the CLI is the same as the one in the `Deep_Q-Network/Config.py`.

For loading and continuing the training you need to change the "model_path" in run(model_path="checkpoints/model_ep10.json", n_episodes=5) in `Deep_Q-Network\Load_model.py` and "load_model" in train(n_episodes=100, checkpoint=10, load_model=None) in `Deep_Q-Network\Train.py`.

If you run the code in the Visual Interface (actual game) and not in the CLI then it might brick the game. If that happens you need to stop teh game from the Task Manager.
This happens because the code wants to load a new scenario after the previous one is done and the game does not like that. 
*This might not happen if it loads the same type of mission or with similar units*

## Folder structure

- `Deep_Q-Network/`
  - `Config.py`
  - `DQN.py`
  - `Env.py`
  - `Train.py`
  - `Load_model.py`
  - `Utils.py`
  - `training_*.txt`
- `OtherFiles/`
  - alternative RL and helper files
- `Q-learning/`
  - older Q-learning implementation
- `Scene/`
  - scenario files for training and testing