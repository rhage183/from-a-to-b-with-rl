# From A to B with RL !

Hello ! This is a repo from a LeWagon project in reinforcement learning where we have attempted (and succeded) at creating agents that could solve some of Gym environments.

We have implemented DQN, doubleDQN, A2C (discrete and continuous action spaces) and PPO agents on two environments (Frozen Lake and Car Racing)

The repo is implicitely separated into two sections (a Frozen Lake (FL) and a  Car Racing (CR)):
Here is a description for it:

car_agent.py/frozen_agent.py : the agent classes that run the environments

network.py : the NN used for the different agents

environment.py : contains methods for running one step of the environment under different conditions

car_race.py/frozen_lake.py : runs the main for each game

buffer.py: Contains a Replay memory class that agents can batch from 

display.py : Contains a Plotter class that allows to track metrics with time

params.py : Hyperparameters used in each game

## Setup

We work with the `rl` virtual environment, based on Python v3.11.9, so let us
first install that python version :

```
  pyenv install 3.11.9
```

If you don't have pyenv, please [install it first](https://github.com/pyenv/pyenv).
If you are on Windows, you can install [pyenv-win](https://pyenv-win.github.io/pyenv-win/)

You then have to create a virtual environment named `rl` :

```
  pyenv virtualenv 3.11.9 rl
````

Then, you can cd to the `from-a-to-b-with-rl` folder, and check if it is activated.

If the virtual environment is not activated automatically upon entering the folder you can run:

```
  pyenv local rl
````

### Packages

The `rl` virtual environment has a few dependencies, notably :

- [pytorch](https://pytorch.org/) for the RL learning
- [numpy](https://numpy.org/) to handle a few numerical outputs
- [gymnasium](https://gymnasium.farama.org/)
- [pygame](https://www.pygame.org/news)
- [moviepy](https://pypi.org/project/moviepy/) to save video from the agent interacting with the environment

You can then decide to install the package itself (XXX
Note, for now, nothing interesting is installed except from the dependencies XXX):

```
  pip install .
````

Or just decide to install the `requirements.txt` file :

```
  pip install -r requirements.txt
```

### Notes on GPU acceleration :

If your GPU is CUDA capable, you will have to adapt your `rl` environment. If you are on Windows, you can type :

```
  pip uninstall torch

  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

If you are on Linux, you can do :

```
  pip uninstall torch

  pip3 install torch torchvision torchaudio
```

If you want to monitor the GPU in the terminal, you can type 

```
  nvidia-smi -l 1
```
