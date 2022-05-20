# Rele-Zoo
![workflow](https://github.com/Ohtar10/rele-zoo/actions/workflows/main.yml/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/rele-zoo/badge/?version=latest)](https://rele-zoo.readthedocs.io/en/latest/?badge=latest)

This is a tool/reference repository of several reinforcement learning (RL) algorithms that I implemented for 
the main purpose of learning RL. I wanted to have a go-to catalog of RL algorithms and applications which also
allowed me to reuse big parts of what implies to run an RL algorithm/task and quickly try out new ideas/algorithms.

Please check out the [documentation page](https://rele-zoo.readthedocs.io/en/latest/?badge=latest) for more detailed usage.

## System requirements
* Unix based systems
* Python >= 3.8
* Conda >= 4.x

## Installation
Check-out the repository and run the following:
```bash
make install-env
make install
conda activate rele-zoo
```

## Usage
### Run tool
Run algorithms and tasks with `relezoo-run`, run the `--help` option to check the default values.

The run tool is powered by [hydra](https://hydra.cc/), by default, the tool will run a REINFORCE algorithm in the OpenAI
Gym cartpole environment and generate some tensorflow logs in the generated `output` folder. You can change any
configuration parameter as needed, for example, to run against `Acrobot-v1` environment instead, just run the tool
like this:
```bash
relezoo-run environments@env_train=acrobot
```
By default, `relezoo-run` runs in train mode, to run in test mode, specify the property and the path to the checkpoints
as per chosen algorithm
```bash
relezoo-run context.mode=play context.epochs=5 context.render=true checkpoints=../../../baselines/reinforce/cartpole/
```

### Shell
For convenience, there is an iphyton shell wrapper where you can run Ad-Hoc experiments directly.
```bash
relezoo shell
```

## References
The development of this project is inspired by:
- [OpenAI Spinning Up](https://spinningup.openai.com/en/latest/user/introduction.html)
- [Stable Baselines 3](https://github.com/DLR-RM/stable-baselines3)
- [Deep Reinforcement Learning Hands-On](https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On)