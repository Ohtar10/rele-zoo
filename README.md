# Rele-Zoo
![workflow](https://github.com/Ohtar10/rele-zoo/actions/workflows/main.yml/badge.svg)

This is a tool/reference repository of several reinforcement learning (RL) algorithms that I implemented for 
the main purpose of learning RL. I wanted to have a go-to catalog of RL algorithms and applications which also
allowed me to reuse big parts of what implies to run an RL algorithm/task and quickly try out new ideas/algorithms.

## System requirements
* Unix based systems
* Python >= 3.8
* Conda >= 4.x

## Installation
Check-out the repository and run the following:
```bash
make install-env
make install
```

## Usage
### Run tool
Run algorithms and tasks with `relezoo-run`, run the `--help` option to check the default values
```bash
relezoo-run --help
...
== Configuration groups ==
Compose your configuration from those groups (algorithm=reinforce)

algorithm: reinforce-continuous, reinforce-discrete
algorithm/policy: simple-continuous, simple-discrete
algorithm/policy/network: simple-fc
environments: acrobot, bipedalwalker, cartpole, mountaincar, parallel-cartpole, pendulum
logger: tensorboard


== Config ==
This is the config generated for this run.
You can override everything, for example:
relezoo algorithm.mode=play environment.name=Acrobot-v1
-------
experiment_name: Relezoo
mode: train
render: false
checkpoints: checkpoints/
episodes: 50
network:
  infer_in_shape: true
  infer_out_shape: true
ray_cpus: 4
ray_memory: 4294967296
ray_dashboard_port: 8265
env_train:
  _target_: relezoo.environments.GymWrapper
  name: CartPole-v1
env_test:
  _target_: relezoo.environments.GymWrapper
  name: CartPole-v1
algorithm:
  policy:
    network:
      _target_: relezoo.networks.simple.SimpleFC
      in_shape: infer
      out_shape: infer
    _target_: relezoo.algorithms.reinforce.discrete.ReinforceDiscretePolicy
    learning_rate: 0.01
    eps_start: 0.0
    eps_min: 0.0
    eps_decay: 0.0
  _target_: relezoo.algorithms.reinforce.discrete.ReinforceDiscrete
  batch_size: 5000
logger:
  _target_: tensorboardX.SummaryWriter
  logdir: tensorboard

-------


...
```
The run tool is powered by [hydra](https://hydra.cc/), by default, the tool will run a REINFORCE algorithm in the OpenAI
Gym cartpole environment and generate some tensorflow logs in the generated `output` folder. You can change any
configuration parameter shown above, for example, to run against `Acrobot-v1` environment instead, just run the tool
like this:
```bash
relezoo-run environments@env_train=acrobot
```
By default, `relezoo-run` runs in train mode, to run in test mode, specify the property and the path to the checkpoints
as per chosen algorithm
```bash
relezoo-run context.mode=play context.episodes=5 context.render=true checkpoints=../../../baselines/reinforce/cartpole/cartpole.cpt
```

### Shell
For convenience, there is an iphyton shell wrapper where you can run Ad-Hoc experiments directly.
```bash
relezoo shell
```

## References
- [OpenAI Spinning Up](https://spinningup.openai.com/en/latest/user/introduction.html)