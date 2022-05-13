Welcome to ReleZoo's documentation!
===================================

**ReleZoo** is a Python library/SDK to run and develop Reinforcement Learning (RL)
algorithms.

It is basically an opinionated abstraction of the several aspects RL experimentation might have
(environments, parallel environments, logging, training loop, evaluation, e.t.c.)
to make easier to run, visualize, and try out new experiments. I developed this as a way **to help me
learn** about RL coming from a more traditional SWE background. Basically I intend this project
to be my own version of Stable_baselines3_.

Check out the :doc:`usage/basic` section for further information, including
how to :ref:`installation` the project.

.. note::

   This project is under active development.

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Usage

   usage/basic
   usage/train
   usage/play
   usage/develop

.. toctree::
   :maxdepth: 2
   :caption: Architecture

   architecture/components
   architecture/engine

.. toctree::
   :maxdepth: 2
   :caption: Algorithms

   modules/algorithms/base
   modules/algorithms/xentropy
   modules/algorithms/reinforce

.. toctree::
   :maxdepth: 2
   :caption: Environments

   modules/environments/base
   modules/environments/gym

.. toctree::
   :maxdepth: 3
   :caption: Logging

   modules/logger/base
   modules/logger/tensorboard
   modules/logger/wandb

.. toctree::
   :maxdepth: 2
   :caption: Networks

   modules/networks/base

.. toctree::
   :maxdepth: 2
   :caption: Utils

   modules/utils/msc

.. toctree::
   :caption: Wandb Reports

   wandb


.. _Stable_baselines3: https://github.com/DLR-RM/stable-baselines3