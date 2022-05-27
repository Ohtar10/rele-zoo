Cross Entropy Method
====================
The Cross-Entropy Method [CEM-GD]_ is a simple and effective algorithm in several occasions.
The idea is basically let the agent interact with the environment collecting rollouts, later
the rollouts are filtered as per their total attained reward, selecting a fixed percentile of
the best ones. The best rollouts are called the "elites", then the idea is to use these elite
rollouts to train a neural network to imitate the actions selected during the observations of
these rollouts. So in essence, the elite rollout's actions become the "labels" for predicted
actions of the current neural network.

The hope is that by copying the best behavior of the best episodes, the agent will start improving
its behavior in subsequent rollouts. This is an iterative process, so the steps are as follows:

#. Collect trajectories using the current policy (neural network).
#. Filter the collected trajectories and select the x percentile of the best ones.
#. Run a training step of the neural network.
    #. Predict actions from observations on elite observations.
    #. Optimize via Gradient Descent the predicted actions against the actual elite actions.
#. Repeat for N epochs.

Algorithm
---------

.. automodule:: relezoo.algorithms.xentropy.core

.. autoclass:: CrossEntropyMethod
    :members:
    :special-members:

Discrete Policies
-----------------

.. automodule:: relezoo.algorithms.xentropy.discrete

.. autoclass:: CrossEntropyDiscretePolicy
    :members:
    :special-members:

Continuous Policies
-------------------
.. automodule:: relezoo.algorithms.xentropy.continuous

.. autoclass:: CrossEntropyContinuousPolicy
    :members:
    :special-members:


.. [CEM-GD] `Cross-Entropy Method with Gradient Descent <https://arxiv.org/abs/2112.07746>`_