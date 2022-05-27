REINFORCE
=========
REINFORCE aka vanilla policy gradients [VPG]_ belongs to the Policy Gradient family. The basic idea is to learn directly
the policy instead of the :math:`Q(s,a)` function. The intuition is that actions that led to a higher
reward at the end of the episode should be reinforced, i.e., encouraged. So, a higher reward as the objective, and
the actions leading to it, in the context of neural networks, hints that this should follow a Gradient **Ascent** instead
of Gradient **Descent** as normally done in supervised learning.

The policy gradient then is estimated by the following formula:

.. math::

    \hat{g} = \frac{1}{|D|}\sum_{\tau \in D}\sum_{t=0}^{T}\nabla_\theta log \pi_\theta (a_t | s_t)R(\tau)

Where :math:`D` is the set of trajectories, :math:`T` is a single trajectory composed of :math:`t_n` time steps,
:math:`R(\tau)` is the reward obtained at trajectory :math:`\tau`, and :math:`\sum_{t=0}^{T}\nabla_\theta log \pi_\theta (a_t | s_t)`
represent the Grad-Log-Prob of a trajectory.

The "loss" function is simply the multiplication of the log probabilities of the batch actions and the corresponding
weight or episode reward. So in essence, the actions of high reward episodes are *reinforced* whereas low reward episode
actions are *discouraged*.

The training procedure is roughly as follows:

#. Collect trajectories using the current policy (neural network).
#. Prepare a batch using the collected trajectories. Each step will use the episode reward as "weight".
    - In case of using reward to go, the episode reward is redistributed evenly across all episode steps.
#. Run a training step of the neural network.
    #. Predict log probabilities of actions from observations.
    #. Compute the loss using the policy gradient formula.
    #. Optimize via Gradient Ascent with the obtained loss.
#. Repeat for N epochs.

Algorithm
---------

.. automodule:: relezoo.algorithms.reinforce.core

.. autoclass:: Reinforce
    :members:
    :special-members:

Discrete Policies
-----------------

This policy has an optional :math:`e`-greedy exploration mechanism

.. automodule:: relezoo.algorithms.reinforce.discrete

.. autoclass:: ReinforceDiscretePolicy
    :members:
    :special-members:

Continuous Policies
-------------------
.. automodule:: relezoo.algorithms.reinforce.continuous

.. autoclass:: ReinforceContinuousPolicy
    :members:
    :special-members:


.. [VPG] `Vanilla Policy Gradients <https://spinningup.openai.com/en/latest/algorithms/vpg.html>`