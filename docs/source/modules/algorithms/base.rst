.. _algorithms:

Algorithms & Policies
=====================

The Policy and the training Algorithm are separate entities. The Algorithms contain the necessary
code to manage the training loop, rollout collection, learn trigger, and logging, whereas the
Policy limits to take action given an observation and defining its learn routine.

This facilitates mixin up a training procedure with different policy variations as well as using
the policy as a first class citizen.

Algorithm
---------

.. automodule:: relezoo.algorithms.base

.. autoclass:: Algorithm
    :members:
    :special-members:

Policy
------
.. autoclass:: Policy
    :members:
    :special-members:

