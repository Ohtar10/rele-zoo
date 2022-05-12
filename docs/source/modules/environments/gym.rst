.. _gym_environments:

Gym Environments
================

For all gym based environments, you can use the classes below to use in ReleZoo.

GymWrapper
----------
Using this class is like using gym environments directly. However, because it must
obey the :py:class:`relezoo.environments.base.Environment` contract, it will have
some guarantees in terms of observation and action spaces that the :py:class:`relezoo.algorithm.base.Algorithm`
assumes in order to work properly. Example of direct usage:

.. code-block::

   import relezoo.environments.gymutils.GymWrapper

   cartpole = GymWrapper('CartPole-v1')
   obs = cartpole.reset()  # obs.shape == (1, 4)
   cartpole.step(1)

.. automodule:: relezoo.environments.gymutils

.. autoclass:: GymWrapper
    :members:

ParallelGym
-----------
``ParallelGym`` relies on `Ray's <https://www.ray.io/>`_ actor system to create as
many agent instances as requested. This class will simply coordinate and aggregate
the interactions between all environments. The idea is that a fixed number of
actors will be launched and each of them will correspond to a separate instance
of the given gym environment. Because the class obey the contract given by
:py:class:`relezoo.environments.base.Environment`, they integrate effortlessly with
:py:class:`relezoo.algorithm.base.Algorithm`, i.e., no matter if an experiment
uses the single or the parallel version of the environment, the interaction is
consistent and no changes would be needed on either part.

Example of direct usage:

.. code-block::

   import relezoo.environments.parallel.ParallelGym

   cartpole = ParallelGym('CartPole-v1', num_envs=5)
   obs = cartpole.reset()  # obs.shape == (5, 4)
   cartpole.step([1, 0, 0, 1, 1])


.. automodule:: relezoo.environments.parallel

.. autoclass:: ParallelGym
    :members:
