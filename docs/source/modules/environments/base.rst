.. _environments:

Environments
============

Environments is one of the three major components of ReleZoo and it represent the RL task to solve.
The make a crucial part of the RL process and hence are considered a major component in ReleZoo.

All the environments must inherit from the base class :py:class:`relezoo.environments.base.Environment`
which contains all the general functionality to work properly with the algorithms
:py:class:`relezoo.algorithms.base.Algorithm`.

Environment Base Class
----------------------

The base class serves as a contract between the algorithms and the underlying environments
which by them selves might have different implementation details. The base class
is inspired by the default `OpenAI Gym Env class <https://github.com/openai/gym/blob/master/gym/core.py>`_.


.. automodule:: relezoo.environments.base

.. autoclass:: Environment
    :members:

