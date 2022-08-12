Components
==========

.. _relezoo_major_components:

ReleZoo Major Components
------------------------
ReleZoo considers three major components for its execution:

- :ref:`environments`: Corresponding to the task environments for the RL agent to train and/or evaluate.
                       The idea is to make then pluggable to the algorithm in question so an algorithm can be tried
                       with several environment in the same way.
- :ref:`algorithms`: Corresponding to the actual RL algorithm for the given task.
- :ref:`logger`: Corresponding to the logging backend for the experiment.

The base classes for each ensure they can communicate with each other using a common contract, so it should be relatively
easy to swap environments, algorithms and loggers for different runs. For example to use the same algorithm to run
on two different environments. However, the actual implementation of each is up to the developer, so you are free to
customize any of the major components by implementing your own! you just need to inherit from the base classes
so the contract between them is honored and they can communicate.

