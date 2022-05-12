.. _logger:

Logger
=======

Logger is one of the three major components of ReleZoo and it represents the logging
mechanism used during an experiment run.

All the logging mechanisms or backends should inherit from the base logging class
:py:class:`relezoo.logging.base.Logging`, the class acts as a contract and so it will
be transparent from the :py:class:`relezoo.algorithm.base.Algorithm` perspective while
logging.

Logging Base Class
------------------
It serves as a common contract between :py:class:`relezoo.algorithm.base.Algorithm` and
the preferred logging backend.

.. automodule:: relezoo.logging.base

.. autoclass:: Logging
    :members:

