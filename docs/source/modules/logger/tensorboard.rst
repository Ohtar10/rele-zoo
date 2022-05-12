Tensorboard
===========

Tensorboard Logging Class
-------------------------

Basic `Tensorboard <https://www.tensorflow.org/tensorboard/>`_ logging mechanism.

Usage example:

.. code-block::

   import relezoo.logging.tensorbaord.TensorboardLogging

   logger = TensorboardLogging('torch.utils.tensorboard.SummaryWriter', log_dir='some/folder')
   logger.log_scalar(name='accuracy', data=0.85, step=1)

.. automodule:: relezoo.logging.tensorboard

.. autoclass:: TensorboardLogging
    :members:
    :special-members:
