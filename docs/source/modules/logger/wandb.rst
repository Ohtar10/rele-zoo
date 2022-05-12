Wandb
=====

Wandb Logging Class
-------------------

`Wandb <https://wandb.ai/>`_ wrapper logging mechanism honoring the :py:class:`relezoo.logging.base.Logging` contract.

Usage example:

.. code-block::

   import relezoo.logging.wandb.WandbLogging

   config = {'epochs': 10, 'learning_rate': 1e-3}
   logger = WandbLogging(config=config, project='ReleZoo', name='my-experiment-name')
   logger.log_scalar(name='accuracy', data=0.85, step=1)

You can also pass any `Wandb init <https://docs.wandb.ai/ref/python/init>`_ accepted parameter in the constructor.

.. automodule:: relezoo.logging.wandb

.. autoclass:: WandbLogging
    :members:
    :special-members:
