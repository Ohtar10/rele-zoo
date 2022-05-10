Train
=====

Run a Training Session
----------------------
As seen in :ref:`installation`, relezoo has a default model and environment configured, the following
command would be the equivalent:

.. code-block:: console

   relezoo-run environments@env_train=cartpole environments@env_test=cartpole algorithm=reinforce-discrete \
   logger=tensorboard context.mode=train context.epochs=50 context.render=false context.eval_every=1 \
   context.mean_reward_window=100 context.checkpoints=checkpoints/

As you can see, you can specify several arguments that would modify the behavior of the run. You can see
the complete list of parameters via ``relezoo-run --help``. However, some components may vary depending on which
major components you choose.

ReleZoo Major Components
------------------------
ReleZoo considers three major components for execution:

- :ref:`environments`: Corresponding to the task environments for the RL agent to train and/or evaluate.
