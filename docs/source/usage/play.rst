Play
====

Play is the mode of using a baseline or checkpoint to load a policy and use it against an environment.
There is no training or logging involved. This mode is basically for enjoyment or evaluation purposes.

Run a Play Session
------------------
By default, ReleZoo runs in training mode. To change this behavior provide the option ``context.mode=play``.
However, play mode makes sense when using a checkpoint or a baseline since the idea is to watch a smart agent
play!. So the full command shouls look like this:

.. code-block:: console

   relezoo-run context.mode=play context.checkpoints=baselines/reinforce/classic_control/cartpole/

**Notes**
- Play mode only uses the ``environments@env_test`` property, so you only need to change this when playing against
other environments.
- Play also generate logs for later inspection.

This will trigger the default environment and algorithm in play or inference mode using the provided
checkpoint as loaded model. The rest of the options can be modified as well.

You must carefully select the correct combination of algorithms, environments and checkpoints, otherwise
the results are not guaranteed.