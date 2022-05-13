Play
====

Play is the mode of using a baseline or checkpoint to load a policy and use it against an environment.
There is no training or logging involved. This mode is basically for enjoyment or evaluation purposes.

Run a Play Session
------------------
By default, ReleZoo runs in training mode. To change this behavior simply run the program like this:

.. code-block:: console

   relezoo-run context.mode=play context.checkpoints=../../../baselines/reinforce/classic_control/cartpole/cartpole.cpt

This will trigger the default environment and algorithm in play or inference mode using the provided
checkpoint as loaded model.

You must carefully select the correct combination of algorithms, environments and checkpoints, otherwise
the results are not guaranteed.