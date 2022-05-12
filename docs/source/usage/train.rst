Train
=====

**Note:** You can find the results for each run in the ``output`` folder at the same folder the command
was executed.

Run a Training Session
----------------------
As seen in :ref:`installation`, relezoo has a default model and environment configured, the following command would be the equivalent:

.. code-block:: console

   relezoo-run environments@env_train=cartpole \
   environments@env_test=cartpole \
   algorithm=reinforce-discrete \
   logger=tensorboard \
   context.mode=train \
   context.epochs=50 \
   context.render=false \
   context.eval_every=1 \
   context.mean_reward_window=100 \
   context.checkpoints=checkpoints/

As you can see, you can specify several arguments that would modify the behavior of the run. You can see the complete list of parameters via ``relezoo-run --help``.
However, some components may vary depending on which :ref:`relezoo_major_components` you choose.

For example, you can run the same algorithm but for the ``acrobot`` environment instead:

.. code-block:: console

   relezoo-run environments@env_train=acrobot \
   environments@env_test=acrobot

Because we only want to run the default algorithm, ``reinforce``, we only need to change the environment. However, since
in ReleZoo the train and test environments can be different, we must specify both environments.

Similarly, we can run the same environment but using a different algorithm:

.. code-block:: console

   relezoo-run algorithm=xentropy-discrete

Using Parallel Environments
---------------------------
ReleZoo has the posibility to run multiple instances of the same environment. For this you can provide the parallel
version of the desired environment:

.. code-block:: console

   relezoo-run environments@parallel-cartpole env_train.num_envs=10

The above command will run a training session with the default algorithm using a parallel version of the cartpole
environment with a total of 10 environments. This will allow the training to collect observations from 10 different
agents simultaneously. However, because we did not changed the ``env_test`` property, the evaluation will still
run in the single agent version of it. This is desirable for example when we want to render a single agent during
evaluation, or if we want to evaluate the agent performance on a slightly different environment.

Running Parallel Jobs
---------------------
You can also run several jobs at the same time. For example running the experiment with different seeds.

**Note:** The default launcher for parallel jobs is based in joblib.

.. code-block:: console

   relezoo-run --multirun context.seed=123,456,789

This will create three different jobs, each with their corresponding seed, the jobs will run in parallel up to
the default concurrency level (2). You can modify this by specifying the launcher property:

.. code-block:: console

   relezoo-run --multirun context.seed=123,456,789,147,258 hydra/launcher.n_jobs=-1

This will create five jobs, and because of the property ``hydra/launcher.n_jobs=-1``, all of them will
run in parallel.

You can mix jobs in parallel with parallel environments, but be careful, this can spawn a significant amount
of processes in your system. For example, 5 jobs running in parallel, each of them using 10 parallel environments
will immediately result in 50 processes for your system.

Changing the Logging Mechanism
------------------------------

If you want to use wandb for example, you can change the property in the run command. However you should also
provide additional properties to initialize it properly. For example:

.. code-block:: console

   relezoo-run logger=wandb logger.project=ReleZoo logger.name=my-experiment-name

This way you will tell to which project you want to submit the metrics and the corresponding experiment name.

Combining Parallel Jobs with Wandb
----------------------------------
Depending on the library versions, there might be some issues when mixing Joblib and Wandb that might prevent you
from running parallel experiments with this logging mechanism. If you encounter an error like:
``ValueError: cannot find context for 'loky'`` chances are you have this problem.
You can check `this issue <https://github.com/wandb/client/issues/1525>`_ for more details.
You can bypass it by specifying the environment variable ``WANDB_START_METHOD="thread"``
or running the command like this:

.. code-block:: console

    WANDB_START_METHOD="thread" relezoo-run --multirun context.sedd=123,456,789


Running Hyperparameter tuning
-----------------------------
ReleZoo relies on `hydra <https://hydra.cc/>`_ for configuration composition, launcher and sweeping. As for sweeping,
ReleZoo uses `Ax <https://ax.dev/>`_ for hyper parameter exploration. However, because each algorithm have different
parameters, there must exist a configuration per algorithm. You can invoke them like this:

.. code-block:: console

   relezoo-run --multirun hydra/sweeper=ax-reinforce hydra.launcher.n_jobs=-1


Running in Headless Mode
------------------------
Rendering the evaluation rollouts is optional and is disabled by default. This is controlled with the property
``context.render``. If you don't care about rendering, then there is no change needed for running in headless
mode, i.e., on a remote server. However, if you want/need to run on a remote server and need to render the
rollouts, you can do the following:

#. Install `xvfb` package. For example: ``sudo apt install -y xvfb libglu1-mesa libglu1-mesa-dev``
#. Run the command creating a virtual display:

.. code-block:: console

  xvfb-run -s "-screen 0 1400x900x24" relezoo-run


Checkpoints
-----------

Resuming work
-------------
