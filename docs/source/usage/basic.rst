Basic
=====

.. _installation:

Installation
------------
***Note:** To use the baselines, you need [Git LFS](https://git-lfs.github.com/) extension installed so you can
pull the binaries from the repo.

To use Relezoo, you must clone the repo and install it via Make:

.. code-block:: console

    git clone https://github.com/Ohtar10/rele-zoo.git
    cd rele-zoo
    make install-env
    make install
    conda activate rele-zoo

Or directly via pip:

.. code-block:: console

    git clone https://github.com/Ohtar10/rele-zoo.git
    cd rele-zoo
    conda env create -f environment.yaml
    conda activate
    pip install .

If you want to use the baselines and you cloned the repository before installing Git LFS, after installing it,
simply run ``git lfs fetch`` or ``git lfs pull`` inside local repository to fetch the actual binaries.


Verify Installation
--------------------------
To check if the installation works works good, first check the ``relezoo-run`` command, and then
execute the default run.

.. code-block:: console

   relezoo-run --help
   == Relezoo-run ==

   This is Relezoo-run!
   You can change different experiment
   configuration groups by appending
   == Configuration groups ==
   Compose your configuration from those groups (algorithm=reinforce)
   By default, relezoo will train a REINFORCE algorithm against cartpole environment,
   so by just invoking ``relezoo-run`` you can check everything is working fine...

By default, relezoo is configured to run a ``reinforce`` algorithm on a single ``cartpole`` environment,
so you can just run the program without any arguments:

.. code-block:: console

   relezoo-run
   [2022-05-10 20:49:01,297][relezoo.engine.runner][INFO] - Running Training Session with config:
   [2022-05-10 20:49:01,300][relezoo.engine.runner][INFO] - ...
   [2022-05-10 20:49:01,301][relezoo.engine.runner][INFO] - Press CTRL + C to cancel the run
   [2022-05-10 20:49:01,301][relezoo.engine.runner][INFO] - A checkpoint will be saved automatically after a successful run or cancel.
   100%|████████████████████████████████████████████████████| 50/50 [00:19<00:00,  2.62it/s, loss=3.35, mean_batch_score=84.50, mean_batch_ep_length=84.50, mean_reward_100=32.82]
   [2022-05-10 20:49:22,318][relezoo.engine.runner][INFO] - Training finished -- Mean reward over 100 epochs: 32.82
   [2022-05-10 20:49:22,318][relezoo.engine.runner][INFO] - Saving current progress...

If you see a similar output, everything is working fine.

Development Mode Install
------------------------

For development and testing, i.e., developing new models or experiments and running the test suites
you can install in dev mode:

.. code-block:: console

    git clone https://github.com/Ohtar10/rele-zoo.git
    cd rele-zoo
    make install-env
    make install-dev
    conda activate rele-zoo

Or directly via pip:

.. code-block:: console

    git clone https://github.com/Ohtar10/rele-zoo.git
    cd rele-zoo
    conda env create -f environment.yaml
    conda activate
    pip install ".[dev]"
