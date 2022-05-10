Usage
=====

.. _installation:

Installation
------------

To use Relezoo, first install it using pip:

.. code-block:: console

    git clone https://github.com/Ohtar10/rele-zoo.git
    cd rele-zoo
    make install-env
    conda activate rele-zoo
    make install

Run the default experiment
--------------------------
By default, relezoo will train a REINFORCE algorithm against cartpole environment,
so by just invoking ``relezoo-run`` you can check everything is working fine:

.. automodule:: relezoo.algorithms.base

.. autoclass:: Algorithm
    :members:

.. automodule:: relezoo.utils.structure

.. autoclass:: Context
    :members:
