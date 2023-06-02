Getting Started
===================

.. toctree::
   :maxdepth: 2

Installation
+++++++++++++++++++

The instructions below are for developers using the `Conda <https://docs.conda.io/en/latest/>`_ package and environment manager.

1. Create the environment by typing the following command from the top directory. The first line of the **environment-dev.yml** file sets the environment name to **thztools**, and the last line uses **pip** to install the **thztools** package in `editable <https://pip.pypa.io/en/stable/cli/pip_install/#install-editable/>`_ mode for development. ::

    conda env create -f envs/environment-dev.yml

2. Activate the **thztools** environment. ::

    conda activate thztools

3. Use the following command to run a Jupyter notebook server from within this environment. ::

    jupyter notebook

4. Deactivate the environment when you are finished using **thztools**. ::

    conda deactivate

More detailed instructions on using Conda to manage environments are available `here <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html/>`_.

