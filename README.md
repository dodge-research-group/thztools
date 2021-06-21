# Data analysis tools for terahertz time-domain spectroscopy

Project to compute estimates for noise model parameters and transfer function parameters.

## Installation

This project is based on the Cookiecutter Data Science [template](https://github.com/drivendata/cookiecutter-data-science). See [here](https://drivendata.github.io/cookiecutter-data-science/) for documentation.

Make sure you create and activate a virtual environment for development with `thztools`. The instructions below are for the Conda package manager.

1. Create the environment by typing the following command from the top directory. The first line of the `environment.yml` file sets the name to `thztools`, and the last line uses `pip` to install the `thztools` package in [editable mode](https://pip.pypa.io/en/stable/cli/pip_install/#install-editable) for development.

    ```
    conda env create -f environment.yml
    ```
2. Activate the `thztools` environment.

    ```
    conda activate thztools
    ```

3. Use the following command to run a Jupyter notebook server from within this environment.

    ```
    jupyter notebook
    ```

4. Deactivate the environment when you are finished using `thztools`.

    ```
    conda deactivate
    ```

More detailed instructions on using Conda to manage environments are available [here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).