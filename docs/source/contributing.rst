How to Contribute
=================

Contributions to THzTools are welcome!
Here are some ways you can contribute:

* Create a
  `discussion topic <https://github.com/dodge-research-group/thztools/discussions>`_
  if you have an idea for a new feature or a general topic for
  discussion.
* Create an `issue <https://github.com/dodge-research-group/thztools/issues>`_
  if you find a bug with THzTools, an issue with its documentation, or have
  a suggestion for improving the package. New feature requests are also welcome!
* If you would like to make improvements to the source code or
  documentation then you may do so directly by opening a
  `pull request <https://github.com/dodge-research-group/thztools/pulls>`_.
  Please review the following section for details about the development process.

Please review our `code of conduct <https://github.com/dodge-research-group/thztools/blob/main/CODE_OF_CONDUCT.md>`_
and follow the guidelines described there to help us maintain clear and
respectful communication.

Development Process
-------------------

Please review the `Contributing to a project
<https://docs.github.com/en/get-started/exploring-projects-on-github
/contributing-to-a-project>`_ instructions on GitHub if you would like to
contribute to the THzTools code base and documentation.

Development Tools
-----------------

THzTools development relies on the following tools:

- Unit tests: `pytest <https://pytest.org>`_
- Code coverage: `coverage <https://coverage.readthedocs.io>`_
- Documentation builder: `Sphinx <https://www.sphinx-doc.org>`_,
  using the `PyData Sphinx Theme
  <https://pydata-sphinx-theme.readthedocs.io>`_ and the
  `numpydoc <https://numpydoc.readthedocs.io>`_ extension
- Documentation tests:
  `doctest <https://docs.python.org/3/library/doctest.html>`_
- Linter: `ruff <https://docs.astral.sh/ruff/linter/>`_
- Code format: `black <https://black.readthedocs.io>`_
- Type checker: `mypy <https://mypy.readthedocs.io>`_

Configuration metadata for each of these tools is stored in ``pyproject.toml``.

Development Environment
-----------------------

THzTools has been developed with the `Hatch <https://hatch.pypa.io>`_ and
`Conda <https://docs.conda.io/en/latest/>`_. Brief instructions for their
installation and use are below.

Hatch
^^^^^

Follow the instructions at this `link <https://hatch.pypa.io/latest/install/>`__
to install Hatch.

Enter the environment by typing the following command from the top
directory of the project (ie, ``~/thztools``)::

  hatch shell

This will update any changes to the dependencies and enable several script
shortcuts that can be used for testing, linting, and building documentation.

To execute tests with ``pytest``::

  hatch run test

To lint the code base with ``ruff``::

  hatch fmt

To build the documentation with ``sphinx`` and test examples with ``doctest``::

  hatch run docs:build

To exit the ``hatch`` environment::

  exit

Conda
^^^^^

Follow the instructions at this
`link <https://docs.anaconda.com/free/miniconda/>`__ to install Miniconda, a
lightweight version of the Conda package and environment manager.

Create the environment by typing the following command from the top directory
of the project (ie, ``~/thztools``). The first line of the
``environment-dev.yml`` file sets the environment name to ``thztools``, and the
last line uses ``pip`` to install the ``thztools`` package in
`editable <https://pip.pypa.io/en/stable/cli/pip_install/#install-editable/>`_
mode for development. ::

  conda env create -f envs/environment-dev.yml

To activate the ``thztools`` environment::

  conda activate thztools

To deactivate the environment when you are finished using ``thztools``::

  conda deactivate

More detailed instructions on using Conda to manage environments are available
`here <https://docs.conda.io/projects/conda/en/stable/user-guide/tasks
/manage-environments.html>`_.
