Contributing Guide
==================

TODO: fix links

Scanpy provides extensive developer documentation, most of which applies
to this project as well. This document will not reproduce the entire
content from there but aims to summarize the most important information
to get you started on contributing.

We assume that you are already familiar with git and making pull requests
on GitHub. If not, please refer to the Scanpy developer guide.

Installing Development Dependencies
-----------------------------------

In addition to the packages needed to *use* this package, you need additional
Python packages to *run tests* and *build the documentation*. It's easy to
install them using pip:

.. code-block:: bash

    cd cell_flow_perturbation
    pip install -e ".[dev,test,doc]"

Code Style
----------

This package uses pre-commit to enforce consistent code styles. On every commit,
pre-commit checks will either automatically fix issues with the code or raise
an error message.

To enable pre-commit locally, simply run:

.. code-block:: bash

    pre-commit install

in the root of the repository. Pre-commit will automatically download all
dependencies when it is run for the first time.

Alternatively, you can rely on the pre-commit.ci service enabled on GitHub.
If you didn't run pre-commit before pushing changes to GitHub, it will
automatically commit fixes to your pull request or show an error message.

If pre-commit.ci added a commit on a branch you were still working on
locally, simply use:

.. code-block:: bash

    git pull --rebase

to integrate the changes into your work. While pre-commit.ci is useful,
we strongly encourage installing and running pre-commit locally first to
understand its usage.

Finally, most editors have an *autoformat on save* feature. Consider
enabling this option for `Ruff`_ and `Prettier`_.

.. _Ruff: https://docs.astral.sh/ruff/integrations/
.. _Prettier: https://prettier.io/docs/en/editors.html

Writing Tests
-------------

.. note::
    Remember to first install the package with `pip install -e '.[dev,test]'`.

This package uses pytest for automated testing. Please write tests for every
function added to the package.

Most IDEs integrate with pytest and provide a GUI to run tests. Alternatively,
you can run all tests from the command line by executing:

.. code-block:: bash

    pytest

in the root of the repository.

Continuous Integration
~~~~~~~~~~~~~~~~~~~~~~

Continuous integration will automatically run the tests on all
pull requests and test against the minimum and maximum supported
Python versions.

Additionally, there's a CI job that tests against pre-releases of
all dependencies (if there are any). The purpose of this check is
to detect incompatibilities of new package versions early on and
give you time to fix the issue or reach out to the developers of
the dependency before the package is released to a wider audience.

Publishing a Release
--------------------

Updating the Version Number
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before making a release, you need to update the version number in
the `pyproject.toml` file. Please adhere to Semantic Versioning, in brief:

    Given a version number MAJOR.MINOR.PATCH, increment the:

    1. MAJOR version when you make incompatible API changes,
    2. MINOR version when you add functionality in a backwards-compatible manner, and
    3. PATCH version when you make backwards-compatible bug fixes.

    Additional labels for pre-release and build metadata are available as
    extensions to the MAJOR.MINOR.PATCH format.

Once you are done, commit and push your changes and navigate to the
"Releases" page of this project on GitHub. Specify `vX.X.X` as a tag name and
create a release. For more information, see `managing GitHub releases`. This will
automatically create a git tag and trigger a GitHub workflow that creates a release on PyPI.

Writing Documentation
----------------------

TODO

Tutorials with myst-nb and Jupyter Notebooks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TODO

Hints
~~~~~

- If you refer to objects from other packages, please add an entry
  to `intersphinx_mapping` in `docs/conf.py`. Only by doing so can Sphinx
  automatically create a link to the external documentation.

- If building the documentation fails because of a missing link that
  is outside your control, you can add an entry to the `nitpick_ignore`
  list in `docs/conf.py`.



Building the Docs Locally
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    cd docs
    make html
    open _build/html/index.html
