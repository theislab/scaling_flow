|PyPI| |Downloads| |CI| |Pre-commit| |Codecov| |Docs|

CellFlow - Modeling Complex Perturbations with Flow Matching
=======================================================

.. image:: docs/_static/img/light_mode_concept_revised.png
    :width: 800px
    :align: center
    :class: only-light

.. image:: docs/_static/img/dark_mode_concept_revised.png
    :width: 800px
    :align: center
    :class: only-dark


**CellFlow** is a framework for predicting phenotypic responses
of complex perturbations on single-cell level

Example Applications
--------------------
- Modeling the effect of single and combinatorial drug treatments
- Predicting the phenutyipc response to genetic perturbations
- Modeling perturbed development of organisms
- Cell fate engineering in the brain
- Optimising organoid protocols
- ... and more, check out the `documentation <https://moscot.readthedocs.io>`_ for more information.


Installation
------------
Install **CellFlow** by running::

    pip install cellflow

In order to install **CellFlow** from in editable mode, run::

    git clone https://github.com/theislab/cell_flow_perturbation
    cd cfp
    pip install -e .

For further instructions how to install jax, please refer to https://github.com/google/jax.

Citing CellFlow
---------------
If you find a model useful for your research, please consider citing our preprint `Klein, Fleck, et al., 2025`_.

.. |Codecov| image:: https://codecov.io/gh/theislab/cell_flow_perturbation/branch/master/graph/badge.svg?token=Rgtm5Tsblo
    :target: https://codecov.io/gh/theislab/cellflow
    :alt: Coverage

.. |PyPI| image:: https://img.shields.io/pypi/v/moscot.svg
    :target: https://pypi.org/project/cellflow/
    :alt: PyPI

.. |CI| image:: https://img.shields.io/github/actions/workflow/status/theislab/cell_flow_perturbation/test.yml?branch=main
    :target: https://github.com/theislab/cell_flow_perturbation/actions
    :alt: CI

.. |Pre-commit| image:: https://results.pre-commit.ci/badge/github/theislab/cell_flow_perturbation/main.svg
   :target: https://results.pre-commit.ci/latest/github/theislab/cell_flow_perturbation/main
   :alt: pre-commit.ci status

.. |Docs| image:: https://img.shields.io/readthedocs/cellflow
    :target: https://cellflow.readthedocs.io/en/latest/
    :alt: Documentation

.. |Downloads| image:: https://static.pepy.tech/badge/cell_flow_perturbation
    :target: https://pepy.tech/project/cell_flow_perturbation
    :alt: Downloads

.. _Klein, Fleck, et al., 2025: TODO
