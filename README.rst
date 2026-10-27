:maintainers:
  `andrewtarzia <https://github.com/andrewtarzia/>`_

Overview
========

A series of case studies showing the first example of toy-model `cgx``-driven
cage structure prediction using graph enumeration.

This project is built on
`cgx <https://cgexplore.readthedocs.io/en/latest/>`_ and
`stk <https://stk.readthedocs.io/en/stable/>`_.


Installation
============

Using ``conda``/``mamba`` (this will give the exact environment used in the paper):

.. code-block:: bash

  mamba env create -f environment.yml

This will install the ``model_enumeration`` in a development state allowing
you to edit the source code, and will install the required dependencies
`openmm <https://openmm.org/>`_ and
`openmmtools <https://openmmtools.readthedocs.io/en/stable/gettingstarted.html>`_
using ``mamba``.

Then use ``just`` to setup the environment:

.. code-block:: bash

  just dev

CREST must be downloaded from
`crest <https://crest-lab.github.io/crest-docs/page/documentation/keywords.html>`_


Projects scripts
================

From ``model_enumeration``, you get the following scripts available in the
command line.

Important:

    In each script, you will have to update the environment variables, namely
    the paths where the data is written to and where some software is.

Each script should be self-contained and once you have installed the
environment with ``just dev``, you should have them as executables. They can
then be run and produce the output used in the manuscript.
Not all scripts were used in the final version, and will be inspirations for
recipes and examples in ``cgx``.

``ligand_prep``
    This script performs conformer analysis of atomistic building blocks to
    collate the minimal model forcefield parameters.

``tri_di_systems``
    This script introduces the parameter optimisation of the minimal model
    forcefield parameters and applies it to the prediction of ditopic +
    tritopic cage structures. Not all aspects were used in the final
    manuscript.

``atomistic_systems``
    Atomistic structure prediction shown at the end of the final manuscript.

``cu18_systems``
    Genetic algorithm-based structure prediction of M9 heteroleptic cage
    structures.

``starship_systems``
    Reproduces the minimal model prediction of starship structures.

``mn_l2n_systems``
    Not mentioned in the manuscript, this script reproduces the relationship
    between preferred MnL2n cages and the bite angle.

``write_chemiscope``
    Produces the dataset visualisation using chemiscope of the output of
    the atomistic case study and genetic algorithm case study.

``rh_isomer_systems``
    This script was not used in the final manuscript but predicts the
    structures of heteroleptic M6L6L6 metal-organic cages.

Acknowledgements
================

Funded by the European Union - Next Generation EU, Mission 4 Component 1
CUP E13C22002930006.
