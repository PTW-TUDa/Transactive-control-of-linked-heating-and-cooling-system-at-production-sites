Transactive control of linked heating and cooling system at production sites
============================================================================

The *Transactive control of linked heating and cooling system at production sites* repository, developed by Fabian Borst as a member of the team of the `ETA-Fabrik <https://www.ptw.tu-darmstadt.de>`_ at Technical University of Darmstadt, is a python project implementing a multi agent system for the transactive control of the thermal supply systems. The concept is part of the correponding doctoral dissertation and validated for the supply systems at the ETA Research factory.

Project structure
-----------------

The project consists of three packages:

- The **experiments** package implements controllers and environments for all virtual and real-world experiments based on the *eta_utility* framework. It follows the standard structure of *eta_x* projects.
- The **multi_agent_system** package holds implementations of all agents. Agents can be traders or markets. Trader agents consist of capacity and pricing assessment models which are imported from the subfolder *models*. All agents (traders and markets) extend the class *base_agent* and all traders extend the class *trader*.
- The **supplementary_material** package holds all other implementations and informations around the doctoral dissertation (e.g. plotting scripts).

Citing this project
--------------------

Please cite this project using our publication:

.. code-block::

    Borst, F. (2024) Transactive control of linked heating and cooling system at production sites (Doctoral dissertation). Technische Universität Darmstadt. Fachbereich Maschinenbau. Unpublished.