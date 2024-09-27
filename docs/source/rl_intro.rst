.. _intro:

RL Quickstart
==============

We explain how to initialize training a RL agent in 

Train a RL agent
----------------

Here is a quick start of reinforcement leanring algorithm using `soft actor-critic (SAC) <https://proceedings.mlr.press/v80/haarnoja18b>`_.

.. code-block:: console

    python solve.py --algo sac --env pendulum-v1

Then we delve deeper into how to initialze the actor-critic algorithm.

Create the environement
^^^^^^^^^^^^^^^^^^^^^^^

Environement is where the dynamics is and agent get samples from the environement.

.. literalinclude:: ../../repr_control/solve.py
    :language: python
    :lines: 88
    :linenos:

Create the replay buffer
^^^^^^^^^^^^^^^^^^^^^^^^

Replay buffer is where the transition data is stored. Only off-policy algorithm needs replay buffer. 
Here off-policy means the policy used in sampling might be different from policy we are optimizing.

.. literalinclude:: ../../repr_control/solve.py
    :language: python
    :lines: 94
    :linenos:

Create the agent
^^^^^^^^^^^^^^^^

.. literalinclude:: ../../repr_control/solve.py
    :language: python
    :lines: 104-109
    :linenos: 


Start training
^^^^^^^^^^^^^^

.. literalinclude:: ../../repr_control/solve.py
    :language: python
    :lines: 135-156
    :linenos: 
