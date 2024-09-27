.. _home:

Welcome to repr-control's documentation!
========================================

**repr-control** is a toolbox to solve nonlinear stochastic control via representation learning. 
User can simply input the **dynamics, rewards, initial distributions** (See :ref:`run_samples` for sample code) of the nonlinear control problem
and get the optimal controller parametrized by a neural network.

The optimal controller is trained via Spectral Dynamics Embedding Control (SDEC) algorithm based on representation learning and reinforcement learning.
For those interested in the details of SDEC algorithm, please check our `papers <https://arxiv.org/abs/2304.03907>`_.

Check out the :doc:`installation` instructions, and the :doc:`usage` section for further information.

We also provide standard RL baselines including `soft actor-critic <https://proceedings.mlr.press/v80/haarnoja18b>`_.
Please check out :doc:`/rl_intro` for 

.. note::

   This project is under active development.

Contents
--------

.. toctree::
   installation
   rl_intro
   usage
