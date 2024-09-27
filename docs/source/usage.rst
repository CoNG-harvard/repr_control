Usage
=====

.. _run_samples:

1. Defining nonlinear dynamics
------------------------------

The dynamics is defined in ``repr_control/define_problem.py``. 
The following items should be defined:

- Dynamics
- Reward function
- Initial distributions
- State and action bounds
- Maximum rollout steps
- Noise level
  
The following is a detailed instruction on how to define the stochastic inverted pendulum dynamics. 

The pendulum dynamics is:

.. math::

   \ddot \theta = \frac{3g}{2l}\sin\theta + \frac{3}{ml^2} T

where :math:`\theta` is the angle, :math:`g` is the gravity constant, :math:`m` is the pendulum mass, :math:`l` is the pendulum length, and :math:`T` is the input torque.
To deal with the unbounded :math:`\theta`, The observation is defined as :math:`[\cos\theta,\sin\theta, \dot \theta]`.

We use euler discretization, combined with the stochastic dynamics,

.. math::

   x' = f(x, u)\Delta t + \epsilon

where :math:`f` is the continuous time nonlinear dynamics, and :math:`\epsilon\sim \mathcal N(0, \sigma^2 I_n)`.


Define problem related constants
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

   .. literalinclude:: define_problem.py
      :language: python
      :lines: 5-18
      :linenos:

   The following constants are defined:

   +--------------+--------------------+----------------------------------------------------------------------------+
   | variable     | format             | meaning                                                                    |
   +==============+====================+============================================================================+
   | state_dim    | int                | state dimension                                                            |
   +--------------+--------------------+----------------------------------------------------------------------------+
   | action_dim   | int                | action dimension                                                           |
   +--------------+--------------------+----------------------------------------------------------------------------+
   | state_range  | [list, list]       | state upper and lower bounds. Sampling will be reset if bound is achieved. |
   +--------------+--------------------+----------------------------------------------------------------------------+
   | action_range | [list, list]       | action upper and lower bounds.                                             |
   +--------------+--------------------+----------------------------------------------------------------------------+
   | max_step     | int                | maximum step per episode.                                                  |
   +--------------+--------------------+----------------------------------------------------------------------------+
   | sigma        | float              | Gaussian noise variance :math:`\sigma^2`.                                  |
   +--------------+--------------------+----------------------------------------------------------------------------+
   | env_name     | str                | Name of the dynamics                                                       |
   +--------------+--------------------+----------------------------------------------------------------------------+

Define dynamics and reward functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
   
   Note that the dynamics must be written in ``pyTorch`` and all the inputs should be ``torch.Tensor``. 
   The dynamics must support **batch operations**, which means 
   the input ``torch.Tensor`` should be in shape ``[batch_size, state_dim]`` and ``[batch_size, action_dim]``.

   Define dynamics:

   .. literalinclude:: define_problem.py
      :language: python
      :lines: 23, 44-59
      :linenos:

   Define rewards:

   .. literalinclude:: define_problem.py
      :language: python
      :lines: 61, 75-79
      :linenos:

2. Start training
-----------------
The training can be started with a single line

.. code-block:: console

      $ python solve.py

Define training hyperparameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The hyper parameters can be set through command line arguments, for example 
  
.. code-block:: console

   $ python solve.py --max_timesteps 2e5 --rf_num 1024


The ``--max_timesteps 2e5`` means the total number of iterations is set to ``2e5``, and ``--rf_num 1024`` means the 
truncated finite dimension of random features are 1024. 

For all the hyperparameters can be tuned, run

.. code-block:: console

   $ python solve.py --help

- Experimental: vectorized solution:

.. code-block:: console

   $ python solve_vec.py --max_timesteps 2e5 --device cuda


use vectorized rollout and evaluation to speed up the training.
   

3. Monitoring and evaluating the training results
----------------------------------

After training starts, the results will look like

.. code-block:: console
   
   repr-control/
   ├── repr-control/
   │   ├── log/ 
   │   │   ├── rfsac/ 
   │   │   │   ├── seed_SEED_DATE-TIME          # folder title
   │   │   │   │   ├── summary/                 # save tensorboard summaries
   │   │   │   │   ├── best_actor.pth           # actor with the best evaluations
   │   │   │   │   ├── best_critic.pth          # critic with the best evaluations
   │   │   │   │   ├── last_actor.pth           # actor after all training steps
   │   │   │   │   ├── last_critic.pth          # critic after all training steps
   └── └── └── └── └── train_params.yaml        # training parameters

Run the follwoing script to evaluate the trained results,

.. code-block:: console

   $ python scripts/eval.py $LOG_PATH

where `$LOG_PATH` is the path of folder title ``seed_SEED_DATE-TIME``.

Monitoring the training process
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: console

   $ tensorboard --logdir $LOG_PATH

You can inspect the training process via tensorboard. 

.. note::

   Monitoring the training process is very helpful for tuning the hyperparameters. 
   Some rules of thumb if you don't have experience playing with the RL hyper parameteters:
   
   - If the value loss is too large, try to scale the rewards to be smaller (or increase the learning rate).
   - If the agent always get stuck, try to adapt the initial distriution to cover more of the state space.

Evaluating the training results: 

.. code-block:: console

   $ python scripts/eval.py $LOG_PATH

I placed a example results in the `examples` folder, you can run the following to see the results,

.. code-block:: console

   $ tensorboard --logdir ./examples/example_results/rfsac/Pendulum/seed_0_2024-07-18-14-50-35

.. code-block:: console

   $ python scripts/eval.py ./examples/example_results/rfsac/Pendulum/seed_0_2024-07-18-14-50-35


4. Use controller elsewhere
----------------------------

   Add the following line to your python code to load training results as a controller,

   .. code-block:: python

      import numpy as np
      from repr_control.scripts.eval import get_controller
      log_path = '$LOG_PATH'
      agent = get_controller(log_path)
   
   To generate control command from states,

   .. code-block:: python

      state = np.zeros([3]) # a sample state with all zero.
      action = agent.select_action(state, explore=False)
 