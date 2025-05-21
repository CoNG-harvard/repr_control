# repr-control: A Toolbox to solve stochastic nonlinear control

[![Documentation](https://img.shields.io/badge/Documentation-Online-blue)](https://repr-control-orgnaization.readthedocs.io/en/latest/)

repr-control is a toolbox to solve nonlinear stochastic control via representation learning. 
User can simply input the [**dynamics, rewards, initial distributions**](repr_control/define_problem.py) of the nonlinear control problem
and get the optimal controller parametrized by a neural network.

### Please refer to our [[documentation]](https://repr-control-orgnaization.readthedocs.io/en/latest/) on how to define custom nonlinear control problems and train the controller.

The optimal controller is trained via Spectral Dynamics Embedding Control (SDEC) algorithm based on representation learning and reinforcement learning.
For those interested in the details of the SDEC algorithm, please check our [papers](https://arxiv.org/abs/2304.03907).

## Installation
1. Install Anaconda and git (if you haven't).
2. Create a new environment,
   
   **Windows** : Open Anaconda prompt.   **Mac** or **Linux** : Open Terminal:
    
    ```shell
    conda create -n repr-control python=3.10
    conda activate repr-control
    ```
3. Install PyTorch dependencies. 
  
    **Windows or Linux**: 

    If you have CUDA-compatible GPUs,
    ```shell
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
    ```
    If you don't have CUDA-compatible GPUs,
    ```shell
    conda install pytorch torchvision torchaudio cpuonly -c pytorch
    ```
   **Mac**:
    ```shell
    conda install pytorch::pytorch torchvision torchaudio -c pytorch
    ```
4. install the toolbox
    ```shell
   git clone https://github.com/CoNG-harvard/repr_control.git
   cd repr_control
   pip install -e .
   ```

Helpful resources: 
- [Anaconda environment](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html)
- [PyTorch installation](https://pytorch.org/get-started/locally/)

## Citations
```
@article{ren2023stochastic,
      title={Stochastic Nonlinear Control via Finite-dimensional Spectral Dynamic Embedding}, 
      author={Tongzheng Ren and Zhaolin Ren and Haitong Ma and Na Li and Bo Dai},
      year={2023},
      eprint={2304.03907},
      archivePrefix={arXiv}
}
```
