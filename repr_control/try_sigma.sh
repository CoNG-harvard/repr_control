#!/bin/bash

# Ensure the script stops if any command fails
set -e

# Optional: Activate a virtual environment (uncomment if needed)
# source /path/to/your/venv/bin/activate

# Running multiple Python scripts
echo "Running script1.py..."
python solve_vec_network_vmap.py --seed 145 --batch_size 32 --rf_num 1024 --rf_sigma 0.0 --max_timesteps 5e3

echo "Running script2.py..."
python solve_vec_network_vmap.py --seed 145 --batch_size 32 --rf_num 1024 --rf_sigma 0.25 --max_timesteps 5e3

# Add more scripts as needed
echo "Running script3.py..."
python solve_vec_network_vmap.py --seed 145 --batch_size 32 --rf_num 1024 --rf_sigma 0.75 --max_timesteps 5e3


# Add more scripts as needed
echo "Running script4.py..."
python solve_vec_network_vmap.py --seed 145 --batch_size 32 --rf_num 1024 --rf_sigma 1.25 --max_timesteps 5e3

# Add more scripts as needed
echo "Running script5.py..."
python solve_vec_network_vmap.py --seed 145 --batch_size 32 --rf_num 1024 --rf_sigma 0.4 --max_timesteps 5e3

# Add more scripts as needed
echo "Running script6.py..."
python solve_vec_network_vmap.py --seed 145 --batch_size 32 --rf_num 1024 --rf_sigma 0.6 --max_timesteps 5e3

# Optional: Deactivate virtual environment (uncomment if used)
# deactivate

echo "All scripts executed successfully."
