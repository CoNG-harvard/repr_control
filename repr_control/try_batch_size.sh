#!/bin/bash

# Ensure the script stops if any command fails
set -e

# Optional: Activate a virtual environment (uncomment if needed)
# source /path/to/your/venv/bin/activate

# Running multiple Python scripts
echo "Running script1.py..."
python solve_vec_network_vmap.py --seed 146 --batch_size 64 --rf_num 1024 --rf_sigma 0.0 --max_timesteps 5e3



# Running multiple Python scripts
echo "Running script1.py..."
python solve_vec_network_vmap.py --seed 146 --batch_size 128 --rf_num 1024 --rf_sigma 0.0 --max_timesteps 5e3


# Running multiple Python scripts
echo "Running script1.py..."
python solve_vec_network_vmap.py --seed 146 --batch_size 256 --rf_num 1024 --rf_sigma 0.0 --max_timesteps 5e3

# Running multiple Python scripts
echo "Running script1.py..."
python solve_vec_network_vmap.py --seed 146 --batch_size 512 --rf_num 1024 --rf_sigma 0.0 --max_timesteps 5e3





# Optional: Deactivate virtual environment (uncomment if used)
# deactivate

echo "All scripts executed successfully."
