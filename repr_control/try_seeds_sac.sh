#!/bin/bash

# Ensure the script stops if any command fails
set -e

# Optional: Activate a virtual environment (uncomment if needed)
# source /path/to/your/venv/bin/activate

# Running multiple Python scripts
echo "Running script1.py..."
python solve_vec_network_vmap_sac.py --seed 165 --batch_size 32  --max_timesteps 3e3 --discount 0.99



# Running multiple Python scripts
echo "Running script1.py..."
python solve_vec_network_vmap_sac.py --seed 166 --batch_size 32 --max_timesteps 3e3 --discount 0.99


# Running multiple Python scripts
echo "Running script1.py..."
python solve_vec_network_vmap_sac.py --seed 167 --batch_size 32  --max_timesteps 3e3 --discount 0.99

# # Running multiple Python scripts
# echo "Running script1.py..."
# python solve_vec_network_vmap_sac.py --seed 149 --batch_size 32 --max_timesteps 5e3

# # Running multiple Python scripts
# echo "Running script1.py..."
# python solve_vec_network_vmap_sac.py --seed 150 --batch_size 32  --max_timesteps 5e3

# # Running multiple Python scripts
# echo "Running script1.py..."
# python solve_vec_network_vmap_sac.py --seed 151 --batch_size 32  --max_timesteps 5e3




# Optional: Deactivate virtual environment (uncomment if used)
# deactivate

echo "All scripts executed successfully."
