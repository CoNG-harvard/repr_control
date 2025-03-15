#!/bin/bash

# Ensure the script stops if any command fails
set -e

# Optional: Activate a virtual environment (uncomment if needed)
# source /path/to/your/venv/bin/activate

# # Running multiple Python scripts
# echo "Running script1.py..."
# python solve_vec_randomized_vmap_replay.py --seed 1025 --has_feature_step --max_timesteps 40000



# Running multiple Python scripts
echo "Running script1.py..."
python solve_vec_randomized_vmap_replay.py --seed 1026 --has_feature_step --max_timesteps 40000


# Running multiple Python scripts
echo "Running script1.py..."
python solve_vec_randomized_vmap_replay.py --seed 1027 --has_feature_step --max_timesteps 40000

echo "Running script1.py..."
python solve_vec_randomized_vmap_replay.py --seed 1028 --has_feature_step --max_timesteps 40000



# Running multiple Python scripts
echo "Running script1.py..."
python solve_vec_randomized_vmap_replay.py --seed 1029 --has_feature_step --max_timesteps 40000


# # Running multiple Python scripts
# echo "Running script1.py..."
# python solve_vec_randomized_vmap.py --seed 175 --has_feature_step --max_timesteps 3000




# Optional: Deactivate virtual environment (uncomment if used)
# deactivate

echo "All scripts executed successfully."
