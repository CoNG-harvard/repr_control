#!/bin/bash

# Ensure the script stops if any command fails
set -e

# Optional: Activate a virtual environment (uncomment if needed)
# source /path/to/your/venv/bin/activate

# Running multiple Python scripts
echo "Running script1.py..."
python solve_vec_randomized_vmap_replay.py --seed 185 --has_feature_step --max_timesteps 5000



# Running multiple Python scripts
echo "Running script1.py..."
python solve_vec_randomized_vmap_replay.py --seed 186 --has_feature_step --max_timesteps 5000


# Running multiple Python scripts
echo "Running script1.py..."
python solve_vec_randomized_vmap_replay.py --seed 187 --has_feature_step --max_timesteps 5000

echo "Running script1.py..."
python solve_vec_randomized_vmap_replay.py --seed 188 --has_feature_step --max_timesteps 5000



# Running multiple Python scripts
echo "Running script1.py..."
python solve_vec_randomized_vmap_replay.py --seed 189 --has_feature_step --max_timesteps 5000


# # Running multiple Python scripts
# echo "Running script1.py..."
# python solve_vec_randomized_vmap.py --seed 175 --has_feature_step --max_timesteps 3000




# Optional: Deactivate virtual environment (uncomment if used)
# deactivate

echo "All scripts executed successfully."
