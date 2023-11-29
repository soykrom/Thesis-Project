import os

# Get the absolute path to the script
script_path = os.path.abspath('actor_sac')
print("Script Path:", script_path)

# Get the directory containing the script
script_dir = os.path.dirname(script_path)
print("Script Directory:", script_dir)

# Get the absolute path to the "models" directory using os.path.join
models_path = os.path.abspath(os.path.join(script_dir, 'models'))
print("Models Path:", models_path)

# Construct the path to "actor_sac"
file_path = os.path.join(models_path, 'sac', 'actor_sac')
print("Final File Path:", file_path)
