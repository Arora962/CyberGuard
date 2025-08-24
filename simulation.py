import subprocess
import os
from datetime import datetime
import json

# Debug: Check for oref0-determine-basal.js existence
def check_script_exists(oref0_dir, script_rel_path):
    script_abs_path = os.path.join(oref0_dir, script_rel_path)
    print(f"Checking for script at: {script_abs_path}")
    if not os.path.isfile(script_abs_path):
        print(f"ERROR: Script not found at {script_abs_path}")
    else:
        print(f"Script found at {script_abs_path}")
    return script_abs_path

# Paths
root = os.path.dirname(os.path.abspath(__file__))
simdata = os.path.join(root, 'simdata')
pred_dir = os.path.join(root, 'predictions')



# Use relative paths from oref0 directory
oref0_dir = os.path.join(root, 'oref0')
iob_json = os.path.relpath(os.path.join(simdata, 'iob.json'), oref0_dir)
currenttemp_json = os.path.relpath(os.path.join(simdata, 'currenttemp.json'), oref0_dir)
glucose_json = os.path.relpath(os.path.join(simdata, 'glucose.json'), oref0_dir)
profile_json = os.path.relpath(os.path.join(simdata, 'profile.json'), oref0_dir)

# Output file
now = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
pred_file = os.path.join(pred_dir, f'prediction-{now}.json')

# Command to run oref0-determine-basal.js
oref0_script_rel = os.path.join('bin', 'oref0-determine-basal.js')
cmd = ['node', oref0_script_rel, iob_json, currenttemp_json, glucose_json, profile_json]

try:
    # Run from the oref0 directory
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=oref0_dir)
    output = result.stdout
    error = result.stderr
    '''print("--- STDOUT ---")
    print(output)
    print("--- STDERR ---")
    print(error)'''
    if result.returncode != 0:
        print(f"oref0 script exited with code {result.returncode}")
except Exception as e:
    output = f"Error running oref0: {str(e)}"
    error = ''

# Write output to file
with open(pred_file, 'w') as f:
    f.write(output)

print(f"Prediction written to {pred_file}")

with open(pred_file) as f:
    data = json.load(f)
    print("\nPrediction File Output:")
    print(json.dumps(data, indent=4))
