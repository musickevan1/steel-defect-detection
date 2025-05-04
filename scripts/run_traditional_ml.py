#!/usr/bin/env python
"""
Wrapper script to run scripts/run_traditional_ml.py
"""

import os
import sys
import subprocess

# Get the directory of this script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Path to the actual script
script_path = os.path.join(script_dir, "scripts", "run_traditional_ml.py")

# Check if the script exists
if not os.path.exists(script_path):
    print(f"Error: Script not found at {script_path}")
    sys.exit(1)

# Run the script with any command-line arguments
cmd = [sys.executable, script_path] + sys.argv[1:]
print(f"Running: {' '.join(cmd)}")
subprocess.run(cmd)
