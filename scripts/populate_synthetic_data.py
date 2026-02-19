#!/usr/bin/env python3
"""
Populate simdata/ with synthetic T1D data.
Use this when you don't have access to real datasets (AZT1D, OhioT1DM, etc.)
or want to test with richer data than the default 3-point static file.
"""

import json
import os
import sys

# Add project root to path
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root)

from data_loader.synthetic_data import generate_synthetic_t1d_data


def main():
    simdata_dir = os.path.join(root, 'simdata')
    glucose_data, training_pairs, profile = generate_synthetic_t1d_data(
        simdata_dir,
        n_glucose_points=500,
        n_training_pairs=400,
    )

    glucose_path = os.path.join(simdata_dir, 'glucose.json')
    with open(glucose_path, 'w') as f:
        json.dump(glucose_data, f, indent=2)

    print(f"Wrote {len(glucose_data)} glucose readings to {glucose_path}")
    print(f"Profile: target_bg={profile['target_bg']}, sens={profile['sens']}, carb_ratio={profile['carb_ratio']}")


if __name__ == '__main__':
    main()
