"""
Loader for AZT1D dataset (Mendeley Data).
Expects CSV files with columns: glucose/CGM, bolus, carbs, etc.
See: https://data.mendeley.com/datasets/gk9m674wcx/1
"""

import csv
import glob
import json
import os
from datetime import datetime


def load_azt1d(data_dir):
    """
    Load AZT1D format data. Supports multiple CSV naming conventions.
    Returns (glucose_data, training_pairs, profile).
    """
    glucose_data = []
    training_pairs = []
    profile = {'target_bg': 110, 'sens': 50, 'carb_ratio': 10}

    csv_files = glob.glob(os.path.join(data_dir, '*.csv'))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    base_ts = int(datetime.now().timestamp() * 1000)

    for path in csv_files[:3]:  # Limit to first 3 files for memory
        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if not rows:
                continue

            # Infer column names (AZT1D may use various names)
            first = rows[0]
            gl_col = next((c for c in first if 'glucose' in c.lower() or 'bgl' in c.lower() or 'cgm' in c.lower() or c == 'sgv'), None)
            bolus_col = next((c for c in first if 'bolus' in c.lower() or 'insulin' in c.lower()), None)
            carb_col = next((c for c in first if 'carb' in c.lower()), None)

            if not gl_col:
                gl_col = next((c for c in first if c in ['value', 'glucose', 'sgv']), list(first.keys())[0])

            for i, row in enumerate(rows):
                try:
                    gl = float(row.get(gl_col, row.get('glucose', row.get('Pre-meal BGL', 120))))
                except (ValueError, TypeError):
                    continue
                gl = max(40, min(400, round(gl)))

                ts = base_ts + i * 5 * 60 * 1000
                glucose_data.append({
                    'date': ts,
                    'glucose': gl,
                    'sgv': gl,
                    'direction': 'Flat',
                    'noise': 1,
                    'filtered': gl,
                    'unfiltered': gl,
                    'rssi': 100,
                    'device': 'azt1d',
                })

                insulin = 0.05
                if bolus_col and row.get(bolus_col):
                    try:
                        insulin = float(row[bolus_col])
                    except (ValueError, TypeError):
                        pass
                elif carb_col and row.get(carb_col):
                    try:
                        insulin = float(row.get(carb_col, 0)) / profile['carb_ratio']
                    except (ValueError, TypeError):
                        pass
                else:
                    # Use correction formula
                    insulin = max(0, (gl - profile['target_bg']) / profile['sens']) if gl > profile['target_bg'] else 0.05

                training_pairs.append((float(gl), float(insulin)))

    if not glucose_data:
        raise ValueError(f"Could not parse any data from {data_dir}")

    return glucose_data, training_pairs, profile
