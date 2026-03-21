"""
Loader for ShangaiT1DM dataset (used in the paper).
The dataset contains CSV files per subject with CGM readings, insulin (CSII basal/bolus), carbs.

Expected directory structure:
  data/external/shanghait1dm/
    1001.csv
    1002.csv
    ...

Columns typically include:
  - Date, Time / datetime
  - CGM glucose (mg/dL or mmol/L)
  - CSII - basal insulin (Novolin R, IU/H)
  - CSII - bolus insulin (Novolin R, IU)
  - Carbohydrate intake (g)

Subject IDs from the paper: 1001, 1002, 1006, 1012
"""

import csv
import glob
import os
from datetime import datetime


# Possible CGM column name variants in the dataset
_CGM_COLS = ['CGM', 'glucose', 'Glucose', 'cgm', 'CGM glucose', 'bg', 'BG', 'cbg']
_BASAL_COLS = ['CSII - basal insulin (Novolin R, IU / H)', 'basal', 'Basal', 'basal_insulin', 'CSII_basal']
_BOLUS_COLS = ['CSII - bolus insulin (Novolin R, IU)', 'bolus', 'Bolus', 'bolus_insulin', 'CSII_bolus']
_CARB_COLS  = ['Carbohydrate intake (g)', 'carbs', 'Carbs', 'CHO', 'carbohydrate']
_TIME_COLS  = ['Date', 'datetime', 'Datetime', 'time', 'Time', 'timestamp']


def _find_col(header, candidates):
    for c in candidates:
        if c in header:
            return c
    # Fuzzy match
    for h in header:
        for c in candidates:
            if c.lower() in h.lower():
                return h
    return None


def _parse_glucose(val, unit_hint='mg/dL'):
    """Convert glucose value, auto-detecting mmol/L vs mg/dL."""
    try:
        v = float(val)
        if v < 30:          # Likely mmol/L → convert
            v = round(v * 18.0182)
        return max(40, min(400, round(v)))
    except (ValueError, TypeError):
        return None


def load_shanghait1dm(data_dir):
    """
    Load ShangaiT1DM CSV files.

    Returns:
        subjects: dict keyed by subject_id →
            {
              'glucose_data': [...],          # simdata-format list
              'training_pairs': [...],        # (glucose, basal, bolus) tuples
              'profile': {...}
            }
        combined_glucose: flat list for OpenAPS-compatible simulation
        combined_pairs: flat list of (glucose, insulin) for LSTM/AP-DRL training
        profile: default profile dict
    """
    subjects = {}
    combined_glucose = []
    combined_pairs = []
    profile = {'target_bg': 110, 'sens': 50, 'carb_ratio': 10}

    csv_files = sorted(glob.glob(os.path.join(data_dir, '*.csv')))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}. "
                                "Place ShangaiT1DM CSV files there (e.g. 1001.csv, 1002.csv ...).")

    base_ts = int(datetime(2023, 1, 1).timestamp() * 1000)

    for path in csv_files:
        subject_id = os.path.splitext(os.path.basename(path))[0]
        glucose_data = []
        training_pairs = []

        with open(path, newline='', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames or []

            cgm_col   = _find_col(header, _CGM_COLS)
            basal_col = _find_col(header, _BASAL_COLS)
            bolus_col = _find_col(header, _BOLUS_COLS)
            time_col  = _find_col(header, _TIME_COLS)

            for i, row in enumerate(reader):
                # --- Glucose ---
                raw_gl = row.get(cgm_col, '') if cgm_col else ''
                if not raw_gl or raw_gl.strip() in ('', 'NaN', 'nan', 'NA'):
                    continue
                gl = _parse_glucose(raw_gl)
                if gl is None:
                    continue

                # --- Timestamp ---
                ts = base_ts + i * 5 * 60 * 1000   # 5-min intervals as fallback
                if time_col and row.get(time_col):
                    raw_ts = row[time_col].strip()
                    for fmt in ('%Y-%m-%d %H:%M:%S', '%Y/%m/%d %H:%M:%S',
                                '%d/%m/%Y %H:%M', '%Y-%m-%dT%H:%M:%S'):
                        try:
                            dt = datetime.strptime(raw_ts, fmt)
                            ts = int(dt.timestamp() * 1000)
                            break
                        except ValueError:
                            continue

                glucose_data.append({
                    'date':      ts,
                    'glucose':   gl,
                    'sgv':       gl,
                    'direction': 'Flat',
                    'noise':     1,
                    'filtered':  gl,
                    'unfiltered': gl,
                    'rssi':      100,
                    'device':    f'shanghait1dm_{subject_id}',
                })

                # --- Insulin labels ---
                basal, bolus = 0.05, 0.0
                if basal_col and row.get(basal_col, '').strip() not in ('', 'NaN', 'nan', 'NA'):
                    try:
                        basal = abs(float(row[basal_col]))
                    except ValueError:
                        pass
                if bolus_col and row.get(bolus_col, '').strip() not in ('', 'NaN', 'nan', 'NA'):
                    try:
                        bolus = abs(float(row[bolus_col]))
                    except ValueError:
                        pass

                training_pairs.append((float(gl), float(basal), float(bolus)))

        if glucose_data:
            subjects[subject_id] = {
                'glucose_data':   glucose_data,
                'training_pairs': training_pairs,
                'profile':        dict(profile),
            }
            combined_glucose.extend(glucose_data)
            # For backward-compat with existing loaders: use total insulin
            for gl, basal, bolus in training_pairs:
                combined_pairs.append((gl, basal + bolus if bolus > 0 else basal))

    if not subjects:
        raise ValueError(f"Could not parse any subject data from {data_dir}")

    return subjects, combined_glucose, combined_pairs, profile
