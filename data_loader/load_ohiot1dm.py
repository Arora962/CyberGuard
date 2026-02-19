"""
Loader for OhioT1DM dataset (XML format).
Expects XML files from OhioT1DM. See: https://webpages.charlotte.edu/rbunescu/data/ohiot1dm/
"""

import os
import xml.etree.ElementTree as ET
from datetime import datetime


def _parse_ts(ts_str):
    """Parse OhioT1DM timestamp to ms."""
    try:
        dt = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
        return int(dt.timestamp() * 1000)
    except Exception:
        return int(datetime.now().timestamp() * 1000)


def load_ohiot1dm(data_dir):
    """
    Load OhioT1DM XML files.
    Returns (glucose_data, training_pairs, profile).
    """
    import glob
    glucose_data = []
    training_pairs = []
    profile = {'target_bg': 110, 'sens': 50, 'carb_ratio': 10}

    xml_files = glob.glob(os.path.join(data_dir, '*.xml'))
    if not xml_files:
        raise FileNotFoundError(f"No XML files found in {data_dir}")

    for path in xml_files[:2]:
        tree = ET.parse(path)
        root = tree.getroot()

        # CGM: typically in 'glucose_level' or similar
        for elem in root.iter():
            tag = elem.tag.lower()
            if 'glucose' in tag or 'cgm' in tag or tag == 'value':
                ts = elem.get('ts', elem.get('timestamp', ''))
                ts_ms = _parse_ts(ts) if ts else 0
                try:
                    gl = float(elem.text or elem.get('value', 120))
                except (ValueError, TypeError):
                    continue
                gl = max(40, min(400, round(gl)))

                glucose_data.append({
                    'date': ts_ms,
                    'glucose': gl,
                    'sgv': gl,
                    'direction': 'Flat',
                    'noise': 1,
                    'filtered': gl,
                    'unfiltered': gl,
                    'rssi': 100,
                    'device': 'ohiot1dm',
                })

                # Estimate insulin from correction formula
                insulin = max(0, (gl - profile['target_bg']) / profile['sens']) if gl > profile['target_bg'] else 0.05
                training_pairs.append((float(gl), float(insulin)))

        # Bolus data for better training pairs
        for elem in root.iter():
            tag = elem.tag.lower()
            if 'bolus' in tag:
                ts = elem.get('ts', '')
                try:
                    amount = float(elem.get('amount', elem.text or 0))
                    # Find nearest glucose reading and update training pair
                    for i, gd in enumerate(glucose_data):
                        if abs(gd['date'] - _parse_ts(ts)) < 15 * 60 * 1000 and i < len(training_pairs):
                            training_pairs[i] = (training_pairs[i][0], amount)
                            break
                except (ValueError, TypeError):
                    pass

    if not glucose_data:
        raise ValueError(f"Could not parse any data from {data_dir}")

    return glucose_data, training_pairs, profile
