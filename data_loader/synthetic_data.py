"""
Generate synthetic T1D data based on typical physiological parameters.
Used when no external dataset (AZT1D, OhioT1DM, etc.) is available.
Parameters are derived from profile.json and typical T1D ranges.
"""

import json
import os
import random
from datetime import datetime, timedelta


def _load_profile(simdata_dir):
    """Load profile from simdata or use defaults."""
    profile_path = os.path.join(simdata_dir, 'profile.json')
    defaults = {
        'target_bg': 110,
        'sens': 50,  # ISF: 1 unit drops glucose by 50 mg/dL
        'carb_ratio': 10,  # 1 unit per 10g carbs
        'min_bg': 70,
        'max_bg': 180,
    }
    if os.path.exists(profile_path):
        with open(profile_path) as f:
            p = json.load(f)
            defaults['target_bg'] = p.get('target_bg', p.get('min_bg', 110))
            defaults['sens'] = p.get('sens', 50)
            cr = p.get('carb_ratio', [{'ratio': 10}])
            defaults['carb_ratio'] = cr[0]['ratio'] if isinstance(cr, list) else cr
            defaults['min_bg'] = p.get('min_bg', 70)
            defaults['max_bg'] = p.get('max_bg', 180)
    return defaults


def generate_synthetic_t1d_data(
    simdata_dir,
    n_glucose_points=500,
    n_training_pairs=200,
    seed=None,
):
    """
    Generate synthetic glucose and insulin data using T1D parameters.

    The insulin-glucose relationship follows the standard correction formula:
        correction_bolus = (glucose - target_bg) / ISF
    with added variation to simulate real-world behavior.

    Returns:
        glucose_data: List of dicts in simdata/glucose.json format
        training_pairs: List of (glucose, insulin) tuples for model training
        profile_params: Dict with target_bg, sens, carb_ratio
    """
    if seed is not None:
        random.seed(seed)

    profile = _load_profile(simdata_dir)
    target_bg = profile['target_bg']
    isf = profile['sens']  # Insulin Sensitivity Factor
    carb_ratio = profile['carb_ratio']

    # Generate glucose values (typical range 60-250 mg/dL with most in 80-180)
    base_time = int(datetime.now().timestamp() * 1000) - (n_glucose_points * 5 * 60 * 1000)
    glucose_data = []
    training_pairs = []

    for i in range(n_glucose_points):
        # Glucose with realistic distribution: mostly 80-160, some excursions
        r = random.random()
        if r < 0.7:
            glucose = random.gauss(120, 25)
        elif r < 0.9:
            glucose = random.gauss(180, 30)  # Hyperglycemia
        else:
            glucose = random.gauss(75, 15)  # Near hypoglycemia

        glucose = max(40, min(400, round(glucose)))
        ts = base_time + i * 5 * 60 * 1000  # 5-min intervals

        direction = random.choice(['Flat', 'FortyFiveUp', 'FortyFiveDown', 'SingleUp', 'SingleDown'])
        glucose_data.append({
            'date': ts,
            'glucose': glucose,
            'sgv': glucose,
            'direction': direction,
            'noise': 1 if 70 <= glucose <= 180 else 2,
            'filtered': glucose,
            'unfiltered': glucose,
            'rssi': 100,
            'device': 'synthetic',
        })

        # Training pairs: insulin = correction + optional meal component
        # Correction: (glucose - target) / ISF when above target
        if glucose > target_bg:
            correction = (glucose - target_bg) / isf
            # Add small random meal component (0-20g carbs equivalent)
            meal_carbs = random.random() * 20 if random.random() < 0.3 else 0
            meal_bolus = meal_carbs / carb_ratio
            insulin = max(0, correction + meal_bolus + random.gauss(0, 0.1))
        else:
            insulin = max(0, random.gauss(0.05, 0.02))  # Small basal-like amount when in/low range

        if i < n_training_pairs:
            training_pairs.append((float(glucose), float(insulin)))

    return glucose_data, training_pairs, profile
