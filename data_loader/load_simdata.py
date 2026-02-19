"""
Load simulation data from simdata/ or external datasets.
Prioritizes: external datasets > synthetic data > existing simdata.
"""

import json
import os


def _load_external_if_exists(root_dir):
    """Check for external dataset files. Returns (glucose_data, training_pairs, profile) or None."""
    external_dir = os.path.join(root_dir, 'data', 'external')
    if not os.path.exists(external_dir):
        return None

    # AZT1D: look for CSV files
    azt1d_dir = os.path.join(external_dir, 'azt1d')
    if os.path.exists(azt1d_dir):
        try:
            from .load_azt1d import load_azt1d
            return load_azt1d(azt1d_dir)
        except Exception:
            pass

    # OhioT1DM: look for XML files
    ohiot1dm_dir = os.path.join(external_dir, 'ohiot1dm')
    if os.path.exists(ohiot1dm_dir):
        try:
            from .load_ohiot1dm import load_ohiot1dm
            return load_ohiot1dm(ohiot1dm_dir)
        except Exception:
            pass

    return None


def load_simdata(root_dir=None):
    """
    Load glucose data and profile for simulation.
    Uses external datasets if available, otherwise synthetic data.

    Returns:
        glucose_data: List of glucose readings (simdata format)
        profile: Dict with target_bg, sens, carb_ratio
        training_pairs: List of (glucose, insulin) for model training
    """
    if root_dir is None:
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    simdata_dir = os.path.join(root_dir, 'simdata')

    # Try external datasets first
    external = _load_external_if_exists(root_dir)
    if external is not None:
        glucose_data, training_pairs, profile = external
        return glucose_data, profile, training_pairs

    # Try existing simdata
    glucose_path = os.path.join(simdata_dir, 'glucose.json')
    profile_path = os.path.join(simdata_dir, 'profile.json')

    if os.path.exists(glucose_path):
        with open(glucose_path) as f:
            glucose_data = json.load(f)
        profile = {}
        if os.path.exists(profile_path):
            with open(profile_path) as f:
                p = json.load(f)
                profile = {
                    'target_bg': p.get('target_bg', p.get('min_bg', 110)),
                    'sens': p.get('sens', 50),
                    'carb_ratio': p.get('carb_ratio', [{'ratio': 10}])[0]['ratio'] if isinstance(p.get('carb_ratio'), list) else 10,
                }
        # Build training pairs from glucose using correction formula
        target = profile.get('target_bg', 110)
        isf = profile.get('sens', 50)
        training_pairs = []
        for g in glucose_data:
            gl = g.get('glucose', g.get('sgv', 120))
            if gl > target:
                insulin = (gl - target) / isf
            else:
                insulin = 0.05
            training_pairs.append((float(gl), float(insulin)))
        # If very few points, augment with synthetic data for better training
        if len(training_pairs) < 50:
            from .synthetic_data import generate_synthetic_t1d_data
            _, synth_pairs, _ = generate_synthetic_t1d_data(simdata_dir, n_glucose_points=300, n_training_pairs=250)
            training_pairs = training_pairs + synth_pairs
        return glucose_data, profile, training_pairs

    # Fall back to synthetic data
    from .synthetic_data import generate_synthetic_t1d_data
    glucose_data, training_pairs, profile = generate_synthetic_t1d_data(simdata_dir)
    profile_params = {
        'target_bg': profile['target_bg'],
        'sens': profile['sens'],
        'carb_ratio': profile['carb_ratio'],
    }
    return glucose_data, profile_params, training_pairs


def load_training_data(root_dir=None):
    """
    Load (glucose, insulin) pairs for model training.
    Returns: (X, y, profile_params) where X=glucose, y=insulin.
    """
    _, profile, pairs = load_simdata(root_dir)
    if not pairs:
        return None, None, profile
    X = [[g] for g, _ in pairs]
    y = [i for _, i in pairs]
    return X, y, profile
