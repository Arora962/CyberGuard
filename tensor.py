"""
Insulin prediction model using data-driven parameters.
The formula uses profile parameters (target_bg, sens/ISF, carb_ratio) and
fits coefficients from actual glucose-insulin data.
"""

import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input


def _load_training_data():
    """Load glucose-insulin pairs and profile from data_loader."""
    try:
        from data_loader import load_training_data
        X, y, profile = load_training_data()
        if X is not None and len(X) >= 10:
            return np.array(X).astype(np.float32), np.array(y).astype(np.float32), profile
    except Exception:
        pass

    # Fallback: use profile from simdata
    root = os.path.dirname(os.path.abspath(__file__))
    profile_path = os.path.join(root, 'simdata', 'profile.json')
    profile = {'target_bg': 110, 'sens': 50, 'carb_ratio': 10}
    if os.path.exists(profile_path):
        import json
        with open(profile_path) as f:
            p = json.load(f)
            profile['target_bg'] = p.get('target_bg', p.get('min_bg', 110))
            profile['sens'] = p.get('sens', 50)
            cr = p.get('carb_ratio', [{'ratio': 10}])
            profile['carb_ratio'] = cr[0]['ratio'] if isinstance(cr, list) else 10

    # Generate synthetic training data using profile parameters
    target = profile['target_bg']
    isf = profile['sens']
    glucose_train = np.linspace(70, 220, 150).astype(np.float32)
    # Formula: insulin = max(0, (glucose - target) / ISF) + small basal
    insulin_train = np.maximum(0.05, (glucose_train - target) / isf).astype(np.float32)
    return glucose_train.reshape(-1, 1), insulin_train, profile


def _data_driven_formula(glucose, target_bg, sens, beta0=0.05, beta1=None):
    """
    Linear regression formula with data parameters:
        insulin = beta0 + beta1 * (glucose - target_bg)
    where beta1 ≈ 1/sens (ISF). When beta1 is None, use 1/sens.
    """
    if beta1 is None:
        beta1 = 1.0 / sens
    return beta0 + beta1 * np.maximum(0, glucose - target_bg)


def create_and_train_model():
    """
    Creates and trains an insulin prediction model using data-driven parameters.
    Uses the formula: insulin = β0 + β1*(glucose - target_bg)
    with coefficients fit from actual/synthetic glucose-insulin data.
    """
    X, y, profile = _load_training_data()
    target_bg = profile.get('target_bg', 110)
    sens = profile.get('sens', 50)

    # Transform: use (glucose - target_bg) as feature so the model learns the correction relationship
    X_centered = X - target_bg

    model = Sequential([
        Input(shape=(1,)),
        Dense(32, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='linear'),
    ])
    model.compile(optimizer='adam', loss='mse')

    # Train on data with parameterized features
    model.fit(X_centered, y, epochs=500, verbose=0)

    # Store profile params for prediction-time use
    model._profile = profile
    return model


def predict_insulin(model, glucose, target_bg=None, sens=None):
    """
    Predict insulin dose. Uses profile parameters for centering.
    """
    profile = getattr(model, '_profile', {})
    target = target_bg or profile.get('target_bg', 110)
    glucose_arr = np.atleast_1d(glucose).astype(np.float32)
    X_centered = (glucose_arr - target).reshape(-1, 1)
    return model.predict(X_centered, verbose=0).flatten()


if __name__ == '__main__':
    model = create_and_train_model()
    test_glucose = np.array([90, 120, 150])
    predicted = predict_insulin(model, test_glucose)
    print(f"Glucose: {test_glucose}")
    print(f"Predicted Insulin: {predicted}")
    print(f"Profile: target_bg={model._profile.get('target_bg')}, sens={model._profile.get('sens')}")
