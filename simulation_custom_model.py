import os
import json
import numpy as np
from datetime import datetime
from tensor import create_and_train_model, predict_insulin

def run_simulation():
    """
    Runs the simulation using the custom TensorFlow model.
    """
    # --- 1. Setup Paths ---
    root = os.path.dirname(os.path.abspath(__file__))
    simdata_dir = os.path.join(root, 'simdata')
    pred_dir = os.path.join(root, 'predictions-new')
    glucose_file = os.path.join(simdata_dir, 'glucose.json')

    # --- 2. Create and Train the Model ---
    print("Training the custom insulin prediction model...")
    model = create_and_train_model()
    print("Model training complete.")

    # --- 3. Load Simulated Glucose Data ---
    try:
        with open(glucose_file, 'r') as f:
            glucose_data = json.load(f)
        # Get the most recent glucose reading
        latest_glucose_reading = glucose_data[-1]['glucose']
        print(f"Using latest glucose reading: {latest_glucose_reading}")
    except (FileNotFoundError, IndexError, KeyError) as e:
        print(f"Error reading glucose data: {e}")
        return

    # --- 4. Predict Insulin Dose ---
    predicted_insulin = predict_insulin(model, latest_glucose_reading)
    if np.isscalar(predicted_insulin):
        predicted_insulin = float(predicted_insulin)
    else:
        predicted_insulin = float(predicted_insulin[0])
    print(f"Predicted insulin dose: {predicted_insulin:.4f} units")

    # --- 5. Save the Prediction ---
    prediction = {
        'timestamp': datetime.now().isoformat(),
        'glucose_input': latest_glucose_reading,
        'predicted_insulin_dose': float(round(predicted_insulin, 4)),
        'model_used': 'tensor.py'
    }
    
    # Create a timestamped filename
    now = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    pred_file_path = os.path.join(pred_dir, f'prediction-{now}.json')

    with open(pred_file_path, 'w') as f:
        json.dump(prediction, f, indent=4)

    print(f"Prediction saved to {pred_file_path}")
    
    # --- 6. Display Aesthetic Output ---
    print("\n--- Custom Model Prediction ---")
    print(json.dumps(prediction, indent=4))
    print("-----------------------------\n")

if __name__ == '__main__':
    run_simulation()
