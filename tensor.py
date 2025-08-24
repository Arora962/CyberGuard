import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from tensorflow.keras.layers import Input

def create_and_train_model():
    """Creates, compiles, and trains a simple insulin prediction model."""
    model = Sequential([
        Input(shape=(1,)),  # Define input shape here
        Dense(32, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')  # Output: Insulin dose
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mse')

    # Generate training data (simulated glucose vs insulin)
    glucose_train = np.linspace(70, 180, 100)  # Simulated glucose range
    insulin_train = (glucose_train - 100) * 0.05  # Simple function for insulin prediction

    # Train the model
    model.fit(glucose_train, insulin_train, epochs=500, verbose=0)
    
    return model

if __name__ == '__main__':
    # Create and train the model
    trained_model = create_and_train_model()

    # Test the model
    test_glucose = np.array([90, 120, 150])
    predicted_insulin = trained_model.predict(test_glucose)

    print(f"Glucose: {test_glucose}, Predicted Insulin: {predicted_insulin.flatten()}")
