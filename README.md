# CyberGuard: Insulin Pump Simulation with OpenAPS and Custom Models

## Overview
CyberGuard is a research-driven project to simulate and analyze insulin pump control systems. This repository contains a working simulation environment that supports two distinct prediction models:
1.  **OpenAPS `oref0` Algorithm**: The standard, community-vetted algorithm for automated insulin delivery.
2.  **Custom TensorFlow Model**: A custom-built neural network (`tensor.py`) that predicts insulin dosage based on glucose levels.

This setup allows for direct comparison and analysis of different control strategies, focusing on security, reliability, and extensibility.

## Project Structure

```
insulin-pump-simulation/
├── openaps/              
├── oref0/                
├── myopenaps/            
├── simdata/              
├── predictions/          
├── predictions-new/                
├── simulation.py        
├── simulation_custom_model.py 
├── tensor.py             
└── README.md
```

## Key Components

-   **`simulation.py`**: Orchestrates a simulation using the standard `oref0` algorithm. It reads data from `simdata/`, runs the prediction, and saves the output to the `predictions/` directory.
-   **`simulation_custom_model.py`**: Runs a simulation using your custom TensorFlow model. It trains the model from `tensor.py`, uses it to predict an insulin dose from `simdata/`, and saves the output to `predictions-new/`.
-   **`tensor.py`**: Defines, trains, and tests a simple neural network to predict insulin doses from glucose values.
-   **`simdata/`**: Contains all the necessary JSON files (e.g., `glucose.json`, `iob.json`, `profile.json`) to feed into the simulations as input.
-   **`predictions/` & `predictions-new/`**: These directories store the timestamped JSON output from the `oref0` and custom model simulations, respectively, allowing for analysis and record-keeping.
-   **`oref0/`**: The core reference implementation of the OpenAPS algorithm, used by `simulation.py`.

## How to Run the Simulations

1.  **Clone the repository**
    ```sh
    git clone <your-repo-url>
    cd insulin-pump-simulation
    ```
2.  **Set up the Python environment**
    Make sure you have all dependencies installed (`tensorflow`, `numpy`) and activate the environment.
    ```sh
    source openaps-env/bin/activate
    ```
3.  **Run the Standard OpenAPS (oref0) Simulation**
    This will use the `oref0` algorithm to predict a temporary basal rate.
    ```sh
    python simulation.py
    ```
    The output will be saved in the `predictions/` directory.

4.  **Run the Custom Model Simulation**
    This will train your `tensor.py` model and use it to predict an insulin dose.
    ```sh
    python simulation_custom_model.py
    ```
    The output will be saved in the `predictions-new/` directory.

## Research Context
- **Inspiration**: The project is inspired by research into the safety and security of medical devices.
- **Our Work**: This repository contains our implementation and experiments for simulating and analyzing different insulin control algorithms.

## How to Contribute
1.  Fork the repository.
2.  Create a new branch for your feature or fix.
3.  Submit a pull request with a clear description of your changes.

## License
This project is for academic and research purposes. See individual folders for their respective licenses.

## Acknowledgements
- OpenAPS community and contributors
- Dr. Anny Leema and Dr. Balakrishnan P
