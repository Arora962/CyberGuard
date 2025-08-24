# CyberGuard: Insulin Pump Simulation with OpenAPS

## Overview
CyberGuard is a research-driven project. The goal is to simulate and analyze the OpenAPS (Open Artificial Pancreas System) environment, focusing on the security, reliability, and extensibility of insulin pump control systems. This repository contains a working simulation of OpenAPS, along with custom scripts, data, and configuration files to support further research and experimentation.

## Project Structure

```
insulin-pump-simulation/
├── tensor.py
├── myopenaps/
│   ├── cgm.ini
│   ├── openaps.ini
│   ├── profile.json
│   ├── pump.ini
│   ├── settings.json
│   └── monitor/
│       ├── glucose.json
│       └── pumphistory.json
├── myopenaps-copy/
│   ├── ... (same as myopenaps, plus oref0/)
├── openaps/
│   ├── ... (OpenAPS source and scripts)
├── openaps-env/
│   ├── ... (Python virtual environment)
└── .gitignore
```

- **myopenaps/** and **myopenaps-copy/**: Simulated OpenAPS configuration, monitoring, and data directories. `myopenaps-copy/oref0/` contains additional scripts and utilities.
- **openaps/**: The OpenAPS source code and command-line tools.
- **openaps-env/**: Python virtual environment for isolated dependencies.
- **tensor.py**: Custom script for simulation or data processing.

## Getting Started

1. **Clone the repository**
   ```sh
   git clone <your-repo-url>
   cd insulin-pump-simulation
   ```
2. **Set up the Python environment**
   ```sh
   source openaps-env/bin/activate
   ```
3. **Explore the OpenAPS simulation**
   - Review and modify configuration files in `myopenaps/` and `myopenaps-copy/`.
   - Use scripts in `openaps/` and `myopenaps-copy/oref0/bin/` for simulation and data analysis.

## Research Context
- **Inspiration**: The project is inspired by the research papers.
- **Our Work**: This repository contains our implementation and experiments based on the concepts and challenges discussed in the papers.

## Key Features
- Simulates OpenAPS workflows and data flows
- Provides a safe environment for testing insulin pump logic
- Supports further research in medical device security and reliability

## How to Contribute
1. Fork the repository
2. Create a new branch for your feature or fix
3. Submit a pull request with a clear description

## License
This project is for academic and research purposes. See individual folders for their respective licenses.

## Acknowledgements
- OpenAPS community and contributors
- Dr. Anny Leema and Dr. Balakrishnan P  

---
For more details, explore the codebase.
