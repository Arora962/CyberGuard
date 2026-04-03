# CyberGuard: Ubiquitous Artificial Pancreas — Blockchain-Secured AI-Driven Digital Twin for IoT-Enabled Insulin Pumps

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python" />
  <img src="https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow" />
  <img src="https://img.shields.io/badge/Blockchain-Private%20PoA-green" />
  <img src="https://img.shields.io/badge/Status-Research%20Prototype-yellow" />
  <img src="https://img.shields.io/badge/Dataset-ShangaiT1DM-purple" />
</p>

> **Research implementation** of the paper:  
> *"Ubiquitous Artificial Pancreas: Blockchain-Secured AI-Driven Digital Twin for IoT-Enabled Insulin Pumps in Type 1 Diabetes Management"*  
> Advisors: Dr. Anny Leema & Dr. Balakrishnan P — VIT University

---

## Overview

CyberGuard is a full-stack simulation of an **Artificial Pancreas (AP) system** that combines three cutting-edge technologies to automate and secure insulin delivery for Type 1 Diabetes (T1D) patients:

1. **Adaptive Predictive Deep Reinforcement Learning (AP-DRL)** — a personalised DQN-based agent that predicts optimal basal and bolus insulin doses using the Bergman Minimal Model of glucose-insulin dynamics.
2. **AI-Driven Digital Twin** — a real-time virtual patient model that integrates LSTM, BiLSTM, and AP-DRL predictions and maintains a live physiological state for each patient.
3. **Private Blockchain with Smart Contracts** — a Proof-of-Authority (PoA) chain where physician-approved insulin doses are immutably recorded before the IoT insulin pump acts on them.

The system implements **Algorithm 1** and the full **Figure 1 workflow** from the paper end-to-end.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     PATIENT IoT LAYER                           │
│  CGM Sensor ──► Glucose Reading ──► IoT Insulin Pump            │
└────────────────────────┬────────────────────────────────────────┘
                         │ Step 1: CGM data stream
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   DIGITAL TWIN (digital_twin.py)                │
│                                                                  │
│  ┌──────────┐  ┌──────────┐  ┌────────────────────────────┐    │
│  │   LSTM   │  │  BiLSTM  │  │    AP-DRL Agent (ap_drl.py)│    │
│  │(baseline)│  │(baseline)│  │  Bergman Model + DQN + ε-  │    │
│  └──────────┘  └──────────┘  │  greedy + Experience Replay │    │
│                               └────────────────────────────┘    │
│                    Step 2: Predict optimal dose                  │
└────────────────────────┬────────────────────────────────────────┘
                         │ Step 3: Propose to blockchain
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              PRIVATE BLOCKCHAIN (blockchain.py)                  │
│                                                                  │
│  SmartContract ──► Physician Review ──► PoA Approval            │
│  Block (hash-chained, HMAC-signed, immutable ledger)            │
│                    Step 4: Physician approves / adjusts          │
└────────────────────────┬────────────────────────────────────────┘
                         │ Step 5: Approved dose to pump
                         ▼
                ┌─────────────────┐
                │  Insulin Pump   │  ◄─── acts on approved dose only
                └─────────────────┘
```

---

## Project Structure

```
insulin-pump-simulation/
│
├── ap_drl.py                     # Core AP-DRL algorithm (Algorithm 1)
│   ├── MinimalModel              # Bergman glucose-insulin dynamics
│   └── APDRLAgent                # DQN + ε-greedy + per-subject hyperparams
│
├── blockchain.py                 # Private PoA blockchain
│   ├── SmartContract             # Access control & dose approval rules
│   ├── Block                     # HMAC-signed, hash-chained block
│   └── PrivateBlockchain         # Full Figure 1 workflow
│
├── digital_twin.py               # AI-driven digital twin (Section 3.3)
│   └── DigitalTwin               # LSTM + BiLSTM + AP-DRL integration
│
├── lstm_models.py                # Baseline comparison models
│   ├── build_lstm_model()        # Stacked LSTM (basal + bolus outputs)
│   └── build_bilstm_model()      # Bidirectional LSTM
│
├── main_pipeline.py              # End-to-end runner (all subjects)
│
├── simulation.py                 # OpenAPS / oref0 simulation
├── simulation_custom_model.py    # Custom TensorFlow model simulation
├── tensor.py                     # Simple neural net baseline
│
├── data_loader/
│   ├── load_shanghait1dm.py      # ShangaiT1DM dataset loader (primary)
│   ├── load_ohiot1dm.py          # OhioT1DM dataset loader
│   ├── load_azt1d.py             # AZT1D dataset loader
│   ├── load_simdata.py           # Simulation data loader
│   └── synthetic_data.py         # Synthetic T1D data generator
│
├── simdata/                      # JSON files for oref0 simulation
│   ├── glucose.json
│   ├── iob.json
│   ├── profile.json
│   └── ...
│
├── output/                       # Per-subject results and blockchain ledger
│   ├── all_results.json
│   ├── blockchain_ledger.json
│   └── results_{subject_id}.json
│
├── predictions/                  # oref0 prediction outputs
├── predictions-new/              # Custom model prediction outputs
├── openaps/                      # OpenAPS framework (submodule)
└── myopenaps/                    # Personal OpenAPS configuration
```

---

## Key Components

### 1. `ap_drl.py` — Adaptive Predictive DRL Agent

Implements **Algorithm 1** from the paper exactly:

- **State vector**: `[G_norm, I_norm, D_norm, A_norm, trend]` — normalised glucose, insulin, diet, activity, and glucose rate of change.
- **Reward function**: `R_t = -|G_target - G(t)| - λ·I(t)` — penalises both glucose deviation and excessive insulin.
- **Bergman Minimal Model**: Simulates glucose-insulin physiology via two coupled ODEs (`dG/dt`, `dX/dt`).
- **DQN**: 3-layer neural network (128→64→32→n_actions) with experience replay, target network, and ε-greedy exploration.
- **Per-subject hyperparameters** (Table 3):

  | Subject | Bolus LR | Bolus γ | Basal LR | Basal γ |
  |---------|----------|---------|----------|---------|
  | 1001    | 0.005    | 0.80    | 0.003    | 0.90    |
  | 1002    | 0.0005   | 0.995   | 0.00001  | 0.99    |
  | 1006    | 0.0001   | 0.95    | 0.05     | 0.90    |
  | 1012    | 0.001    | 0.90    | 0.001    | 0.995   |

### 2. `blockchain.py` — Private Proof-of-Authority Blockchain

Implements **Section 3.4** and **Figure 1**:

- **SmartContract**: Enforces access rules — only registered physicians can validate; dose changes >2 IU always require explicit physician approval.
- **Block**: SHA-256 hash-chained with HMAC-SHA256 signatures per block.
- **5-step workflow**: AP-DRL proposes → smart contract fires → physician reviews → block added to chain → pump reads approved dose.

### 3. `digital_twin.py` — Patient Digital Twin

Implements **Section 3.3**:

- Maintains continuous glucose history and approved dose history per patient.
- Trains LSTM, BiLSTM, and AP-DRL models on subject-specific data.
- Provides a `status()` dashboard for healthcare professionals.
- Implements the full IoT → AI → Blockchain feedback loop.

### 4. `lstm_models.py` — Baseline Models

Dual-output LSTM and BiLSTM architectures (predicting both basal and bolus) used for comparison in **Table 2** of the paper.

---

## Dataset

The system is validated on the **ShangaiT1DM** dataset — the primary dataset used in the paper. Additional loaders are included for:

- **OhioT1DM** (XML format)
- **AZT1D** (CSV format)
- **Synthetic data** (generated automatically as fallback)

---

## Installation

```bash
git clone https://github.com/Arora962/CyberGuard.git
cd CyberGuard/insulin-pump-simulation

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate          # Linux/macOS
# venv\Scripts\activate           # Windows

# Install dependencies
pip install tensorflow numpy
```

---

## Usage

### Run the Full Pipeline (all 4 subjects, all models)

```bash
python main_pipeline.py
```

This trains all models per subject and prints a **Table 2-style** performance report.

### Run with specific subjects or options

```bash
python main_pipeline.py --subjects 1001 1012 --episodes 20 --synthetic
```

| Flag | Description |
|------|-------------|
| `--subjects` | Space-separated subject IDs (default: 1001 1002 1006 1012) |
| `--episodes` | AP-DRL training episodes (default: 30) |
| `--synthetic` | Force synthetic data even if dataset is available |

### Run the oref0 (OpenAPS) Simulation

```bash
python simulation.py
```

Output saved to `predictions/`.

### Run the Custom TensorFlow Model Simulation

```bash
python simulation_custom_model.py
```

Output saved to `predictions-new/`.

### Populate Synthetic Patient Data

```bash
python scripts/populate_synthetic_data.py
```

---

## Results (Table 2 equivalent)

After running `main_pipeline.py`, results are printed per subject in this format:

```
──────────────────────────────────────────────────────────────────────
  Performance Table — Subject 1001
  Method         Bolus MAE  Bolus RMSE  Basal MAE  Basal RMSE
  ──────────────────────────────────────────────────────────────────
  LSTM              0.XXXX      0.XXXX     0.XXXX      0.XXXX
  BiLSTM            0.XXXX      0.XXXX     0.XXXX      0.XXXX
  AP-DRL            0.XXXX      0.XXXX     0.XXXX      0.XXXX
──────────────────────────────────────────────────────────────────────
```

Full results saved to `output/all_results.json`. Blockchain transaction ledger saved to `output/blockchain_ledger.json`.

---

## Security Design

| Feature | Implementation |
|---------|---------------|
| Immutable audit trail | SHA-256 hash chaining |
| Physician authentication | HMAC-SHA256 signed blocks (PoA) |
| Access control | Smart contract `can_read()` / `can_validate()` |
| Dose safety threshold | Smart contract rejects changes >2 IU without approval |
| Chain integrity | `is_valid()` verifies full chain on demand |

---

## Research Context

This repository implements the complete system from the paper in a simulatable form suitable for:

- Replicating Table 2 (LSTM vs BiLSTM vs AP-DRL performance comparison)
- Demonstrating the blockchain security workflow (Figure 1)
- Personalised per-patient insulin delivery with subject-specific DRL hyperparameters (Table 3)
- Validating the Digital Twin concept for T1D management

---

## Acknowledgements

- OpenAPS community and the `oref0` reference implementation
- ShangaiT1DM, OhioT1DM, and AZT1D dataset creators
- Dr. Anny Leema and Dr. Balakrishnan P — VIT University
- VIT IPR & TTCELL — VIT University
---

## License

This project is for academic and research purposes. See individual subfolders for their respective licenses. The CyberGuard system itself (ap_drl.py, blockchain.py, digital_twin.py, lstm_models.py, main_pipeline.py) is under active patent filing — please contact the authors before commercial use.