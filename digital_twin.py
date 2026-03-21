"""
digital_twin.py
===============
AI-Driven Digital Twin for the Ubiquitous Artificial Pancreas system.
Implements Section 3.3 of the paper.

The Digital Twin:
  1. Maintains a real-time virtual model of the patient's physiological state
  2. Integrates LSTM, BiLSTM, and AP-DRL models
  3. Provides the healthcare professional with the patient's current status
  4. Proposes optimal basal/bolus insulin doses to the blockchain
  5. Receives feedback from the blockchain (approved dose) and updates itself

Architecture (Figure 1, steps 2a–2c):
  IoT Pump → CGM data → Digital Twin → proposes dose → Blockchain
  Blockchain approved dose → Digital Twin (feedback loop)
  Healthcare professional monitors Digital Twin dashboard
"""

import json
import os
from datetime import datetime, timezone

import numpy as np


class DigitalTwin:
    """
    Patient-specific digital twin.

    Parameters
    ----------
    subject_id  : str  — patient identifier
    profile     : dict — {'target_bg': 110, 'sens': 50, 'carb_ratio': 10}
    blockchain  : PrivateBlockchain instance (or None for standalone use)
    models_dir  : str  — directory for persisting trained model weights
    seq_len     : int  — LSTM/BiLSTM look-back window (time steps)
    """

    def __init__(self, subject_id, profile=None, blockchain=None,
                 models_dir=None, seq_len=12):
        self.subject_id = str(subject_id)
        self.profile = profile or {'target_bg': 110, 'sens': 50, 'carb_ratio': 10}
        self.blockchain = blockchain
        self.models_dir = models_dir or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'models', subject_id
        )
        self.seq_len = seq_len

        # Internal state (Algorithm 1, line 22 feedback loop)
        self.glucose_history  = []    # raw mg/dL readings
        self.insulin_history  = []    # approved insulin doses
        self.prediction_log   = []    # timestamped predictions

        # Model references (loaded lazily)
        self._lstm    = None
        self._bilstm  = None
        self._ap_basal = None
        self._ap_bolus = None
        self._models_trained = False

    # ── Data ingestion (Step 2a: IoT pump updates Digital Twin) ──────────────

    def ingest_cgm(self, glucose_reading: float,
                   diet_g: float = 0.0, activity_min: float = 0.0):
        """
        Receive a new CGM glucose reading from the IoT pump.
        Updates internal history.
        """
        self.glucose_history.append(float(glucose_reading))
        return self

    def ingest_approved_dose(self, basal: float, bolus: float):
        """
        Receive the blockchain-approved insulin dose as feedback.
        Corresponds to Algorithm 1 line 22 (feedback loop).
        """
        self.insulin_history.append({'basal': basal, 'bolus': bolus})
        return self

    # ── Model training ────────────────────────────────────────────────────────

    def train_models(self, glucose_arr, basal_arr, bolus_arr,
                     epochs_lstm=30, episodes_apdrl=20, verbose=True):
        """
        Train LSTM, BiLSTM and AP-DRL on subject-specific data.
        Called once (or periodically) when sufficient data is available.
        """
        from lstm_models import DeepInsulinPredictor
        from ap_drl import APDRLAgent

        if verbose:
            print(f"\n[DigitalTwin {self.subject_id}] Training LSTM…")
        self._lstm = DeepInsulinPredictor(model_type='lstm', seq_len=self.seq_len)
        self._lstm.fit(glucose_arr, basal_arr, bolus_arr,
                       epochs=epochs_lstm, verbose=0)

        if verbose:
            print(f"[DigitalTwin {self.subject_id}] Training BiLSTM…")
        self._bilstm = DeepInsulinPredictor(model_type='bilstm', seq_len=self.seq_len)
        self._bilstm.fit(glucose_arr, basal_arr, bolus_arr,
                         epochs=epochs_lstm, verbose=0)

        if verbose:
            print(f"[DigitalTwin {self.subject_id}] Training AP-DRL (basal)…")
        self._ap_basal = APDRLAgent(subject_id=self.subject_id, mode='basal')
        self._ap_basal.train(glucose_arr, basal_arr, bolus_arr,
                             episodes=episodes_apdrl, verbose=verbose)

        if verbose:
            print(f"[DigitalTwin {self.subject_id}] Training AP-DRL (bolus)…")
        self._ap_bolus = APDRLAgent(subject_id=self.subject_id, mode='bolus')
        self._ap_bolus.train(glucose_arr, basal_arr, bolus_arr,
                             episodes=episodes_apdrl, verbose=verbose)

        self._models_trained = True
        if verbose:
            print(f"[DigitalTwin {self.subject_id}] All models trained ✓")
        return self

    # ── Prediction (Step 2b: propose insulin level) ───────────────────────────

    def predict_insulin(self, glucose: float = None,
                        prev_glucose: float = None) -> dict:
        """
        Compute optimal basal and bolus insulin predictions from all models.
        Uses the most recent history if glucose not supplied.

        Returns dict with predictions from each model + AP-DRL consensus.
        """
        if glucose is None:
            if not self.glucose_history:
                glucose = self.profile.get('target_bg', 110)
            else:
                glucose = self.glucose_history[-1]

        if prev_glucose is None and len(self.glucose_history) >= 2:
            prev_glucose = self.glucose_history[-2]

        predictions = {'glucose': glucose, 'timestamp': datetime.now(timezone.utc).isoformat()}

        # ── AP-DRL predictions (primary model) ────────────────────────────────
        if self._ap_basal is not None:
            cur_ins = self.insulin_history[-1]['basal'] if self.insulin_history else None
            predictions['ap_drl_basal'] = self._ap_basal.predict(
                glucose, cur_ins, prev_glucose=prev_glucose)
        else:
            # Fallback: correction formula
            target = self.profile.get('target_bg', 110)
            isf    = self.profile.get('sens', 50)
            predictions['ap_drl_basal'] = max(0.02, (glucose - target) / isf) if glucose > target else 0.05

        if self._ap_bolus is not None:
            cur_ins = self.insulin_history[-1]['bolus'] if self.insulin_history else None
            predictions['ap_drl_bolus'] = self._ap_bolus.predict(
                glucose, cur_ins, prev_glucose=prev_glucose)
        else:
            target = self.profile.get('target_bg', 110)
            isf    = self.profile.get('sens', 50)
            predictions['ap_drl_bolus'] = max(0.0, (glucose - target) / isf) if glucose > target else 0.0

        # ── LSTM / BiLSTM predictions (baselines) ────────────────────────────
        if self._lstm is not None and len(self.glucose_history) >= self.seq_len:
            window = self.glucose_history[-self.seq_len:]
            try:
                ba, bo = self._lstm.predict(window)
                predictions['lstm_basal'] = ba
                predictions['lstm_bolus'] = bo
            except Exception:
                pass

        if self._bilstm is not None and len(self.glucose_history) >= self.seq_len:
            window = self.glucose_history[-self.seq_len:]
            try:
                ba, bo = self._bilstm.predict(window)
                predictions['bilstm_basal'] = ba
                predictions['bilstm_bolus'] = bo
            except Exception:
                pass

        self.prediction_log.append(predictions)
        return predictions

    # ── Blockchain interaction (Steps 2b → 3 → 4 → 5) ───────────────────────

    def propose_to_blockchain(self, glucose: float = None) -> dict:
        """
        Predict insulin and submit proposal to blockchain.
        Corresponds to Step 2b in Figure 1.
        Returns the transaction id and predictions.
        """
        preds = self.predict_insulin(glucose)
        current = self.insulin_history[-1] if self.insulin_history else {'basal': 0.0, 'bolus': 0.0}

        if self.blockchain is None:
            print("[DigitalTwin] No blockchain attached — returning predictions only.")
            return {'tx_id': None, 'predictions': preds}

        tx_id = self.blockchain.add_pending_transaction({
            'subject_id':      self.subject_id,
            'glucose_reading': preds['glucose'],
            'predicted_basal': preds['ap_drl_basal'],
            'predicted_bolus': preds['ap_drl_bolus'],
            'current_basal':   current.get('basal', 0.0),
            'current_bolus':   current.get('bolus', 0.0),
        })
        return {'tx_id': tx_id, 'predictions': preds}

    def apply_approved_dose(self) -> dict:
        """
        Read the latest approved dose from the blockchain and update state.
        Corresponds to Steps 5 / feedback loop.
        """
        if self.blockchain is None:
            return self.insulin_history[-1] if self.insulin_history else {'basal': 0.05, 'bolus': 0.0}

        dose = self.blockchain.latest_approved_dose(self.subject_id)
        self.ingest_approved_dose(dose['basal'], dose['bolus'])
        print(f"  [DigitalTwin {self.subject_id}] Applied approved dose: "
              f"basal={dose['basal']:.4f} IU/H, bolus={dose['bolus']:.4f} IU")
        return dose

    # ── Evaluation ───────────────────────────────────────────────────────────

    def evaluate_all_models(self, glucose_arr, basal_arr, bolus_arr) -> dict:
        """
        Compute MAE / RMSE for all trained models.
        Matches Table 2 in the paper.
        """
        results = {}

        if self._lstm is not None:
            results['LSTM'] = self._lstm.evaluate(glucose_arr, basal_arr, bolus_arr)

        if self._bilstm is not None:
            results['BiLSTM'] = self._bilstm.evaluate(glucose_arr, basal_arr, bolus_arr)

        if self._ap_basal is not None:
            results['AP-DRL_basal'] = self._ap_basal.evaluate(glucose_arr, basal_arr, bolus_arr)

        if self._ap_bolus is not None:
            results['AP-DRL_bolus'] = self._ap_bolus.evaluate(glucose_arr, basal_arr, bolus_arr)

        return results

    # ── Status dashboard (for healthcare professional UI) ────────────────────

    def status(self) -> dict:
        """Return a concise status dict for healthcare professional monitoring."""
        latest_glucose  = self.glucose_history[-1]  if self.glucose_history  else None
        latest_insulin  = self.insulin_history[-1]  if self.insulin_history  else None
        latest_pred     = self.prediction_log[-1]   if self.prediction_log   else None

        return {
            'subject_id':       self.subject_id,
            'timestamp':        datetime.now(timezone.utc).isoformat(),
            'latest_glucose':   latest_glucose,
            'latest_insulin':   latest_insulin,
            'latest_prediction': latest_pred,
            'history_length':   len(self.glucose_history),
            'models_trained':   self._models_trained,
            'profile':          self.profile,
        }
