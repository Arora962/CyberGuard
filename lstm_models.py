"""
LSTM and Bidirectional LSTM models for glucose-insulin dynamics prediction.
Used as baseline comparison models alongside AP-DRL (as in the paper).

Models:
  - LSTMModel: standard LSTM for sequence-based insulin prediction
  - BiLSTMModel: Bidirectional LSTM (captures both past and future context)

Both predict:
  - CSII basal insulin (IU/H)   → regression output 1
  - CSII bolus insulin (IU)     → regression output 2
"""

import numpy as np

try:
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (
        LSTM, Bidirectional, Dense, Input, Dropout, BatchNormalization
    )
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


# ─── Sequence preparation ────────────────────────────────────────────────────

def prepare_sequences(glucose_series, basal_series, bolus_series,
                       seq_len=12, step=1):
    """
    Build overlapping windows of length `seq_len` from time-series data.

    Input:
        glucose_series: array of shape (N,)  — normalised glucose readings
        basal_series:   array of shape (N,)  — normalised basal insulin
        bolus_series:   array of shape (N,)  — normalised bolus insulin

    Returns:
        X: (samples, seq_len, 1)   — glucose windows
        y_basal: (samples,)
        y_bolus: (samples,)
    """
    X, y_basal, y_bolus = [], [], []
    for i in range(0, len(glucose_series) - seq_len, step):
        X.append(glucose_series[i: i + seq_len])
        y_basal.append(basal_series[i + seq_len])
        y_bolus.append(bolus_series[i + seq_len])

    X = np.array(X, dtype=np.float32).reshape(-1, seq_len, 1)
    return X, np.array(y_basal, dtype=np.float32), np.array(y_bolus, dtype=np.float32)


def normalise(arr):
    mu, sigma = arr.mean(), arr.std() + 1e-8
    return (arr - mu) / sigma, mu, sigma


def denormalise(arr, mu, sigma):
    return arr * sigma + mu


# ─── LSTM model ──────────────────────────────────────────────────────────────

def build_lstm_model(seq_len=12, units=64, dropout=0.2):
    """
    Standard stacked-LSTM model.
    Outputs: [basal_pred, bolus_pred]
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow is required for LSTM model.")

    inp = Input(shape=(seq_len, 1), name='glucose_seq')
    x = LSTM(units, return_sequences=True, name='lstm_1')(inp)
    x = Dropout(dropout)(x)
    x = LSTM(units // 2, return_sequences=False, name='lstm_2')(x)
    x = BatchNormalization()(x)
    x = Dense(32, activation='relu')(x)
    basal = Dense(1, activation='linear', name='basal')(x)
    bolus = Dense(1, activation='linear', name='bolus')(x)
    model = Model(inputs=inp, outputs=[basal, bolus], name='LSTMModel')
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss={'basal': 'mse', 'bolus': 'mse'},
        metrics={'basal': 'mae', 'bolus': 'mae'},
    )
    return model


def build_bilstm_model(seq_len=12, units=64, dropout=0.2):
    """
    Bidirectional LSTM model.
    Outputs: [basal_pred, bolus_pred]
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow is required for BiLSTM model.")

    inp = Input(shape=(seq_len, 1), name='glucose_seq')
    x = Bidirectional(LSTM(units, return_sequences=True), name='bilstm_1')(inp)
    x = Dropout(dropout)(x)
    x = Bidirectional(LSTM(units // 2, return_sequences=False), name='bilstm_2')(x)
    x = BatchNormalization()(x)
    x = Dense(32, activation='relu')(x)
    basal = Dense(1, activation='linear', name='basal')(x)
    bolus = Dense(1, activation='linear', name='bolus')(x)
    model = Model(inputs=inp, outputs=[basal, bolus], name='BiLSTMModel')
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss={'basal': 'mse', 'bolus': 'mse'},
        metrics={'basal': 'mae', 'bolus': 'mae'},
    )
    return model


# ─── Training wrapper ─────────────────────────────────────────────────────────

class DeepInsulinPredictor:
    """
    Wrapper that trains LSTM or BiLSTM on subject-specific data and
    provides predict() with the same interface as AP-DRL.
    """

    def __init__(self, model_type='lstm', seq_len=12, units=64):
        self.model_type = model_type.lower()
        self.seq_len = seq_len
        self.units = units
        self.model = None
        self._gl_mu = self._gl_sigma = None
        self._ba_mu = self._ba_sigma = None
        self._bo_mu = self._bo_sigma = None

    def fit(self, glucose_arr, basal_arr, bolus_arr,
            epochs=50, batch_size=32, verbose=0):
        """
        Train on arrays of shape (N,).
        glucose_arr in mg/dL, basal/bolus in IU.
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow required.")

        gl_norm, self._gl_mu, self._gl_sigma = normalise(glucose_arr)
        ba_norm, self._ba_mu, self._ba_sigma = normalise(basal_arr)
        bo_norm, self._bo_mu, self._bo_sigma = normalise(bolus_arr)

        X, y_ba, y_bo = prepare_sequences(gl_norm, ba_norm, bo_norm, self.seq_len)

        if self.model_type == 'bilstm':
            self.model = build_bilstm_model(self.seq_len, self.units)
        else:
            self.model = build_lstm_model(self.seq_len, self.units)

        es = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
        self.model.fit(
            X, {'basal': y_ba, 'bolus': y_bo},
            epochs=epochs, batch_size=batch_size,
            verbose=verbose, callbacks=[es],
        )
        return self

    def predict(self, glucose_window):
        """
        glucose_window: list/array of last `seq_len` glucose readings (mg/dL)
        Returns: (basal_pred, bolus_pred) in IU
        """
        if self.model is None:
            raise RuntimeError("Call fit() before predict().")

        gl_norm = (np.array(glucose_window, dtype=np.float32) - self._gl_mu) / self._gl_sigma
        X = gl_norm.reshape(1, self.seq_len, 1)
        ba_pred, bo_pred = self.model.predict(X, verbose=0)

        basal = float(denormalise(ba_pred[0, 0], self._ba_mu, self._ba_sigma))
        bolus = float(denormalise(bo_pred[0, 0], self._bo_mu, self._bo_sigma))
        return max(0.0, basal), max(0.0, bolus)

    def evaluate(self, glucose_arr, basal_arr, bolus_arr):
        """
        Returns dict with MAE and RMSE for basal and bolus.
        """
        preds_ba, preds_bo = [], []
        for i in range(self.seq_len, len(glucose_arr)):
            window = glucose_arr[i - self.seq_len: i]
            ba, bo = self.predict(window)
            preds_ba.append(ba)
            preds_bo.append(bo)

        actual_ba = basal_arr[self.seq_len:]
        actual_bo = bolus_arr[self.seq_len:]
        n = len(preds_ba)

        mae_ba  = float(np.mean(np.abs(np.array(preds_ba)  - actual_ba[:n])))
        rmse_ba = float(np.sqrt(np.mean((np.array(preds_ba)  - actual_ba[:n])**2)))
        mae_bo  = float(np.mean(np.abs(np.array(preds_bo)  - actual_bo[:n])))
        rmse_bo = float(np.sqrt(np.mean((np.array(preds_bo)  - actual_bo[:n])**2)))

        return {
            'basal': {'mae': round(mae_ba, 4), 'rmse': round(rmse_ba, 4)},
            'bolus': {'mae': round(mae_bo, 4), 'rmse': round(rmse_bo, 4)},
        }


# ─── Quick test ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import numpy as np
    np.random.seed(42)
    N = 300
    gl   = np.random.normal(130, 30, N).clip(70, 250)
    ba   = np.maximum(0.05, (gl - 110) / 50 + np.random.normal(0, 0.05, N))
    bo   = np.maximum(0.0,  (gl - 120) / 40 + np.random.normal(0, 0.1,  N))

    for mtype in ('lstm', 'bilstm'):
        m = DeepInsulinPredictor(model_type=mtype, seq_len=12)
        m.fit(gl, ba, bo, epochs=10, verbose=0)
        metrics = m.evaluate(gl, ba, bo)
        print(f"\n[{mtype.upper()}] metrics: {metrics}")
        window = gl[:12]
        pred_ba, pred_bo = m.predict(window)
        print(f"  Sample predict — basal={pred_ba:.4f} IU/H, bolus={pred_bo:.4f} IU")
