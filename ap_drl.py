"""
Adaptive Predictive Deep Reinforcement Learning (AP-DRL)
=========================================================
Core algorithm from the paper:
  "Ubiquitous Artificial Pancreas: Blockchain-Secured AI-Driven
   Digital Twin for IoT-Enabled Insulin Pumps in Type 1 Diabetes Management"

Implements Algorithm 1 from the paper exactly:
  - State: [G_norm, I_norm, D_norm, A_norm, trend]
  - Action: discrete insulin dosage levels
  - Reward: -|G_target - G(t)| - λ·I(t)
  - Model: Minimal Model for glucose-insulin dynamics (Bergman)
  - Learning: Deep Q-Network (DQN) with experience replay
  - Policy: ε-greedy with decay
  - Personalisation: subject-specific learning_rate and gamma

Subject-specific hyperparameters (from Table 3 of the paper):
  Subject 1001 bolus: lr=0.005,  γ=0.80  | basal: lr=0.003,  γ=0.90
  Subject 1002 bolus: lr=0.0005, γ=0.995 | basal: lr=0.00001,γ=0.99
  Subject 1006 bolus: lr=0.0001, γ=0.95  | basal: lr=0.05,   γ=0.90
  Subject 1012 bolus: lr=0.001,  γ=0.90  | basal: lr=0.001,  γ=0.995
"""

import json
import random
from collections import deque

import numpy as np

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Input
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


# ─── Subject-specific hyperparameters from Table 3 ───────────────────────────

SUBJECT_HYPERPARAMS = {
    '1001': {'bolus': {'lr': 0.005,   'gamma': 0.80},  'basal': {'lr': 0.003,   'gamma': 0.90}},
    '1002': {'bolus': {'lr': 0.0005,  'gamma': 0.995}, 'basal': {'lr': 0.00001, 'gamma': 0.99}},
    '1006': {'bolus': {'lr': 0.0001,  'gamma': 0.95},  'basal': {'lr': 0.05,    'gamma': 0.90}},
    '1012': {'bolus': {'lr': 0.001,   'gamma': 0.90},  'basal': {'lr': 0.001,   'gamma': 0.995}},
}

# Default hyperparameters for unknown subjects
DEFAULT_HYPERPARAMS = {'bolus': {'lr': 0.001, 'gamma': 0.90}, 'basal': {'lr': 0.001, 'gamma': 0.95}}


# ─── Bergman Minimal Model (glucose-insulin dynamics) ────────────────────────

class MinimalModel:
    """
    Bergman Minimal Model for glucose-insulin dynamics.
    Implements Algorithm 1, lines 6 and 20.

    dG/dt = -p1·G(t) - X(t)·G(t) + p2
    dX/dt = -p3·X(t) + p4·(I(t) - Ib)

    State:
        G  : plasma glucose (mg/dL)
        X  : remote insulin effect
        Ib : basal insulin

    Typical parameter ranges (from literature):
        p1: 0.028–0.036 /min
        p2: glucose production rate (≈ p1 * Gb)
        p3: 0.01–0.05 /min
        p4: insulin action rate
    """

    def __init__(self, Gb=120.0, Ib=0.05,
                 p1=0.03, p3=0.025, p4=1.5e-4,
                 dt=5.0):
        """
        Gb : basal glucose (mg/dL)
        Ib : basal insulin (IU/H)
        dt : time step in minutes
        """
        self.Gb = Gb
        self.Ib = Ib
        self.p1 = p1
        self.p2 = p1 * Gb          # Steady-state constraint: dG/dt=0 → p2 = p1*Gb
        self.p3 = p3
        self.p4 = p4
        self.dt = dt / 60.0        # Convert minutes → hours for IU/H compatibility

        # State
        self.G = Gb
        self.X = 0.0

    def reset(self, G0=None):
        self.G = G0 if G0 is not None else self.Gb
        self.X = 0.0
        return self.G

    def step(self, insulin):
        """
        Advance one time step given insulin dose (IU or IU/H).
        Returns new glucose level.
        """
        dG = -self.p1 * self.G - self.X * self.G + self.p2
        dX = -self.p3 * self.X + self.p4 * (insulin - self.Ib)
        self.G += dG * self.dt
        self.X += dX * self.dt
        self.G = max(20.0, min(600.0, self.G))   # Physiological limits
        return self.G


# ─── DQN Network ─────────────────────────────────────────────────────────────

def _build_dqn(state_dim, n_actions, lr):
    """Q-network used for both online and target networks."""
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow is required for AP-DRL.")

    model = Sequential([
        Input(shape=(state_dim,)),
        Dense(128, activation='relu'),
        Dense(64,  activation='relu'),
        Dense(32,  activation='relu'),
        Dense(n_actions, activation='linear'),
    ], name='DQN')
    model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
    return model


# ─── AP-DRL Agent ─────────────────────────────────────────────────────────────

class APDRLAgent:
    """
    Adaptive Predictive Deep Reinforcement Learning agent.
    Implements Algorithm 1 from the paper in full.

    Parameters
    ----------
    subject_id  : str   — used to look up personalised hyperparameters
    mode        : str   — 'bolus' or 'basal'
    target_bg   : float — target blood glucose (mg/dL), default 100
    lambda_      : float — penalty weight on insulin in reward
    n_actions   : int   — number of discrete insulin dose levels
    max_dose    : float — maximum single dose (IU)
    replay_size : int   — experience replay buffer capacity
    batch_size  : int   — minibatch size for gradient descent
    """

    STATE_DIM = 5   # [G_norm, I_norm, D_norm, A_norm, trend]

    def __init__(self, subject_id='default', mode='bolus',
                 target_bg=100.0, lambda_=0.1,
                 n_actions=11, max_dose=5.0,
                 replay_size=10_000, batch_size=32,
                 epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995):

        self.subject_id = str(subject_id)
        self.mode = mode
        self.target_bg = target_bg
        self.lambda_ = lambda_
        self.n_actions = n_actions
        self.max_dose = max_dose
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Personalised hyperparameters (Algorithm 1, line 19)
        hp = SUBJECT_HYPERPARAMS.get(self.subject_id, DEFAULT_HYPERPARAMS)
        mode_hp = hp.get(mode, DEFAULT_HYPERPARAMS['bolus'])
        self.lr    = mode_hp['lr']
        self.gamma = mode_hp['gamma']

        # Discrete action space: {0, Δ1, Δ2, …, Δn}
        self.action_space = np.linspace(0.0, max_dose, n_actions, dtype=np.float32)

        # Replay memory (Algorithm 1, line 7)
        self.memory = deque(maxlen=replay_size)

        # Q-networks (online + target)
        self._online = None
        self._target = None
        self._step_count = 0
        self._target_update_freq = 50   # Sync target network every N steps

        # Minimal model (Algorithm 1, line 6)
        self.minimal_model = MinimalModel()

        # Running stats for normalisation
        self._gl_mu, self._gl_sigma = 120.0, 30.0
        self._in_mu, self._in_sigma = 0.5, 0.5

    # ── Network initialisation ────────────────────────────────────────────────

    def _ensure_networks(self):
        if self._online is None:
            if not TF_AVAILABLE:
                raise ImportError("TensorFlow is required.")
            self._online = _build_dqn(self.STATE_DIM, self.n_actions, self.lr)
            self._target = _build_dqn(self.STATE_DIM, self.n_actions, self.lr)
            self._target.set_weights(self._online.get_weights())

    # ── State construction (Algorithm 1, line 3) ─────────────────────────────

    def _make_state(self, glucose, insulin, diet=0.0, activity=0.0, prev_glucose=None):
        """
        State vector: [G_norm, I_norm, D_norm, A_norm, trend]
        trend = (G_t - G_{t-1}) / sigma_G  (rate of change)
        """
        G_norm = (glucose - self._gl_mu) / self._gl_sigma
        I_norm = (insulin  - self._in_mu) / self._in_sigma
        D_norm = diet / 50.0       # normalise by typical meal (50g carbs)
        A_norm = activity / 60.0   # normalise by 60 min exercise
        trend  = (glucose - prev_glucose) / self._gl_sigma if prev_glucose is not None else 0.0
        return np.array([G_norm, I_norm, D_norm, A_norm, trend], dtype=np.float32)

    # ── Reward function (Algorithm 1, line 5) ────────────────────────────────

    def _reward(self, glucose, insulin_given):
        """Rt = -|G_target - G(t)| - λ·I(t)"""
        return -abs(self.target_bg - glucose) - self.lambda_ * insulin_given

    # ── Action selection (Algorithm 1, line 11) ───────────────────────────────

    def select_action(self, state):
        """ε-greedy policy."""
        self._ensure_networks()
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        q = self._online.predict(state[np.newaxis], verbose=0)[0]
        return int(np.argmax(q))

    def action_to_dose(self, action_idx):
        return float(self.action_space[action_idx])

    # ── Experience replay (Algorithm 1, lines 13–15) ──────────────────────────

    def remember(self, state, action, reward, next_state, done=False):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return None

        self._ensure_networks()
        batch = random.sample(self.memory, self.batch_size)
        states      = np.array([e[0] for e in batch], dtype=np.float32)
        actions     = np.array([e[1] for e in batch], dtype=np.int32)
        rewards     = np.array([e[2] for e in batch], dtype=np.float32)
        next_states = np.array([e[3] for e in batch], dtype=np.float32)
        dones       = np.array([e[4] for e in batch], dtype=np.float32)

        # Bellman target (Algorithm 1, line 15)
        q_next   = self._target.predict(next_states, verbose=0)
        q_target = self._online.predict(states, verbose=0)

        for i in range(self.batch_size):
            target_val = rewards[i]
            if not dones[i]:
                target_val += self.gamma * np.max(q_next[i])
            q_target[i][actions[i]] = target_val

        loss = self._online.train_on_batch(states, q_target)

        # Epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Sync target network
        self._step_count += 1
        if self._step_count % self._target_update_freq == 0:
            self._target.set_weights(self._online.get_weights())

        return float(loss) if not isinstance(loss, (list, tuple)) else float(loss[0])

    # ── Training loop (Algorithm 1, lines 8–18) ───────────────────────────────

    def train(self, glucose_arr, basal_arr, bolus_arr,
              episodes=30, verbose=True):
        """
        Train on subject-specific time-series data.

        glucose_arr: array (N,) in mg/dL
        basal_arr:   array (N,) in IU/H
        bolus_arr:   array (N,) in IU
        """
        self._ensure_networks()
        insulin_arr = basal_arr if self.mode == 'basal' else bolus_arr

        # Update normalisation stats
        self._gl_mu, self._gl_sigma = float(np.mean(glucose_arr)), float(np.std(glucose_arr) + 1e-8)
        self._in_mu, self._in_sigma = float(np.mean(insulin_arr)), float(np.std(insulin_arr) + 1e-8)

        # Update minimal model basal reference
        self.minimal_model.Gb = self._gl_mu
        self.minimal_model.Ib = self._in_mu
        self.minimal_model.p2 = self.minimal_model.p1 * self._gl_mu

        history = []
        N = len(glucose_arr)

        for episode in range(episodes):
            total_reward = 0.0
            self.minimal_model.reset(glucose_arr[0])
            prev_g = None

            # Shuffle start point (except first episode)
            start = random.randint(0, max(0, N // 4)) if episode > 0 else 0

            for t in range(start, N - 1):
                g = glucose_arr[t]
                ins = insulin_arr[t]
                state = self._make_state(g, ins, prev_glucose=prev_g)

                action = self.select_action(state)
                dose   = self.action_to_dose(action)

                # Simulate glucose response (Algorithm 1, line 12 / 20)
                g_next = self.minimal_model.step(dose)
                reward = self._reward(g_next, dose)

                next_ins = insulin_arr[min(t + 1, N - 1)]
                next_state = self._make_state(g_next, next_ins, prev_glucose=g)

                self.remember(state, action, reward, next_state)
                loss = self.replay()
                total_reward += reward
                prev_g = g

            avg_reward = total_reward / (N - start)
            history.append({'episode': episode, 'avg_reward': avg_reward, 'epsilon': self.epsilon})
            if verbose and (episode % 5 == 0 or episode == episodes - 1):
                print(f"  [AP-DRL {self.subject_id}/{self.mode}] "
                      f"Ep {episode+1:3d}/{episodes} | "
                      f"avg_reward={avg_reward:.2f} | ε={self.epsilon:.4f}")

        return history

    # ── Prediction (Algorithm 1, line 21) ────────────────────────────────────

    def predict(self, glucose, current_insulin=None, diet=0.0, activity=0.0, prev_glucose=None):
        """
        Predict optimal insulin dose for current state.
        Returns: dose (float, IU or IU/H)
        """
        self._ensure_networks()
        ins = current_insulin if current_insulin is not None else self._in_mu
        state = self._make_state(glucose, ins, diet, activity, prev_glucose)
        q = self._online.predict(state[np.newaxis], verbose=0)[0]
        action = int(np.argmax(q))
        return self.action_to_dose(action)

    # ── Evaluation metrics ───────────────────────────────────────────────────

    def evaluate(self, glucose_arr, basal_arr, bolus_arr):
        """
        Returns MAE and RMSE for basal and bolus predictions.
        """
        insulin_arr = basal_arr if self.mode == 'basal' else bolus_arr
        preds, actuals = [], []
        prev_g = None

        for t in range(len(glucose_arr) - 1):
            pred = self.predict(glucose_arr[t], insulin_arr[t], prev_glucose=prev_g)
            preds.append(pred)
            actuals.append(insulin_arr[t + 1])
            prev_g = glucose_arr[t]

        preds   = np.array(preds)
        actuals = np.array(actuals)
        mae     = float(np.mean(np.abs(preds - actuals)))
        rmse    = float(np.sqrt(np.mean((preds - actuals) ** 2)))
        return {'mae': round(mae, 4), 'rmse': round(rmse, 4)}

    # ── Serialisation ────────────────────────────────────────────────────────

    def save(self, path):
        """Save model weights and agent metadata to a directory."""
        import os, json
        os.makedirs(path, exist_ok=True)
        if self._online is not None:
            self._online.save_weights(os.path.join(path, 'online.weights.h5'))
        meta = {
            'subject_id': self.subject_id,
            'mode':       self.mode,
            'target_bg':  self.target_bg,
            'lambda_':    self.lambda_,
            'n_actions':  self.n_actions,
            'max_dose':   self.max_dose,
            'epsilon':    self.epsilon,
            'lr':         self.lr,
            'gamma':      self.gamma,
            'gl_mu':      self._gl_mu,
            'gl_sigma':   self._gl_sigma,
            'in_mu':      self._in_mu,
            'in_sigma':   self._in_sigma,
        }
        with open(os.path.join(path, 'meta.json'), 'w') as f:
            json.dump(meta, f, indent=2)
        print(f"  AP-DRL saved to {path}")

    def load(self, path):
        """Restore agent from a saved directory."""
        import os, json
        meta_path = os.path.join(path, 'meta.json')
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"No meta.json in {path}")
        with open(meta_path) as f:
            meta = json.load(f)
        self.subject_id = meta['subject_id']
        self.mode       = meta['mode']
        self.target_bg  = meta['target_bg']
        self.epsilon    = meta['epsilon']
        self._gl_mu     = meta['gl_mu']
        self._gl_sigma  = meta['gl_sigma']
        self._in_mu     = meta['in_mu']
        self._in_sigma  = meta['in_sigma']
        self._ensure_networks()
        weights_path = os.path.join(path, 'online.weights.h5')
        if os.path.exists(weights_path):
            self._online.load_weights(weights_path)
            self._target.set_weights(self._online.get_weights())
        return self


# ─── Quick self-test ──────────────────────────────────────────────────────────

if __name__ == '__main__':
    np.random.seed(42)
    N = 200
    gl   = np.random.normal(140, 35, N).clip(70, 280).astype(np.float32)
    ba   = np.maximum(0.02, (gl - 110) / 60 + np.random.normal(0, 0.03, N)).astype(np.float32)
    bo   = np.maximum(0.0, (gl - 120) / 40 + np.random.normal(0, 0.1, N)).astype(np.float32)

    for sid in ['1001', '1012']:
        for mode in ['basal', 'bolus']:
            agent = APDRLAgent(subject_id=sid, mode=mode, n_actions=11, max_dose=5.0)
            print(f"\n=== Subject {sid} / {mode} | lr={agent.lr} γ={agent.gamma} ===")
            agent.train(gl, ba, bo, episodes=5, verbose=True)
            metrics = agent.evaluate(gl, ba, bo)
            print(f"  MAE={metrics['mae']:.4f}  RMSE={metrics['rmse']:.4f}")
