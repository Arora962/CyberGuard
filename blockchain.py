"""
blockchain.py
=============
Simulated Private Blockchain for the Ubiquitous Artificial Pancreas system.

Implements the architecture described in Section 3.4 of the paper:
  - Private blockchain (Proof of Authority consensus)
  - Smart contracts for data access control
  - Asymmetric encryption simulation (RSA-like using hashlib)
  - Immutable block storage (JSON-file backed ledger)
  - Physician approval workflow before insulin level is committed

Block structure:
  {
    "index":       int,
    "timestamp":   ISO-8601 string,
    "data":        { ... },          ← glucose / predicted insulin / approved insulin
    "previous_hash": str,
    "hash":        str,
    "validator":   str,              ← physician/hospital ID (PoA)
    "signature":   str,              ← hex digest simulating cryptographic signature
    "approved":    bool,
    "nonce":       int
  }
"""

import hashlib
import json
import os
import time
from datetime import datetime, timezone


# ─── Utility helpers ─────────────────────────────────────────────────────────

def _sha256(data: str) -> str:
    return hashlib.sha256(data.encode('utf-8')).hexdigest()


def _sign(private_key: str, payload: str) -> str:
    """Simulate cryptographic signature: HMAC-SHA256(private_key, payload)."""
    import hmac
    return hmac.new(private_key.encode(), payload.encode(), hashlib.sha256).hexdigest()


def _verify_signature(public_key: str, payload: str, signature: str) -> bool:
    """Verify simulated signature."""
    expected = _sign(public_key, payload)     # In real PoA public_key ≡ private_key for HMAC
    return expected == signature


# ─── Smart Contract ──────────────────────────────────────────────────────────

class SmartContract:
    """
    Self-executing access control contract.
    Rules:
      - Only registered validators (physicians/hospitals) can approve blocks.
      - Only authorised users (patient + their physicians) can read data.
      - Insulin adjustments > MAX_ADJUSTMENT require physician approval.

    Corresponds to Section 3.4 "Smart Contracts" in the paper.
    """

    MAX_ADJUSTMENT_IU = 2.0   # Doses beyond this ALWAYS require approval

    def __init__(self, authorised_validators: list, authorised_readers: list):
        self.validators = set(authorised_validators)
        self.readers    = set(authorised_readers)

    def can_validate(self, entity_id: str) -> bool:
        return entity_id in self.validators

    def can_read(self, entity_id: str) -> bool:
        return entity_id in self.readers

    def requires_approval(self, predicted_dose: float, current_dose: float) -> bool:
        """Fire approval event if dose change is significant (Section 3 step 3)."""
        return abs(predicted_dose - current_dose) > self.MAX_ADJUSTMENT_IU

    def enforce(self, action: str, entity_id: str, **kwargs) -> dict:
        """
        Enforce contract rules.
        Returns {'allowed': bool, 'reason': str}.
        """
        if action == 'validate':
            if self.can_validate(entity_id):
                return {'allowed': True, 'reason': 'Authorised validator'}
            return {'allowed': False, 'reason': f'{entity_id} is not a registered validator'}

        if action == 'read':
            if self.can_read(entity_id):
                return {'allowed': True, 'reason': 'Authorised reader'}
            return {'allowed': False, 'reason': f'{entity_id} does not have read access'}

        if action == 'write_insulin':
            pred  = kwargs.get('predicted_dose', 0.0)
            curr  = kwargs.get('current_dose', 0.0)
            if self.requires_approval(pred, curr) and not kwargs.get('approved', False):
                return {'allowed': False,
                        'reason': f'Dose change {abs(pred-curr):.2f} IU exceeds threshold — physician approval required'}
            return {'allowed': True, 'reason': 'Within safe adjustment range or approved'}

        return {'allowed': False, 'reason': f'Unknown action: {action}'}


# ─── Block ───────────────────────────────────────────────────────────────────

class Block:
    def __init__(self, index, data, previous_hash, validator,
                 private_key, approved=False):
        self.index         = index
        self.timestamp     = datetime.now(timezone.utc).isoformat()
        self.data          = data
        self.previous_hash = previous_hash
        self.validator     = validator
        self.approved      = approved
        self.nonce         = 0
        self.hash          = self._compute_hash()
        self.signature     = _sign(private_key, self.hash)

    def _compute_hash(self) -> str:
        block_string = json.dumps({
            'index':         self.index,
            'timestamp':     self.timestamp,
            'data':          self.data,
            'previous_hash': self.previous_hash,
            'validator':     self.validator,
            'approved':      self.approved,
            'nonce':         self.nonce,
        }, sort_keys=True)
        return _sha256(block_string)

    def to_dict(self) -> dict:
        return {
            'index':         self.index,
            'timestamp':     self.timestamp,
            'data':          self.data,
            'previous_hash': self.previous_hash,
            'hash':          self.hash,
            'validator':     self.validator,
            'signature':     self.signature,
            'approved':      self.approved,
            'nonce':         self.nonce,
        }


# ─── Blockchain ──────────────────────────────────────────────────────────────

class PrivateBlockchain:
    """
    Private blockchain with Proof-of-Authority consensus.

    Usage (see Section 3.4 and Figure 1 of the paper):
      1. AP-DRL predicts insulin dose → add_pending_transaction()
      2. Smart contract fires event → physician is notified
      3. Physician approves/modifies → approve_transaction()
      4. Block is added → insulin pump reads latest_approved_dose()

    Parameters
    ----------
    ledger_path   : str  — path to persistent JSON ledger file
    validators    : dict — {entity_id: private_key} of authorised hospitals/physicians
    authorised_readers : list — patient IDs + physician IDs that may read data
    """

    GENESIS_DATA = {
        'type':    'genesis',
        'message': 'Ubiquitous Artificial Pancreas — Private Blockchain Genesis Block',
    }

    def __init__(self, ledger_path: str,
                 validators: dict = None,
                 authorised_readers: list = None):

        self.ledger_path = ledger_path
        os.makedirs(os.path.dirname(ledger_path) if os.path.dirname(ledger_path) else '.', exist_ok=True)

        self.validators  = validators  or {'hospital_1': 'secret_key_h1',
                                           'physician_1': 'secret_key_p1'}
        self.readers     = set(authorised_readers or list(self.validators.keys()) + ['patient_default'])

        self.contract = SmartContract(
            authorised_validators=list(self.validators.keys()),
            authorised_readers=list(self.readers),
        )

        self.chain   = []
        self.pending = []          # Transactions awaiting physician approval

        if os.path.exists(ledger_path):
            self._load()
        else:
            self._create_genesis()

    # ── Internal ─────────────────────────────────────────────────────────────

    def _create_genesis(self):
        validator_id  = list(self.validators.keys())[0]
        private_key   = self.validators[validator_id]
        genesis = Block(0, self.GENESIS_DATA, '0' * 64, validator_id, private_key, approved=True)
        self.chain.append(genesis.to_dict())
        self._save()

    def _save(self):
        with open(self.ledger_path, 'w') as f:
            json.dump({'chain': self.chain, 'pending': self.pending}, f, indent=2)

    def _load(self):
        with open(self.ledger_path) as f:
            data = json.load(f)
        self.chain   = data.get('chain', [])
        self.pending = data.get('pending', [])
        if not self.chain:
            self._create_genesis()

    def _latest_block(self) -> dict:
        return self.chain[-1]

    # ── Chain integrity ──────────────────────────────────────────────────────

    def is_valid(self) -> bool:
        """Verify immutability of the entire chain."""
        for i in range(1, len(self.chain)):
            curr = self.chain[i]
            prev = self.chain[i - 1]
            if curr['previous_hash'] != prev['hash']:
                return False
            # Recompute hash
            check_str = json.dumps({
                'index':         curr['index'],
                'timestamp':     curr['timestamp'],
                'data':          curr['data'],
                'previous_hash': curr['previous_hash'],
                'validator':     curr['validator'],
                'approved':      curr['approved'],
                'nonce':         curr['nonce'],
            }, sort_keys=True)
            if _sha256(check_str) != curr['hash']:
                return False
        return True

    # ── Step 2a / 2b: AI proposes insulin level ──────────────────────────────

    def add_pending_transaction(self, transaction: dict) -> str:
        """
        AP-DRL proposes a new bolus/basal insulin level.
        Stored in pending pool; smart contract fires event.

        transaction should contain:
          subject_id, glucose_reading, predicted_basal, predicted_bolus,
          current_basal, current_bolus, timestamp (optional)
        """
        if 'timestamp' not in transaction:
            transaction['timestamp'] = datetime.now(timezone.utc).isoformat()
        tx_id = _sha256(json.dumps(transaction, sort_keys=True))
        transaction['tx_id'] = tx_id
        transaction['type']  = 'insulin_proposal'

        # Smart contract check
        result = self.contract.enforce(
            'write_insulin',
            'system',
            predicted_dose=transaction.get('predicted_bolus', 0),
            current_dose=transaction.get('current_bolus', 0),
        )
        transaction['requires_approval'] = not result['allowed']
        transaction['contract_note']     = result['reason']
        self.pending.append(transaction)
        self._save()
        print(f"  [Blockchain] Pending tx {tx_id[:8]}… | {result['reason']}")
        return tx_id

    # ── Step 3 / 4: Physician approves / rejects ─────────────────────────────

    def approve_transaction(self, tx_id: str, physician_id: str,
                            approved: bool = True,
                            override_basal: float = None,
                            override_bolus: float = None) -> dict:
        """
        Physician (identified by physician_id) reviews and approves/rejects
        the pending transaction.  Corresponds to Step 4 in Figure 1.

        If approved:
          - The block is added to the chain with the physician's signature.
          - If override values are given, those replace the AP-DRL prediction.
        """
        # PoA: check physician is authorised validator
        result = self.contract.enforce('validate', physician_id)
        if not result['allowed']:
            return {'success': False, 'error': result['reason']}

        # Find transaction
        tx = next((t for t in self.pending if t.get('tx_id') == tx_id), None)
        if tx is None:
            return {'success': False, 'error': f'Transaction {tx_id} not found in pending pool'}

        private_key = self.validators[physician_id]

        # Apply physician overrides (Step 5 in Figure 1)
        if override_basal is not None:
            tx['approved_basal'] = override_basal
        else:
            tx['approved_basal'] = tx.get('predicted_basal', 0.0)

        if override_bolus is not None:
            tx['approved_bolus'] = override_bolus
        else:
            tx['approved_bolus'] = tx.get('predicted_bolus', 0.0)

        tx['physician_id'] = physician_id
        tx['approved']     = approved
        tx['decision_ts']  = datetime.now(timezone.utc).isoformat()

        if approved:
            block = Block(
                index         = len(self.chain),
                data          = tx,
                previous_hash = self._latest_block()['hash'],
                validator     = physician_id,
                private_key   = private_key,
                approved      = True,
            )
            self.chain.append(block.to_dict())

        # Remove from pending regardless of decision
        self.pending = [t for t in self.pending if t.get('tx_id') != tx_id]
        self._save()

        action = 'approved' if approved else 'rejected'
        print(f"  [Blockchain] Tx {tx_id[:8]}… {action} by {physician_id} → "
              f"basal={tx['approved_basal']:.4f} IU/H, bolus={tx['approved_bolus']:.4f} IU")

        return {
            'success':        True,
            'action':         action,
            'approved_basal': tx['approved_basal'],
            'approved_bolus': tx['approved_bolus'],
            'block_index':    len(self.chain) - 1 if approved else None,
        }

    # ── Step 5: Insulin pump reads latest approved dose ──────────────────────

    def latest_approved_dose(self, subject_id: str = None) -> dict:
        """
        Insulin pump reads the most recent approved block.
        Corresponds to Step 5 in Figure 1.
        """
        for block in reversed(self.chain):
            if not block.get('approved'):
                continue
            data = block.get('data', {})
            if data.get('type') == 'genesis':
                continue
            if subject_id and data.get('subject_id') != subject_id:
                continue
            return {
                'basal':       data.get('approved_basal', 0.0),
                'bolus':       data.get('approved_bolus', 0.0),
                'block_index': block['index'],
                'timestamp':   block['timestamp'],
                'validator':   block['validator'],
            }
        return {'basal': 0.05, 'bolus': 0.0, 'block_index': 0,
                'timestamp': datetime.now(timezone.utc).isoformat(), 'validator': 'genesis'}

    # ── Read access (smart contract enforced) ────────────────────────────────

    def read_chain(self, reader_id: str) -> list:
        """Return chain only if reader_id is authorised."""
        result = self.contract.enforce('read', reader_id)
        if not result['allowed']:
            raise PermissionError(result['reason'])
        return self.chain

    def chain_summary(self) -> dict:
        approved   = [b for b in self.chain if b.get('approved') and b['data'].get('type') != 'genesis']
        return {
            'total_blocks':    len(self.chain),
            'approved_blocks': len(approved),
            'pending':         len(self.pending),
            'chain_valid':     self.is_valid(),
        }

    # ── Convenience: auto-approve (for simulation / testing) ─────────────────

    def auto_approve_all(self, physician_id: str = None) -> list:
        """
        Approve all pending transactions automatically.
        Useful for automated simulation runs.
        """
        pid = physician_id or list(self.validators.keys())[0]
        results = []
        for tx in list(self.pending):
            r = self.approve_transaction(tx['tx_id'], pid)
            results.append(r)
        return results


# ─── Quick demo ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import tempfile, os

    ledger = os.path.join(tempfile.gettempdir(), 'test_ledger.json')
    if os.path.exists(ledger):
        os.remove(ledger)

    bc = PrivateBlockchain(
        ledger_path=ledger,
        validators={'hospital_vit': 'key_vit_2024', 'physician_anny': 'key_anny_2024'},
        authorised_readers=['patient_1001', 'hospital_vit', 'physician_anny'],
    )

    print("\n=== Chain summary (genesis only) ===")
    print(json.dumps(bc.chain_summary(), indent=2))

    # AP-DRL proposes dose
    tx_id = bc.add_pending_transaction({
        'subject_id':      '1001',
        'glucose_reading': 145.0,
        'predicted_basal': 0.19,
        'predicted_bolus': 0.41,
        'current_basal':   0.10,
        'current_bolus':   0.00,
    })

    # Physician approves with minor override
    result = bc.approve_transaction(
        tx_id, 'physician_anny', approved=True,
        override_basal=0.20,   # Physician adjusts basal slightly
    )
    print("\nApproval result:", json.dumps(result, indent=2))

    # Insulin pump reads latest dose
    dose = bc.latest_approved_dose(subject_id='1001')
    print("\nInsulin pump reads:", json.dumps(dose, indent=2))

    print("\n=== Chain valid:", bc.is_valid(), "===")
    print("=== Summary ===")
    print(json.dumps(bc.chain_summary(), indent=2))
