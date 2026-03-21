"""
main_pipeline.py
================
End-to-end pipeline for the Ubiquitous Artificial Pancreas system.
Implements the complete workflow described in the paper (Sections 3–4).

Workflow:
  1. Load ShangaiT1DM (or synthetic) data per subject
  2. Train Digital Twin (LSTM + BiLSTM + AP-DRL) on subject data
  3. Simulate IoT pump → Digital Twin → Blockchain loop
  4. Evaluate all models and print Table-2-style results
  5. Save predictions and blockchain ledger

Usage:
  python main_pipeline.py
  python main_pipeline.py --subjects 1001 1012 --episodes 20 --synthetic
"""

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone

import numpy as np


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _print_section(title: str):
    bar = '─' * 60
    print(f"\n{bar}\n  {title}\n{bar}")


def _results_table(results: dict, subject_id: str):
    """Print Table-2-style performance metrics."""
    print(f"\n{'─'*70}")
    print(f"  Performance Table — Subject {subject_id}")
    print(f"  {'Method':<14} {'Bolus MAE':>10} {'Bolus RMSE':>11} {'Basal MAE':>10} {'Basal RMSE':>11}")
    print(f"  {'─'*66}")

    # LSTM
    if 'LSTM' in results:
        r = results['LSTM']
        ba = r.get('basal', {})
        bo = r.get('bolus', {})
        print(f"  {'LSTM':<14} {bo.get('mae', '-'):>10.4f} {bo.get('rmse', '-'):>11.4f} "
              f"{ba.get('mae', '-'):>10.4f} {ba.get('rmse', '-'):>11.4f}")

    # BiLSTM
    if 'BiLSTM' in results:
        r = results['BiLSTM']
        ba = r.get('basal', {})
        bo = r.get('bolus', {})
        print(f"  {'BiLSTM':<14} {bo.get('mae', '-'):>10.4f} {bo.get('rmse', '-'):>11.4f} "
              f"{ba.get('mae', '-'):>10.4f} {ba.get('rmse', '-'):>11.4f}")

    # AP-DRL
    basal_m = results.get('AP-DRL_basal', {})
    bolus_m = results.get('AP-DRL_bolus', {})
    print(f"  {'AP-DRL':<14} {bolus_m.get('mae', '-'):>10.4f} {bolus_m.get('rmse', '-'):>11.4f} "
          f"{basal_m.get('mae', '-'):>10.4f} {basal_m.get('rmse', '-'):>11.4f}")
    print(f"{'─'*70}")


# ─── Synthetic data fallback ──────────────────────────────────────────────────

def _generate_subject_data(subject_id: str, n: int = 300, seed: int = 42):
    """Generate realistic synthetic subject data for testing."""
    rng = np.random.default_rng(seed + int(subject_id) if subject_id.isdigit() else seed)
    profile = {'target_bg': 110, 'sens': 50, 'carb_ratio': 10}

    # Different glucose patterns per subject (simulate personalisation)
    mean_g = {
        '1001': 145.0, '1002': 128.0, '1006': 155.0, '1012': 132.0,
    }.get(subject_id, 140.0)

    glucose = np.clip(rng.normal(mean_g, 30, n), 60, 280).astype(np.float32)
    basal   = np.maximum(0.02, (glucose - 110) / 50 + rng.normal(0, 0.03, n)).astype(np.float32)
    bolus   = np.maximum(0.0,  np.where(rng.random(n) < 0.3,
                                         (glucose - 110) / 40 + rng.normal(0, 0.1, n), 0.0)).astype(np.float32)
    return glucose, basal, bolus, profile


# ─── Per-subject pipeline ─────────────────────────────────────────────────────

def run_subject(subject_id: str, glucose_arr, basal_arr, bolus_arr,
                profile: dict, blockchain, output_dir: str,
                epochs_lstm: int = 30, episodes_apdrl: int = 20,
                verbose: bool = True):
    """
    Full Digital Twin + Blockchain pipeline for one subject.
    """
    from digital_twin import DigitalTwin

    _print_section(f"Subject {subject_id}")

    # ── 1. Create Digital Twin ─────────────────────────────────────────────
    twin = DigitalTwin(
        subject_id  = subject_id,
        profile     = profile,
        blockchain  = blockchain,
        models_dir  = os.path.join(output_dir, 'models', subject_id),
        seq_len     = 12,
    )

    # ── 2. Train all models ────────────────────────────────────────────────
    twin.train_models(
        glucose_arr, basal_arr, bolus_arr,
        epochs_lstm   = epochs_lstm,
        episodes_apdrl = episodes_apdrl,
        verbose        = verbose,
    )

    # ── 3. Evaluate (Table 2) ─────────────────────────────────────────────
    results = twin.evaluate_all_models(glucose_arr, basal_arr, bolus_arr)
    _results_table(results, subject_id)

    # ── 4. Simulate IoT pump → Digital Twin → Blockchain loop ─────────────
    _print_section(f"IoT Pump → Blockchain Simulation — Subject {subject_id}")
    simulation_steps = min(5, len(glucose_arr) - 1)
    physician_id = list(blockchain.validators.keys())[0]

    for step in range(simulation_steps):
        glucose = float(glucose_arr[step])
        print(f"\n  [Step {step+1}] IoT Pump CGM reading: {glucose:.1f} mg/dL")

        # IoT pump sends glucose to digital twin (Step 2a)
        twin.ingest_cgm(glucose)

        # Digital twin proposes insulin to blockchain (Step 2b → 3)
        proposal = twin.propose_to_blockchain(glucose)
        tx_id    = proposal['tx_id']
        preds    = proposal['predictions']
        print(f"  [Step {step+1}] AP-DRL proposes → basal={preds['ap_drl_basal']:.4f}, "
              f"bolus={preds['ap_drl_bolus']:.4f}")

        # Physician approves (Step 4) — auto-approve in simulation
        if tx_id:
            approval = blockchain.approve_transaction(tx_id, physician_id, approved=True)
            if approval['success']:
                print(f"  [Step {step+1}] Physician {physician_id} approved ✓")

        # Insulin pump reads latest approved dose (Step 5)
        dose = twin.apply_approved_dose()

    # ── 5. Save results ────────────────────────────────────────────────────
    subject_output = {
        'subject_id': subject_id,
        'timestamp':  datetime.now(timezone.utc).isoformat(),
        'metrics':    results,
        'twin_status': twin.status(),
        'blockchain_summary': blockchain.chain_summary(),
    }
    out_path = os.path.join(output_dir, f'results_{subject_id}.json')
    with open(out_path, 'w') as f:
        json.dump(subject_output, f, indent=2, default=str)
    print(f"\n  Results saved → {out_path}")

    return results


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Ubiquitous Artificial Pancreas — Full Pipeline')
    parser.add_argument('--subjects',    nargs='+', default=['1001', '1002', '1006', '1012'],
                        help='Subject IDs to process')
    parser.add_argument('--data_dir',   default='data/external/shanghait1dm',
                        help='Path to ShangaiT1DM CSV files')
    parser.add_argument('--output_dir', default='output',
                        help='Output directory for results')
    parser.add_argument('--epochs_lstm',     type=int, default=30,
                        help='Epochs for LSTM/BiLSTM training')
    parser.add_argument('--episodes_apdrl', type=int, default=20,
                        help='Episodes for AP-DRL training')
    parser.add_argument('--synthetic', action='store_true',
                        help='Use synthetic data (skip ShangaiT1DM loading)')
    parser.add_argument('--quiet',      action='store_true', help='Reduce verbosity')
    args = parser.parse_args()

    verbose = not args.quiet
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Blockchain setup ──────────────────────────────────────────────────
    from blockchain import PrivateBlockchain

    ledger_path = os.path.join(args.output_dir, 'blockchain_ledger.json')
    if os.path.exists(ledger_path):
        os.remove(ledger_path)

    all_readers = args.subjects + ['physician_vit', 'hospital_vit']
    blockchain = PrivateBlockchain(
        ledger_path=ledger_path,
        validators={
            'physician_vit': 'secret_key_physician_vit_2024',
            'hospital_vit':  'secret_key_hospital_vit_2024',
        },
        authorised_readers=all_readers,
    )
    _print_section("Blockchain initialised")
    print(json.dumps(blockchain.chain_summary(), indent=2))

    # ── Load data ─────────────────────────────────────────────────────────
    subjects_data = {}

    if not args.synthetic:
        try:
            root = os.path.dirname(os.path.abspath(__file__))
            data_dir = os.path.join(root, args.data_dir)
            from data_loader.load_shanghait1dm import load_shanghait1dm
            subjects, _, _, profile = load_shanghait1dm(data_dir)
            for sid, sdata in subjects.items():
                if sid in args.subjects:
                    pairs = sdata['training_pairs']
                    gl    = np.array([p[0] for p in pairs], dtype=np.float32)
                    ba    = np.array([p[1] for p in pairs], dtype=np.float32)
                    bo    = np.array([p[2] for p in pairs], dtype=np.float32)
                    subjects_data[sid] = (gl, ba, bo, sdata['profile'])
            if subjects_data:
                _print_section(f"ShangaiT1DM loaded: {list(subjects_data.keys())}")
            else:
                print("[Warning] No matching subjects found in ShangaiT1DM — falling back to synthetic data.")
        except Exception as e:
            print(f"[Warning] Could not load ShangaiT1DM ({e}) — using synthetic data.")

    # Fill missing subjects with synthetic data
    for sid in args.subjects:
        if sid not in subjects_data:
            gl, ba, bo, prof = _generate_subject_data(sid)
            subjects_data[sid] = (gl, ba, bo, prof)
            if verbose:
                print(f"  [Synthetic data] Subject {sid}: {len(gl)} time steps")

    # ── Run pipeline per subject ──────────────────────────────────────────
    all_results = {}
    for sid in args.subjects:
        if sid not in subjects_data:
            continue
        gl, ba, bo, prof = subjects_data[sid]
        results = run_subject(
            subject_id     = sid,
            glucose_arr    = gl,
            basal_arr      = ba,
            bolus_arr      = bo,
            profile        = prof,
            blockchain     = blockchain,
            output_dir     = args.output_dir,
            epochs_lstm    = args.epochs_lstm,
            episodes_apdrl = args.episodes_apdrl,
            verbose        = verbose,
        )
        all_results[sid] = results

    # ── Final summary ─────────────────────────────────────────────────────
    _print_section("Final Blockchain Summary")
    print(json.dumps(blockchain.chain_summary(), indent=2))
    print(f"\n✓ Blockchain valid: {blockchain.is_valid()}")

    _print_section("All-Subject Summary (AP-DRL)")
    print(f"\n  {'Subject':<10} {'Bolus MAE':>10} {'Bolus RMSE':>11} {'Basal MAE':>10} {'Basal RMSE':>11}")
    print(f"  {'─'*56}")
    for sid, res in all_results.items():
        ba = res.get('AP-DRL_basal', {})
        bo = res.get('AP-DRL_bolus', {})
        print(f"  {sid:<10} {bo.get('mae','-'):>10.4f} {bo.get('rmse','-'):>11.4f} "
              f"{ba.get('mae','-'):>10.4f} {ba.get('rmse','-'):>11.4f}")

    summary_path = os.path.join(args.output_dir, 'all_results.json')
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Full results saved → {summary_path}")


if __name__ == '__main__':
    main()
