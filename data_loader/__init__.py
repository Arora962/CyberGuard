"""
Data loaders for T1D datasets (AZT1D, OhioT1DM, OpenAPS, DiaData).
Falls back to synthetic data when no external data is available.
"""

from .load_simdata import load_simdata, load_training_data

__all__ = ['load_simdata', 'load_training_data']
