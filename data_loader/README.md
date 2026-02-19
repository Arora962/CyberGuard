# T1D Dataset Loaders

This module supports loading data from multiple Type 1 Diabetes datasets for use in the insulin pump simulation.

## Supported Datasets

### 1. AZT1D Dataset
- **Source**: [Mendeley Data](https://data.mendeley.com/datasets/gk9m674wcx/1)
- **Contents**: CGM data, insulin dosing, carb intake, device mode labels from 25 individuals
- **Format**: CSV (download from Mendeley)
- **Usage**: Place downloaded files in `data/external/azt1d/` and run the loader

### 2. OhioT1DM Dataset
- **Source**: [IEEE Dataport](https://ieee-dataport.org/open-access/ohio-type-1-diabetes-dataset) or [Request DUA](https://ohio.qualtrics.com/jfe/form/SV_02QtWEVm7ARIKIl)
- **Contents**: 8 weeks of CGM (5-min), insulin, meals, exercise, sleep per subject
- **Format**: XML files per subject
- **Usage**: Place XML files in `data/external/ohiot1dm/` after obtaining access

### 3. OpenAPS Data Commons
- **Source**: [Open Humans](https://www.openhumans.org/activity/openaps-data-commons/)
- **Contents**: Anonymized OpenAPS user data
- **Requires**: Free Open Humans account

### 4. DiaData
- **Source**: [arXiv paper](https://arxiv.org/abs/2508.09160) - check for dataset release
- **Contents**: ~2,510 subjects, 149M CGM points

## Quick Start (Synthetic Data)

When no external data is available, the loader generates **synthetic data** based on typical T1D parameters from your `simdata/profile.json`:

```python
from data_loader import load_simdata

glucose_data, profile, training_pairs = load_simdata()
```

## Using Real Data

1. Download data from one of the sources above
2. Place in `data/external/<dataset_name>/`
3. Set `DATA_SOURCE` in your config or pass to the loader
4. Run the simulation - it will auto-detect and convert the format
