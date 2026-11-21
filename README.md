# SemPQ-IoT: Semantic Encoding, Post-Quantum Security and Lyapunov Scheduling for Wearable ECG

Code accompanying the article by **M. A. Azzawi, H. M. Al-Juboori, R. Hassan, M. F. Nasrudin, and A. A. Abdulhussein**.

![SemPQ-IoT: Semantic Encoding](sempq-iot.gif)

## Overview

This repository contains the simulation code for the article's three layers, together with the scripts needed to regenerate every table and figure.

| Layer | Component | Description |
| --- | --- | --- |
| **L1** | Semantic ECG encoding | Semantic ECG encoder with rate control, evaluated using the MIT-BIH inter-patient DS1/DS2 split and externally validated on the St Petersburg INCART database. |
| **L2** | PQ-Lite security | Hybrid X25519 + ML-KEM-512 handshake with ML-DSA-44 gateway authentication, a symmetric ratchet, and an AES-GCM record layer; includes ProVerif models and a node cost model. |
| **L3** | Lyapunov scheduling | Drift-plus-penalty scheduling of rate, power, and security tier subject to a threat-dependent minimum security tier. |

> **Simulation-only results:** All experiments use open datasets. On-device computational cost and energy estimates come from a model based on published cycle counts and datasheet currents; they are **not hardware measurements**.

## 1. Environment

**Requirements:** Python 3.12, Linux, and a CPU. No GPU is required.

### Python dependencies

Create a virtual environment and install the pinned dependencies:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --require-hashes -r requirements.lock
pip install --no-deps -r requirements-torch.txt --index-url https://download.pytorch.org/whl/cpu
```

### liboqs 0.16.0

Build and install **liboqs 0.16.0** with the required ML-KEM and ML-DSA algorithms:

```bash
git clone --depth 1 --branch 0.16.0 https://github.com/open-quantum-safe/liboqs /tmp/liboqs

cmake -S /tmp/liboqs -B /tmp/liboqs/build -GNinja -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON -DOQS_BUILD_ONLY_LIB=ON \
  -DOQS_MINIMAL_BUILD="KEM_ml_kem_512;KEM_ml_kem_768;SIG_ml_dsa_44" \
  -DCMAKE_INSTALL_PREFIX=/opt/liboqs

cmake --build /tmp/liboqs/build -j4
sudo cmake --install /tmp/liboqs/build

export OQS_INSTALL_PATH=/opt/liboqs
export LD_LIBRARY_PATH=/opt/liboqs/lib:$LD_LIBRARY_PATH
```

The workflow also requires **CMake** and **Ninja** for building liboqs, the **AWS CLI** (`aws`) for downloading datasets, and **ProVerif 2.05** on your `PATH` for formal verification.

Verify the installation:

```bash
python -m pytest -q
```

## 2. Data

No datasets are stored in this repository. The data stage downloads three public datasets from the [PhysioNet](https://physionet.org/) open-data mirror on Amazon S3 (`physionet-open`). No AWS account is required.

| Dataset | Version | S3 location |
| --- | --- | --- |
| MIT-BIH Arrhythmia Database | 1.0.0 | `s3://physionet-open/mitdb/1.0.0/` |
| PTB-XL | 1.0.3 | `s3://physionet-open/ptb-xl/1.0.3/` |
| St Petersburg INCART 12-lead Arrhythmia Database | 1.0.0 | `s3://physionet-open/incartdb/1.0.0/` |

Download and prepare the data:

```bash
python experiments/wp0_data_stats.py source=s3
```

Downloaded files are written to `data/raw/`. Set the `SEMPQ_DATA_RAW` environment variable to use another directory. Each file is verified against the SHA-256 checksum list published by PhysioNet, and the verification results are recorded in `data/manifests/`.

Patient-level split files are stored in `data/splits/`, with checksums in `data/splits/SHA256SUMS`. These checksums are verified whenever the splits are loaded. You can also verify them manually:

```bash
cd data/splits && sha256sum -c SHA256SUMS
```

Use of each dataset is subject to the applicable licence listed on [PhysioNet](https://physionet.org/).

## 3. Reproducing the Results

Run the complete reproduction workflow:

```bash
python reproduce.py
```

List the 55 encoder-stage runs:

```bash
python reproduce.py --list
```

Run selected stages:

```bash
python reproduce.py --stage wp2 --stage wp3inputs --stage wp3
```

### Pipeline stages

The workflow executes the following stages in dependency order:

| Stage | Purpose |
| --- | --- |
| `data` | Dataset preparation |
| `wp1` | Training and evaluation, with five seeds per method |
| `wp1agg` | Aggregation of encoder results |
| `wp1stats` | Encoder statistics |
| `wp2` | PQ-Lite security evaluation |
| `wp1energy` | Encoder energy modelling |
| `formal` | Formal verification using ProVerif |
| `wp3inputs` | Scheduler input preparation |
| `wp3` | Lyapunov scheduler evaluation |
| `wp3e2e` | End-to-end scheduler evaluation |
| `wp3lat` | Latency evaluation |
| `wp3bound` | Bound evaluation |
| `wp3traces` | Scheduler traces |
| `ext` | External validation of frozen models on INCART |
| `extstats` | External-validation statistics |

Generated outputs are written to `results/`, `tables/`, and `figures/`. The `ext` stage evaluates the frozen models produced by the `wp1` stage of the same run, using models stored under `results/wp1/`.

### Reproducibility notes

All runs are CPU-only and single-threaded per process, with fixed random seeds. Rerunning on the same CPU model reproduces the reported numbers exactly. Small numerical differences in encoder training may occur across CPU models.

The results reported in the article were generated on an **AMD EPYC 9654** processor.

## 4. Repository Contents

```text
src/             l1_semantic, l2_pqlite, l3_scheduler, channel, data, utils
experiments/     Entry point for each stage, table, and figure
configs/         Hydra configurations; constants.yaml documents external constants and sources
formal/          ProVerif models, expected verdicts, and checker
tests/           Unit tests
tools/           Release scripts for rented CPU machines and model deposit preparation
data/splits/     Patient-level split files and checksums
data/manifests/  Downloaded-dataset verification records
reproduce.py     One-command result reproduction
```

The scripts under `tools/` are not required for local reproduction.
