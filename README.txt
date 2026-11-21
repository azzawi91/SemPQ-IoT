SemPQ-IoT: Semantic Encoding, Post-Quantum Security and Lyapunov Scheduling for Wearable ECG

Code accompanying the article by M. A. Azzawi, H. M. Al-Juboori, R. Hassan, M. F. Nasrudin
and A. A. Abdulhussein.

![SemPQ-IoT: Semantic Encoding, Post-Quantum Security and Lyapunov Scheduling for Wearable ECG](https://azzawi91.github.io/assets/research/sempq-iot.gif)

This repository holds the simulation code of the three layers of the article and the script
that regenerates every table and figure:

  L1  semantic ECG encoder with rate control (MIT-BIH inter-patient split DS1/DS2; external
      validation on the St Petersburg INCART database)
  L2  PQ-Lite, a hybrid X25519 + ML-KEM-512 handshake with ML-DSA-44 gateway authentication,
      a symmetric ratchet and an AES-GCM record layer; ProVerif models; node cost model
  L3  Lyapunov drift-plus-penalty scheduler of rate, power and security tier under a
      threat-dependent tier floor

Everything is simulation on open data. The on-device cost and energy figures are a model
built from published cycle counts and datasheet currents; they are not hardware measurements.


1. ENVIRONMENT

Python 3.12 on Linux, CPU only.

  python3.12 -m venv .venv && . .venv/bin/activate
  pip install --require-hashes -r requirements.lock
  pip install --no-deps -r requirements-torch.txt --index-url https://download.pytorch.org/whl/cpu

liboqs 0.16.0 (ML-KEM and ML-DSA only), built once:

  git clone --depth 1 --branch 0.16.0 https://github.com/open-quantum-safe/liboqs /tmp/liboqs
  cmake -S /tmp/liboqs -B /tmp/liboqs/build -GNinja -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON -DOQS_BUILD_ONLY_LIB=ON \
    -DOQS_MINIMAL_BUILD="KEM_ml_kem_512;KEM_ml_kem_768;SIG_ml_dsa_44" \
    -DCMAKE_INSTALL_PREFIX=/opt/liboqs
  cmake --build /tmp/liboqs/build -j4 && sudo cmake --install /tmp/liboqs/build
  export OQS_INSTALL_PATH=/opt/liboqs
  export LD_LIBRARY_PATH=/opt/liboqs/lib:$LD_LIBRARY_PATH

Also needed: cmake and ninja for the liboqs build, the AWS command line interface (aws) for
the data download, and ProVerif 2.05 on the PATH for the formal-verification stage.

Check the installation:

  python -m pytest -q


2. DATA

No dataset is stored in this repository. The data stage downloads the three databases from
the PhysioNet open-data mirror on Amazon S3 (bucket physionet-open; public, no account):

  MIT-BIH Arrhythmia Database 1.0.0            s3://physionet-open/mitdb/1.0.0/
  PTB-XL 1.0.3                                 s3://physionet-open/ptb-xl/1.0.3/
  St Petersburg INCART 12-lead Database 1.0.0  s3://physionet-open/incartdb/1.0.0/

  python experiments/wp0_data_stats.py source=s3

The files are written under data/raw/ (set SEMPQ_DATA_RAW to use another folder) and every
file is checked against the SHA-256 list that PhysioNet publishes with the database; the
outcome is recorded in data/manifests/. The patient-level split files are in data/splits/
with their checksums in data/splits/SHA256SUMS, and they are verified on every load:

  cd data/splits && sha256sum -c SHA256SUMS

Use of the databases is subject to the licences stated on physionet.org.


3. REPRODUCING THE RESULTS

  python reproduce.py            regenerates every result: all stages, in order
  python reproduce.py --list     the 55 runs of the encoder stage
  python reproduce.py --stage wp2 --stage wp3inputs --stage wp3      selected stages

Stages: data, wp1 (training and evaluation, 5 seeds per method), wp1agg, wp1stats, wp2,
wp1energy, formal (ProVerif), wp3inputs, wp3, wp3e2e, wp3lat, wp3bound, wp3traces, ext
(frozen models on INCART) and extstats. Outputs are written to results/, tables/ and
figures/. The ext stage evaluates the models that the wp1 stage of the same run wrote under
results/wp1/.

Every run is CPU-only and single-threaded per process, and the seeds are fixed. A rerun on
the same CPU model reproduces the numbers exactly; small differences across CPU models are
expected for the encoder training. The numbers in the article were produced on an AMD EPYC
9654 processor.


4. CONTENTS

  src/            l1_semantic, l2_pqlite, l3_scheduler, channel, data, utils
  experiments/    one entry point per stage, table or figure
  configs/        Hydra configurations; constants.yaml lists every external constant with
                  its source
  formal/         ProVerif models, expected verdicts and the checker
  tests/          unit tests
  tools/          scripts that ran the release on rented CPU machines and prepared the
                  model deposit (not needed for a local reproduction)
  data/splits/    patient-level split files and their checksums
  data/manifests/ record of the downloaded databases
  reproduce.py    one-command reproduction
