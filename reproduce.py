"""One-command reproduction of every table and figure (brief WP4).

    python reproduce.py                    # all stages, in order, on this machine
    python reproduce.py --stage wp1 --jobs 4
    python reproduce.py --list

Stages (each writes under results/, tables/, figures/):
  data       download + verify MIT-BIH and PTB-XL (PhysioNet open-data mirror), re-verify the
             committed split hashes, data statistics, channel model curves        (WP0)
  wp1        train and evaluate every WP1 run: configuration B (9 methods x 5 seeds) and
             configuration A (proposed encoder x 5 seeds)                          (WP1)
             and DS1 4-fold patient CV of the proposed encoder x 5 seeds (scheduler inputs)
  wp1agg     aggregate WP1 runs: means, Pareto flags, paired tests, tables, figures
  wp1stats   record-level bootstrap, DS2 without record 202, whole-record loss over the link
  wp2        PQ-Lite and comparators: sizes, cost model, records, link simulation  (WP2)
  wp1energy  per-beat node energy of every L1 method (device + record + radio)
  formal     ProVerif models and verdict check (needs `proverif` on PATH)          (WP2)
  wp3inputs  build results/wp3/inputs.yaml from this run's WP1 CV / WP2 results
  wp3        scheduler, baselines, ablations, V-sweep, link-budget sweep           (WP3)
  wp3e2e     end-to-end trace over DS2 beats: realised macro-F1 per policy
  wp3lat     end-to-end beat latency model over the DS2 trace (decision 2026-10-13)
  wp3bound   sample-path utility upper bound for the implemented scheduler (Proposition 3)
  wp3traces  per-slot traces, convergence and policy-heatmap figures
  ext        frozen WP1 models on the external INCART cohort, pre-registered noninferiority
             (decision 2026-10-13; reads the models of EXT_REF: by default the result branch
             origin/r/release-a2a4d9a where this checkout has it, otherwise "local", the models the
             wp1 stage of this run wrote under results/wp1/)

Requirements: requirements.lock + requirements-torch.txt, liboqs 0.16.0 (OQS_INSTALL_PATH),
the aws CLI for the data download, ProVerif 2.05 for `formal`. Every run is CPU-only and
single-threaded per process; seeds are fixed, so a rerun on the same CPU model reproduces
the numbers exactly (small differences across CPU models are expected for WP1 training).
"""
from __future__ import annotations

import argparse
import itertools
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent
PY = sys.executable
SEEDS = [0, 1, 2, 3, 4]
CONFIG_B_METHODS = ["semantic", "raw", "dwt", "jscc", "label", "rr_only",
                    "semantic_norec", "semantic_noaug", "semantic_notpl"]


def sh(args: list[str], log: Path | None = None, env=None):
    print("+", " ".join(args), flush=True)
    if log is None:
        subprocess.run(args, cwd=REPO, check=True, env=env)
        return
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w") as f:
        subprocess.run(args, cwd=REPO, check=True, stdout=f, stderr=subprocess.STDOUT, env=env)


def wp1_jobs():
    jobs = [("wp1_mitbih_configB", m, s, []) for m, s in itertools.product(CONFIG_B_METHODS, SEEDS)]
    jobs += [("wp1_mitbih", "semantic", s, ["run_name=semantic_configA_mitbih", "+run_label=semantic_configA"])
             for s in SEEDS]
    # DS1 patient CV of the proposed encoder (scheduler inputs without test data; decision 2026-10-12)
    jobs += [("wp1_mitbih_configB", "semantic", s, ["dev.enabled=true", "dev.cv_folds=4", "dev.cv_epochs=15",
                                                     "dev.variant=configB_cv", "run_name=semantic_cv_mitbih",
                                                     "+run_label=semantic_cv"]) for s in SEEDS]
    return jobs


def job_id(job) -> str:
    cfg, method, seed, extra = job
    tag = [e for e in extra if e.startswith("+run_label=")][0].split("=")[1] if extra else method
    return f"{tag}:{seed}"


def run_wp1_job(job):
    cfg, method, seed, extra = job
    tag = job_id(job).split(":")[0]
    sh([PY, "experiments/wp1_train.py", "--config-name", cfg, f"method={method}", f"seed={seed}", *extra],
       log=REPO / "results" / "wp1" / "logs" / f"train_{tag}_seed{seed}.log")


EXT_RELEASE = "origin/r/release-a2a4d9a"


def ext_ref() -> str:
    """Where the external evaluation takes the frozen models from: EXT_REF when set; the result branch of the
    release when this checkout has it; otherwise "local" (results/wp1/ of this run), so that a copy of the code
    without result branches still runs every stage."""
    ref = os.environ.get("EXT_REF")
    if ref:
        return ref
    has = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--verify", "--quiet", EXT_RELEASE + "^{commit}"],
                         capture_output=True).returncode == 0
    return EXT_RELEASE if has else "local"


def stage(name: str, jobs: int, only: str | None):
    if name == "data":
        sh([PY, "experiments/wp0_data_stats.py", "source=s3", "mode=auto"])
        sh([PY, "experiments/wp0_channel_curves.py", "seed=0"])
    elif name == "wp1":
        js = wp1_jobs()
        if only:  # job id "<method or semantic_configA>:<seed>"
            js = [j for j in js if job_id(j) == only]
            if not js:
                raise SystemExit(f"no WP1 job {only}; see --list")
        with ThreadPoolExecutor(max_workers=jobs) as ex:
            list(ex.map(run_wp1_job, js))
    elif name == "wp1agg":
        sh([PY, "experiments/wp1_aggregate.py", "--results", str(REPO), "--tag", "release"])
    elif name == "wp1stats":
        sh([PY, "experiments/wp1_stats.py"])
    elif name == "wp2":
        sh([PY, "experiments/wp2_pqlite.py"])
    elif name == "wp1energy":
        sh([PY, "experiments/wp1_energy.py"])
    elif name == "formal":
        sh([PY, "formal/make_variants.py"])
        out = REPO / "formal" / "outputs"
        out.mkdir(parents=True, exist_ok=True)
        for f in sorted((REPO / "formal" / "build").glob("*.pv")):
            with open(out / (f.name + ".out"), "w") as fh:
                subprocess.run(["proverif", str(f)], cwd=REPO, stdout=fh, stderr=subprocess.STDOUT)
        sh([PY, "formal/check.py"])
    elif name == "wp3inputs":
        (REPO / "results" / "wp3").mkdir(parents=True, exist_ok=True)
        sh([PY, "experiments/wp3_build_inputs.py", "--wp1-summary", "results/wp1/aggregate/summary.json",
            "--wp2-dir", "results/wp2", "--wp1-cv-dir", "results/wp1_dev/configB_cv",
            "--out", "results/wp3/inputs.yaml"])
    elif name == "wp3":
        sh([PY, "experiments/wp3_scheduler.py", "inputs_path=results/wp3/inputs.yaml"])
    elif name == "wp3e2e":
        sh([PY, "experiments/wp3_e2e.py", "inputs_path=results/wp3/inputs.yaml"])
    elif name == "wp3lat":
        sh([PY, "experiments/wp3_latency.py", "inputs_path=results/wp3/inputs.yaml"])
    elif name == "wp3bound":
        sh([PY, "experiments/wp3_bound.py", "inputs_path=results/wp3/inputs.yaml"])
    elif name == "wp3traces":
        sh([PY, "experiments/wp3_traces.py", "inputs_path=results/wp3/inputs.yaml"])
    elif name == "extstats":  # statistics only, from the per-beat INCART predictions of the base release
        sh([PY, "experiments/wp1_external.py", "--stats-only", "--ref", ext_ref()])
    elif name == "ext":
        sh([PY, "experiments/wp1_external.py", "--ref", ext_ref()])
    else:
        raise SystemExit(f"unknown stage {name}")


STAGES = ["data", "wp1", "wp1agg", "wp1stats", "wp2", "wp1energy", "formal", "wp3inputs", "wp3", "wp3e2e", "wp3lat", "wp3bound", "wp3traces", "ext", "extstats"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=STAGES, action="append")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--only", help="wp1 only: job id '<method>:<seed>' (configuration A: 'semantic_configA:<seed>')")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        for j in wp1_jobs():
            print(job_id(j))
        return
    for s in a.stage or STAGES:
        stage(s, a.jobs, a.only)


if __name__ == "__main__":
    main()
