#!/usr/bin/env bash
# Encoder-size selection sweep (decision 2026-10-13) on one RunPod CPU pod: DS1 4-fold patient CV only,
# DS2 never loaded. 6 candidates x 5 seeds. Pushes outputs to r/wp1-encsize-<sha7>.
# Pod env: GH_TOKEN, GH_REPO (RunPod secret references only), JOBS (default 15), MAX_SECONDS (default 7200).
set -uo pipefail
cd "$(dirname "$0")/../.."
exec > >(tee -a /root/encsize_pod.log) 2>&1

stop_pod() {
  python3 - <<'PY' || true
import json, os, urllib.request
k, p = os.environ.get("RUNPOD_API_KEY"), os.environ.get("RUNPOD_POD_ID")
if k and p:
    q = {"query": 'mutation { podStop(input:{podId:"%s"}) { id } }' % p}
    req = urllib.request.Request("https://api.runpod.io/graphql?api_key=" + k, json.dumps(q).encode(),
                                 {"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=60)
    print("pod stop requested")
PY
}
(sleep "${MAX_SECONDS:-7200}"; echo "HARD STOP after ${MAX_SECONDS:-7200} s"; stop_pod) &

SHA7=$(git rev-parse --short=7 HEAD)
BR="r/wp1-encsize-$SHA7"
echo "encoder-size sweep of $(git rev-parse HEAD) -> $BR on pod ${RUNPOD_POD_ID:-?} at $(date -u +%FT%TZ)"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq git build-essential curl ca-certificates python3.12-venv python3-dev \
  >/dev/null || { echo SETUP_FAILED apt; stop_pod; exit 1; }
python3 -m venv /root/venv && . /root/venv/bin/activate
pip install -q --upgrade pip
pip install -q --require-hashes -r requirements.lock
pip install -q --no-deps -r requirements-torch.txt --index-url https://download.pytorch.org/whl/cpu
pip install -q awscli || { echo SETUP_FAILED pip; stop_pod; exit 1; }
echo "SETUP_OK $(date -u +%T)"

export SEMPQ_DATA_RAW=/root/raw SEMPQ_DATA_CACHE=/root/cache
echo "== data start $(date -u +%T)"
python reproduce.py --stage data > data.log 2>&1 && echo "== data success $(date -u +%T)" || { echo "== data failure"; tail -30 data.log; }
git diff --exit-code data/splits > /dev/null || echo "WARNING: split files changed"
echo "== sweep start $(date -u +%T)"
python experiments/wp1_encsize.py --run --jobs "${JOBS:-15}" > encsize.log 2>&1
echo "== sweep end $(date -u +%T)"; tail -12 encsize.log
n=$(ls results/wp1_dev/enc_*/semantic_seed*.json 2>/dev/null | wc -l); echo "CV runs: $n of 30"

{
  echo "$(git rev-parse HEAD) runpod pod ${RUNPOD_POD_ID:-?}"
  echo "cpu: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs) x $(nproc)"
  echo "cv_runs: $n"
} > ENCSIZE_RUN.txt
git config user.name "sempq-pod"; git config user.email "sempq-pod@users.noreply.github.com"
git checkout -q -B "$BR"
find results -type f -size +95M -print -delete
git add -f results/wp1_dev results/wp0 ENCSIZE_RUN.txt data.log encsize.log 2>/dev/null || true
git commit -qm "r/wp1-encsize: DS1-CV encoder-size sweep of $(git rev-parse HEAD)" || echo "nothing to commit"
git push -qf origin "$BR" && echo "pushed $BR"
echo "ALL_DONE $(date -u +%FT%TZ)"
stop_pod
