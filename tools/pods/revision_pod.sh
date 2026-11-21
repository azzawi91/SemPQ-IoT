#!/usr/bin/env bash
# Revision run (decision 2026-10-13, external review) on one RunPod CPU pod. WP1 is NOT retrained: the WP1
# outputs and models of the release of record (BASE_REF, default r/release-a2a4d9a) are imported unchanged, then
# the stages touched by the revision are run from this commit:
#   wp2 (RFC 8773 comparator), wp1energy (BLE connection event + sleep), wp3inputs, wp3 (combined baselines),
#   wp3e2e, wp3lat (end-to-end latency), wp3traces, ext (INCART, frozen models, pre-registered NI).
# Outputs go to r/release-<sha7>; RELEASE_RUN.txt records the base. No Zenodo action (models unchanged).
# Pod env: GH_TOKEN (secret reference), BASE_REF, MAX_SECONDS (hard stop, default 5400), STAGES (space-separated
# subset to run on top of the base, default: all revision stages; outputs of stages not run come from the base).
set -uo pipefail
cd "$(dirname "$0")/../.."
exec > >(tee -a /root/revision_pod.log) 2>&1
stop_pod() { python3 -c "
import json,os,urllib.request
k,p=os.environ.get('RUNPOD_API_KEY'),os.environ.get('RUNPOD_POD_ID')
if k and p:
    q={'query':'mutation { podStop(input:{podId:\"%s\"}) { id } }'%p}
    urllib.request.urlopen(urllib.request.Request('https://api.runpod.io/graphql?api_key=' + k, json.dumps(q).encode(),{'Content-Type':'application/json'}),timeout=60); print('pod stop requested')
" || true; }
(sleep "${MAX_SECONDS:-5400}"; echo "HARD STOP"; stop_pod) &
BASE=${BASE_REF:-r/release-a2a4d9a}
SHA7=$(git rev-parse --short=7 HEAD)
BR="r/release-$SHA7"
echo "revision run of $(git rev-parse HEAD) on base $BASE -> $BR, pod ${RUNPOD_POD_ID:-?}, $(date -u +%FT%TZ)"
if git ls-remote --exit-code origin "refs/heads/$BR" >/dev/null; then echo "$BR exists: nothing to do"; stop_pod; exit 0; fi
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq git build-essential cmake ninja-build curl ca-certificates \
  python3.12-venv python3-dev libssl-dev >/dev/null || { echo SETUP_FAILED apt; stop_pod; exit 1; }
git clone -q --depth 1 --branch 0.16.0 https://github.com/open-quantum-safe/liboqs /root/liboqs
cmake -S /root/liboqs -B /root/liboqs/build -GNinja -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
  -DOQS_BUILD_ONLY_LIB=ON -DOQS_MINIMAL_BUILD="KEM_ml_kem_512;KEM_ml_kem_768;SIG_ml_dsa_44" \
  -DCMAKE_INSTALL_PREFIX=/opt/liboqs >/dev/null
cmake --build /root/liboqs/build -j8 >/dev/null && cmake --install /root/liboqs/build >/dev/null \
  || { echo SETUP_FAILED liboqs; stop_pod; exit 1; }
export OQS_INSTALL_PATH=/opt/liboqs LD_LIBRARY_PATH=/opt/liboqs/lib
python3 -m venv /root/venv && . /root/venv/bin/activate
pip install -q --upgrade pip
pip install -q --require-hashes -r requirements.lock
pip install -q --no-deps -r requirements-torch.txt --index-url https://download.pytorch.org/whl/cpu
pip install -q awscli || { echo SETUP_FAILED pip; stop_pod; exit 1; }
python -c "import oqs; oqs.KeyEncapsulation('ML-KEM-512')" || { echo SETUP_FAILED oqs; stop_pod; exit 1; }
echo "SETUP_OK $(date -u +%T)"

git fetch -q origin "$BASE:refs/remotes/origin/$BASE" || { echo BASE_FETCH_FAILED; stop_pod; exit 1; }
git checkout "origin/$BASE" -- results tables figures formal/outputs RELEASE_RUN.txt
mv RELEASE_RUN.txt BASE_RELEASE_RUN.txt
# keep the base's stage logs (and its own base record, if it was itself a revision run)
git ls-tree --name-only "origin/$BASE" | grep -E '\.log$|^BASE_RELEASE_RUN\.txt$' | sed 's/^BASE_RELEASE_RUN.txt$//' | \
  xargs -r git checkout "origin/$BASE" --
export SEMPQ_DATA_RAW=/root/raw SEMPQ_DATA_CACHE=/root/cache EXT_REF="origin/$BASE"
declare -A ST
stage() { local name=$1; shift; echo "== $name start $(date -u +%T)"
  if "$@" > "$name.log" 2>&1; then ST[$name]=success; else ST[$name]=failure; fi
  echo "== $name ${ST[$name]} $(date -u +%T)"; }
ALL="wp2 wp1energy wp3inputs wp3 wp3e2e wp3lat wp3bound wp3traces ext extstats"
RUN=" ${STAGES:-$ALL} "
want() { [[ "$RUN" == *" $1 "* ]]; }
# INCART evaluation is independent of the scheduler chain: run it alongside
if want ext; then (python reproduce.py --stage ext 2>&1 | tee ext.log | sed -u 's/^/[ext] /'; echo "${PIPESTATUS[0]}" > /root/ext.rc) & PE=$!; fi
for st in wp2 wp1energy wp3inputs wp3 wp3e2e wp3lat wp3bound wp3traces extstats; do
  if want "$st"; then
    [ "$st" = wp3inputs ] && cp results/wp3/inputs.yaml /root/inputs_base.yaml
    stage "$st" python reproduce.py --stage "$st"
    if [ "$st" = wp3inputs ]; then
      if cmp -s results/wp3/inputs.yaml /root/inputs_base.yaml; then echo "wp3 inputs identical to base"; else echo "WARNING: wp3 inputs differ from base"; fi
    fi
  fi
done
if want ext; then wait $PE; if [ "$(cat /root/ext.rc)" = 0 ]; then ST[ext]=success; else ST[ext]=failure; fi; fi
{
  echo "refs/heads/$(git rev-parse --abbrev-ref HEAD 2>/dev/null) $(git rev-parse HEAD) runpod pod ${RUNPOD_POD_ID:-?}"
  echo "base: $BASE ($(git rev-parse "origin/$BASE")); WP1 outputs and models imported unchanged"
  echo "cpu: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs) x $(nproc)"
  echo -n "jobs:"; for k in $ALL; do echo -n " $k=${ST[$k]:-not_run}"; done; echo
} > RELEASE_RUN.txt
cat RELEASE_RUN.txt
find results tables figures -type f -size +95M -print -delete
git config user.name "sempq-pod"; git config user.email "sempq-pod@users.noreply.github.com"
git checkout -q -B "$BR"
git add -f results tables figures formal/outputs data/manifests RELEASE_RUN.txt BASE_RELEASE_RUN.txt ./*.log 2>/dev/null || true
git commit -qm "r/release (revision): outputs of $(git rev-parse HEAD) on base $BASE (RunPod CPU pod)" || echo "nothing to commit"
git push -qf origin "$BR" && echo "pushed $BR"
echo "ALL_DONE $(date -u +%FT%TZ)"
stop_pod
