#!/usr/bin/env bash
# WP4 release on one RunPod CPU pod (decision 2026-10-12: GitHub Actions blocked by billing, repo stays private).
# Runs every stage of reproduce.py from the checked-out commit, pushes all outputs to r/release-<sha7>,
# then creates the Zenodo DRAFT of the WP1 models (never published). Mirrors .github/workflows/release.yml.
#
# Pod env: GH_TOKEN, GH_REPO, ZENODO_TOKEN (RunPod secret references only), WP1_JOBS (default 13),
#          MAX_SECONDS (hard stop, default 6900). Started from the repo root of a clone at the release commit.
set -uo pipefail
cd "$(dirname "$0")/../.."
REPO_DIR=$PWD
exec > >(tee -a /root/release_pod.log) 2>&1

stop_pod() {
  python3 - <<'EOF' || true
import json, os, urllib.request
k, p = os.environ.get("RUNPOD_API_KEY"), os.environ.get("RUNPOD_POD_ID")
if k and p:
    q = {"query": 'mutation { podStop(input:{podId:"%s"}) { id } }' % p}
    req = urllib.request.Request("https://api.runpod.io/graphql?api_key=" + k, json.dumps(q).encode(),
                                 {"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=60)
    print("pod stop requested")
else:
    print("no pod API key in the environment; the pod is stopped from outside")
EOF
}
(sleep "${MAX_SECONDS:-6900}"; echo "HARD STOP after ${MAX_SECONDS:-6900} s"; stop_pod) &

SHA7=$(git rev-parse --short=7 HEAD)
BR="r/release-$SHA7"
echo "release of $(git rev-parse HEAD) -> $BR on pod ${RUNPOD_POD_ID:-?} at $(date -u +%FT%TZ)"

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq git build-essential cmake ninja-build curl ca-certificates \
  python3.12-venv python3-dev ocaml-nox ocaml-findlib libssl-dev >/dev/null || { echo SETUP_FAILED apt; stop_pod; exit 1; }

if git ls-remote --exit-code origin "refs/heads/$BR" >/dev/null; then
  echo "$BR already exists: computation skipped"
else
  # liboqs 0.16.0, same options as .github/actions/setup-env
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

  # ProVerif 2.05 from source, as in the release workflow
  (cd /root && curl -sSL --fail -o proverif.tar.gz https://bblanche.gitlabpages.inria.fr/proverif/proverif2.05.tar.gz \
     && sha256sum proverif.tar.gz > "$REPO_DIR/proverif_source_sha256.txt" \
     && tar xzf proverif.tar.gz && cd proverif2.05 && ./build -nointeract >/dev/null 2>&1)
  export PATH="/root/proverif2.05:$PATH"
  command -v proverif >/dev/null || { echo SETUP_FAILED proverif; stop_pod; exit 1; }
  echo "SETUP_OK $(date -u +%T)"

  export SEMPQ_DATA_RAW=/root/raw SEMPQ_DATA_CACHE=/root/cache
  declare -A ST
  stage() { local name=$1; shift; echo "== $name start $(date -u +%T)"
    if "$@" > "$name.log" 2>&1; then ST[$name]=success; else ST[$name]=failure; fi
    echo "== $name ${ST[$name]} $(date -u +%T)"; }

  stage data python reproduce.py --stage data
  git diff --exit-code data/splits > /dev/null || { echo "split files changed"; ST[data]=failure; }

  # WP2 and ProVerif are independent of WP1: run them alongside, one core each
  (python reproduce.py --stage wp2 > wp2.log 2>&1; echo $? > /root/wp2.rc) &
  P2=$!
  (python3 reproduce.py --stage formal > formal.log 2>&1; echo $? > /root/formal.rc) &
  PF=$!
  stage wp1 python reproduce.py --stage wp1 --jobs "${WP1_JOBS:-13}"
  wait $P2 $PF
  [ "$(cat /root/wp2.rc)" = 0 ] && ST[wp2]=success || ST[wp2]=failure
  [ "$(cat /root/formal.rc)" = 0 ] && ST[formal]=success || ST[formal]=failure
  n=$(ls results/wp1/*_mitbih/seed?.json 2>/dev/null | wc -l); echo "WP1 runs: $n"
  ncv=$(ls results/wp1_dev/configB_cv/semantic_seed*.json 2>/dev/null | wc -l); echo "CV runs: $ncv"
  [ "$n" -eq 50 ] && [ "$ncv" -eq 5 ] || ST[wp1]=failure

  stage wp1agg python reproduce.py --stage wp1agg
  stage wp1stats python reproduce.py --stage wp1stats
  stage wp1energy python reproduce.py --stage wp1energy
  stage wp3inputs python reproduce.py --stage wp3inputs
  stage wp3 python reproduce.py --stage wp3
  stage wp3e2e python reproduce.py --stage wp3e2e

  {
    echo "refs/heads/$(git rev-parse --abbrev-ref HEAD 2>/dev/null) $(git rev-parse HEAD) runpod pod ${RUNPOD_POD_ID:-?}"
    echo "cpu: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs) x $(nproc)"
    echo -n "jobs:"; for k in data wp1 wp2 formal wp1agg wp1stats wp1energy wp3inputs wp3 wp3e2e; do echo -n " $k=${ST[$k]:-not_run}"; done; echo
  } > RELEASE_RUN.txt
  cat RELEASE_RUN.txt

  # nothing over 95 MB goes to git
  find results tables figures formal/outputs -type f -size +95M -print -delete
  git config user.name "sempq-pod"; git config user.email "sempq-pod@users.noreply.github.com"
  git checkout -q -B "$BR"
  git add -f results tables figures formal/outputs data/manifests RELEASE_RUN.txt \
    ./*.log proverif_source_sha256.txt 2>/dev/null || true
  git commit -qm "r/release: outputs of $(git rev-parse HEAD) (RunPod CPU pod)" || echo "nothing to commit"
  git push -qf origin "$BR" && echo "pushed $BR"
fi

# Zenodo draft only when every stage succeeded, and only once (r/zenodo marks an existing draft)
if git ls-remote origin "refs/heads/$BR" >/dev/null && \
   git fetch -q origin "$BR" && git show "origin/$BR:RELEASE_RUN.txt" | grep -q "jobs: data=success wp1=success wp2=success formal=success wp1agg=success wp1stats=success wp1energy=success wp3inputs=success wp3=success wp3e2e=success"; then
  if git fetch -q origin r/zenodo 2>/dev/null && git show origin/r/zenodo:ZENODO_DRAFT.json | grep -q "\"$BR\""; then
    echo "r/zenodo already records $BR: nothing to do"
  else
    [ -d /root/venv ] && . /root/venv/bin/activate
    timeout 1200 python3 tools/zenodo_deposit.py --repo "$GH_REPO" --release-branch "$BR" && echo ZENODO_DONE || echo ZENODO_FAILED
  fi
else
  echo "release incomplete: Zenodo draft not created"
fi
echo "ALL_DONE $(date -u +%FT%TZ)"
stop_pod
