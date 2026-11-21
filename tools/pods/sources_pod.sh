#!/usr/bin/env bash
# Fetch the nRF52840 Product Specification v1.1 (same URL and SHA-256 as r/sources:sources/MANIFEST.json) and push
# its full text to r/sources-full, so radio timing and start-up values can be pinned with page citations
# (decision 2026-10-13: BLE connection-event cost). Pod env: GH_TOKEN (secret reference), MAX_SECONDS.
set -uo pipefail
cd "$(dirname "$0")/../.."
stop_pod() { python3 -c "
import json,os,urllib.request
k,p=os.environ.get('RUNPOD_API_KEY'),os.environ.get('RUNPOD_POD_ID')
if k and p:
    q={'query':'mutation { podStop(input:{podId:\"%s\"}) { id } }'%p}
    urllib.request.urlopen(urllib.request.Request('https://api.runpod.io/graphql?api_key=' + k, json.dumps(q).encode(),{'Content-Type':'application/json'}),timeout=60); print('pod stop requested')
" || true; }
(sleep "${MAX_SECONDS:-1200}"; echo HARD STOP; stop_pod) &
apt-get update -qq && apt-get install -y -qq poppler-utils >/dev/null
URL=https://cdn.sparkfun.com/assets/d/2/6/2/6/nRF52840_PS_v1.1.pdf
WANT=c619e336b9c0610663273041f057f2537a65fd408ce0c5b8214a26de2aa88422
curl -sSL --fail -o /root/ps.pdf "$URL" || { echo FETCH_FAILED; stop_pod; exit 1; }
GOT=$(sha256sum /root/ps.pdf | cut -d' ' -f1)
echo "sha256 $GOT"
[ "$GOT" = "$WANT" ] || { echo SHA_MISMATCH; stop_pod; exit 1; }
mkdir -p sources
for p in $(seq 1 619); do echo "===== page $p ====="; pdftotext -layout -f $p -l $p /root/ps.pdf - ; done > sources/nrf52840_ps_v1_1_full.txt
echo "{\"url\": \"$URL\", \"sha256\": \"$GOT\", \"tool\": \"$(pdftotext -v 2>&1 | head -1)\"}" > sources/nrf52840_ps_v1_1_full.json
git config user.name sempq-pod; git config user.email sempq-pod@users.noreply.github.com
git checkout -q --orphan r/sources-full && git rm -rqf --cached . && git add -f sources/nrf52840_ps_v1_1_full.txt sources/nrf52840_ps_v1_1_full.json \
  && git commit -qm "nRF52840 PS v1.1 full text (pdftotext -layout), sha256 $GOT" && git push -qf origin r/sources-full && echo PUSHED
echo ALL_DONE; stop_pod
