#!/usr/bin/env bash
# Zenodo DRAFT of the WP1 models of a release branch, on a small RunPod CPU pod (never publishes).
# Pod env: GH_TOKEN, GH_REPO, ZENODO_TOKEN (RunPod secret references only), CODE_SHA, RELEASE_BRANCH.
# The log goes to branch r/log-zenodo so the outcome is readable after the pod is gone.
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq >/dev/null && apt-get install -y -qq git ca-certificates >/dev/null
cd /root
git clone -q "https://x-access-token:${GH_TOKEN}@github.com/${GH_REPO}.git" code && cd code && git checkout -q "$CODE_SHA"
{
  echo "zenodo pod ${RUNPOD_POD_ID:-?} at $(date -u +%FT%TZ), code $(git rev-parse --short HEAD), release $RELEASE_BRANCH"
  timeout 1200 python3 tools/zenodo_deposit.py --repo "$GH_REPO" --release-branch "$RELEASE_BRANCH" && echo ZENODO_DONE || echo ZENODO_FAILED
} > /root/zenodo.log 2>&1
cat /root/zenodo.log
git clone -q "https://x-access-token:${GH_TOKEN}@github.com/${GH_REPO}.git" /root/logrepo && cd /root/logrepo \
  && git checkout -q --orphan r/log-zenodo && git rm -rq --cached . && cp /root/zenodo.log ZENODO_LOG.txt \
  && git add ZENODO_LOG.txt && git -c user.name=sempq-pod -c user.email=pod@users.noreply.github.com commit -qm "zenodo pod log" \
  && git push -qf origin r/log-zenodo && echo "log pushed"
echo ALL_DONE
sleep infinity
