"""Create an UNPUBLISHED Zenodo draft with the trained WP1 models of a release run.

    ZENODO_TOKEN=... python tools/zenodo_deposit.py --release-branch r/release-<sha>

Runs on a short-lived pod (decision 2026-10-11): clones the results branch, zips
results/wp1/<run>_mitbih/seed*_model.pt with their seed*.json metrics and the release metadata,
creates a draft deposition, uploads the zip, and writes the draft id and link to branch
r/zenodo. It never publishes; Dr. Yazan reviews and presses Publish.
The token is read from the environment only and is never printed or written to disk.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

API = "https://zenodo.org/api"


def req(method: str, url: str, token: str, data=None, headers=None):
    h = {"Authorization": f"Bearer {token}", **(headers or {})}
    body = None
    if data is not None and not isinstance(data, (bytes, bytearray)):
        body = json.dumps(data).encode()
        h["Content-Type"] = "application/json"
    elif data is not None:
        body = data
    for attempt in range(5):  # Zenodo answers 502/503/504 under load: back off and retry
        r = urllib.request.Request(url, data=body, method=method, headers=h)
        try:
            with urllib.request.urlopen(r, timeout=600) as resp:
                txt = resp.read()
            return json.loads(txt) if txt else {}
        except urllib.error.HTTPError as e:  # report Zenodo's reason; the token is never part of it
            msg = f"Zenodo {method} {url.split('?')[0]} -> HTTP {e.code}: {e.read()[:300]!r}"
            if e.code in (429, 500, 502, 503, 504) and attempt < 4:
                print(msg + f"; retry in {30 * (attempt + 1)} s", flush=True)
                time.sleep(30 * (attempt + 1))
                continue
            raise RuntimeError(msg) from None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="azzawi91/SemPQ-IoT")
    ap.add_argument("--release-branch", required=True)
    ap.add_argument("--dry-run", action="store_true", help="build the zip and metadata only")
    a = ap.parse_args()
    token = os.environ.get("ZENODO_TOKEN", "")
    gh = os.environ.get("GH_TOKEN", "")
    work = Path(tempfile.mkdtemp())
    url = f"https://x-access-token:{gh}@github.com/{a.repo}.git" if gh else f"https://github.com/{a.repo}.git"
    subprocess.run(["git", "clone", "-q", "--depth", "1", "-b", a.release_branch, url, str(work / "rel")], check=True)
    rel = work / "rel"
    models = sorted((rel / "results" / "wp1").glob("*_mitbih/seed*_model.pt"))
    if not models:
        sys.exit("no model files on the release branch")
    zpath = work / "sempq_iot_wp1_models.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for m in models:
            run = m.parent.name
            seed = m.name.split("_")[0]
            z.write(m, f"{run}/{m.name}")
            z.write(m.parent / f"{seed}.json", f"{run}/{seed}.json")
        for extra in ("RELEASE_RUN.txt", "results/wp1/aggregate/summary.json"):
            if (rel / extra).exists():
                z.write(rel / extra, extra)
        z.writestr("README.txt",
                   "SemPQ-IoT v5 - trained WP1 models (semantic ECG encoder and baselines), MIT-BIH DS1 -> DS2.\n"
                   "Each <run>/seedK_model.pt holds the evaluated state_dict, the full config and the code commit;\n"
                   "seedK.json holds the test metrics. Code: https://github.com/" + a.repo + "\n")
    meta = {"metadata": {
        "upload_type": "software",
        "title": "SemPQ-IoT v5: trained semantic ECG encoder and baseline models (WP1 release)",
        "description": "Weights and per-seed test metrics of the WP1 models (configuration B: 9 methods x 5 seeds; "
                       "configuration A ablation: 5 seeds) trained on MIT-BIH DS1 and evaluated on DS2. "
                       f"Produced by the release run on branch {a.release_branch}.",
        "creators": [{"name": "Azzawi, Mustafa A."}, {"name": "Al-Juboori, Haider M."},
                     {"name": "Hassan, Rosilah"}, {"name": "Nasrudin, Mohammad Faidzul"},
                     {"name": "Abdulhussein, Ansam A."}],
        "license": "MIT", "access_right": "open", "keywords": ["ECG", "semantic communication", "post-quantum", "IoT"]}}
    summary = {"models": len(models), "zip_bytes": zpath.stat().st_size, "branch": a.release_branch}
    if a.dry_run or not token:
        summary["draft"] = None
        print(json.dumps(summary))
        return
    # reuse an unpublished draft with the same title (a rerun must not leave duplicate drafts)
    title = meta["metadata"]["title"]
    existing = [d for d in req("GET", f"{API}/deposit/depositions?size=20&sort=mostrecent", token)
                if d.get("title") == title and not d.get("submitted", False)]
    dep = existing[0] if existing else req("POST", f"{API}/deposit/depositions", token, {})
    print(("reusing draft " if existing else "created draft ") + str(dep["id"]), flush=True)
    req("PUT", f"{API}/deposit/depositions/{dep['id']}", token, meta)
    dep = req("GET", f"{API}/deposit/depositions/{dep['id']}", token)
    # a newer release replaces the archive of an earlier one in the same unpublished draft
    for f in req("GET", f"{API}/deposit/depositions/{dep['id']}/files", token):
        if f.get("filename") == zpath.name:
            req("DELETE", f"{API}/deposit/depositions/{dep['id']}/files/{f['id']}", token)
            print("replaced the earlier model archive", flush=True)
    bucket = dep["links"]["bucket"]
    req("PUT", f"{bucket}/{zpath.name}", token, zpath.read_bytes(), {"Content-Type": "application/octet-stream"})
    print("model archive uploaded", flush=True)
    summary.update({"draft_id": dep["id"], "draft_url": dep["links"]["html"], "published": False})
    out = work / "zout"
    subprocess.run(["git", "clone", "-q", url, str(out)], check=True)
    subprocess.run(["git", "-C", str(out), "checkout", "-q", "--orphan", "r/zenodo"], check=True)
    subprocess.run(["git", "-C", str(out), "rm", "-rq", "--cached", "."], check=False)
    (out / "ZENODO_DRAFT.json").write_text(json.dumps(summary, indent=1) + "\n")
    subprocess.run(["git", "-C", str(out), "add", "ZENODO_DRAFT.json"], check=True)
    subprocess.run(["git", "-C", str(out), "-c", "user.name=sempq-pod", "-c", "user.email=pod@users.noreply.github.com",
                    "commit", "-qm", f"r/zenodo: draft for {a.release_branch}"], check=True)
    subprocess.run(["git", "-C", str(out), "push", "-qf", "origin", "r/zenodo"], check=True)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
