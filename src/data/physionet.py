"""Download and integrity-check PhysioNet databases at a pinned version.

Each database is fetched as the versioned project archive, unpacked under ``data/raw/``
(gitignored), and every file is checked against the SHA256SUMS.txt that PhysioNet
publishes with the project. A provenance manifest (archive hash, file count, mismatches)
is written to ``data/manifests/<name>.json`` and committed.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PhysioNetDB:
    name: str
    slug: str
    version: str
    archive_url: str

    @property
    def files_url(self) -> str:
        return f"https://physionet.org/files/{self.slug}/{self.version}/"


DATABASES = {
    "mitdb": PhysioNetDB(
        name="mitdb",
        slug="mitdb",
        version="1.0.0",
        archive_url="https://physionet.org/static/published-projects/mitdb/"
        "mit-bih-arrhythmia-database-1.0.0.zip",
    ),
    "ptbxl": PhysioNetDB(
        name="ptbxl",
        slug="ptb-xl",
        version="1.0.3",
        archive_url="https://physionet.org/static/published-projects/ptb-xl/"
        "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip",
    ),
    # external test cohort only (decision 2026-10-13); fetched from the open-data mirror (source="s3")
    "incartdb": PhysioNetDB(
        name="incartdb",
        slug="incartdb",
        version="1.0.0",
        archive_url="https://physionet.org/static/published-projects/incartdb/"
        "st-petersburg-incart-12-lead-arrhythmia-database-1.0.0.zip",
    ),
}


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "sempq-iot-wp0"})
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)
    tmp.replace(dest)


def _find_root(extract_dir: Path) -> Path:
    """The archive holds one top-level folder; return the folder that contains SHA256SUMS.txt."""
    hits = sorted(extract_dir.rglob("SHA256SUMS.txt"), key=lambda p: len(p.parts))
    if not hits:
        raise FileNotFoundError(f"SHA256SUMS.txt not found under {extract_dir}")
    return hits[0].parent


def verify_against_sums(root: Path, exclude_prefixes: tuple[str, ...] = ()) -> dict:
    """Check every file listed in PhysioNet's SHA256SUMS.txt (minus excluded subtrees)."""
    sums = (root / "SHA256SUMS.txt").read_text().splitlines()
    checked, missing, mismatched, skipped = 0, [], [], 0
    for line in sums:
        if not line.strip():
            continue
        digest, rel = line.split(maxsplit=1)
        rel = rel.lstrip("*")
        if rel.startswith(tuple(exclude_prefixes)):
            skipped += 1
            continue
        p = root / rel
        if not p.exists():
            missing.append(rel)
            continue
        if sha256_file(p) != digest:
            mismatched.append(rel)
        checked += 1
    return {
        "files_listed": len([s for s in sums if s.strip()]),
        "files_checked": checked,
        "files_excluded": skipped,
        "excluded_prefixes": list(exclude_prefixes),
        "missing": missing,
        "mismatched": mismatched,
        "sha256sums_txt": sha256_file(root / "SHA256SUMS.txt"),
    }


# Subtrees not needed by this project (PTB-XL 500 Hz signals); skipped by the s3 source.
EXCLUDE = {"mitdb": (), "ptbxl": ("records500/",), "incartdb": ()}


def _s3_sync(db: PhysioNetDB, dest: Path, exclude: tuple[str, ...]) -> None:
    """Sync from PhysioNet's AWS Open Data mirror (anonymous). Requires the aws CLI."""
    cmd = ["aws", "s3", "sync", "--no-sign-request", "--only-show-errors",
           f"s3://physionet-open/{db.slug}/{db.version}/", str(dest)]
    for e in exclude:
        cmd += ["--exclude", f"{e}*"]
    subprocess.run(cmd, check=True)


def fetch(name: str, raw_dir: Path, manifest_dir: Path, source: str = "archive") -> Path:
    """Download (if absent), unpack, verify. Returns the database root directory.

    source="archive": versioned project zip from physionet.org (all files verified).
    source="s3": PhysioNet's open-data bucket, excluding EXCLUDE[name] subtrees; only the
    downloaded files are verified, and the exclusion is recorded in the manifest.
    """
    db = DATABASES[name]
    raw_dir = Path(raw_dir)
    archive = raw_dir / f"{db.name}-{db.version}.zip"
    extract_dir = raw_dir / f"{db.name}-{db.version}"
    exclude: tuple[str, ...] = ()
    if source == "s3":
        exclude = EXCLUDE[name]
        root = extract_dir
        if not (root / "SHA256SUMS.txt").exists():
            _s3_sync(db, root, exclude)
        summary = verify_against_sums(root, exclude)
        return _finish(name, db, summary, None, source, manifest_dir, root)
    if source != "archive":
        raise ValueError(source)
    if not extract_dir.exists():
        if not archive.exists():
            _download(db.archive_url, archive)
        with zipfile.ZipFile(archive) as z:
            z.extractall(extract_dir)
    try:
        root = _find_root(extract_dir)
    except FileNotFoundError:  # archive without the checksum list: fetch it from the files page
        subdirs = [p for p in extract_dir.iterdir() if p.is_dir()]
        root = subdirs[0] if len(subdirs) == 1 else extract_dir
        _download(db.files_url + "SHA256SUMS.txt", root / "SHA256SUMS.txt")
    summary = verify_against_sums(root)
    return _finish(name, db, summary, archive, source, manifest_dir, root)


def _finish(name, db, summary, archive, source, manifest_dir, root) -> Path:
    if summary["missing"] or summary["mismatched"]:
        raise RuntimeError(
            f"{name}: integrity check failed "
            f"({len(summary['missing'])} missing, {len(summary['mismatched'])} mismatched)"
        )
    manifest = {
        "database": db.slug,
        "version": db.version,
        "source": source,
        "url": db.archive_url if source == "archive" else f"s3://physionet-open/{db.slug}/{db.version}/",
        "archive_sha256": sha256_file(archive) if archive is not None and archive.exists() else None,
        **summary,
    }
    manifest_dir = Path(manifest_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / f"{name}.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return root
