"""Compare ProVerif 'RESULT' lines with formal/expected.json; exit 1 on any mismatch or gap."""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
expected = json.loads((HERE / "expected.json").read_text())
ok, report = True, []
for model, items in expected.items():
    out = (HERE / "outputs" / (model + ".out")).read_text()
    results = re.findall(r"^RESULT (.*) is (true|false|cannot be proved)\.?$", out, re.M)
    norm = [(re.sub(r"_\d+", "", re.sub(r"\s+", "", q)), v) for q, v in results]  # drop ProVerif renaming
    for pattern, want in items:
        p = re.sub(r"\s+", "", pattern)
        hits = [v for q, v in norm if p in q]
        status = "MISSING" if not hits else ("OK" if all(v == want for v in hits) else "MISMATCH")
        ok &= status == "OK"
        report.append(f"{model:20s} {status:9s} expect {want:6s} got {','.join(hits) or '-':12s} {pattern}")
    report.append(f"{model:20s} total RESULT lines: {len(results)}")
print("\n".join(report))
(HERE / "outputs" / "SUMMARY.txt").write_text("\n".join(report) + "\n")
sys.exit(0 if ok else 1)
