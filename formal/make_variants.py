"""Generate model variants from formal/pqlite.pv and record the expected verdict of each query.

pqlite.pv           base model (Dolev-Yao attacker)
pqlite_quantum.pv   attacker can additionally compute discrete logarithms (X25519 broken, the
                    "harvest now, decrypt later" setting); ML-KEM is unbroken.
Expected verdicts are in formal/expected.json; formal/check.py compares ProVerif's output.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
base = (HERE / "pqlite.pv").read_text()
EXTRA_Q = "query attacker(secretC) phase 1.                              (* classical FS *)\n"
base = base.replace("query attacker(secretH) phase 1.", EXTRA_Q + "query attacker(secretH) phase 1.", 1)
(HERE / "build").mkdir(exist_ok=True)
(HERE / "build" / "pqlite.pv").write_text(base)
anchor = "equation forall x: exponent, y: exponent; exp(exp(g, x), y) = exp(exp(g, y), x).\n"
assert anchor in base
quantum = base.replace(anchor, anchor + "(* quantum variant: discrete logarithms are computable *)\n"
                       "reduc forall x: exponent; dlog(exp(g, x)) = x.\n", 1)
(HERE / "build" / "pqlite_quantum.pv").write_text(quantum)

pcs = (HERE / "pqlite_pcs.pv").read_text()
(HERE / "build" / "pqlite_pcs_passive.pv").write_text(pcs)
active = (pcs.replace("  out(cp, ek);\n  (* PASSIVE: read the peer's response from the authentic copy *)\n  in(cp, ct: bitstring);",
                      "  in(c, ct: bitstring);")
             .replace("  in(cp, ek: kempkey);", "  in(c, ek: kempkey);")
             .replace("  out(c, ct);\n  out(cp, ct).", "  out(c, ct)."))
assert active != pcs
(HERE / "build" / "pqlite_pcs_active.pv").write_text(active)

T, F = "true", "false"
expected = {
    "pqlite.pv": [
        ("not attacker_p1(secretC[])", T),     # reported in the last phase: covers phases 0 and 1
        ("not attacker_p1(secretH[])", T),
        ("inj-event(NodeAccept", T),
        ("inj-event(GwAccept", T),
        ("event(GwAccept(nN,nG,CLASSICAL,k))", T),
        ("event(NodeAccept(nN,nG,HYBRID,k))", T),
    ],
    "pqlite_quantum.pv": [
        ("not attacker_p1(secretC[])", F),     # classical: PSK leaked + DH broken -> exposed (sanity)
        ("not attacker_p1(secretH[])", T),     # hybrid: protected by ML-KEM
        ("inj-event(NodeAccept", T),
        ("inj-event(GwAccept", T),
        ("event(GwAccept(nN,nG,CLASSICAL,k))", T),
        ("event(NodeAccept(nN,nG,HYBRID,k))", T),
    ],
    "pqlite_pcs_passive.pv": [
        ("not attacker_p2(secret1[])", T),     # earlier record stays secret (deleted message key)
        ("not attacker_p2(secret2[])", T),     # healed after compromise
    ],
    "pqlite_pcs_active.pv": [
        ("not attacker_p2(secret1[])", T),
        ("not attacker_p2(secret2[])", F),     # an active attacker holding the state can MITM the re-key
    ],
}
(HERE / "expected.json").write_text(json.dumps(expected, indent=1) + "\n")
print("variants written")
