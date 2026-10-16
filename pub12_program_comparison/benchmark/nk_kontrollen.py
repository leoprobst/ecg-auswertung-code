"""Controls for NeuroKit2 (equality counting only, no values).

  nk_kontrollen.py wdh  <raw_a.jsonl> <raw_b.jsonl>        repeat run against the first run (same keys)
  nk_kontrollen.py wirk <raw.jsonl> <dataset>              efficacy count VV1/VV2 against VV0 (count only)
  nk_kontrollen.py g1   <raw.jsonl> <dataset>              factor 1 (VV0_G1) against the run without scaling (VV0), same records
Equal = status, beat marks (indices) and lead errors identical (times: same ms value, as defined in the pre-registered protocol).
"""
import json
import sys
from pathlib import Path


def lade(p):
    d = {}
    for line in Path(p).read_text(encoding="utf8").splitlines():
        if line.strip():
            r = json.loads(line)
            d[r["key"]] = r
    return d


def kern(r):
    return {"status": r.get("status"), "leads": r.get("leads"), "lead_fehler": r.get("lead_fehler"), "fs": r.get("fs"), "grund": r.get("grund")}


def vergleiche(a, b, paare):
    gleich, abw = 0, []
    for ka, kb in paare:
        if ka in a and kb in b:
            if kern(a[ka]) == kern(b[kb]):
                gleich += 1
            else:
                abw.append(ka)
    n = sum(1 for ka, kb in paare if ka in a and kb in b)
    return {"verglichen": n, "gleich": gleich, "abweichend": len(abw), "abweichende_erste": abw[:10], "bestanden": n > 0 and not abw}


if __name__ == "__main__":
    m = sys.argv[1]
    if m == "wdh":
        a, b = lade(sys.argv[2]), lade(sys.argv[3])
        res = vergleiche(a, b, [(k, k) for k in a])
        res["nur_a"] = len(set(a) - set(b))
        res["nur_b"] = len(set(b) - set(a))
    elif m == "g1":
        a = lade(sys.argv[2])
        ds = sys.argv[3]
        paare = [(k, k.rsplit("|", 1)[0] + "|VV0_G1") for k in a if k.startswith(ds + "|") and k.endswith("|VV0")]
        res = vergleiche(a, a, paare)
    elif m == "wirk":
        # Number of records whose beat marks change relative to VV0 (pure count, no values)
        a = lade(sys.argv[2])
        ds = sys.argv[3]
        res = {}
        for vv in ("VV1", "VV2"):
            paare = [(k, k.rsplit("|", 1)[0] + "|" + vv) for k in a if k.startswith(ds + "|") and k.endswith("|VV0")]
            v = vergleiche(a, a, paare)
            res[vv] = {"verglichen": v["verglichen"], "geaendert": v["abweichend"]}
    else:
        raise SystemExit(__doc__)
    print(json.dumps(res, indent=1))
