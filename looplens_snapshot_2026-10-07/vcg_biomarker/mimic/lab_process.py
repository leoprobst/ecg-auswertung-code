"""Single local case process, not a production workflow/supervisor."""
from __future__ import annotations

import json
import os
import sys
import traceback


def main() -> int:
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    import psutil
    process = psutil.Process()
    if os.name == "nt":
        process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    else:
        process.nice(10)
    request = json.loads(sys.stdin.readline())
    if request.get("schema_version") != "1.0.0":
        raise ValueError("Unsupported laboratory request")
    # Production has priority. Do not start if the 2-GiB reserve plus the
    # laboratory's conservative admission estimate is unavailable.
    if psutil.virtual_memory().available < 4 * 1024**3:
        raise RuntimeError("Labor wartet: weniger als 4 GiB frei (2 GiB Reserve + 2 GiB Laborbudget)")
    from threadpoolctl import threadpool_limits
    from .config import MimicProjectConfig
    from .ecg_lab import ECGCaseLibrary
    from .lab_workspace import resolve_powerline
    library = ECGCaseLibrary(MimicProjectConfig.from_dict(request["config"]))
    case = library.load_case(request["case_id"])
    def checkpoint():
        if request.get("cancel_path") and os.path.isfile(request["cancel_path"]):
            raise RuntimeError("Laboranalyse an sicherem Phasenübergang abgebrochen")
        if psutil.virtual_memory().available < 2 * 1024**3 or process.memory_info().rss > 2 * 1024**3:
            raise RuntimeError("Laborbudget erreicht; Produktionslauf bleibt unverändert. Später erneut starten.")
    with threadpool_limits(limits=1):
        payload = library.analyze_case(case.case_id, powerline_hz=resolve_powerline(request["powerline"], case.metadata), vcg_method=request.get("vcg_method", "kors"), resource_checkpoint=checkpoint, ischemia_context=request.get('ischemia_context'))
    print("LAB_RESULT=" + json.dumps({"case_id": case.case_id, "analysis_id": payload["analysis_id"]}), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
