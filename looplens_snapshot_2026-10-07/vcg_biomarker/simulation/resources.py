"""Paging-pressure observations; no promise to control other Windows processes."""
from __future__ import annotations

import os


class PagingMonitor:
    """Windows hard page input (also includes mapped-file reads), conservative guard.

    A unavailable counter is reported as unknown, never a false zero-spill proof.
    This does not disable the OS pagefile or change global memory policy.
    """
    def __init__(self):
        self.query = None
        self.error = None
        self.pressure_samples = 0
        if os.name != "nt":
            self.error = "windows_counter_not_available"
            return
        try:
            import ctypes as c
            from ctypes import wintypes as w
            self.api = c.WinDLL("pdh")
            self.api.PdhOpenQueryW.argtypes = [w.LPCWSTR, c.c_size_t, c.POINTER(w.HANDLE)]
            self.api.PdhAddEnglishCounterW.argtypes = [w.HANDLE, w.LPCWSTR, c.c_size_t, c.POINTER(w.HANDLE)]
            self.api.PdhCollectQueryData.argtypes = [w.HANDLE]
            self.api.PdhCloseQuery.argtypes = [w.HANDLE]

            class Value(c.Structure):
                _fields_ = [("status", w.DWORD), ("value", c.c_double)]
            self.Value = Value
            self.api.PdhGetFormattedCounterValue.argtypes = [w.HANDLE, w.DWORD, c.c_void_p, c.POINTER(Value)]
            self.query, self.counter = w.HANDLE(), w.HANDLE()
            if self.api.PdhOpenQueryW(None, 0, c.byref(self.query)):
                raise OSError("PDH query failed")
            if self.api.PdhAddEnglishCounterW(self.query, r"\Memory\Pages Input/sec", 0, c.byref(self.counter)):
                raise OSError("PDH hard-page counter unavailable")
            self.api.PdhCollectQueryData(self.query)
        except (OSError, AttributeError) as exc:
            self.error = str(exc)
            self.close()

    def sample(self):
        rate = None
        if self.query:
            import ctypes as c
            value = self.Value()
            if (not self.api.PdhCollectQueryData(self.query)
                    and not self.api.PdhGetFormattedCounterValue(self.counter, 0x200, None, c.byref(value))
                    and value.status in (0, 1)):
                rate = max(0., float(value.value))
        self.pressure_samples = self.pressure_samples+1 if rate is not None and rate > 256 else 0
        return {"hard_page_inputs_per_second": rate, "paging_pressure": self.pressure_samples >= 2,
                "paging_counter_error": self.error,
                "paging_counter_scope": "system hard page inputs, not specific to pagefile or this job"}

    def close(self):
        if self.query:
            self.api.PdhCloseQuery(self.query)
            self.query = None


def actionable_paging_pressure(available_gib, worker_gib, hard_page_pressure):
    """Treat hard-page activity as pressure only with little physical headroom.

    Windows counts ordinary mapped-file reads too. Those alone must not shed
    workers while RAM is plentiful. The 2 GiB reserve plus one peak worker
    (at least 1 GiB) is an explicit precautionary band, not a swap detector.
    """
    return bool(hard_page_pressure and available_gib < 2. + max(1., worker_gib))


def capacity_target(cpus, system_cpu_percent, own_cpu_cores, available_gib, worker_gib, maximum, paging_pressure=False,
                    active_workers=0, active_resident_gib=0.):
    import math
    from .learning import admission_target
    # Active resident memory has ALREADY been deducted from available RAM. Reserve
    # their unrealized peak growth, then budget additional workers, not the same
    # resident pages a second time. Unknown residency=0 is conservative.
    peak = max(.5, worker_gib)
    unrealized = max(0., active_workers*peak-max(0., active_resident_gib))
    ram = max(0, active_workers+math.floor((available_gib-2.-unrealized)/peak))
    cpu = admission_target(cpus, system_cpu_percent, own_cpu_cores, 1e9, peak, maximum)
    target = min(maximum, cpu, ram)
    return min(target, 1) if actionable_paging_pressure(available_gib, peak, paging_pressure) else target
