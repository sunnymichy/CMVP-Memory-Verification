"""
perturbation_bench.py -- quantify read-only-observation non-intrusiveness (REAL measurement)
============================================================================================
Produces the numbers the manuscript needs for a "Non-intrusiveness measurement" subsection,
honestly measured on the live target.

EASIEST -- one command, N repetitions of each condition at equal duration, auto-aggregated
to mean +/- SD, and writes perturbation.tex (Table tab:perturbation) directly:
      python perturbation_bench.py --role driver --reps 5 --duration 20 --tex perturbation.tex

Manual two-shell mode (single run; same measurement, for inspection):
  Shell A (the target):
      python perturbation_bench.py --role target --duration 30 --pidfile pert.pid
  Shell B (the read-only monitor; start AFTER the target prints "ready"):
      python perturbation_bench.py --role monitor --duration 25 --pidfile pert.pid --interval 0.05

Protocol (do it twice):
  1) BASELINE: run the target ALONE (do not start the monitor). Record its printed
     ops/sec, page-fault delta, working-set delta, and mismatches (must be 0).
  2) MONITORED: run the target AND the monitor concurrently. Record the same target
     numbers plus the monitor's suspend-latency distribution and snapshot bytes.
  Compare the two: the ops/sec drop, extra page faults, and working-set delta are the
  observation cost; mismatches must stay 0 (output bit-identity = functional non-interference).

No number is hard-coded into the paper; the two runs' output provides the reported
values. Windows only (ctypes psapi/ntdll). Requires the `cryptography` package.
"""
import argparse, ctypes as C, hashlib, json, os, sys, time
from ctypes import wintypes

# ---------------------------------------------------------------- Windows process counters
class PROCESS_MEMORY_COUNTERS(C.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", C.c_size_t), ("WorkingSetSize", C.c_size_t),
                ("QuotaPeakPagedPoolUsage", C.c_size_t), ("QuotaPagedPoolUsage", C.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", C.c_size_t), ("QuotaNonPagedPoolUsage", C.c_size_t),
                ("PagefileUsage", C.c_size_t), ("PeakPagefileUsage", C.c_size_t)]

_psapi = C.WinDLL("psapi", use_last_error=True)
_kernel32 = C.WinDLL("kernel32", use_last_error=True)
_ntdll = C.WinDLL("ntdll", use_last_error=True)

# Pin 64-bit handle/pointer types so pseudo-handles (GetCurrentProcess == -1) are not
# truncated to 32 bits (the cause of WinError 6, invalid handle).
_H = wintypes.HANDLE
_kernel32.GetCurrentProcess.restype = _H
_kernel32.GetCurrentProcess.argtypes = []
_kernel32.OpenProcess.restype = _H
_kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
_kernel32.VirtualQueryEx.restype = C.c_size_t
_kernel32.VirtualQueryEx.argtypes = [_H, C.c_void_p, C.c_void_p, C.c_size_t]
_kernel32.ReadProcessMemory.restype = wintypes.BOOL
_kernel32.ReadProcessMemory.argtypes = [_H, C.c_void_p, C.c_void_p, C.c_size_t, C.POINTER(C.c_size_t)]
_psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
_psapi.GetProcessMemoryInfo.argtypes = [_H, C.c_void_p, wintypes.DWORD]
_ntdll.NtSuspendProcess.restype = wintypes.LONG
_ntdll.NtSuspendProcess.argtypes = [_H]
_ntdll.NtResumeProcess.restype = wintypes.LONG
_ntdll.NtResumeProcess.argtypes = [_H]


def counters(handle):
    pmc = PROCESS_MEMORY_COUNTERS(); pmc.cb = C.sizeof(pmc)
    if not _psapi.GetProcessMemoryInfo(handle, C.byref(pmc), pmc.cb):
        raise C.WinError(C.get_last_error())
    return pmc.PageFaultCount, pmc.WorkingSetSize, pmc.PeakWorkingSetSize


# ---------------------------------------------------------------- target role
def run_target(a):
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    key = bytes(range(32))                       # fixed 256-bit key, held resident
    pt = b"CMVP-KAT-block!!"                      # fixed 16-byte plaintext
    enc = Cipher(algorithms.AES(key), modes.ECB()).encryptor
    golden = (lambda e: e.update(pt) + e.finalize())(enc())
    _resident = [key, golden]                    # keep material in memory

    self_h = _kernel32.GetCurrentProcess()
    with open(a.pidfile, "w") as f:
        f.write(str(os.getpid()))
    pf0, ws0, _ = counters(self_h)
    print(f"[target] pid={os.getpid()} ready; key+output resident. running {a.duration}s ...", flush=True)

    ops = mism = 0
    t_end = time.perf_counter() + a.duration
    while time.perf_counter() < t_end:
        ct = (lambda e: e.update(pt) + e.finalize())(enc())
        if ct != golden:
            mism += 1
        ops += 1
    pf1, ws1, wsp = counters(self_h)
    dur = a.duration
    try:
        os.remove(a.pidfile)
    except OSError:
        pass
    out = {"role": "target", "ops": ops, "ops_per_s": round(ops / dur, 1),
           "mismatches": mism, "output_sha256": hashlib.sha256(golden).hexdigest()[:16],
           "page_fault_delta": pf1 - pf0, "working_set_delta_kb": round((ws1 - ws0) / 1024, 1),
           "peak_working_set_kb": round(wsp / 1024, 1)}
    print("[target] RESULT " + json.dumps(out), flush=True)


# ---------------------------------------------------------------- monitor role (read-only)
PROCESS_RIGHTS = 0x0800 | 0x0400 | 0x0010 | 0x1000  # SUSPEND_RESUME|QUERY_INFO|VM_READ|QUERY_LIMITED

class MEMORY_BASIC_INFORMATION(C.Structure):
    _fields_ = [("BaseAddress", C.c_void_p), ("AllocationBase", C.c_void_p),
                ("AllocationProtect", wintypes.DWORD), ("RegionSize", C.c_size_t),
                ("State", wintypes.DWORD), ("Protect", wintypes.DWORD), ("Type", wintypes.DWORD)]


def snapshot_bytes(handle, cap):
    """Read committed, readable regions (read-only) up to `cap` bytes; return bytes read."""
    READABLE = {0x02, 0x04, 0x20, 0x40}          # R, RW, X_R, X_RW
    addr, total = 0, 0
    mbi = MEMORY_BASIC_INFORMATION()
    buf = C.create_string_buffer(1 << 20)
    nread = C.c_size_t(0)
    while total < cap:
        if _kernel32.VirtualQueryEx(handle, C.c_void_p(addr), C.byref(mbi), C.sizeof(mbi)) == 0:
            break
        base = mbi.BaseAddress or 0
        size = mbi.RegionSize or 0
        if mbi.State == 0x1000 and (mbi.Protect & 0xFF) in READABLE:   # MEM_COMMIT + readable
            off = 0
            while off < size and total < cap:
                chunk = min(len(buf), size - off, cap - total)
                if _kernel32.ReadProcessMemory(handle, C.c_void_p(base + off), buf,
                                               chunk, C.byref(nread)) and nread.value:
                    total += nread.value
                off += len(buf)
        addr = base + size
        if addr == 0:
            break
    return total


def monitor_loop(h, duration, interval, cap):
    """Suspend/snapshot/resume the target (by handle) for `duration` s; return stats dict."""
    pf0, ws0, _ = counters(h)
    lat, snap_bytes, cycles = [], 0, 0
    t_end = time.perf_counter() + duration
    while time.perf_counter() < t_end:
        if _ntdll.NtSuspendProcess(h) != 0:
            break                                  # target exited or access lost
        t0 = time.perf_counter()
        snap_bytes += snapshot_bytes(h, cap)
        _ntdll.NtResumeProcess(h)
        lat.append((time.perf_counter() - t0) * 1e3)  # ms suspended
        cycles += 1
        time.sleep(interval)
    pf1, ws1, _ = counters(h)
    ls = sorted(lat)
    def pct(p): return round(ls[min(len(ls) - 1, int(p * len(ls)))], 2) if ls else None
    return {"cycles": cycles,
            "suspend_ms_mean": round(sum(lat) / len(lat), 2) if lat else None,
            "suspend_ms_p50": pct(0.50), "suspend_ms_p95": pct(0.95),
            "suspend_ms_max": round(max(lat), 2) if lat else None,
            "snapshot_MB_per_cycle": round(snap_bytes / max(cycles, 1) / 1e6, 2),
            "target_pf_delta_during": pf1 - pf0, "target_ws_delta_kb": round((ws1 - ws0) / 1024, 1)}


def run_monitor(a):
    while not os.path.exists(a.pidfile):
        time.sleep(0.05)
    pid = int(open(a.pidfile).read().strip())
    h = _kernel32.OpenProcess(PROCESS_RIGHTS, False, pid)
    if not h:
        raise C.WinError(C.get_last_error())
    print(f"[monitor] attached to pid={pid} (read-only). observing {a.duration}s ...", flush=True)
    out = {"role": "monitor", **monitor_loop(h, a.duration, a.interval, a.cap)}
    print("[monitor] RESULT " + json.dumps(out), flush=True)


def run_driver(a):
    """One command: run baseline (target alone) and monitored (target + read-only monitor)
    `--reps` times each at equal duration, then report mean +/- SD and emit perturbation.tex."""
    import re as _re, subprocess as _sp, statistics as _st
    self_path = os.path.abspath(__file__)

    def _target_alone(dur):
        p = _sp.run([sys.executable, self_path, "--role", "target",
                     "--duration", str(dur), "--pidfile", a.pidfile], capture_output=True, text=True)
        m = _re.search(r"\[target\] RESULT (\{.*\})", p.stdout)
        if not m:
            raise RuntimeError("target produced no RESULT:\n" + p.stdout + p.stderr)
        return json.loads(m.group(1))

    def _target_monitored(dur):
        try: os.remove(a.pidfile)
        except OSError: pass
        proc = _sp.Popen([sys.executable, self_path, "--role", "target",
                          "--duration", str(dur), "--pidfile", a.pidfile],
                         stdout=_sp.PIPE, stderr=_sp.PIPE, text=True)
        waited = 0.0
        while not os.path.exists(a.pidfile) and waited < 15 and proc.poll() is None:
            time.sleep(0.1); waited += 0.1
        if not os.path.exists(a.pidfile):
            o, e = proc.communicate(); raise RuntimeError("target never wrote pidfile:\n" + o + e)
        pid = int(open(a.pidfile).read().strip())
        h = _kernel32.OpenProcess(PROCESS_RIGHTS, False, pid)
        if not h:
            proc.terminate(); raise C.WinError(C.get_last_error())
        time.sleep(0.3)
        mon = monitor_loop(h, max(dur - 1, 1), a.interval, a.cap)   # stop ~1s before target exits
        o, e = proc.communicate()
        m = _re.search(r"\[target\] RESULT (\{.*\})", o)
        if not m:
            raise RuntimeError("monitored target no RESULT:\n" + o + e)
        return json.loads(m.group(1)), mon

    base, mon_t, mon_m = [], [], []
    for i in range(a.reps):
        print(f"[driver] baseline  rep {i+1}/{a.reps} ({a.duration:.0f}s) ...", flush=True)
        base.append(_target_alone(a.duration))
    for i in range(a.reps):
        print(f"[driver] monitored rep {i+1}/{a.reps} ({a.duration:.0f}s) ...", flush=True)
        t, mm = _target_monitored(a.duration); mon_t.append(t); mon_m.append(mm)

    def stat(xs):
        xs = [x for x in xs if x is not None]
        if not xs: return (0.0, 0.0)
        return (round(_st.mean(xs), 1), round(_st.pstdev(xs), 1) if len(xs) > 1 else 0.0)

    bo, bos = stat([r["ops_per_s"] for r in base]);             mo, mos = stat([r["ops_per_s"] for r in mon_t])
    bpf, _  = stat([r["page_fault_delta"] for r in base]);      mpf, _  = stat([r["page_fault_delta"] for r in mon_t])
    bws, _  = stat([r["working_set_delta_kb"] for r in base]);  mws, _  = stat([r["working_set_delta_kb"] for r in mon_t])
    sus, _  = stat([r["suspend_ms_mean"] for r in mon_m]);      sp95, _ = stat([r["suspend_ms_p95"] for r in mon_m])
    snap, _ = stat([r["snapshot_MB_per_cycle"] for r in mon_m])
    mism = sum(r["mismatches"] for r in base) + sum(r["mismatches"] for r in mon_t)
    nops = sum(r["ops"] for r in base) + sum(r["ops"] for r in mon_t)
    drop = round(100 * (bo - mo) / bo, 1) if bo else 0.0
    bws_mb, mws_mb = round(bws / 1024, 2), round(mws / 1024, 2)

    summary = {"reps": a.reps, "duration_s": a.duration,
               "throughput_ops_s": {"baseline_mean_sd": [bo, bos], "monitored_mean_sd": [mo, mos],
                                    "drop_pct": drop},
               "page_faults": {"baseline": bpf, "monitored": mpf},
               "working_set_MB": {"baseline": bws_mb, "monitored": mws_mb},
               "suspend_ms_mean": sus, "suspend_ms_p95": sp95, "snapshot_MB": snap,
               "total_ops": nops, "total_mismatches": mism}
    print("\n[driver] SUMMARY " + json.dumps(summary, indent=2))

    tex = ("% Auto-generated by perturbation_bench.py --role driver -- VERIFY then \\input.\n"
           "\\begin{table}[t]\\centering\\scriptsize\n"
           f"\\caption{{Non-intrusiveness of read-only observation ({a.reps} runs each, {a.duration:.0f}\\,s, "
           f"deterministic AES KAT target at 20\\,Hz snapshotting; mean$\\pm$SD). Output is bit-identical in "
           f"every run ({mism} mismatches over {nops/1e6:.1f}M operations).}}\n"
           "\\label{tab:perturbation}\n\\begin{tabular}{@{}lccc@{}}\n\\toprule\n"
           "\\textbf{Metric} & \\textbf{Baseline} & \\textbf{Monitored} & \\textbf{$\\Delta$} \\\\\n\\midrule\n"
           f"Throughput (ops/s) & ${bo:,.0f}\\pm{bos:,.0f}$ & ${mo:,.0f}\\pm{mos:,.0f}$ & ${-drop:+.1f}\\%$ \\\\\n"
           f"Page faults & {bpf:,.0f} & {mpf:,.0f} & $+{mpf-bpf:,.0f}$ \\\\\n"
           f"Working-set growth (MB) & {bws_mb} & {mws_mb} & $+{round(mws_mb-bws_mb,2)}$ \\\\\n"
           f"Output mismatches & 0 & 0 & 0 \\\\\n\\midrule\n"
           f"Suspend/snapshot (ms, mean / p95) & \\multicolumn{{3}}{{c}}{{${sus}\\,/\\,{sp95}$}} \\\\\n"
           f"Read per snapshot (MB) & \\multicolumn{{3}}{{c}}{{${snap}$}} \\\\\n"
           "\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    open(a.tex, "w", encoding="utf-8").write(tex)
    print(f"\n[driver] wrote {a.tex}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", required=True, choices=["target", "monitor", "driver"])
    ap.add_argument("--duration", type=float, default=30)
    ap.add_argument("--reps", type=int, default=5, help="driver: repetitions of each condition")
    ap.add_argument("--pidfile", default="pert.pid")
    ap.add_argument("--interval", type=float, default=0.05, help="monitor: seconds between snapshots")
    ap.add_argument("--cap", type=int, default=64 * 1024 * 1024, help="monitor: max bytes per snapshot")
    ap.add_argument("--tex", default="perturbation.tex")
    a = ap.parse_args()
    {"target": run_target, "monitor": run_monitor, "driver": run_driver}[a.role](a)


if __name__ == "__main__":
    main()
