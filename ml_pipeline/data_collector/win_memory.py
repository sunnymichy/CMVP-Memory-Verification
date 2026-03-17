"""
win_memory.py
Windows process memory reading and scanning utility.

Provides wrappers for ReadProcessMemory / VirtualQueryEx to analyze
memory of the current process or external processes.
"""

import ctypes
import ctypes.wintypes as wt
import os
import struct
from typing import List, Optional, Tuple

kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)

# ─── Constants ───
PROCESS_VM_READ = 0x0010
PROCESS_QUERY_INFORMATION = 0x0400
MEM_COMMIT = 0x1000
MEM_IMAGE = 0x1000000
MEM_MAPPED = 0x40000
MEM_PRIVATE = 0x20000

PAGE_NOACCESS = 0x01
PAGE_GUARD = 0x100
PAGE_EXECUTE = 0x10
PAGE_EXECUTE_READ = 0x20
PAGE_EXECUTE_READWRITE = 0x40
PAGE_EXECUTE_WRITECOPY = 0x80

READABLE_PROTECTIONS = {
    0x02,   # PAGE_READONLY
    0x04,   # PAGE_READWRITE
    0x08,   # PAGE_WRITECOPY
    PAGE_EXECUTE_READ,
    PAGE_EXECUTE_READWRITE,
    PAGE_EXECUTE_WRITECOPY,
}


class MEMORY_BASIC_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("BaseAddress", ctypes.c_void_p),
        ("AllocationBase", ctypes.c_void_p),
        ("AllocationProtect", wt.DWORD),
        ("RegionSize", ctypes.c_size_t),
        ("State", wt.DWORD),
        ("Protect", wt.DWORD),
        ("Type", wt.DWORD),
    ]


# ─── Windows API Prototypes ───
kernel32.OpenProcess.restype = wt.HANDLE
kernel32.OpenProcess.argtypes = [wt.DWORD, wt.BOOL, wt.DWORD]

kernel32.CloseHandle.restype = wt.BOOL
kernel32.CloseHandle.argtypes = [wt.HANDLE]

kernel32.ReadProcessMemory.restype = wt.BOOL
kernel32.ReadProcessMemory.argtypes = [
    wt.HANDLE, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_size_t)
]

kernel32.VirtualQueryEx.restype = ctypes.c_size_t
kernel32.VirtualQueryEx.argtypes = [
    wt.HANDLE, ctypes.c_void_p,
    ctypes.POINTER(MEMORY_BASIC_INFORMATION), ctypes.c_size_t
]

kernel32.GetCurrentProcess.restype = wt.HANDLE
kernel32.GetCurrentProcess.argtypes = []


# ─── Memory Region Type Classification ───
# Paper F6: 0=Unknown, 1=DLL Data, 2=Stack/Heap(Private), 3=Other

def classify_region_type(mbi: MEMORY_BASIC_INFORMATION) -> int:
    """Classify the memory region type (F6) as defined in the paper from MEMORY_BASIC_INFORMATION."""
    if mbi.Type == MEM_IMAGE:
        return 1    # DLL/EXE image region -> DLL Data
    elif mbi.Type == MEM_PRIVATE:
        return 2    # Stack/Heap (Private)
    elif mbi.Type == MEM_MAPPED:
        return 3    # Mapped Memory -> Other
    else:
        return 0    # Unknown


def is_readable(mbi: MEMORY_BASIC_INFORMATION) -> bool:
    """Check whether the given memory region is readable."""
    if mbi.State != MEM_COMMIT:
        return False
    if mbi.Protect & PAGE_GUARD:
        return False
    if mbi.Protect == PAGE_NOACCESS:
        return False
    return (mbi.Protect & 0xFF) in READABLE_PROTECTIONS


# ─── Memory Region Enumeration ───

class MemoryRegion:
    """Process memory region information."""
    __slots__ = ('base', 'size', 'protect', 'mem_type', 'region_type')

    def __init__(self, base: int, size: int, protect: int,
                 mem_type: int, region_type: int):
        self.base = base
        self.size = size
        self.protect = protect
        self.mem_type = mem_type
        self.region_type = region_type     # Paper F6 value

    def __repr__(self):
        type_names = {0: 'Unknown', 1: 'DLL', 2: 'Private', 3: 'Other'}
        return (f"Region(0x{self.base:X}, size={self.size}, "
                f"type={type_names.get(self.region_type, '?')})")


def enumerate_regions(handle: wt.HANDLE,
                      max_addr: int = 0x7FFFFFFFFFFF) -> List[MemoryRegion]:
    """Enumerate readable memory regions of a process."""
    regions = []
    addr = 0
    mbi = MEMORY_BASIC_INFORMATION()
    mbi_size = ctypes.sizeof(mbi)

    while addr < max_addr:
        ret = kernel32.VirtualQueryEx(handle, ctypes.c_void_p(addr),
                                      ctypes.byref(mbi), mbi_size)
        if ret == 0:
            break

        base = mbi.BaseAddress or 0
        size = mbi.RegionSize or 0

        if is_readable(mbi) and size > 0:
            regions.append(MemoryRegion(
                base=base,
                size=size,
                protect=mbi.Protect,
                mem_type=mbi.Type,
                region_type=classify_region_type(mbi),
            ))

        next_addr = base + size
        if next_addr <= addr:
            break
        addr = next_addr

    return regions


def read_memory(handle: wt.HANDLE, address: int, size: int) -> Optional[bytes]:
    """Read process memory. Returns None on failure."""
    buf = (ctypes.c_ubyte * size)()
    bytes_read = ctypes.c_size_t(0)
    ok = kernel32.ReadProcessMemory(
        handle, ctypes.c_void_p(address),
        ctypes.byref(buf), size, ctypes.byref(bytes_read)
    )
    if ok and bytes_read.value > 0:
        return bytes(buf[:bytes_read.value])
    return None


# ─── Pattern Search ───

def scan_for_pattern(handle: wt.HANDLE, pattern: bytes,
                     regions: Optional[List[MemoryRegion]] = None,
                     ) -> List[Tuple[int, int]]:
    """
    Search for a byte pattern in process memory.

    Returns:
        [(address, region_type), ...] list of found addresses and their region types
    """
    if regions is None:
        regions = enumerate_regions(handle)

    found = []
    plen = len(pattern)

    for region in regions:
        data = read_memory(handle, region.base, region.size)
        if data is None:
            continue

        offset = 0
        while True:
            idx = data.find(pattern, offset)
            if idx == -1:
                break
            found.append((region.base + idx, region.region_type))
            offset = idx + 1

    return found


def read_own_memory(address: int, size: int) -> bytes:
    """Read current process memory directly (ctypes.string_at)."""
    return ctypes.string_at(address, size)


def get_own_process_handle() -> wt.HANDLE:
    """Return the current process handle."""
    return kernel32.GetCurrentProcess()


def get_region_type_at(handle: wt.HANDLE, address: int) -> int:
    """Return the memory region type (F6) at a specific address."""
    mbi = MEMORY_BASIC_INFORMATION()
    ret = kernel32.VirtualQueryEx(handle, ctypes.c_void_p(address),
                                  ctypes.byref(mbi), ctypes.sizeof(mbi))
    if ret == 0:
        return 0
    return classify_region_type(mbi)


# ─── Snapshot ───

class MemorySnapshot:
    """Memory state snapshot for a list of specific addresses."""

    def __init__(self):
        self.entries = {}   # {address: bytes}
        self.timestamp = 0.0

    def capture(self, handle: wt.HANDLE,
                addresses: List[Tuple[int, int]]):
        """
        Capture memory at the specified list of addresses.
        addresses: [(address, size), ...]
        """
        import time
        self.timestamp = time.time()
        self.entries = {}
        for addr, size in addresses:
            data = read_memory(handle, addr, size)
            if data is not None:
                self.entries[addr] = data

    def get(self, address: int) -> Optional[bytes]:
        return self.entries.get(address)


def take_snapshots(handle: wt.HANDLE,
                   addresses: List[Tuple[int, int]],
                   count: int = 20,
                   interval_ms: int = 500) -> List[MemorySnapshot]:
    """
    Collect time-series memory snapshots.

    Args:
        handle: Process handle
        addresses: [(address, size), ...] list of addresses to capture
        count: Number of snapshots (default 20)
        interval_ms: Capture interval in milliseconds
    """
    import time
    snapshots = []
    for i in range(count):
        snap = MemorySnapshot()
        snap.capture(handle, addresses)
        snapshots.append(snap)
        if i < count - 1:
            time.sleep(interval_ms / 1000.0)
    return snapshots


def count_changes(snapshots: List[MemorySnapshot], address: int) -> int:
    """Count the number of changes at the given address across snapshots."""
    changes = 0
    prev = None
    for snap in snapshots:
        curr = snap.get(address)
        if curr is None:
            continue
        if prev is not None and curr != prev:
            changes += 1
        prev = curr
    return changes


# ─── Standalone Test ───
if __name__ == '__main__':
    h = get_own_process_handle()
    regions = enumerate_regions(h)
    print(f"Readable memory regions: {len(regions)}")

    type_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    total_size = 0
    for r in regions:
        type_counts[r.region_type] += 1
        total_size += r.size

    type_names = {0: 'Unknown', 1: 'DLL/Image', 2: 'Private', 3: 'Other'}
    for t, c in sorted(type_counts.items()):
        print(f"  {type_names[t]}: {c}")
    print(f"  Total memory: {total_size / (1024*1024):.1f} MB")

    # Pattern search test
    test_pattern = b'\xDE\xAD\xBE\xEF' * 8   # 32-byte test pattern
    buf = (ctypes.c_ubyte * 32)(*test_pattern)
    buf_addr = ctypes.addressof(buf)
    print(f"\nTest pattern address: 0x{buf_addr:X}")

    found = scan_for_pattern(h, test_pattern, regions)
    print(f"Search results: {len(found)} found")
    for addr, rtype in found:
        print(f"  0x{addr:X} (region type: {type_names[rtype]})")
