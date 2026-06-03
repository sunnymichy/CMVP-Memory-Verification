"""
win_memory.py
Windows 프로세스 메모리 읽기 및 스캔 유틸리티.

ReadProcessMemory / VirtualQueryEx 래퍼를 제공하여
자체 프로세스 또는 외부 프로세스의 메모리를 분석한다.
"""

import ctypes
import ctypes.wintypes as wt
import os
import struct
from typing import List, Optional, Tuple

kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)

# ─── 상수 ───
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


# ─── Windows API 프로토타입 ───
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


# ─── 메모리 영역 유형 판별 ───
# 논문 F6: 0=Unknown, 1=DLL Data, 2=Stack/Heap(Private), 3=Other

def classify_region_type(mbi: MEMORY_BASIC_INFORMATION) -> int:
    """MEMORY_BASIC_INFORMATION으로부터 논문의 메모리 영역 유형(F6)을 판별."""
    if mbi.Type == MEM_IMAGE:
        return 1    # DLL/EXE 이미지 영역 → DLL Data
    elif mbi.Type == MEM_PRIVATE:
        return 2    # Stack/Heap (Private)
    elif mbi.Type == MEM_MAPPED:
        return 3    # Mapped Memory → Other
    else:
        return 0    # Unknown


def is_readable(mbi: MEMORY_BASIC_INFORMATION) -> bool:
    """해당 영역이 읽기 가능한지 확인."""
    if mbi.State != MEM_COMMIT:
        return False
    if mbi.Protect & PAGE_GUARD:
        return False
    if mbi.Protect == PAGE_NOACCESS:
        return False
    return (mbi.Protect & 0xFF) in READABLE_PROTECTIONS


# ─── 메모리 영역 열거 ───

class MemoryRegion:
    """프로세스 메모리 영역 정보."""
    __slots__ = ('base', 'size', 'protect', 'mem_type', 'region_type')

    def __init__(self, base: int, size: int, protect: int,
                 mem_type: int, region_type: int):
        self.base = base
        self.size = size
        self.protect = protect
        self.mem_type = mem_type
        self.region_type = region_type     # 논문 F6 값

    def __repr__(self):
        type_names = {0: 'Unknown', 1: 'DLL', 2: 'Private', 3: 'Other'}
        return (f"Region(0x{self.base:X}, size={self.size}, "
                f"type={type_names.get(self.region_type, '?')})")


def enumerate_regions(handle: wt.HANDLE,
                      max_addr: int = 0x7FFFFFFFFFFF) -> List[MemoryRegion]:
    """프로세스의 읽기 가능한 메모리 영역을 열거한다."""
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
    """프로세스 메모리를 읽는다. 실패 시 None 반환."""
    buf = (ctypes.c_ubyte * size)()
    bytes_read = ctypes.c_size_t(0)
    ok = kernel32.ReadProcessMemory(
        handle, ctypes.c_void_p(address),
        ctypes.byref(buf), size, ctypes.byref(bytes_read)
    )
    if ok and bytes_read.value > 0:
        return bytes(buf[:bytes_read.value])
    return None


# ─── 패턴 검색 ───

def scan_for_pattern(handle: wt.HANDLE, pattern: bytes,
                     regions: Optional[List[MemoryRegion]] = None,
                     ) -> List[Tuple[int, int]]:
    """
    프로세스 메모리에서 바이트 패턴을 검색한다.

    Returns:
        [(address, region_type), ...] 검색된 주소와 영역 유형 목록
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
    """자체 프로세스 메모리를 직접 읽는다 (ctypes.string_at)."""
    return ctypes.string_at(address, size)


def get_own_process_handle() -> wt.HANDLE:
    """자체 프로세스 핸들을 반환한다."""
    return kernel32.GetCurrentProcess()


def get_region_type_at(handle: wt.HANDLE, address: int) -> int:
    """특정 주소의 메모리 영역 유형(F6)을 반환한다."""
    mbi = MEMORY_BASIC_INFORMATION()
    ret = kernel32.VirtualQueryEx(handle, ctypes.c_void_p(address),
                                  ctypes.byref(mbi), ctypes.sizeof(mbi))
    if ret == 0:
        return 0
    return classify_region_type(mbi)


# ─── 스냅샷 ───

class MemorySnapshot:
    """특정 주소 목록의 메모리 상태 스냅샷."""

    def __init__(self):
        self.entries = {}   # {address: bytes}
        self.timestamp = 0.0

    def capture(self, handle: wt.HANDLE,
                addresses: List[Tuple[int, int]]):
        """
        지정 주소 목록의 메모리를 캡처한다.
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
    시계열 메모리 스냅샷을 수집한다.

    Args:
        handle: 프로세스 핸들
        addresses: [(address, size), ...] 캡처할 주소 목록
        count: 스냅샷 수 (기본 20)
        interval_ms: 캡처 주기 (밀리초)
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
    """스냅샷 간 해당 주소의 변경 횟수를 계산한다."""
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


# ─── 단독 테스트 ───
if __name__ == '__main__':
    h = get_own_process_handle()
    regions = enumerate_regions(h)
    print(f"읽기 가능 메모리 영역: {len(regions)}개")

    type_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    total_size = 0
    for r in regions:
        type_counts[r.region_type] += 1
        total_size += r.size

    type_names = {0: 'Unknown', 1: 'DLL/Image', 2: 'Private', 3: 'Other'}
    for t, c in sorted(type_counts.items()):
        print(f"  {type_names[t]}: {c}개")
    print(f"  총 메모리: {total_size / (1024*1024):.1f} MB")

    # 패턴 검색 테스트
    test_pattern = b'\xDE\xAD\xBE\xEF' * 8   # 32바이트 테스트 패턴
    buf = (ctypes.c_ubyte * 32)(*test_pattern)
    buf_addr = ctypes.addressof(buf)
    print(f"\n테스트 패턴 주소: 0x{buf_addr:X}")

    found = scan_for_pattern(h, test_pattern, regions)
    print(f"검색 결과: {len(found)}개 발견")
    for addr, rtype in found:
        print(f"  0x{addr:X} (영역 유형: {type_names[rtype]})")
