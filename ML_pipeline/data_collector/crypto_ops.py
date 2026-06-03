"""
crypto_ops.py
암호 라이브러리별 키/IV/암호문/평문 생성 및 메모리 주소 추적.

지원 라이브러리:
  1) cryptography (OpenSSL 백엔드)
  2) PyCryptodome
  3) Windows CNG (ctypes)
  4) PyNaCl (libsodium 백엔드)
  5) pyaes (순수 Python AES)

각 함수는 알려진(known) 키 값을 사용하여 암호 연산을 수행하고,
메모리에서 해당 바이트 패턴을 검색하여 ground truth 주소를 반환한다.
"""

import os
import ctypes
import ctypes.wintypes as wt
import struct
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from win_memory import (
    get_own_process_handle,
    enumerate_regions,
    scan_for_pattern,
    get_region_type_at,
    MemoryRegion,
)


@dataclass
class CryptoSample:
    """수집된 암호 데이터 표본."""
    label: str              # KEY / IV / CIPHERTEXT / PLAINTEXT / NON_CRYPTO
    data: bytes             # 실제 바이트 데이터
    address: int            # 메모리 주소
    region_type: int        # 0-3 (논문 F6)
    algorithm: str          # 예: AES-256-CBC
    library: str            # 예: OpenSSL
    description: str = ""   # 추가 설명


# ─── 라이브러리 공용 도우미 ───

def _keep_alive_list():
    """GC 방지용 참조 보관 리스트."""
    if not hasattr(_keep_alive_list, '_refs'):
        _keep_alive_list._refs = []
    return _keep_alive_list._refs


def _pin_bytes(data: bytes) -> Tuple[int, ctypes.Array]:
    """
    바이트 데이터를 ctypes 버퍼에 복사하여 주소를 고정한다.
    GC로 인한 주소 이동을 방지.
    """
    buf = (ctypes.c_ubyte * len(data))(*data)
    _keep_alive_list().append(buf)  # prevent GC
    return ctypes.addressof(buf), buf


def _find_in_memory(pattern: bytes, handle=None, regions=None,
                    max_regions: int = 80,
                    ) -> List[Tuple[int, int]]:
    """메모리에서 패턴을 검색. [(address, region_type), ...]
    성능을 위해 Private(힙) 영역만 우선 스캔하고 최대 영역 수를 제한한다."""
    if handle is None:
        handle = get_own_process_handle()
    if regions is None:
        all_regions = enumerate_regions(handle)
        # Private(힙/스택) 영역만 선별 (키가 주로 존재하는 곳)
        regions = [r for r in all_regions if r.region_type == 2]
        if len(regions) > max_regions:
            regions = regions[:max_regions]
    return scan_for_pattern(handle, pattern, regions)


# ═══════════════════════════════════════════════════════════
# 1. cryptography (OpenSSL 백엔드)
# ═══════════════════════════════════════════════════════════

def collect_openssl_aes(key_size: int = 32) -> List[CryptoSample]:
    """OpenSSL AES 키/IV/암호문/평문 수집."""
    try:
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    except ImportError:
        print("  [건너뜀] cryptography 패키지 미설치")
        return []

    samples = []
    handle = get_own_process_handle()
    regions = enumerate_regions(handle)

    algo_name = f"AES-{key_size*8}-CBC"

    # 알려진 값 생성
    key_bytes = os.urandom(key_size)
    iv_bytes = os.urandom(16)
    plaintext = os.urandom(64)

    # ctypes 버퍼에 고정 (GC 방지)
    key_addr, key_buf = _pin_bytes(key_bytes)
    iv_addr, iv_buf = _pin_bytes(iv_bytes)
    pt_addr, pt_buf = _pin_bytes(plaintext)

    # 암호화 수행
    cipher = Cipher(algorithms.AES(key_bytes), modes.CBC(iv_bytes))
    encryptor = cipher.encryptor()
    ciphertext = encryptor.update(plaintext) + encryptor.finalize()
    ct_addr, ct_buf = _pin_bytes(ciphertext)

    # ── KEY 표본 ──
    # 고정 버퍼 주소 (확실한 ground truth)
    samples.append(CryptoSample(
        label='KEY', data=key_bytes, address=key_addr,
        region_type=get_region_type_at(handle, key_addr),
        algorithm=algo_name, library='OpenSSL',
        description=f'pinned buffer {key_size}B'
    ))
    # 메모리 내 추가 사본 검색 (라이브러리 내부 복사본)
    found = _find_in_memory(key_bytes, handle, regions)
    for addr, rtype in found:
        if addr != key_addr:
            samples.append(CryptoSample(
                label='KEY', data=key_bytes, address=addr,
                region_type=rtype, algorithm=algo_name,
                library='OpenSSL', description='library internal copy'
            ))

    # ── IV 표본 ──
    samples.append(CryptoSample(
        label='IV', data=iv_bytes, address=iv_addr,
        region_type=get_region_type_at(handle, iv_addr),
        algorithm=algo_name, library='OpenSSL',
        description='pinned buffer 16B'
    ))
    for addr, rtype in _find_in_memory(iv_bytes, handle, regions):
        if addr != iv_addr:
            samples.append(CryptoSample(
                label='IV', data=iv_bytes, address=addr,
                region_type=rtype, algorithm=algo_name,
                library='OpenSSL', description='library internal copy'
            ))

    # ── PLAINTEXT 표본 ──
    samples.append(CryptoSample(
        label='PLAINTEXT', data=plaintext, address=pt_addr,
        region_type=get_region_type_at(handle, pt_addr),
        algorithm=algo_name, library='OpenSSL',
        description='input buffer 64B'
    ))

    # ── CIPHERTEXT 표본 ──
    samples.append(CryptoSample(
        label='CIPHERTEXT', data=ciphertext, address=ct_addr,
        region_type=get_region_type_at(handle, ct_addr),
        algorithm=algo_name, library='OpenSSL',
        description='output buffer'
    ))

    # 참조 유지 (GC 방지)
    _keep_alive_list().extend([cipher, encryptor, ciphertext])

    return samples


def collect_openssl_chacha20() -> List[CryptoSample]:
    """OpenSSL ChaCha20-Poly1305 키/IV 수집."""
    try:
        from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
    except ImportError:
        return []

    samples = []
    handle = get_own_process_handle()

    key_bytes = os.urandom(32)
    nonce = os.urandom(12)
    plaintext = os.urandom(48)

    key_addr, _ = _pin_bytes(key_bytes)
    nonce_addr, _ = _pin_bytes(nonce)
    pt_addr, _ = _pin_bytes(plaintext)

    chacha = ChaCha20Poly1305(key_bytes)
    ciphertext = chacha.encrypt(nonce, plaintext, None)
    ct_addr, _ = _pin_bytes(ciphertext)

    samples.append(CryptoSample(
        label='KEY', data=key_bytes, address=key_addr,
        region_type=get_region_type_at(handle, key_addr),
        algorithm='ChaCha20-Poly1305', library='OpenSSL',
    ))
    samples.append(CryptoSample(
        label='IV', data=nonce, address=nonce_addr,
        region_type=get_region_type_at(handle, nonce_addr),
        algorithm='ChaCha20-Poly1305', library='OpenSSL',
        description='nonce 12B'
    ))
    samples.append(CryptoSample(
        label='PLAINTEXT', data=plaintext, address=pt_addr,
        region_type=get_region_type_at(handle, pt_addr),
        algorithm='ChaCha20-Poly1305', library='OpenSSL',
    ))
    samples.append(CryptoSample(
        label='CIPHERTEXT', data=ciphertext, address=ct_addr,
        region_type=get_region_type_at(handle, ct_addr),
        algorithm='ChaCha20-Poly1305', library='OpenSSL',
    ))

    _keep_alive_list().extend([chacha, ciphertext])
    return samples


def collect_openssl_rsa(key_size: int = 2048) -> List[CryptoSample]:
    """OpenSSL RSA 키 수집."""
    try:
        from cryptography.hazmat.primitives.asymmetric import rsa, padding
        from cryptography.hazmat.primitives import hashes, serialization
    except ImportError:
        return []

    samples = []
    handle = get_own_process_handle()
    regions = enumerate_regions(handle)

    private_key = rsa.generate_private_key(
        public_exponent=65537, key_size=key_size
    )

    # DER 형식으로 직렬화하여 바이트 추출
    priv_der = private_key.private_bytes(
        serialization.Encoding.DER,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption()
    )
    pub_der = private_key.public_key().public_bytes(
        serialization.Encoding.DER,
        serialization.PublicFormat.SubjectPublicKeyInfo
    )

    priv_addr, _ = _pin_bytes(priv_der)
    pub_addr, _ = _pin_bytes(pub_der)

    samples.append(CryptoSample(
        label='KEY', data=priv_der, address=priv_addr,
        region_type=get_region_type_at(handle, priv_addr),
        algorithm=f'RSA-{key_size}', library='OpenSSL',
        description=f'private key DER {len(priv_der)}B'
    ))
    samples.append(CryptoSample(
        label='KEY', data=pub_der, address=pub_addr,
        region_type=get_region_type_at(handle, pub_addr),
        algorithm=f'RSA-{key_size}', library='OpenSSL',
        description=f'public key DER {len(pub_der)}B'
    ))

    # 라이브러리 내부 키 사본 검색 (큰 키는 부분 매칭)
    if len(priv_der) <= 512:
        for addr, rtype in _find_in_memory(priv_der, handle, regions):
            if addr != priv_addr:
                samples.append(CryptoSample(
                    label='KEY', data=priv_der, address=addr,
                    region_type=rtype, algorithm=f'RSA-{key_size}',
                    library='OpenSSL', description='library copy'
                ))

    _keep_alive_list().extend([private_key, priv_der, pub_der])
    return samples


def collect_openssl_ec() -> List[CryptoSample]:
    """OpenSSL ECDSA P-256 키 수집."""
    try:
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.hazmat.primitives import serialization
    except ImportError:
        return []

    samples = []
    handle = get_own_process_handle()

    private_key = ec.generate_private_key(ec.SECP256R1())

    priv_der = private_key.private_bytes(
        serialization.Encoding.DER,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption()
    )
    # 비압축 공개키 (65바이트)
    pub_bytes = private_key.public_key().public_bytes(
        serialization.Encoding.X962,
        serialization.PublicFormat.UncompressedPoint
    )

    priv_addr, _ = _pin_bytes(priv_der)
    pub_addr, _ = _pin_bytes(pub_bytes)

    samples.append(CryptoSample(
        label='KEY', data=priv_der, address=priv_addr,
        region_type=get_region_type_at(handle, priv_addr),
        algorithm='ECDSA-P256', library='OpenSSL',
        description=f'private key DER {len(priv_der)}B'
    ))
    samples.append(CryptoSample(
        label='KEY', data=pub_bytes, address=pub_addr,
        region_type=get_region_type_at(handle, pub_addr),
        algorithm='ECDSA-P256', library='OpenSSL',
        description=f'public key uncompressed {len(pub_bytes)}B'
    ))

    _keep_alive_list().extend([private_key, priv_der, pub_bytes])
    return samples


def collect_openssl_hmac() -> List[CryptoSample]:
    """OpenSSL HMAC-SHA256 키 수집."""
    try:
        from cryptography.hazmat.primitives import hmac, hashes
    except ImportError:
        return []

    samples = []
    handle = get_own_process_handle()

    key_bytes = os.urandom(32)
    message = os.urandom(64)

    key_addr, _ = _pin_bytes(key_bytes)
    msg_addr, _ = _pin_bytes(message)

    h = hmac.HMAC(key_bytes, hashes.SHA256())
    h_copy = h.copy()
    h.update(message)
    mac = h.finalize()
    mac_addr, _ = _pin_bytes(mac)

    samples.append(CryptoSample(
        label='KEY', data=key_bytes, address=key_addr,
        region_type=get_region_type_at(handle, key_addr),
        algorithm='HMAC-SHA256', library='OpenSSL',
    ))
    samples.append(CryptoSample(
        label='PLAINTEXT', data=message, address=msg_addr,
        region_type=get_region_type_at(handle, msg_addr),
        algorithm='HMAC-SHA256', library='OpenSSL',
    ))
    samples.append(CryptoSample(
        label='CIPHERTEXT', data=mac, address=mac_addr,
        region_type=get_region_type_at(handle, mac_addr),
        algorithm='HMAC-SHA256', library='OpenSSL',
        description='MAC output 32B'
    ))

    _keep_alive_list().extend([h_copy, mac])
    return samples


def collect_openssl_aes_gcm() -> List[CryptoSample]:
    """OpenSSL AES-256-GCM 수집."""
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError:
        return []

    samples = []
    handle = get_own_process_handle()

    key_bytes = os.urandom(32)
    nonce = os.urandom(12)
    plaintext = os.urandom(48)

    key_addr, _ = _pin_bytes(key_bytes)
    nonce_addr, _ = _pin_bytes(nonce)
    pt_addr, _ = _pin_bytes(plaintext)

    aesgcm = AESGCM(key_bytes)
    ciphertext = aesgcm.encrypt(nonce, plaintext, None)
    ct_addr, _ = _pin_bytes(ciphertext)

    samples.append(CryptoSample(
        label='KEY', data=key_bytes, address=key_addr,
        region_type=get_region_type_at(handle, key_addr),
        algorithm='AES-256-GCM', library='OpenSSL',
    ))
    samples.append(CryptoSample(
        label='IV', data=nonce, address=nonce_addr,
        region_type=get_region_type_at(handle, nonce_addr),
        algorithm='AES-256-GCM', library='OpenSSL',
        description='GCM nonce 12B'
    ))
    samples.append(CryptoSample(
        label='PLAINTEXT', data=plaintext, address=pt_addr,
        region_type=get_region_type_at(handle, pt_addr),
        algorithm='AES-256-GCM', library='OpenSSL',
    ))
    samples.append(CryptoSample(
        label='CIPHERTEXT', data=ciphertext, address=ct_addr,
        region_type=get_region_type_at(handle, ct_addr),
        algorithm='AES-256-GCM', library='OpenSSL',
        description='ciphertext + GCM tag'
    ))

    _keep_alive_list().extend([aesgcm, ciphertext])
    return samples


def collect_openssl_ecdsa_p384() -> List[CryptoSample]:
    """OpenSSL ECDSA P-384 키 수집."""
    try:
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.hazmat.primitives import serialization
    except ImportError:
        return []

    samples = []
    handle = get_own_process_handle()

    private_key = ec.generate_private_key(ec.SECP384R1())

    priv_der = private_key.private_bytes(
        serialization.Encoding.DER,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption()
    )
    pub_bytes = private_key.public_key().public_bytes(
        serialization.Encoding.X962,
        serialization.PublicFormat.UncompressedPoint
    )

    priv_addr, _ = _pin_bytes(priv_der)
    pub_addr, _ = _pin_bytes(pub_bytes)

    samples.append(CryptoSample(
        label='KEY', data=priv_der, address=priv_addr,
        region_type=get_region_type_at(handle, priv_addr),
        algorithm='ECDSA-P384', library='OpenSSL',
        description=f'private key DER {len(priv_der)}B'
    ))
    samples.append(CryptoSample(
        label='KEY', data=pub_bytes, address=pub_addr,
        region_type=get_region_type_at(handle, pub_addr),
        algorithm='ECDSA-P384', library='OpenSSL',
        description=f'public key uncompressed {len(pub_bytes)}B'
    ))

    _keep_alive_list().extend([private_key, priv_der, pub_bytes])
    return samples


def collect_openssl_aes_ctr() -> List[CryptoSample]:
    """OpenSSL AES-256-CTR 수집."""
    try:
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    except ImportError:
        return []

    samples = []
    handle = get_own_process_handle()

    key_bytes = os.urandom(32)
    nonce = os.urandom(16)
    plaintext = os.urandom(64)

    key_addr, _ = _pin_bytes(key_bytes)
    nonce_addr, _ = _pin_bytes(nonce)
    pt_addr, _ = _pin_bytes(plaintext)

    cipher = Cipher(algorithms.AES(key_bytes), modes.CTR(nonce))
    encryptor = cipher.encryptor()
    ciphertext = encryptor.update(plaintext) + encryptor.finalize()
    ct_addr, _ = _pin_bytes(ciphertext)

    samples.append(CryptoSample(
        label='KEY', data=key_bytes, address=key_addr,
        region_type=get_region_type_at(handle, key_addr),
        algorithm='AES-256-CTR', library='OpenSSL',
    ))
    samples.append(CryptoSample(
        label='IV', data=nonce, address=nonce_addr,
        region_type=get_region_type_at(handle, nonce_addr),
        algorithm='AES-256-CTR', library='OpenSSL',
        description='CTR nonce 16B'
    ))
    samples.append(CryptoSample(
        label='PLAINTEXT', data=plaintext, address=pt_addr,
        region_type=get_region_type_at(handle, pt_addr),
        algorithm='AES-256-CTR', library='OpenSSL',
    ))
    samples.append(CryptoSample(
        label='CIPHERTEXT', data=ciphertext, address=ct_addr,
        region_type=get_region_type_at(handle, ct_addr),
        algorithm='AES-256-CTR', library='OpenSSL',
    ))

    _keep_alive_list().extend([cipher, encryptor, ciphertext])
    return samples


# ═══════════════════════════════════════════════════════════
# 2. PyCryptodome
# ═══════════════════════════════════════════════════════════

def collect_pycryptodome_aes(key_size: int = 32) -> List[CryptoSample]:
    """PyCryptodome AES-CBC 키/IV 수집."""
    try:
        from Crypto.Cipher import AES
    except ImportError:
        print("  [건너뜀] pycryptodome 미설치")
        return []

    samples = []
    handle = get_own_process_handle()
    regions = enumerate_regions(handle)
    algo_name = f"AES-{key_size*8}-CBC"

    key_bytes = os.urandom(key_size)
    iv_bytes = os.urandom(16)
    # 평문은 16바이트 배수
    plaintext = os.urandom(64)

    key_addr, _ = _pin_bytes(key_bytes)
    iv_addr, _ = _pin_bytes(iv_bytes)
    pt_addr, _ = _pin_bytes(plaintext)

    cipher = AES.new(key_bytes, AES.MODE_CBC, iv=iv_bytes)
    ciphertext = cipher.encrypt(plaintext)
    ct_addr, _ = _pin_bytes(ciphertext)

    samples.append(CryptoSample(
        label='KEY', data=key_bytes, address=key_addr,
        region_type=get_region_type_at(handle, key_addr),
        algorithm=algo_name, library='PyCryptodome',
    ))
    # 라이브러리 내부 키 사본
    for addr, rtype in _find_in_memory(key_bytes, handle, regions):
        if addr != key_addr:
            samples.append(CryptoSample(
                label='KEY', data=key_bytes, address=addr,
                region_type=rtype, algorithm=algo_name,
                library='PyCryptodome', description='internal copy'
            ))

    samples.append(CryptoSample(
        label='IV', data=iv_bytes, address=iv_addr,
        region_type=get_region_type_at(handle, iv_addr),
        algorithm=algo_name, library='PyCryptodome',
    ))
    samples.append(CryptoSample(
        label='PLAINTEXT', data=plaintext, address=pt_addr,
        region_type=get_region_type_at(handle, pt_addr),
        algorithm=algo_name, library='PyCryptodome',
    ))
    samples.append(CryptoSample(
        label='CIPHERTEXT', data=ciphertext, address=ct_addr,
        region_type=get_region_type_at(handle, ct_addr),
        algorithm=algo_name, library='PyCryptodome',
    ))

    _keep_alive_list().extend([cipher, ciphertext])
    return samples


def collect_pycryptodome_aes_gcm() -> List[CryptoSample]:
    """PyCryptodome AES-256-GCM 수집."""
    try:
        from Crypto.Cipher import AES
    except ImportError:
        return []

    samples = []
    handle = get_own_process_handle()

    key_bytes = os.urandom(32)
    nonce = os.urandom(12)     # GCM 표준 12바이트
    plaintext = os.urandom(48)

    key_addr, _ = _pin_bytes(key_bytes)
    nonce_addr, _ = _pin_bytes(nonce)
    pt_addr, _ = _pin_bytes(plaintext)

    cipher = AES.new(key_bytes, AES.MODE_GCM, nonce=nonce)
    ciphertext, tag = cipher.encrypt_and_digest(plaintext)
    ct_addr, _ = _pin_bytes(ciphertext)
    tag_addr, _ = _pin_bytes(tag)

    samples.append(CryptoSample(
        label='KEY', data=key_bytes, address=key_addr,
        region_type=get_region_type_at(handle, key_addr),
        algorithm='AES-256-GCM', library='PyCryptodome',
    ))
    samples.append(CryptoSample(
        label='IV', data=nonce, address=nonce_addr,
        region_type=get_region_type_at(handle, nonce_addr),
        algorithm='AES-256-GCM', library='PyCryptodome',
        description='GCM nonce 12B'
    ))
    samples.append(CryptoSample(
        label='PLAINTEXT', data=plaintext, address=pt_addr,
        region_type=get_region_type_at(handle, pt_addr),
        algorithm='AES-256-GCM', library='PyCryptodome',
    ))
    samples.append(CryptoSample(
        label='CIPHERTEXT', data=ciphertext, address=ct_addr,
        region_type=get_region_type_at(handle, ct_addr),
        algorithm='AES-256-GCM', library='PyCryptodome',
    ))
    samples.append(CryptoSample(
        label='CIPHERTEXT', data=tag, address=tag_addr,
        region_type=get_region_type_at(handle, tag_addr),
        algorithm='AES-256-GCM', library='PyCryptodome',
        description='GCM auth tag 16B'
    ))

    _keep_alive_list().extend([cipher, ciphertext, tag])
    return samples


def collect_pycryptodome_des3() -> List[CryptoSample]:
    """PyCryptodome 3DES-CBC 수집."""
    try:
        from Crypto.Cipher import DES3
    except ImportError:
        return []

    samples = []
    handle = get_own_process_handle()

    key_bytes = DES3.adjust_key_parity(os.urandom(24))
    iv_bytes = os.urandom(8)
    plaintext = os.urandom(32)  # 8바이트 배수

    key_addr, _ = _pin_bytes(key_bytes)
    iv_addr, _ = _pin_bytes(iv_bytes)
    pt_addr, _ = _pin_bytes(plaintext)

    cipher = DES3.new(key_bytes, DES3.MODE_CBC, iv=iv_bytes)
    ciphertext = cipher.encrypt(plaintext)
    ct_addr, _ = _pin_bytes(ciphertext)

    samples.append(CryptoSample(
        label='KEY', data=key_bytes, address=key_addr,
        region_type=get_region_type_at(handle, key_addr),
        algorithm='3DES-CBC', library='PyCryptodome',
    ))
    samples.append(CryptoSample(
        label='IV', data=iv_bytes, address=iv_addr,
        region_type=get_region_type_at(handle, iv_addr),
        algorithm='3DES-CBC', library='PyCryptodome',
        description='DES IV 8B'
    ))
    samples.append(CryptoSample(
        label='PLAINTEXT', data=plaintext, address=pt_addr,
        region_type=get_region_type_at(handle, pt_addr),
        algorithm='3DES-CBC', library='PyCryptodome',
    ))
    samples.append(CryptoSample(
        label='CIPHERTEXT', data=ciphertext, address=ct_addr,
        region_type=get_region_type_at(handle, ct_addr),
        algorithm='3DES-CBC', library='PyCryptodome',
    ))

    _keep_alive_list().extend([cipher, ciphertext])
    return samples


def collect_pycryptodome_rsa(key_size: int = 2048) -> List[CryptoSample]:
    """PyCryptodome RSA 키 수집."""
    try:
        from Crypto.PublicKey import RSA
    except ImportError:
        return []

    samples = []
    handle = get_own_process_handle()

    key = RSA.generate(key_size)
    priv_der = key.export_key('DER')
    pub_der = key.publickey().export_key('DER')

    priv_addr, _ = _pin_bytes(priv_der)
    pub_addr, _ = _pin_bytes(pub_der)

    samples.append(CryptoSample(
        label='KEY', data=priv_der, address=priv_addr,
        region_type=get_region_type_at(handle, priv_addr),
        algorithm=f'RSA-{key_size}', library='PyCryptodome',
        description=f'private DER {len(priv_der)}B'
    ))
    samples.append(CryptoSample(
        label='KEY', data=pub_der, address=pub_addr,
        region_type=get_region_type_at(handle, pub_addr),
        algorithm=f'RSA-{key_size}', library='PyCryptodome',
        description=f'public DER {len(pub_der)}B'
    ))

    _keep_alive_list().extend([key, priv_der, pub_der])
    return samples


def collect_pycryptodome_chacha20() -> List[CryptoSample]:
    """PyCryptodome ChaCha20 수집."""
    try:
        from Crypto.Cipher import ChaCha20
    except ImportError:
        return []

    samples = []
    handle = get_own_process_handle()

    key_bytes = os.urandom(32)
    nonce = os.urandom(8)
    plaintext = os.urandom(48)

    key_addr, _ = _pin_bytes(key_bytes)
    nonce_addr, _ = _pin_bytes(nonce)
    pt_addr, _ = _pin_bytes(plaintext)

    cipher = ChaCha20.new(key=key_bytes, nonce=nonce)
    ciphertext = cipher.encrypt(plaintext)
    ct_addr, _ = _pin_bytes(ciphertext)

    samples.append(CryptoSample(
        label='KEY', data=key_bytes, address=key_addr,
        region_type=get_region_type_at(handle, key_addr),
        algorithm='ChaCha20', library='PyCryptodome',
    ))
    samples.append(CryptoSample(
        label='IV', data=nonce, address=nonce_addr,
        region_type=get_region_type_at(handle, nonce_addr),
        algorithm='ChaCha20', library='PyCryptodome',
        description='nonce 8B'
    ))
    samples.append(CryptoSample(
        label='PLAINTEXT', data=plaintext, address=pt_addr,
        region_type=get_region_type_at(handle, pt_addr),
        algorithm='ChaCha20', library='PyCryptodome',
    ))
    samples.append(CryptoSample(
        label='CIPHERTEXT', data=ciphertext, address=ct_addr,
        region_type=get_region_type_at(handle, ct_addr),
        algorithm='ChaCha20', library='PyCryptodome',
    ))

    _keep_alive_list().extend([cipher, ciphertext])
    return samples


def collect_pycryptodome_salsa20() -> List[CryptoSample]:
    """PyCryptodome Salsa20 수집."""
    try:
        from Crypto.Cipher import Salsa20
    except ImportError:
        return []

    samples = []
    handle = get_own_process_handle()

    key_bytes = os.urandom(32)
    nonce = os.urandom(8)
    plaintext = os.urandom(48)

    key_addr, _ = _pin_bytes(key_bytes)
    nonce_addr, _ = _pin_bytes(nonce)
    pt_addr, _ = _pin_bytes(plaintext)

    cipher = Salsa20.new(key=key_bytes, nonce=nonce)
    ciphertext = cipher.encrypt(plaintext)
    ct_addr, _ = _pin_bytes(ciphertext)

    samples.append(CryptoSample(
        label='KEY', data=key_bytes, address=key_addr,
        region_type=get_region_type_at(handle, key_addr),
        algorithm='Salsa20', library='PyCryptodome',
    ))
    samples.append(CryptoSample(
        label='IV', data=nonce, address=nonce_addr,
        region_type=get_region_type_at(handle, nonce_addr),
        algorithm='Salsa20', library='PyCryptodome',
        description='nonce 8B'
    ))
    samples.append(CryptoSample(
        label='PLAINTEXT', data=plaintext, address=pt_addr,
        region_type=get_region_type_at(handle, pt_addr),
        algorithm='Salsa20', library='PyCryptodome',
    ))
    samples.append(CryptoSample(
        label='CIPHERTEXT', data=ciphertext, address=ct_addr,
        region_type=get_region_type_at(handle, ct_addr),
        algorithm='Salsa20', library='PyCryptodome',
    ))

    _keep_alive_list().extend([cipher, ciphertext])
    return samples


def collect_pycryptodome_aes_ctr() -> List[CryptoSample]:
    """PyCryptodome AES-256-CTR 수집."""
    try:
        from Crypto.Cipher import AES
    except ImportError:
        return []

    samples = []
    handle = get_own_process_handle()

    key_bytes = os.urandom(32)
    nonce = os.urandom(8)
    plaintext = os.urandom(64)

    key_addr, _ = _pin_bytes(key_bytes)
    nonce_addr, _ = _pin_bytes(nonce)
    pt_addr, _ = _pin_bytes(plaintext)

    cipher = AES.new(key_bytes, AES.MODE_CTR, nonce=nonce)
    ciphertext = cipher.encrypt(plaintext)
    ct_addr, _ = _pin_bytes(ciphertext)

    samples.append(CryptoSample(
        label='KEY', data=key_bytes, address=key_addr,
        region_type=get_region_type_at(handle, key_addr),
        algorithm='AES-256-CTR', library='PyCryptodome',
    ))
    samples.append(CryptoSample(
        label='IV', data=nonce, address=nonce_addr,
        region_type=get_region_type_at(handle, nonce_addr),
        algorithm='AES-256-CTR', library='PyCryptodome',
        description='CTR nonce 8B'
    ))
    samples.append(CryptoSample(
        label='PLAINTEXT', data=plaintext, address=pt_addr,
        region_type=get_region_type_at(handle, pt_addr),
        algorithm='AES-256-CTR', library='PyCryptodome',
    ))
    samples.append(CryptoSample(
        label='CIPHERTEXT', data=ciphertext, address=ct_addr,
        region_type=get_region_type_at(handle, ct_addr),
        algorithm='AES-256-CTR', library='PyCryptodome',
    ))

    _keep_alive_list().extend([cipher, ciphertext])
    return samples


# ═══════════════════════════════════════════════════════════
# 3. Windows CNG (bcrypt.dll via ctypes)
# ═══════════════════════════════════════════════════════════

def collect_cng_aes(key_size: int = 32) -> List[CryptoSample]:
    """Windows CNG BCrypt AES 키 수집."""
    try:
        bcrypt = ctypes.WinDLL('bcrypt')
    except OSError:
        print("  [건너뜀] bcrypt.dll 로드 실패")
        return []

    samples = []
    handle = get_own_process_handle()
    regions = enumerate_regions(handle)
    algo_name = f"AES-{key_size*8}-CBC"

    # BCryptOpenAlgorithmProvider
    hAlg = wt.HANDLE()
    status = bcrypt.BCryptOpenAlgorithmProvider(
        ctypes.byref(hAlg),
        'AES',      # BCRYPT_AES_ALGORITHM
        None, 0
    )
    if status != 0:
        print(f"  [CNG 오류] BCryptOpenAlgorithmProvider: 0x{status:08X}")
        return []

    # 체이닝 모드 설정 (CBC)
    mode = 'ChainingModeCBC'
    mode_buf = ctypes.create_unicode_buffer(mode)
    bcrypt.BCryptSetProperty(
        hAlg,
        'ChainingMode',
        ctypes.byref(mode_buf),
        len(mode) * 2 + 2,
        0
    )

    # 키 생성
    key_bytes = os.urandom(key_size)
    key_addr, key_buf = _pin_bytes(key_bytes)

    hKey = wt.HANDLE()
    key_obj_len = wt.DWORD()
    cb_result = wt.DWORD()
    bcrypt.BCryptGetProperty(
        hAlg, 'ObjectLength',
        ctypes.byref(key_obj_len), ctypes.sizeof(key_obj_len),
        ctypes.byref(cb_result), 0
    )

    key_obj = (ctypes.c_ubyte * key_obj_len.value)()
    status = bcrypt.BCryptGenerateSymmetricKey(
        hAlg, ctypes.byref(hKey),
        ctypes.byref(key_obj), key_obj_len.value,
        key_buf, key_size, 0
    )
    if status != 0:
        print(f"  [CNG 오류] BCryptGenerateSymmetricKey: 0x{status:08X}")
        bcrypt.BCryptCloseAlgorithmProvider(hAlg, 0)
        return []

    # ── KEY 표본 ──
    samples.append(CryptoSample(
        label='KEY', data=key_bytes, address=key_addr,
        region_type=get_region_type_at(handle, key_addr),
        algorithm=algo_name, library='Windows CNG',
        description=f'pinned key {key_size}B'
    ))
    # CNG 내부 키 사본 검색
    for addr, rtype in _find_in_memory(key_bytes, handle, regions):
        if addr != key_addr:
            samples.append(CryptoSample(
                label='KEY', data=key_bytes, address=addr,
                region_type=rtype, algorithm=algo_name,
                library='Windows CNG', description='CNG internal copy'
            ))

    # 암호화
    iv_bytes = os.urandom(16)
    iv_addr, iv_buf = _pin_bytes(iv_bytes)
    plaintext = os.urandom(64)
    pt_addr, pt_buf = _pin_bytes(plaintext)

    # IV를 별도 버퍼로 복사 (CNG가 IV를 수정함)
    iv_copy = (ctypes.c_ubyte * 16)(*iv_bytes)

    ct_len = wt.DWORD()
    bcrypt.BCryptEncrypt(
        hKey, pt_buf, len(plaintext),
        None, iv_copy, 16,
        None, 0, ctypes.byref(ct_len), 0
    )
    ct_buf = (ctypes.c_ubyte * ct_len.value)()

    # IV 재설정
    iv_copy2 = (ctypes.c_ubyte * 16)(*iv_bytes)
    status = bcrypt.BCryptEncrypt(
        hKey, pt_buf, len(plaintext),
        None, iv_copy2, 16,
        ct_buf, ct_len.value, ctypes.byref(ct_len), 0
    )
    if status == 0:
        ciphertext = bytes(ct_buf[:ct_len.value])
        ct_addr_pin, _ = _pin_bytes(ciphertext)

        samples.append(CryptoSample(
            label='IV', data=iv_bytes, address=iv_addr,
            region_type=get_region_type_at(handle, iv_addr),
            algorithm=algo_name, library='Windows CNG',
        ))
        samples.append(CryptoSample(
            label='PLAINTEXT', data=plaintext, address=pt_addr,
            region_type=get_region_type_at(handle, pt_addr),
            algorithm=algo_name, library='Windows CNG',
        ))
        samples.append(CryptoSample(
            label='CIPHERTEXT', data=ciphertext, address=ct_addr_pin,
            region_type=get_region_type_at(handle, ct_addr_pin),
            algorithm=algo_name, library='Windows CNG',
        ))

    # 정리
    bcrypt.BCryptDestroyKey(hKey)
    bcrypt.BCryptCloseAlgorithmProvider(hAlg, 0)

    _keep_alive_list().extend([key_obj, iv_copy, ct_buf])
    return samples


# ═══════════════════════════════════════════════════════════
# 4. PyNaCl (libsodium 백엔드)
# ═══════════════════════════════════════════════════════════

def collect_pynacl_secretbox() -> List[CryptoSample]:
    """PyNaCl SecretBox (XSalsa20-Poly1305) 수집."""
    try:
        from nacl.secret import SecretBox
        from nacl.utils import random as nacl_random
    except ImportError:
        print("  [건너뜀] pynacl 미설치")
        return []

    samples = []
    handle = get_own_process_handle()

    key_bytes = nacl_random(SecretBox.KEY_SIZE)  # 32 bytes
    key_addr, _ = _pin_bytes(key_bytes)

    box = SecretBox(key_bytes)
    plaintext = os.urandom(48)
    pt_addr, _ = _pin_bytes(plaintext)

    encrypted = box.encrypt(plaintext)
    nonce = encrypted.nonce       # 24 bytes (XSalsa20)
    ciphertext = encrypted.ciphertext
    nonce_addr, _ = _pin_bytes(nonce)
    ct_addr, _ = _pin_bytes(ciphertext)

    samples.append(CryptoSample(
        label='KEY', data=key_bytes, address=key_addr,
        region_type=get_region_type_at(handle, key_addr),
        algorithm='XSalsa20-Poly1305', library='PyNaCl',
    ))
    samples.append(CryptoSample(
        label='IV', data=nonce, address=nonce_addr,
        region_type=get_region_type_at(handle, nonce_addr),
        algorithm='XSalsa20-Poly1305', library='PyNaCl',
        description='XSalsa20 nonce 24B'
    ))
    samples.append(CryptoSample(
        label='PLAINTEXT', data=plaintext, address=pt_addr,
        region_type=get_region_type_at(handle, pt_addr),
        algorithm='XSalsa20-Poly1305', library='PyNaCl',
    ))
    samples.append(CryptoSample(
        label='CIPHERTEXT', data=ciphertext, address=ct_addr,
        region_type=get_region_type_at(handle, ct_addr),
        algorithm='XSalsa20-Poly1305', library='PyNaCl',
    ))

    _keep_alive_list().extend([box, encrypted])
    return samples


def collect_pynacl_sealedbox() -> List[CryptoSample]:
    """PyNaCl SealedBox (Curve25519 공개키 암호화) 수집."""
    try:
        from nacl.public import PrivateKey, SealedBox
    except ImportError:
        return []

    samples = []
    handle = get_own_process_handle()

    private_key = PrivateKey.generate()
    priv_bytes = bytes(private_key)         # 32 bytes
    pub_bytes = bytes(private_key.public_key)  # 32 bytes

    priv_addr, _ = _pin_bytes(priv_bytes)
    pub_addr, _ = _pin_bytes(pub_bytes)

    sealed_box = SealedBox(private_key.public_key)
    plaintext = os.urandom(48)
    pt_addr, _ = _pin_bytes(plaintext)

    ciphertext = bytes(sealed_box.encrypt(plaintext))
    ct_addr, _ = _pin_bytes(ciphertext)

    samples.append(CryptoSample(
        label='KEY', data=priv_bytes, address=priv_addr,
        region_type=get_region_type_at(handle, priv_addr),
        algorithm='Curve25519-XSalsa20', library='PyNaCl',
        description='Curve25519 private key 32B'
    ))
    samples.append(CryptoSample(
        label='KEY', data=pub_bytes, address=pub_addr,
        region_type=get_region_type_at(handle, pub_addr),
        algorithm='Curve25519-XSalsa20', library='PyNaCl',
        description='Curve25519 public key 32B'
    ))
    samples.append(CryptoSample(
        label='PLAINTEXT', data=plaintext, address=pt_addr,
        region_type=get_region_type_at(handle, pt_addr),
        algorithm='Curve25519-XSalsa20', library='PyNaCl',
    ))
    samples.append(CryptoSample(
        label='CIPHERTEXT', data=ciphertext, address=ct_addr,
        region_type=get_region_type_at(handle, ct_addr),
        algorithm='Curve25519-XSalsa20', library='PyNaCl',
    ))

    _keep_alive_list().extend([private_key, sealed_box, ciphertext])
    return samples


def collect_pynacl_ed25519() -> List[CryptoSample]:
    """PyNaCl Ed25519 서명키 수집."""
    try:
        from nacl.signing import SigningKey
    except ImportError:
        return []

    samples = []
    handle = get_own_process_handle()

    signing_key = SigningKey.generate()
    priv_bytes = bytes(signing_key)            # 32 bytes (seed)
    pub_bytes = bytes(signing_key.verify_key)  # 32 bytes

    priv_addr, _ = _pin_bytes(priv_bytes)
    pub_addr, _ = _pin_bytes(pub_bytes)

    message = os.urandom(64)
    msg_addr, _ = _pin_bytes(message)

    signed = signing_key.sign(message)
    signature = signed.signature  # 64 bytes
    sig_addr, _ = _pin_bytes(signature)

    samples.append(CryptoSample(
        label='KEY', data=priv_bytes, address=priv_addr,
        region_type=get_region_type_at(handle, priv_addr),
        algorithm='Ed25519', library='PyNaCl',
        description='signing key seed 32B'
    ))
    samples.append(CryptoSample(
        label='KEY', data=pub_bytes, address=pub_addr,
        region_type=get_region_type_at(handle, pub_addr),
        algorithm='Ed25519', library='PyNaCl',
        description='verify key 32B'
    ))
    samples.append(CryptoSample(
        label='PLAINTEXT', data=message, address=msg_addr,
        region_type=get_region_type_at(handle, msg_addr),
        algorithm='Ed25519', library='PyNaCl',
    ))
    samples.append(CryptoSample(
        label='CIPHERTEXT', data=signature, address=sig_addr,
        region_type=get_region_type_at(handle, sig_addr),
        algorithm='Ed25519', library='PyNaCl',
        description='Ed25519 signature 64B'
    ))

    _keep_alive_list().extend([signing_key, signed])
    return samples


def collect_pynacl_box() -> List[CryptoSample]:
    """PyNaCl Box (Curve25519 키 교환 + XSalsa20-Poly1305) 수집."""
    try:
        from nacl.public import PrivateKey, Box
    except ImportError:
        return []

    samples = []
    handle = get_own_process_handle()

    alice_key = PrivateKey.generate()
    bob_key = PrivateKey.generate()

    alice_priv = bytes(alice_key)
    bob_pub = bytes(bob_key.public_key)

    alice_addr, _ = _pin_bytes(alice_priv)
    bob_pub_addr, _ = _pin_bytes(bob_pub)

    box = Box(alice_key, bob_key.public_key)
    plaintext = os.urandom(48)
    pt_addr, _ = _pin_bytes(plaintext)

    encrypted = box.encrypt(plaintext)
    nonce = encrypted.nonce        # 24 bytes
    ciphertext = encrypted.ciphertext

    nonce_addr, _ = _pin_bytes(nonce)
    ct_addr, _ = _pin_bytes(ciphertext)

    samples.append(CryptoSample(
        label='KEY', data=alice_priv, address=alice_addr,
        region_type=get_region_type_at(handle, alice_addr),
        algorithm='Curve25519-Box', library='PyNaCl',
        description='Curve25519 private key 32B'
    ))
    samples.append(CryptoSample(
        label='KEY', data=bob_pub, address=bob_pub_addr,
        region_type=get_region_type_at(handle, bob_pub_addr),
        algorithm='Curve25519-Box', library='PyNaCl',
        description='peer public key 32B'
    ))
    samples.append(CryptoSample(
        label='IV', data=nonce, address=nonce_addr,
        region_type=get_region_type_at(handle, nonce_addr),
        algorithm='Curve25519-Box', library='PyNaCl',
        description='nonce 24B'
    ))
    samples.append(CryptoSample(
        label='PLAINTEXT', data=plaintext, address=pt_addr,
        region_type=get_region_type_at(handle, pt_addr),
        algorithm='Curve25519-Box', library='PyNaCl',
    ))
    samples.append(CryptoSample(
        label='CIPHERTEXT', data=ciphertext, address=ct_addr,
        region_type=get_region_type_at(handle, ct_addr),
        algorithm='Curve25519-Box', library='PyNaCl',
    ))

    _keep_alive_list().extend([alice_key, bob_key, box, encrypted])
    return samples


def collect_pynacl_generichash() -> List[CryptoSample]:
    """PyNaCl BLAKE2b 키 해시 수집."""
    try:
        from nacl.hash import blake2b
        from nacl.encoding import RawEncoder
    except ImportError:
        return []

    samples = []
    handle = get_own_process_handle()

    key_bytes = os.urandom(32)
    message = os.urandom(64)

    key_addr, _ = _pin_bytes(key_bytes)
    msg_addr, _ = _pin_bytes(message)

    digest = blake2b(message, key=key_bytes, encoder=RawEncoder)
    dig_addr, _ = _pin_bytes(digest)

    samples.append(CryptoSample(
        label='KEY', data=key_bytes, address=key_addr,
        region_type=get_region_type_at(handle, key_addr),
        algorithm='BLAKE2b', library='PyNaCl',
    ))
    samples.append(CryptoSample(
        label='PLAINTEXT', data=message, address=msg_addr,
        region_type=get_region_type_at(handle, msg_addr),
        algorithm='BLAKE2b', library='PyNaCl',
    ))
    samples.append(CryptoSample(
        label='CIPHERTEXT', data=digest, address=dig_addr,
        region_type=get_region_type_at(handle, dig_addr),
        algorithm='BLAKE2b', library='PyNaCl',
        description='BLAKE2b keyed hash 32B'
    ))

    _keep_alive_list().extend([digest])
    return samples


# ═══════════════════════════════════════════════════════════
# 5. pyaes (순수 Python AES 구현)
# ═══════════════════════════════════════════════════════════

def collect_pyaes_ctr(key_size: int = 32) -> List[CryptoSample]:
    """pyaes AES-CTR 수집 (순수 Python 구현)."""
    try:
        import pyaes
    except ImportError:
        print("  [건너뜀] pyaes 미설치")
        return []

    samples = []
    handle = get_own_process_handle()
    algo_name = f"AES-{key_size*8}-CTR"

    key_bytes = os.urandom(key_size)
    iv_bytes = os.urandom(16)
    plaintext = os.urandom(64)

    key_addr, _ = _pin_bytes(key_bytes)
    iv_addr, _ = _pin_bytes(iv_bytes)
    pt_addr, _ = _pin_bytes(plaintext)

    counter = pyaes.Counter(initial_value=int.from_bytes(iv_bytes, 'big'))
    aes = pyaes.AESModeOfOperationCTR(key_bytes, counter=counter)
    ciphertext = aes.encrypt(plaintext)
    ct_addr, _ = _pin_bytes(ciphertext)

    samples.append(CryptoSample(
        label='KEY', data=key_bytes, address=key_addr,
        region_type=get_region_type_at(handle, key_addr),
        algorithm=algo_name, library='pyaes',
    ))
    samples.append(CryptoSample(
        label='IV', data=iv_bytes, address=iv_addr,
        region_type=get_region_type_at(handle, iv_addr),
        algorithm=algo_name, library='pyaes',
        description='CTR initial value 16B'
    ))
    samples.append(CryptoSample(
        label='PLAINTEXT', data=plaintext, address=pt_addr,
        region_type=get_region_type_at(handle, pt_addr),
        algorithm=algo_name, library='pyaes',
    ))
    samples.append(CryptoSample(
        label='CIPHERTEXT', data=ciphertext, address=ct_addr,
        region_type=get_region_type_at(handle, ct_addr),
        algorithm=algo_name, library='pyaes',
    ))

    _keep_alive_list().extend([aes, ciphertext])
    return samples


def collect_pyaes_cbc(key_size: int = 32) -> List[CryptoSample]:
    """pyaes AES-CBC 수집 (순수 Python 구현)."""
    try:
        import pyaes
    except ImportError:
        return []

    samples = []
    handle = get_own_process_handle()
    algo_name = f"AES-{key_size*8}-CBC"

    key_bytes = os.urandom(key_size)
    iv_bytes = os.urandom(16)
    plaintext = os.urandom(64)  # 16바이트 배수

    key_addr, _ = _pin_bytes(key_bytes)
    iv_addr, _ = _pin_bytes(iv_bytes)
    pt_addr, _ = _pin_bytes(plaintext)

    aes = pyaes.AESModeOfOperationCBC(key_bytes, iv=iv_bytes)
    ciphertext = b''
    for i in range(0, len(plaintext), 16):
        ciphertext += aes.encrypt(plaintext[i:i+16])
    ct_addr, _ = _pin_bytes(ciphertext)

    samples.append(CryptoSample(
        label='KEY', data=key_bytes, address=key_addr,
        region_type=get_region_type_at(handle, key_addr),
        algorithm=algo_name, library='pyaes',
    ))
    samples.append(CryptoSample(
        label='IV', data=iv_bytes, address=iv_addr,
        region_type=get_region_type_at(handle, iv_addr),
        algorithm=algo_name, library='pyaes',
    ))
    samples.append(CryptoSample(
        label='PLAINTEXT', data=plaintext, address=pt_addr,
        region_type=get_region_type_at(handle, pt_addr),
        algorithm=algo_name, library='pyaes',
    ))
    samples.append(CryptoSample(
        label='CIPHERTEXT', data=ciphertext, address=ct_addr,
        region_type=get_region_type_at(handle, ct_addr),
        algorithm=algo_name, library='pyaes',
    ))

    _keep_alive_list().extend([aes, ciphertext])
    return samples


# ═══════════════════════════════════════════════════════════
# 6. NON_CRYPTO 표본 수집
# ═══════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════
# 5b. mbedTLS  (UNSEEN library: NOT in the training corpus)
#     Requires: pip install python-mbedtls
#     Used to test out-of-distribution field detection. The structure-aware
#     verifier (csp_verify.py) is library-agnostic, so mbedTLS RSA/EC private
#     keys (exported as DER) are still detected -- demonstrating generalization
#     beyond the trained libraries.
# ═══════════════════════════════════════════════════════════

def collect_mbedtls_aes(key_size: int = 32) -> List[CryptoSample]:
    """mbedTLS AES-CBC key/IV (raw symmetric key; schedule layout is library-specific)."""
    try:
        from mbedtls import cipher
    except ImportError:
        print("  [건너뜀] python-mbedtls 미설치 (pip install python-mbedtls)")
        return []
    samples = []
    handle = get_own_process_handle()
    algo = f"AES-{key_size*8}-CBC"
    key = os.urandom(key_size)
    iv = os.urandom(16)
    pt = os.urandom(64)
    key_addr, _ = _pin_bytes(key)
    iv_addr, _ = _pin_bytes(iv)
    pt_addr, _ = _pin_bytes(pt)
    ct = b""
    try:
        try:
            mode = cipher.Mode.CBC          # newer python-mbedtls
        except Exception:
            mode = cipher.MODE_CBC          # older API
        c = cipher.AES.new(key, mode, iv)
        ct = c.encrypt(pt)
    except Exception as e:
        print(f"  [mbedTLS AES 오류] {e}")
    samples.append(CryptoSample(label='KEY', data=key, address=key_addr,
                                region_type=get_region_type_at(handle, key_addr),
                                algorithm=algo, library='mbedTLS'))
    samples.append(CryptoSample(label='IV', data=iv, address=iv_addr,
                                region_type=get_region_type_at(handle, iv_addr),
                                algorithm=algo, library='mbedTLS'))
    samples.append(CryptoSample(label='PLAINTEXT', data=pt, address=pt_addr,
                                region_type=get_region_type_at(handle, pt_addr),
                                algorithm=algo, library='mbedTLS'))
    if ct:
        ct_addr, _ = _pin_bytes(ct)
        samples.append(CryptoSample(label='CIPHERTEXT', data=ct, address=ct_addr,
                                    region_type=get_region_type_at(handle, ct_addr),
                                    algorithm=algo, library='mbedTLS'))
    _keep_alive_list().append((key, iv, pt, ct))
    return samples


def collect_mbedtls_rsa(key_size: int = 2048) -> List[CryptoSample]:
    """mbedTLS RSA private key, exported to DER (structurally verifiable)."""
    try:
        from mbedtls import pk
    except ImportError:
        return []
    handle = get_own_process_handle()
    try:
        rsa = pk.RSA()
        rsa.generate(key_size)
        priv = rsa.export_key(format="DER")
    except Exception as e:
        print(f"  [mbedTLS RSA 오류] {e}")
        return []
    priv_addr, _ = _pin_bytes(priv)
    _keep_alive_list().append((rsa, priv))
    return [CryptoSample(label='KEY', data=priv, address=priv_addr,
                         region_type=get_region_type_at(handle, priv_addr),
                         algorithm=f'RSA-{key_size}', library='mbedTLS',
                         description=f'private key DER {len(priv)}B')]


def collect_mbedtls_ec() -> List[CryptoSample]:
    """mbedTLS EC private key, exported to DER (structurally verifiable)."""
    try:
        from mbedtls import pk
    except ImportError:
        return []
    handle = get_own_process_handle()
    try:
        try:
            ecc = pk.ECC()                          # default curve
        except Exception:
            ecc = pk.ECC(pk.Curve.SECP256R1)
        ecc.generate()
        priv = ecc.export_key(format="DER")
    except Exception as e:
        print(f"  [mbedTLS EC 오류] {e}")
        return []
    priv_addr, _ = _pin_bytes(priv)
    _keep_alive_list().append((ecc, priv))
    return [CryptoSample(label='KEY', data=priv, address=priv_addr,
                         region_type=get_region_type_at(handle, priv_addr),
                         algorithm='ECDSA-P256', library='mbedTLS',
                         description=f'private key DER {len(priv)}B')]


# ═══════════════════════════════════════════════════════════
# 5c. PurePy  (UNSEEN libraries, pure-Python, no C build needed)
#     pip install ecdsa rsa
#     `rsa` and `ecdsa` are independent implementations NOT in the training corpus;
#     they export private keys as DER, which the (library-agnostic) structure-aware
#     verifier detects -- a clean out-of-distribution generalization test.
# ═══════════════════════════════════════════════════════════

def collect_purepy_rsa(key_size: int = 2048) -> List[CryptoSample]:
    """`rsa` package RSA private key as PKCS#1 DER (structurally verifiable)."""
    try:
        import rsa as _rsa
    except ImportError:
        print("  [건너뜀] rsa 미설치 (pip install rsa)")
        return []
    handle = get_own_process_handle()
    try:
        _pub, priv = _rsa.newkeys(key_size)
        der = priv.save_pkcs1(format="DER")
    except Exception as e:
        print(f"  [PurePy RSA 오류] {e}")
        return []
    a, _ = _pin_bytes(der)
    _keep_alive_list().append((priv, der))
    return [CryptoSample(label='KEY', data=der, address=a,
                         region_type=get_region_type_at(handle, a),
                         algorithm=f'RSA-{key_size}', library='PurePy',
                         description=f'private key DER {len(der)}B')]


def collect_purepy_ecdsa() -> List[CryptoSample]:
    """`ecdsa` package EC private key as SEC1 DER (structurally verifiable)."""
    try:
        from ecdsa import SigningKey, NIST256p
    except ImportError:
        print("  [건너뜀] ecdsa 미설치 (pip install ecdsa)")
        return []
    handle = get_own_process_handle()
    try:
        sk = SigningKey.generate(curve=NIST256p)
        der = sk.to_der()
    except Exception as e:
        print(f"  [PurePy ECDSA 오류] {e}")
        return []
    a, _ = _pin_bytes(der)
    _keep_alive_list().append((sk, der))
    return [CryptoSample(label='KEY', data=der, address=a,
                         region_type=get_region_type_at(handle, a),
                         algorithm='ECDSA-P256', library='PurePy',
                         description=f'private key DER {len(der)}B')]


def collect_non_crypto(count: int = 50,
                       crypto_patterns: Optional[List[bytes]] = None
                       ) -> List[CryptoSample]:
    """
    비암호 메모리 영역에서 NON_CRYPTO 표본을 수집한다.

    다양한 엔트로피 분포를 확보하기 위해:
    - 일반 힙 데이터 (저엔트로피)
    - 문자열 데이터 (중간 엔트로피)
    - 반복 패턴 (매우 저엔트로피)
    - 구조체/포인터 (저~중간 엔트로피)
    - 압축 데이터 시뮬레이션 (고엔트로피, 위양성 유발)
    """
    if crypto_patterns is None:
        crypto_patterns = []

    samples = []
    handle = get_own_process_handle()
    regions = enumerate_regions(handle)

    # (a) 프로세스 메모리에서 랜덤 블록 샘플링
    import random
    random.seed(42)

    # 읽기 가능 영역 중 Private(힙) 영역 선택
    heap_regions = [r for r in regions if r.region_type == 2 and r.size >= 256]
    other_regions = [r for r in regions if r.region_type in (1, 3) and r.size >= 256]
    pool = heap_regions + other_regions

    collected = 0
    block_sizes = [8, 16, 24, 32, 48, 64, 128, 256]
    attempts = 0
    max_attempts = count * 20

    while collected < count and attempts < max_attempts:
        attempts += 1
        if not pool:
            break

        region = random.choice(pool)
        size = random.choice(block_sizes)
        if region.size <= size:
            continue

        offset = random.randint(0, region.size - size)
        # 8바이트 정렬
        offset = (offset // 8) * 8
        addr = region.base + offset

        from win_memory import read_memory
        data = read_memory(handle, addr, size)
        if data is None or len(data) < size:
            continue

        # 전부 0인 블록 제외
        if all(b == 0 for b in data):
            continue

        # 알려진 암호 패턴과 겹치는지 확인
        is_crypto = False
        for pattern in crypto_patterns:
            if pattern in data or data in pattern:
                is_crypto = True
                break
        if is_crypto:
            continue

        samples.append(CryptoSample(
            label='NON_CRYPTO', data=data, address=addr,
            region_type=region.region_type,
            algorithm='N/A', library='memory',
            description=f'random block {size}B'
        ))
        collected += 1

    # (b) 의도적 고엔트로피 비암호 데이터 (위양성 유도용)
    # 압축 데이터 시뮬레이션
    high_ent_count = min(count // 5, 20)
    for i in range(high_ent_count):
        # 높은 엔트로피이지만 완전히 랜덤은 아닌 데이터
        size = random.choice([16, 32, 64])
        data = bytes([((i * 7 + j * 13 + j * j) % 256) for j in range(size)])
        addr, _ = _pin_bytes(data)
        samples.append(CryptoSample(
            label='NON_CRYPTO', data=data, address=addr,
            region_type=get_region_type_at(handle, addr),
            algorithm='N/A', library='synthetic',
            description='high entropy non-crypto'
        ))

    # (c) 문자열 데이터 (중간 엔트로피)
    string_samples = [
        b"The quick brown fox jumps over the lazy dog 1234567890!@#$",
        b"AAAABBBBCCCCDDDDEEEEFFFFGGGGHHHH",
        b"0123456789ABCDEF" * 2,
        b"\x00\x01\x02\x03\x04\x05\x06\x07" * 4,
        b"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n<html>",
    ]
    for s in string_samples:
        size = min(len(s), 64)
        data = s[:size]
        addr, _ = _pin_bytes(data)
        samples.append(CryptoSample(
            label='NON_CRYPTO', data=data, address=addr,
            region_type=get_region_type_at(handle, addr),
            algorithm='N/A', library='synthetic',
            description='string data'
        ))

    return samples


# ═══════════════════════════════════════════════════════════
# 전체 수집 오케스트레이터
# ═══════════════════════════════════════════════════════════

ALL_COLLECTORS = [
    # 1. cryptography (OpenSSL 백엔드) — 10개
    ("OpenSSL AES-128-CBC", lambda: collect_openssl_aes(16)),
    ("OpenSSL AES-192-CBC", lambda: collect_openssl_aes(24)),
    ("OpenSSL AES-256-CBC", lambda: collect_openssl_aes(32)),
    ("OpenSSL AES-256-GCM", collect_openssl_aes_gcm),
    ("OpenSSL AES-256-CTR", collect_openssl_aes_ctr),
    ("OpenSSL ChaCha20-Poly1305", collect_openssl_chacha20),
    ("OpenSSL RSA-2048", lambda: collect_openssl_rsa(2048)),
    ("OpenSSL ECDSA-P256", collect_openssl_ec),
    ("OpenSSL ECDSA-P384", collect_openssl_ecdsa_p384),
    ("OpenSSL HMAC-SHA256", collect_openssl_hmac),
    # 2. PyCryptodome — 9개
    ("PyCryptodome AES-128-CBC", lambda: collect_pycryptodome_aes(16)),
    ("PyCryptodome AES-192-CBC", lambda: collect_pycryptodome_aes(24)),
    ("PyCryptodome AES-256-CBC", lambda: collect_pycryptodome_aes(32)),
    ("PyCryptodome AES-256-GCM", collect_pycryptodome_aes_gcm),
    ("PyCryptodome AES-256-CTR", collect_pycryptodome_aes_ctr),
    ("PyCryptodome 3DES-CBC", collect_pycryptodome_des3),
    ("PyCryptodome ChaCha20", collect_pycryptodome_chacha20),
    ("PyCryptodome Salsa20", collect_pycryptodome_salsa20),
    ("PyCryptodome RSA-2048", lambda: collect_pycryptodome_rsa(2048)),
    # 3. Windows CNG — 3개
    ("Windows CNG AES-128-CBC", lambda: collect_cng_aes(16)),
    ("Windows CNG AES-192-CBC", lambda: collect_cng_aes(24)),
    ("Windows CNG AES-256-CBC", lambda: collect_cng_aes(32)),
    # 4. PyNaCl (libsodium) — 5개
    ("PyNaCl SecretBox", collect_pynacl_secretbox),
    ("PyNaCl SealedBox", collect_pynacl_sealedbox),
    ("PyNaCl Ed25519", collect_pynacl_ed25519),
    ("PyNaCl Box", collect_pynacl_box),
    ("PyNaCl BLAKE2b", collect_pynacl_generichash),
    # 5. pyaes (순수 Python) — 6개
    ("pyaes AES-128-CTR", lambda: collect_pyaes_ctr(16)),
    ("pyaes AES-192-CTR", lambda: collect_pyaes_ctr(24)),
    ("pyaes AES-256-CTR", lambda: collect_pyaes_ctr(32)),
    ("pyaes AES-128-CBC", lambda: collect_pyaes_cbc(16)),
    ("pyaes AES-192-CBC", lambda: collect_pyaes_cbc(24)),
    ("pyaes AES-256-CBC", lambda: collect_pyaes_cbc(32)),
    # 6. mbedTLS — UNSEEN library (pip install python-mbedtls); not in training corpus
    ("mbedTLS AES-128-CBC", lambda: collect_mbedtls_aes(16)),
    ("mbedTLS AES-256-CBC", lambda: collect_mbedtls_aes(32)),
    ("mbedTLS RSA-2048", lambda: collect_mbedtls_rsa(2048)),
    ("mbedTLS ECDSA-P256", collect_mbedtls_ec),
    # 7. PurePy — UNSEEN pure-Python libraries (pip install ecdsa rsa); not in training
    ("PurePy RSA-2048", lambda: collect_purepy_rsa(2048)),
    ("PurePy ECDSA-P256", collect_purepy_ecdsa),
]


def collect_all(repetitions: int = 5,
                non_crypto_count: int = 50) -> List[CryptoSample]:
    """
    모든 암호 라이브러리에서 표본을 수집한다.

    Args:
        repetitions: 각 수집기를 반복 실행할 횟수 (다양한 키 값 생성)
        non_crypto_count: NON_CRYPTO 표본 수
    """
    all_samples = []
    crypto_patterns = []

    for rep in range(repetitions):
        print(f"\n── 반복 {rep+1}/{repetitions} ──")
        for name, collector_fn in ALL_COLLECTORS:
            try:
                samples = collector_fn()
                all_samples.extend(samples)
                for s in samples:
                    if s.label in ('KEY', 'IV'):
                        crypto_patterns.append(s.data)
                if samples:
                    print(f"  {name}: {len(samples)}개 수집")
            except Exception as e:
                print(f"  {name}: 오류 - {e}")

    # NON_CRYPTO 수집
    print(f"\n── NON_CRYPTO 수집 (목표: {non_crypto_count}개) ──")
    nc_samples = collect_non_crypto(non_crypto_count, crypto_patterns)
    all_samples.extend(nc_samples)
    print(f"  NON_CRYPTO: {len(nc_samples)}개 수집")

    return all_samples


if __name__ == '__main__':
    print("암호 표본 수집 테스트")
    print("=" * 50)
    samples = collect_all(repetitions=1, non_crypto_count=10)
    print(f"\n총 수집: {len(samples)}개")

    label_counts = {}
    lib_counts = {}
    for s in samples:
        label_counts[s.label] = label_counts.get(s.label, 0) + 1
        lib_counts[s.library] = lib_counts.get(s.library, 0) + 1

    print("\n클래스별:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}")
    print("\n라이브러리별:")
    for lib, count in sorted(lib_counts.items()):
        print(f"  {lib}: {count}")
