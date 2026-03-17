"""
crypto_ops.py
Per-library generation of key/IV/ciphertext/plaintext and memory address tracking.

Supported libraries:
  1) cryptography (OpenSSL backend)
  2) PyCryptodome
  3) Windows CNG (ctypes)
  4) PyNaCl (libsodium backend)
  5) pyaes (pure Python AES)

Each function performs cryptographic operations using known key values
and searches for the corresponding byte patterns in memory to return ground truth addresses.
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
    """Collected cryptographic data sample."""
    label: str              # KEY / IV / CIPHERTEXT / PLAINTEXT / NON_CRYPTO
    data: bytes             # actual byte data
    address: int            # memory address
    region_type: int        # 0-3 (paper F6)
    algorithm: str          # e.g.: AES-256-CBC
    library: str            # e.g.: OpenSSL
    description: str = ""   # additional description


# ─── Common library helpers ───

def _keep_alive_list():
    """Reference storage list to prevent garbage collection."""
    if not hasattr(_keep_alive_list, '_refs'):
        _keep_alive_list._refs = []
    return _keep_alive_list._refs


def _pin_bytes(data: bytes) -> Tuple[int, ctypes.Array]:
    """
    Copy byte data into a ctypes buffer to pin its address.
    Prevents address relocation due to garbage collection.
    """
    buf = (ctypes.c_ubyte * len(data))(*data)
    _keep_alive_list().append(buf)  # prevent GC
    return ctypes.addressof(buf), buf


def _find_in_memory(pattern: bytes, handle=None, regions=None,
                    max_regions: int = 80,
                    ) -> List[Tuple[int, int]]:
    """Search for a pattern in memory. Returns [(address, region_type), ...]
    For performance, scans only Private (heap) regions first and limits the maximum number of regions."""
    if handle is None:
        handle = get_own_process_handle()
    if regions is None:
        all_regions = enumerate_regions(handle)
        # Select only Private (heap/stack) regions (where keys typically reside)
        regions = [r for r in all_regions if r.region_type == 2]
        if len(regions) > max_regions:
            regions = regions[:max_regions]
    return scan_for_pattern(handle, pattern, regions)


# ═══════════════════════════════════════════════════════════
# 1. cryptography (OpenSSL backend)
# ═══════════════════════════════════════════════════════════

def collect_openssl_aes(key_size: int = 32) -> List[CryptoSample]:
    """Collect OpenSSL AES key/IV/ciphertext/plaintext."""
    try:
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    except ImportError:
        print("  [Skipped] cryptography package not installed")
        return []

    samples = []
    handle = get_own_process_handle()
    regions = enumerate_regions(handle)

    algo_name = f"AES-{key_size*8}-CBC"

    # Generate known values
    key_bytes = os.urandom(key_size)
    iv_bytes = os.urandom(16)
    plaintext = os.urandom(64)

    # Pin to ctypes buffer (prevent GC)
    key_addr, key_buf = _pin_bytes(key_bytes)
    iv_addr, iv_buf = _pin_bytes(iv_bytes)
    pt_addr, pt_buf = _pin_bytes(plaintext)

    # Perform encryption
    cipher = Cipher(algorithms.AES(key_bytes), modes.CBC(iv_bytes))
    encryptor = cipher.encryptor()
    ciphertext = encryptor.update(plaintext) + encryptor.finalize()
    ct_addr, ct_buf = _pin_bytes(ciphertext)

    # ── KEY sample ──
    # Pinned buffer address (definite ground truth)
    samples.append(CryptoSample(
        label='KEY', data=key_bytes, address=key_addr,
        region_type=get_region_type_at(handle, key_addr),
        algorithm=algo_name, library='OpenSSL',
        description=f'pinned buffer {key_size}B'
    ))
    # Search for additional copies in memory (library internal copies)
    found = _find_in_memory(key_bytes, handle, regions)
    for addr, rtype in found:
        if addr != key_addr:
            samples.append(CryptoSample(
                label='KEY', data=key_bytes, address=addr,
                region_type=rtype, algorithm=algo_name,
                library='OpenSSL', description='library internal copy'
            ))

    # ── IV sample ──
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

    # ── PLAINTEXT sample ──
    samples.append(CryptoSample(
        label='PLAINTEXT', data=plaintext, address=pt_addr,
        region_type=get_region_type_at(handle, pt_addr),
        algorithm=algo_name, library='OpenSSL',
        description='input buffer 64B'
    ))

    # ── CIPHERTEXT sample ──
    samples.append(CryptoSample(
        label='CIPHERTEXT', data=ciphertext, address=ct_addr,
        region_type=get_region_type_at(handle, ct_addr),
        algorithm=algo_name, library='OpenSSL',
        description='output buffer'
    ))

    # Retain references (prevent GC)
    _keep_alive_list().extend([cipher, encryptor, ciphertext])

    return samples


def collect_openssl_chacha20() -> List[CryptoSample]:
    """Collect OpenSSL ChaCha20-Poly1305 key/IV."""
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
    """Collect OpenSSL RSA keys."""
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

    # Serialize to DER format to extract bytes
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

    # Search for library internal key copies (partial matching for large keys)
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
    """Collect OpenSSL ECDSA P-256 keys."""
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
    # Uncompressed public key (65 bytes)
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
    """Collect OpenSSL HMAC-SHA256 key."""
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
    """Collect OpenSSL AES-256-GCM samples."""
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
    """Collect OpenSSL ECDSA P-384 keys."""
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
    """Collect OpenSSL AES-256-CTR samples."""
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
    """Collect PyCryptodome AES-CBC key/IV."""
    try:
        from Crypto.Cipher import AES
    except ImportError:
        print("  [Skipped] pycryptodome not installed")
        return []

    samples = []
    handle = get_own_process_handle()
    regions = enumerate_regions(handle)
    algo_name = f"AES-{key_size*8}-CBC"

    key_bytes = os.urandom(key_size)
    iv_bytes = os.urandom(16)
    # Plaintext must be a multiple of 16 bytes
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
    # Library internal key copies
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
    """Collect PyCryptodome AES-256-GCM samples."""
    try:
        from Crypto.Cipher import AES
    except ImportError:
        return []

    samples = []
    handle = get_own_process_handle()

    key_bytes = os.urandom(32)
    nonce = os.urandom(12)     # GCM standard 12 bytes
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
    """Collect PyCryptodome 3DES-CBC samples."""
    try:
        from Crypto.Cipher import DES3
    except ImportError:
        return []

    samples = []
    handle = get_own_process_handle()

    key_bytes = DES3.adjust_key_parity(os.urandom(24))
    iv_bytes = os.urandom(8)
    plaintext = os.urandom(32)  # multiple of 8 bytes

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
    """Collect PyCryptodome RSA keys."""
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
    """Collect PyCryptodome ChaCha20 samples."""
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
    """Collect PyCryptodome Salsa20 samples."""
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
    """Collect PyCryptodome AES-256-CTR samples."""
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
    """Collect Windows CNG BCrypt AES keys."""
    try:
        bcrypt = ctypes.WinDLL('bcrypt')
    except OSError:
        print("  [Skipped] Failed to load bcrypt.dll")
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
        print(f"  [CNG Error] BCryptOpenAlgorithmProvider: 0x{status:08X}")
        return []

    # Set chaining mode (CBC)
    mode = 'ChainingModeCBC'
    mode_buf = ctypes.create_unicode_buffer(mode)
    bcrypt.BCryptSetProperty(
        hAlg,
        'ChainingMode',
        ctypes.byref(mode_buf),
        len(mode) * 2 + 2,
        0
    )

    # Generate key
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
        print(f"  [CNG Error] BCryptGenerateSymmetricKey: 0x{status:08X}")
        bcrypt.BCryptCloseAlgorithmProvider(hAlg, 0)
        return []

    # ── KEY sample ──
    samples.append(CryptoSample(
        label='KEY', data=key_bytes, address=key_addr,
        region_type=get_region_type_at(handle, key_addr),
        algorithm=algo_name, library='Windows CNG',
        description=f'pinned key {key_size}B'
    ))
    # Search for CNG internal key copies
    for addr, rtype in _find_in_memory(key_bytes, handle, regions):
        if addr != key_addr:
            samples.append(CryptoSample(
                label='KEY', data=key_bytes, address=addr,
                region_type=rtype, algorithm=algo_name,
                library='Windows CNG', description='CNG internal copy'
            ))

    # Encryption
    iv_bytes = os.urandom(16)
    iv_addr, iv_buf = _pin_bytes(iv_bytes)
    plaintext = os.urandom(64)
    pt_addr, pt_buf = _pin_bytes(plaintext)

    # Copy IV to a separate buffer (CNG modifies the IV in place)
    iv_copy = (ctypes.c_ubyte * 16)(*iv_bytes)

    ct_len = wt.DWORD()
    bcrypt.BCryptEncrypt(
        hKey, pt_buf, len(plaintext),
        None, iv_copy, 16,
        None, 0, ctypes.byref(ct_len), 0
    )
    ct_buf = (ctypes.c_ubyte * ct_len.value)()

    # Reset IV
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

    # Cleanup
    bcrypt.BCryptDestroyKey(hKey)
    bcrypt.BCryptCloseAlgorithmProvider(hAlg, 0)

    _keep_alive_list().extend([key_obj, iv_copy, ct_buf])
    return samples


# ═══════════════════════════════════════════════════════════
# 4. PyNaCl (libsodium backend)
# ═══════════════════════════════════════════════════════════

def collect_pynacl_secretbox() -> List[CryptoSample]:
    """Collect PyNaCl SecretBox (XSalsa20-Poly1305) samples."""
    try:
        from nacl.secret import SecretBox
        from nacl.utils import random as nacl_random
    except ImportError:
        print("  [Skipped] pynacl not installed")
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
    """Collect PyNaCl SealedBox (Curve25519 public key encryption) samples."""
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
    """Collect PyNaCl Ed25519 signing keys."""
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
    """Collect PyNaCl Box (Curve25519 key exchange + XSalsa20-Poly1305) samples."""
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
    """Collect PyNaCl BLAKE2b keyed hash samples."""
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
# 5. pyaes (pure Python AES implementation)
# ═══════════════════════════════════════════════════════════

def collect_pyaes_ctr(key_size: int = 32) -> List[CryptoSample]:
    """Collect pyaes AES-CTR samples (pure Python implementation)."""
    try:
        import pyaes
    except ImportError:
        print("  [Skipped] pyaes not installed")
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
    """Collect pyaes AES-CBC samples (pure Python implementation)."""
    try:
        import pyaes
    except ImportError:
        return []

    samples = []
    handle = get_own_process_handle()
    algo_name = f"AES-{key_size*8}-CBC"

    key_bytes = os.urandom(key_size)
    iv_bytes = os.urandom(16)
    plaintext = os.urandom(64)  # multiple of 16 bytes

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
# 6. NON_CRYPTO sample collection
# ═══════════════════════════════════════════════════════════

def collect_non_crypto(count: int = 50,
                       crypto_patterns: Optional[List[bytes]] = None
                       ) -> List[CryptoSample]:
    """
    Collect NON_CRYPTO samples from non-cryptographic memory regions.

    To ensure diverse entropy distributions:
    - General heap data (low entropy)
    - String data (medium entropy)
    - Repeating patterns (very low entropy)
    - Structs/pointers (low to medium entropy)
    - Simulated compressed data (high entropy, induces false positives)
    """
    if crypto_patterns is None:
        crypto_patterns = []

    samples = []
    handle = get_own_process_handle()
    regions = enumerate_regions(handle)

    # (a) Random block sampling from process memory
    import random
    random.seed(42)

    # Select Private (heap) regions among readable regions
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
        # 8-byte alignment
        offset = (offset // 8) * 8
        addr = region.base + offset

        from win_memory import read_memory
        data = read_memory(handle, addr, size)
        if data is None or len(data) < size:
            continue

        # Exclude all-zero blocks
        if all(b == 0 for b in data):
            continue

        # Check for overlap with known cryptographic patterns
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

    # (b) Intentional high-entropy non-crypto data (to induce false positives)
    # Simulated compressed data
    high_ent_count = min(count // 5, 20)
    for i in range(high_ent_count):
        # High entropy but not completely random data
        size = random.choice([16, 32, 64])
        data = bytes([((i * 7 + j * 13 + j * j) % 256) for j in range(size)])
        addr, _ = _pin_bytes(data)
        samples.append(CryptoSample(
            label='NON_CRYPTO', data=data, address=addr,
            region_type=get_region_type_at(handle, addr),
            algorithm='N/A', library='synthetic',
            description='high entropy non-crypto'
        ))

    # (c) String data (medium entropy)
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
# Overall collection orchestrator
# ═══════════════════════════════════════════════════════════

ALL_COLLECTORS = [
    # 1. cryptography (OpenSSL backend) — 10 collectors
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
    # 2. PyCryptodome — 9 collectors
    ("PyCryptodome AES-128-CBC", lambda: collect_pycryptodome_aes(16)),
    ("PyCryptodome AES-192-CBC", lambda: collect_pycryptodome_aes(24)),
    ("PyCryptodome AES-256-CBC", lambda: collect_pycryptodome_aes(32)),
    ("PyCryptodome AES-256-GCM", collect_pycryptodome_aes_gcm),
    ("PyCryptodome AES-256-CTR", collect_pycryptodome_aes_ctr),
    ("PyCryptodome 3DES-CBC", collect_pycryptodome_des3),
    ("PyCryptodome ChaCha20", collect_pycryptodome_chacha20),
    ("PyCryptodome Salsa20", collect_pycryptodome_salsa20),
    ("PyCryptodome RSA-2048", lambda: collect_pycryptodome_rsa(2048)),
    # 3. Windows CNG — 3 collectors
    ("Windows CNG AES-128-CBC", lambda: collect_cng_aes(16)),
    ("Windows CNG AES-192-CBC", lambda: collect_cng_aes(24)),
    ("Windows CNG AES-256-CBC", lambda: collect_cng_aes(32)),
    # 4. PyNaCl (libsodium) — 5 collectors
    ("PyNaCl SecretBox", collect_pynacl_secretbox),
    ("PyNaCl SealedBox", collect_pynacl_sealedbox),
    ("PyNaCl Ed25519", collect_pynacl_ed25519),
    ("PyNaCl Box", collect_pynacl_box),
    ("PyNaCl BLAKE2b", collect_pynacl_generichash),
    # 5. pyaes (pure Python) — 6 collectors
    ("pyaes AES-128-CTR", lambda: collect_pyaes_ctr(16)),
    ("pyaes AES-192-CTR", lambda: collect_pyaes_ctr(24)),
    ("pyaes AES-256-CTR", lambda: collect_pyaes_ctr(32)),
    ("pyaes AES-128-CBC", lambda: collect_pyaes_cbc(16)),
    ("pyaes AES-192-CBC", lambda: collect_pyaes_cbc(24)),
    ("pyaes AES-256-CBC", lambda: collect_pyaes_cbc(32)),
]


def collect_all(repetitions: int = 5,
                non_crypto_count: int = 50) -> List[CryptoSample]:
    """
    Collect samples from all cryptographic libraries.

    Args:
        repetitions: Number of times to run each collector (to generate diverse key values)
        non_crypto_count: Number of NON_CRYPTO samples
    """
    all_samples = []
    crypto_patterns = []

    for rep in range(repetitions):
        print(f"\n── Repetition {rep+1}/{repetitions} ──")
        for name, collector_fn in ALL_COLLECTORS:
            try:
                samples = collector_fn()
                all_samples.extend(samples)
                for s in samples:
                    if s.label in ('KEY', 'IV'):
                        crypto_patterns.append(s.data)
                if samples:
                    print(f"  {name}: {len(samples)} samples collected")
            except Exception as e:
                print(f"  {name}: Error - {e}")

    # Collect NON_CRYPTO
    print(f"\n── NON_CRYPTO collection (target: {non_crypto_count} samples) ──")
    nc_samples = collect_non_crypto(non_crypto_count, crypto_patterns)
    all_samples.extend(nc_samples)
    print(f"  NON_CRYPTO: {len(nc_samples)} samples collected")

    return all_samples


if __name__ == '__main__':
    print("Cryptographic sample collection test")
    print("=" * 50)
    samples = collect_all(repetitions=1, non_crypto_count=10)
    print(f"\nTotal collected: {len(samples)} samples")

    label_counts = {}
    lib_counts = {}
    for s in samples:
        label_counts[s.label] = label_counts.get(s.label, 0) + 1
        lib_counts[s.library] = lib_counts.get(s.library, 0) + 1

    print("\nBy class:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}")
    print("\nBy library:")
    for lib, count in sorted(lib_counts.items()):
        print(f"  {lib}: {count}")
