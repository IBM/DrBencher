# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""Security/Cyber API utilities for cybersecurity benchmark.

Provides hardcoded reference data for cryptographic algorithms and MITRE ATT&CK
entities, with simple dict lookup functions. No external API calls needed —
all data is pinned from NIST standards and ATT&CK STIX v15.

Also provides Wikipedia entity clue fetching and Wikidata QID resolution
for the chain-based grounding pipeline.
"""

import json
import os
import re
import requests
import time
from typing import Any, Dict, List, Optional

from .tool_util import _kg_cache_get, _kg_cache_set

# Disk cache for security's Wikipedia clue fetches (the only networked lookups;
# crypto/ATT&CK data are in-memory reference tables). Reuses the shared
# SHA-256(key)->JSON, 1-week-TTL cache helpers from tool_util.
SECURITY_CACHE_DIR = "cache/security_cache"


# ---------------------------------------------------------------------------
# Cryptographic Reference Data (NIST / ISO / IETF standards)
# ---------------------------------------------------------------------------

CRYPTO_REFERENCE_DATA: Dict[str, Dict[str, Any]] = {
    # --- Block Ciphers ---
    "AES-128": {"type": "block_cipher", "family": "AES", "key_size": 128, "block_size": 128, "rounds": 10, "security_strength": 128, "year": 2001, "wiki": "Advanced Encryption Standard"},
    "AES-192": {"type": "block_cipher", "family": "AES", "key_size": 192, "block_size": 128, "rounds": 12, "security_strength": 192, "year": 2001, "wiki": "Advanced Encryption Standard"},
    "AES-256": {"type": "block_cipher", "family": "AES", "key_size": 256, "block_size": 128, "rounds": 14, "security_strength": 256, "year": 2001, "wiki": "Advanced Encryption Standard"},
    "DES": {"type": "block_cipher", "family": "DES", "key_size": 56, "block_size": 64, "rounds": 16, "security_strength": 56, "year": 1977, "wiki": "Data Encryption Standard"},
    "3DES": {"type": "block_cipher", "family": "DES", "key_size": 168, "block_size": 64, "rounds": 48, "security_strength": 112, "year": 1998, "wiki": "Triple DES"},
    "Blowfish": {"type": "block_cipher", "family": "Blowfish", "key_size": 448, "block_size": 64, "rounds": 16, "security_strength": 128, "year": 1993, "wiki": "Blowfish (cipher)"},
    "Twofish": {"type": "block_cipher", "family": "Twofish", "key_size": 256, "block_size": 128, "rounds": 16, "security_strength": 256, "year": 1998, "wiki": "Twofish"},
    "Serpent": {"type": "block_cipher", "family": "Serpent", "key_size": 256, "block_size": 128, "rounds": 32, "security_strength": 256, "year": 1998, "wiki": "Serpent (cipher)"},
    "Camellia-256": {"type": "block_cipher", "family": "Camellia", "key_size": 256, "block_size": 128, "rounds": 24, "security_strength": 256, "year": 2000, "wiki": "Camellia (cipher)"},
    "CAST-128": {"type": "block_cipher", "family": "CAST", "key_size": 128, "block_size": 64, "rounds": 16, "security_strength": 128, "year": 1996, "wiki": "CAST-128"},
    "IDEA": {"type": "block_cipher", "family": "IDEA", "key_size": 128, "block_size": 64, "rounds": 8, "security_strength": 128, "year": 1991, "wiki": "International Data Encryption Algorithm"},
    "RC5-64": {"type": "block_cipher", "family": "RC5", "key_size": 128, "block_size": 64, "rounds": 12, "security_strength": 128, "year": 1994, "wiki": "RC5"},
    "ARIA-256": {"type": "block_cipher", "family": "ARIA", "key_size": 256, "block_size": 128, "rounds": 16, "security_strength": 256, "year": 2003, "wiki": "ARIA (cipher)"},
    "SM4": {"type": "block_cipher", "family": "SM4", "key_size": 128, "block_size": 128, "rounds": 32, "security_strength": 128, "year": 2006, "wiki": "SM4 (cipher)"},
    "SEED": {"type": "block_cipher", "family": "SEED", "key_size": 128, "block_size": 128, "rounds": 16, "security_strength": 128, "year": 1998, "wiki": "SEED"},
    # --- Stream Ciphers ---
    "ChaCha20": {"type": "stream_cipher", "family": "ChaCha", "key_size": 256, "nonce_size": 96, "rounds": 20, "state_size": 512, "security_strength": 256, "year": 2008, "wiki": "Salsa20#ChaCha_variant"},
    "Salsa20": {"type": "stream_cipher", "family": "Salsa", "key_size": 256, "nonce_size": 64, "rounds": 20, "state_size": 512, "security_strength": 256, "year": 2005, "wiki": "Salsa20"},
    "RC4": {"type": "stream_cipher", "family": "RC4", "key_size": 256, "nonce_size": 0, "rounds": 1, "state_size": 2064, "security_strength": 0, "year": 1987, "wiki": "RC4"},
    "Rabbit": {"type": "stream_cipher", "family": "Rabbit", "key_size": 128, "nonce_size": 64, "rounds": 4, "state_size": 513, "security_strength": 128, "year": 2003, "wiki": "Rabbit (cipher)"},
    "Trivium": {"type": "stream_cipher", "family": "Trivium", "key_size": 80, "nonce_size": 80, "rounds": 1152, "state_size": 288, "security_strength": 80, "year": 2005, "wiki": "Trivium (cipher)"},
    # --- Hash Functions ---
    "SHA-1": {"type": "hash", "family": "SHA", "output_size": 160, "block_size": 512, "rounds": 80, "security_strength": 80, "year": 1995, "wiki": "SHA-1"},
    "SHA-256": {"type": "hash", "family": "SHA-2", "output_size": 256, "block_size": 512, "rounds": 64, "security_strength": 128, "year": 2001, "wiki": "SHA-2"},
    "SHA-384": {"type": "hash", "family": "SHA-2", "output_size": 384, "block_size": 1024, "rounds": 80, "security_strength": 192, "year": 2001, "wiki": "SHA-2"},
    "SHA-512": {"type": "hash", "family": "SHA-2", "output_size": 512, "block_size": 1024, "rounds": 80, "security_strength": 256, "year": 2001, "wiki": "SHA-2"},
    "SHA3-256": {"type": "hash", "family": "SHA-3", "output_size": 256, "block_size": 1088, "rounds": 24, "security_strength": 128, "year": 2015, "wiki": "SHA-3"},
    "SHA3-512": {"type": "hash", "family": "SHA-3", "output_size": 512, "block_size": 576, "rounds": 24, "security_strength": 256, "year": 2015, "wiki": "SHA-3"},
    "MD5": {"type": "hash", "family": "MD", "output_size": 128, "block_size": 512, "rounds": 64, "security_strength": 0, "year": 1992, "wiki": "MD5"},
    "BLAKE2b": {"type": "hash", "family": "BLAKE", "output_size": 512, "block_size": 1024, "rounds": 12, "security_strength": 256, "year": 2012, "wiki": "BLAKE (hash function)"},
    "BLAKE2s": {"type": "hash", "family": "BLAKE", "output_size": 256, "block_size": 512, "rounds": 10, "security_strength": 128, "year": 2012, "wiki": "BLAKE (hash function)"},
    "RIPEMD-160": {"type": "hash", "family": "RIPEMD", "output_size": 160, "block_size": 512, "rounds": 80, "security_strength": 80, "year": 1996, "wiki": "RIPEMD"},
    "Whirlpool": {"type": "hash", "family": "Whirlpool", "output_size": 512, "block_size": 512, "rounds": 10, "security_strength": 256, "year": 2000, "wiki": "Whirlpool (hash function)"},
    "SM3": {"type": "hash", "family": "SM", "output_size": 256, "block_size": 512, "rounds": 64, "security_strength": 128, "year": 2010, "wiki": "SM3 (hash function)"},
    # --- Public Key ---
    "RSA-2048": {"type": "public_key", "family": "RSA", "key_size": 2048, "security_strength": 112, "year": 1977, "wiki": "RSA (cryptosystem)"},
    "RSA-3072": {"type": "public_key", "family": "RSA", "key_size": 3072, "security_strength": 128, "year": 1977, "wiki": "RSA (cryptosystem)"},
    "RSA-4096": {"type": "public_key", "family": "RSA", "key_size": 4096, "security_strength": 152, "year": 1977, "wiki": "RSA (cryptosystem)"},
    "ECDSA-P256": {"type": "public_key", "family": "ECDSA", "key_size": 256, "security_strength": 128, "year": 1999, "wiki": "Elliptic Curve Digital Signature Algorithm"},
    "ECDSA-P384": {"type": "public_key", "family": "ECDSA", "key_size": 384, "security_strength": 192, "year": 1999, "wiki": "Elliptic Curve Digital Signature Algorithm"},
    "Ed25519": {"type": "public_key", "family": "EdDSA", "key_size": 256, "security_strength": 128, "year": 2011, "wiki": "EdDSA"},
    "Ed448": {"type": "public_key", "family": "EdDSA", "key_size": 448, "security_strength": 224, "year": 2015, "wiki": "EdDSA"},
    "DSA-2048": {"type": "public_key", "family": "DSA", "key_size": 2048, "security_strength": 112, "year": 1991, "wiki": "Digital Signature Algorithm"},
    "ElGamal-2048": {"type": "public_key", "family": "ElGamal", "key_size": 2048, "security_strength": 112, "year": 1985, "wiki": "ElGamal encryption"},
    # --- Key Exchange ---
    "DH-2048": {"type": "key_exchange", "family": "DH", "key_size": 2048, "security_strength": 112, "year": 1976, "wiki": "Diffie\u2013Hellman key exchange"},
    "ECDH-P256": {"type": "key_exchange", "family": "ECDH", "key_size": 256, "security_strength": 128, "year": 1999, "wiki": "Elliptic-curve Diffie\u2013Hellman"},
    "X25519": {"type": "key_exchange", "family": "X25519", "key_size": 256, "security_strength": 128, "year": 2014, "wiki": "Curve25519"},
    "X448": {"type": "key_exchange", "family": "X448", "key_size": 448, "security_strength": 224, "year": 2015, "wiki": "Curve448"},
    "Kyber-512": {"type": "key_exchange", "family": "Kyber", "key_size": 1632, "security_strength": 128, "year": 2022, "wiki": "Kyber"},
    "Kyber-1024": {"type": "key_exchange", "family": "Kyber", "key_size": 3168, "security_strength": 256, "year": 2022, "wiki": "Kyber"},
    # --- AEAD ---
    "AES-GCM-128": {"type": "aead", "family": "AES-GCM", "key_size": 128, "nonce_size": 96, "tag_size": 128, "block_size": 128, "rounds": 10, "security_strength": 128, "year": 2007, "wiki": "Galois/Counter Mode"},
    "AES-GCM-256": {"type": "aead", "family": "AES-GCM", "key_size": 256, "nonce_size": 96, "tag_size": 128, "block_size": 128, "rounds": 14, "security_strength": 256, "year": 2007, "wiki": "Galois/Counter Mode"},
    "ChaCha20-Poly1305": {"type": "aead", "family": "ChaCha-Poly", "key_size": 256, "nonce_size": 96, "tag_size": 128, "rounds": 20, "security_strength": 256, "year": 2014, "wiki": "ChaCha20-Poly1305"},
    "AES-CCM": {"type": "aead", "family": "AES-CCM", "key_size": 128, "nonce_size": 56, "tag_size": 128, "block_size": 128, "rounds": 10, "security_strength": 128, "year": 2004, "wiki": "CCM mode"},
    # --- MAC ---
    "HMAC-SHA256": {"type": "mac", "family": "HMAC", "key_size": 256, "output_size": 256, "block_size": 512, "security_strength": 128, "year": 1996, "wiki": "HMAC"},
    "HMAC-SHA512": {"type": "mac", "family": "HMAC", "key_size": 512, "output_size": 512, "block_size": 1024, "security_strength": 256, "year": 1996, "wiki": "HMAC"},
    "CMAC-AES": {"type": "mac", "family": "CMAC", "key_size": 128, "output_size": 128, "block_size": 128, "security_strength": 64, "year": 2006, "wiki": "One-key MAC"},
    "Poly1305": {"type": "mac", "family": "Poly1305", "key_size": 256, "output_size": 128, "security_strength": 128, "year": 2005, "wiki": "Poly1305"},
    # --- PQC Digital Signatures (NIST FIPS 204 / FIPS 205) ---
    # key_size = signing (secret) key size in bytes; output_size = signature size in bytes
    "ML-DSA-44": {"type": "pqc_signature", "family": "ML-DSA", "key_size": 2560, "output_size": 2420, "security_strength": 128, "year": 2024, "wiki": "CRYSTALS-Dilithium"},
    "ML-DSA-65": {"type": "pqc_signature", "family": "ML-DSA", "key_size": 4032, "output_size": 3309, "security_strength": 192, "year": 2024, "wiki": "CRYSTALS-Dilithium"},
    "ML-DSA-87": {"type": "pqc_signature", "family": "ML-DSA", "key_size": 4896, "output_size": 4627, "security_strength": 256, "year": 2024, "wiki": "CRYSTALS-Dilithium"},
    "SLH-DSA-128f": {"type": "pqc_signature", "family": "SLH-DSA", "key_size": 64, "output_size": 17088, "security_strength": 128, "year": 2024, "wiki": "SPHINCS"},
    "SLH-DSA-128s": {"type": "pqc_signature", "family": "SLH-DSA", "key_size": 64, "output_size": 7856, "security_strength": 128, "year": 2024, "wiki": "SPHINCS"},
    "SLH-DSA-256f": {"type": "pqc_signature", "family": "SLH-DSA", "key_size": 128, "output_size": 49856, "security_strength": 256, "year": 2024, "wiki": "SPHINCS"},
    "Falcon-512": {"type": "pqc_signature", "family": "Falcon", "key_size": 1281, "output_size": 666, "security_strength": 128, "year": 2020, "wiki": "Falcon (signature scheme)"},
    "Falcon-1024": {"type": "pqc_signature", "family": "Falcon", "key_size": 2305, "output_size": 1280, "security_strength": 256, "year": 2020, "wiki": "Falcon (signature scheme)"},
    # --- PQC Key Encapsulation Mechanisms (NIST Round 4 candidates) ---
    # key_size = public key size in bytes
    "BIKE-L1": {"type": "pqc_kem", "family": "BIKE", "key_size": 1541, "security_strength": 128, "year": 2022, "wiki": "BIKE (cryptography)"},
    "BIKE-L3": {"type": "pqc_kem", "family": "BIKE", "key_size": 3083, "security_strength": 192, "year": 2022, "wiki": "BIKE (cryptography)"},
    "HQC-128": {"type": "pqc_kem", "family": "HQC", "key_size": 2249, "security_strength": 128, "year": 2022, "wiki": "Hamming quasi-cyclic"},
    "HQC-192": {"type": "pqc_kem", "family": "HQC", "key_size": 4522, "security_strength": 192, "year": 2022, "wiki": "Hamming quasi-cyclic"},
    "HQC-256": {"type": "pqc_kem", "family": "HQC", "key_size": 7245, "security_strength": 256, "year": 2022, "wiki": "Hamming quasi-cyclic"},
    "Classic-McEliece-348864": {"type": "pqc_kem", "family": "Classic-McEliece", "key_size": 261120, "security_strength": 128, "year": 2017, "wiki": "McEliece cryptosystem"},
    "Classic-McEliece-460896": {"type": "pqc_kem", "family": "Classic-McEliece", "key_size": 524160, "security_strength": 192, "year": 2017, "wiki": "McEliece cryptosystem"},
    "FrodoKEM-640": {"type": "pqc_kem", "family": "FrodoKEM", "key_size": 9616, "security_strength": 128, "year": 2022, "wiki": "FrodoKEM"},
    # --- Lightweight Ciphers (NIST LWC / ISO 29192) ---
    "ASCON-128": {"type": "lightweight_cipher", "family": "ASCON", "key_size": 128, "nonce_size": 128, "tag_size": 128, "block_size": 64, "rounds": 6, "security_strength": 128, "year": 2014, "wiki": "Ascon (cipher)"},
    "ASCON-128a": {"type": "lightweight_cipher", "family": "ASCON", "key_size": 128, "nonce_size": 128, "tag_size": 128, "block_size": 128, "rounds": 8, "security_strength": 128, "year": 2014, "wiki": "Ascon (cipher)"},
    "GIFT-128": {"type": "lightweight_cipher", "family": "GIFT", "key_size": 128, "block_size": 128, "rounds": 40, "security_strength": 128, "year": 2017, "wiki": "GIFT (block cipher)"},
    "PRESENT-80": {"type": "lightweight_cipher", "family": "PRESENT", "key_size": 80, "block_size": 64, "rounds": 31, "security_strength": 80, "year": 2007, "wiki": "PRESENT"},
    "PRESENT-128": {"type": "lightweight_cipher", "family": "PRESENT", "key_size": 128, "block_size": 64, "rounds": 31, "security_strength": 128, "year": 2007, "wiki": "PRESENT"},
    "SIMON-128-128": {"type": "lightweight_cipher", "family": "SIMON", "key_size": 128, "block_size": 128, "rounds": 68, "security_strength": 128, "year": 2013, "wiki": "Simon (cipher)"},
    "SPECK-128-128": {"type": "lightweight_cipher", "family": "SPECK", "key_size": 128, "block_size": 128, "rounds": 32, "security_strength": 128, "year": 2013, "wiki": "Speck (cipher)"},
    "SKINNY-128-256": {"type": "lightweight_cipher", "family": "SKINNY", "key_size": 256, "block_size": 128, "rounds": 48, "security_strength": 256, "year": 2016, "wiki": "SKINNY (cipher)"},
    "PRINCE": {"type": "lightweight_cipher", "family": "PRINCE", "key_size": 128, "block_size": 64, "rounds": 12, "security_strength": 64, "year": 2012, "wiki": "Prince (cipher)"},
    "CLEFIA-128": {"type": "lightweight_cipher", "family": "CLEFIA", "key_size": 128, "block_size": 128, "rounds": 18, "security_strength": 128, "year": 2007, "wiki": "CLEFIA"},
    # --- Elliptic Curves (NIST / IETF / Brainpool) ---
    # key_size = field size in bits
    "P-256": {"type": "elliptic_curve", "family": "NIST", "key_size": 256, "security_strength": 128, "year": 1999, "wiki": "Elliptic-curve cryptography"},
    "P-384": {"type": "elliptic_curve", "family": "NIST", "key_size": 384, "security_strength": 192, "year": 1999, "wiki": "Elliptic-curve cryptography"},
    "P-521": {"type": "elliptic_curve", "family": "NIST", "key_size": 521, "security_strength": 256, "year": 1999, "wiki": "Elliptic-curve cryptography"},
    "Curve25519": {"type": "elliptic_curve", "family": "Bernstein", "key_size": 255, "security_strength": 128, "year": 2005, "wiki": "Curve25519"},
    "Curve448": {"type": "elliptic_curve", "family": "Hamburg", "key_size": 448, "security_strength": 224, "year": 2015, "wiki": "Curve448"},
    "secp256k1": {"type": "elliptic_curve", "family": "Koblitz", "key_size": 256, "security_strength": 128, "year": 2000, "wiki": "Secp256k1"},
    "brainpoolP256r1": {"type": "elliptic_curve", "family": "Brainpool", "key_size": 256, "security_strength": 128, "year": 2005, "wiki": "Brainpool curve"},
    "brainpoolP384r1": {"type": "elliptic_curve", "family": "Brainpool", "key_size": 384, "security_strength": 192, "year": 2005, "wiki": "Brainpool curve"},
    "brainpoolP512r1": {"type": "elliptic_curve", "family": "Brainpool", "key_size": 512, "security_strength": 256, "year": 2005, "wiki": "Brainpool curve"},
    "FourQ": {"type": "elliptic_curve", "family": "Microsoft", "key_size": 256, "security_strength": 128, "year": 2015, "wiki": "FourQ"},
    # --- Extendable-Output Functions (SHA-3 / NIST SP 800-185) ---
    "SHAKE128": {"type": "xof", "family": "SHA-3", "output_size": 256, "block_size": 1344, "rounds": 24, "security_strength": 128, "year": 2015, "wiki": "SHA-3"},
    "SHAKE256": {"type": "xof", "family": "SHA-3", "output_size": 512, "block_size": 1088, "rounds": 24, "security_strength": 256, "year": 2015, "wiki": "SHA-3"},
    "cSHAKE128": {"type": "xof", "family": "SHA-3", "output_size": 256, "block_size": 1344, "rounds": 24, "security_strength": 128, "year": 2016, "wiki": "SHA-3"},
    "cSHAKE256": {"type": "xof", "family": "SHA-3", "output_size": 512, "block_size": 1088, "rounds": 24, "security_strength": 256, "year": 2016, "wiki": "SHA-3"},
    "KMAC128": {"type": "xof", "family": "SHA-3", "key_size": 128, "output_size": 256, "block_size": 1344, "rounds": 24, "security_strength": 128, "year": 2016, "wiki": "SHA-3"},
    "KMAC256": {"type": "xof", "family": "SHA-3", "key_size": 256, "output_size": 512, "block_size": 1088, "rounds": 24, "security_strength": 256, "year": 2016, "wiki": "SHA-3"},
    "TupleHash128": {"type": "xof", "family": "SHA-3", "output_size": 256, "block_size": 1344, "rounds": 24, "security_strength": 128, "year": 2016, "wiki": "SHA-3"},
    "TupleHash256": {"type": "xof", "family": "SHA-3", "output_size": 512, "block_size": 1088, "rounds": 24, "security_strength": 256, "year": 2016, "wiki": "SHA-3"},
}


# ---------------------------------------------------------------------------
# MITRE ATT&CK Reference Data (pinned from ATT&CK STIX v15)
# ---------------------------------------------------------------------------

ATTACK_REFERENCE_DATA: Dict[str, Dict[str, Any]] = {
    # --- Threat Groups ---
    "APT28": {"type": "group", "attack_id": "G0007", "technique_count": 67, "sub_technique_count": 47, "tactic_count": 11, "software_count": 28, "first_seen": 2004, "wiki": "Fancy Bear", "country": "Russia", "notable_campaign": "DNC hack 2016", "target_sectors": ["government", "military", "media"]},
    "APT29": {"type": "group", "attack_id": "G0016", "technique_count": 54, "sub_technique_count": 41, "tactic_count": 10, "software_count": 21, "first_seen": 2008, "wiki": "Cozy Bear", "country": "Russia", "notable_campaign": "SolarWinds supply chain attack", "target_sectors": ["government", "think tanks", "technology"]},
    "Lazarus Group": {"type": "group", "attack_id": "G0032", "technique_count": 72, "sub_technique_count": 52, "tactic_count": 11, "software_count": 35, "first_seen": 2009, "wiki": "Lazarus Group", "country": "North Korea", "notable_campaign": "Sony Pictures hack 2014", "target_sectors": ["financial", "entertainment", "cryptocurrency"]},
    "Turla": {"type": "group", "attack_id": "G0010", "technique_count": 56, "sub_technique_count": 40, "tactic_count": 10, "software_count": 25, "first_seen": 1996, "wiki": "Turla (malware)", "country": "Russia", "notable_campaign": "Agent.BTZ USB worm operation", "target_sectors": ["government", "military", "diplomatic"]},
    "Sandworm Team": {"type": "group", "attack_id": "G0034", "technique_count": 48, "sub_technique_count": 35, "tactic_count": 9, "software_count": 18, "first_seen": 2009, "wiki": "Sandworm (hacker group)", "country": "Russia", "notable_campaign": "NotPetya 2017", "target_sectors": ["energy", "government", "critical infrastructure"]},
    "APT41": {"type": "group", "attack_id": "G0096", "technique_count": 60, "sub_technique_count": 44, "tactic_count": 10, "software_count": 30, "first_seen": 2012, "wiki": "Double Dragon (hacking group)", "country": "China", "notable_campaign": "Video game supply chain attacks", "target_sectors": ["gaming", "healthcare", "technology"]},
    "Kimsuky": {"type": "group", "attack_id": "G0094", "technique_count": 45, "sub_technique_count": 32, "tactic_count": 9, "software_count": 15, "first_seen": 2012, "wiki": "Kimsuky", "country": "North Korea", "notable_campaign": "Korea Hydro & Nuclear Power hack", "target_sectors": ["nuclear", "think tanks", "academia"]},
    "FIN7": {"type": "group", "attack_id": "G0046", "technique_count": 50, "sub_technique_count": 36, "tactic_count": 10, "software_count": 22, "first_seen": 2013, "wiki": "FIN7", "country": "Russia", "notable_campaign": "Restaurant and hospitality POS attacks", "target_sectors": ["retail", "hospitality", "financial"]},
    "Wizard Spider": {"type": "group", "attack_id": "G0102", "technique_count": 42, "sub_technique_count": 30, "tactic_count": 9, "software_count": 17, "first_seen": 2016, "wiki": "Wizard Spider", "country": "Russia", "notable_campaign": "Ryuk ransomware campaigns", "target_sectors": ["healthcare", "government", "education"]},
    "Hafnium": {"type": "group", "attack_id": "G0125", "technique_count": 20, "sub_technique_count": 14, "tactic_count": 7, "software_count": 8, "first_seen": 2021, "wiki": "Hafnium (hacker group)", "country": "China", "notable_campaign": "Microsoft Exchange Server exploitation 2021", "target_sectors": ["legal", "defense", "infectious disease research"]},
    "APT1": {"type": "group", "attack_id": "G0006", "technique_count": 30, "sub_technique_count": 18, "tactic_count": 8, "software_count": 16, "first_seen": 2006, "wiki": "PLA Unit 61398", "country": "China", "notable_campaign": "Mandiant APT1 report 2013", "target_sectors": ["aerospace", "technology", "telecommunications"]},
    "Equation Group": {"type": "group", "attack_id": "G0020", "technique_count": 25, "sub_technique_count": 15, "tactic_count": 7, "software_count": 12, "first_seen": 2001, "wiki": "Equation Group", "country": "United States", "notable_campaign": "Stuxnet development", "target_sectors": ["government", "military", "telecommunications"]},
    "OilRig": {"type": "group", "attack_id": "G0049", "technique_count": 55, "sub_technique_count": 38, "tactic_count": 10, "software_count": 20, "first_seen": 2014, "wiki": "OilRig (hacker group)", "country": "Iran", "notable_campaign": "DNSpionage campaign", "target_sectors": ["energy", "government", "financial"]},
    "Gamaredon Group": {"type": "group", "attack_id": "G0047", "technique_count": 35, "sub_technique_count": 25, "tactic_count": 8, "software_count": 10, "first_seen": 2013, "wiki": "Gamaredon", "country": "Russia", "notable_campaign": "Ukraine government targeting", "target_sectors": ["government", "military", "law enforcement"]},
    "Mustang Panda": {"type": "group", "attack_id": "G0129", "technique_count": 38, "sub_technique_count": 28, "tactic_count": 9, "software_count": 12, "first_seen": 2017, "wiki": "Mustang Panda", "country": "China", "notable_campaign": "Southeast Asian government espionage", "target_sectors": ["government", "nonprofits", "religious organizations"]},
    "LAPSUS$": {"type": "group", "attack_id": "G1004", "technique_count": 18, "sub_technique_count": 12, "tactic_count": 6, "software_count": 5, "first_seen": 2021, "wiki": "Lapsus$", "country": "United Kingdom", "notable_campaign": "Nvidia and Samsung source code theft", "target_sectors": ["technology", "gaming", "telecommunications"]},
    "Scattered Spider": {"type": "group", "attack_id": "G1015", "technique_count": 22, "sub_technique_count": 16, "tactic_count": 7, "software_count": 8, "first_seen": 2022, "wiki": "Scattered Spider", "country": "United States", "notable_campaign": "MGM Resorts social engineering attack", "target_sectors": ["telecommunications", "hospitality", "technology"]},
    "MuddyWater": {"type": "group", "attack_id": "G0069", "technique_count": 40, "sub_technique_count": 28, "tactic_count": 9, "software_count": 14, "first_seen": 2017, "wiki": "MuddyWater (hacker group)", "country": "Iran", "notable_campaign": "Middle East government espionage", "target_sectors": ["government", "telecommunications", "oil and gas"]},
    "Winnti Group": {"type": "group", "attack_id": "G0044", "technique_count": 35, "sub_technique_count": 24, "tactic_count": 8, "software_count": 18, "first_seen": 2010, "wiki": "Winnti Group", "country": "China", "notable_campaign": "Gaming industry supply chain attacks", "target_sectors": ["gaming", "technology", "pharmaceutical"]},
    "DarkSide": {"type": "group", "attack_id": "G0102", "technique_count": 28, "sub_technique_count": 20, "tactic_count": 8, "software_count": 6, "first_seen": 2020, "wiki": "DarkSide (hacking group)", "country": "Russia", "notable_campaign": "Colonial Pipeline ransomware attack", "target_sectors": ["energy", "manufacturing", "financial"]},
    "Volt Typhoon": {"type": "group", "attack_id": "G1017", "technique_count": 32, "sub_technique_count": 22, "tactic_count": 8, "software_count": 10, "first_seen": 2021, "wiki": "Volt Typhoon", "country": "China", "notable_campaign": "US critical infrastructure pre-positioning", "target_sectors": ["critical infrastructure", "communications", "energy"]},
    "BlackTech": {"type": "group", "attack_id": "G0098", "technique_count": 30, "sub_technique_count": 20, "tactic_count": 8, "software_count": 12, "first_seen": 2010, "wiki": "BlackTech", "country": "China", "notable_campaign": "Router firmware manipulation", "target_sectors": ["technology", "government", "defense"]},
    "APT33": {"type": "group", "attack_id": "G0064", "technique_count": 28, "sub_technique_count": 18, "tactic_count": 7, "software_count": 11, "first_seen": 2013, "wiki": "APT33", "country": "Iran", "notable_campaign": "Shamoon-linked destructive attacks", "target_sectors": ["aerospace", "energy", "petrochemical"]},
    "Charming Kitten": {"type": "group", "attack_id": "G0058", "technique_count": 38, "sub_technique_count": 26, "tactic_count": 9, "software_count": 14, "first_seen": 2014, "wiki": "Charming Kitten", "country": "Iran", "notable_campaign": "Social media credential harvesting", "target_sectors": ["academia", "journalism", "human rights"]},
    "Andariel": {"type": "group", "attack_id": "G0138", "technique_count": 25, "sub_technique_count": 18, "tactic_count": 7, "software_count": 9, "first_seen": 2015, "wiki": "Andariel (hacker group)", "country": "North Korea", "notable_campaign": "South Korean defense contractor espionage", "target_sectors": ["defense", "financial", "healthcare"]},
    # --- Techniques ---
    "Phishing": {"type": "technique", "attack_id": "T1566", "sub_technique_count": 4, "group_count": 45, "software_count": 3, "mitigation_count": 4, "data_source_count": 3, "tactic": "initial-access", "wiki": "Phishing", "platform": ["Windows", "macOS", "Linux", "SaaS", "Office 365"], "permission_required": "user"},
    "Command and Scripting Interpreter": {"type": "technique", "attack_id": "T1059", "sub_technique_count": 8, "group_count": 62, "software_count": 55, "mitigation_count": 5, "data_source_count": 4, "tactic": "execution", "wiki": "Scripting language", "platform": ["Windows", "macOS", "Linux"], "permission_required": "user"},
    "Ingress Tool Transfer": {"type": "technique", "attack_id": "T1105", "sub_technique_count": 0, "group_count": 80, "software_count": 90, "mitigation_count": 3, "data_source_count": 2, "tactic": "command-and-control", "wiki": "Ingress (software)", "platform": ["Windows", "macOS", "Linux"], "permission_required": "user"},
    "Masquerading": {"type": "technique", "attack_id": "T1036", "sub_technique_count": 9, "group_count": 40, "software_count": 48, "mitigation_count": 4, "data_source_count": 3, "tactic": "defense-evasion", "wiki": "Masquerade attack", "platform": ["Windows", "macOS", "Linux", "Containers"], "permission_required": "user"},
    "OS Credential Dumping": {"type": "technique", "attack_id": "T1003", "sub_technique_count": 8, "group_count": 50, "software_count": 30, "mitigation_count": 6, "data_source_count": 4, "tactic": "credential-access", "wiki": "Credential dumping", "platform": ["Windows", "Linux"], "permission_required": "administrator"},
    "Valid Accounts": {"type": "technique", "attack_id": "T1078", "sub_technique_count": 4, "group_count": 55, "software_count": 10, "mitigation_count": 7, "data_source_count": 3, "tactic": "defense-evasion", "wiki": "Authentication", "platform": ["Windows", "macOS", "Linux", "SaaS", "Azure AD"], "permission_required": "user"},
    "Scheduled Task/Job": {"type": "technique", "attack_id": "T1053", "sub_technique_count": 5, "group_count": 35, "software_count": 25, "mitigation_count": 5, "data_source_count": 3, "tactic": "execution", "wiki": "Job scheduler", "platform": ["Windows", "macOS", "Linux", "Containers"], "permission_required": "administrator"},
    "Remote Services": {"type": "technique", "attack_id": "T1021", "sub_technique_count": 6, "group_count": 48, "software_count": 15, "mitigation_count": 6, "data_source_count": 3, "tactic": "lateral-movement", "wiki": "Remote desktop software", "platform": ["Windows", "macOS", "Linux"], "permission_required": "user"},
    "Archive Collected Data": {"type": "technique", "attack_id": "T1560", "sub_technique_count": 3, "group_count": 30, "software_count": 20, "mitigation_count": 2, "data_source_count": 3, "tactic": "collection", "wiki": "Data compression", "platform": ["Windows", "macOS", "Linux"], "permission_required": "user"},
    "Exfiltration Over C2 Channel": {"type": "technique", "attack_id": "T1041", "sub_technique_count": 0, "group_count": 35, "software_count": 40, "mitigation_count": 2, "data_source_count": 2, "tactic": "exfiltration", "wiki": "Data exfiltration", "platform": ["Windows", "macOS", "Linux"], "permission_required": "user"},
    "Obfuscated Files or Information": {"type": "technique", "attack_id": "T1027", "sub_technique_count": 12, "group_count": 55, "software_count": 60, "mitigation_count": 3, "data_source_count": 3, "tactic": "defense-evasion", "wiki": "Obfuscation (software)", "platform": ["Windows", "macOS", "Linux"], "permission_required": "user"},
    "Process Injection": {"type": "technique", "attack_id": "T1055", "sub_technique_count": 12, "group_count": 30, "software_count": 45, "mitigation_count": 4, "data_source_count": 3, "tactic": "defense-evasion", "wiki": "Code injection", "platform": ["Windows", "macOS", "Linux"], "permission_required": "administrator"},
    "Data Encrypted for Impact": {"type": "technique", "attack_id": "T1486", "sub_technique_count": 0, "group_count": 20, "software_count": 35, "mitigation_count": 3, "data_source_count": 3, "tactic": "impact", "wiki": "Ransomware", "platform": ["Windows", "macOS", "Linux"], "permission_required": "user"},
    "System Information Discovery": {"type": "technique", "attack_id": "T1082", "sub_technique_count": 0, "group_count": 60, "software_count": 70, "mitigation_count": 1, "data_source_count": 3, "tactic": "discovery", "wiki": "Fingerprint (computing)", "platform": ["Windows", "macOS", "Linux"], "permission_required": "user"},
    "Account Discovery": {"type": "technique", "attack_id": "T1087", "sub_technique_count": 4, "group_count": 40, "software_count": 25, "mitigation_count": 2, "data_source_count": 3, "tactic": "discovery", "wiki": "System administrator", "platform": ["Windows", "macOS", "Linux", "Azure AD"], "permission_required": "user"},
    "File and Directory Discovery": {"type": "technique", "attack_id": "T1083", "sub_technique_count": 0, "group_count": 45, "software_count": 55, "mitigation_count": 1, "data_source_count": 2, "tactic": "discovery", "wiki": "File system", "platform": ["Windows", "macOS", "Linux"], "permission_required": "user"},
    "Exploitation for Privilege Escalation": {"type": "technique", "attack_id": "T1068", "sub_technique_count": 0, "group_count": 20, "software_count": 15, "mitigation_count": 5, "data_source_count": 2, "tactic": "privilege-escalation", "wiki": "Privilege escalation", "platform": ["Windows", "macOS", "Linux"], "permission_required": "user"},
    "Supply Chain Compromise": {"type": "technique", "attack_id": "T1195", "sub_technique_count": 3, "group_count": 12, "software_count": 5, "mitigation_count": 3, "data_source_count": 2, "tactic": "initial-access", "wiki": "Supply chain attack", "platform": ["Windows", "macOS", "Linux"], "permission_required": "user"},
    "Brute Force": {"type": "technique", "attack_id": "T1110", "sub_technique_count": 4, "group_count": 20, "software_count": 10, "mitigation_count": 5, "data_source_count": 3, "tactic": "credential-access", "wiki": "Brute-force attack", "platform": ["Windows", "macOS", "Linux", "SaaS", "Azure AD"], "permission_required": "user"},
    "Create Account": {"type": "technique", "attack_id": "T1136", "sub_technique_count": 3, "group_count": 18, "software_count": 8, "mitigation_count": 4, "data_source_count": 3, "tactic": "persistence", "wiki": "User account", "platform": ["Windows", "macOS", "Linux", "Azure AD"], "permission_required": "administrator"},
    "Exploit Public-Facing Application": {"type": "technique", "attack_id": "T1190", "sub_technique_count": 0, "group_count": 42, "software_count": 5, "mitigation_count": 5, "data_source_count": 3, "tactic": "initial-access", "wiki": "Web application security", "platform": ["Windows", "Linux", "Containers", "Network"], "permission_required": "user"},
    "Indicator Removal": {"type": "technique", "attack_id": "T1070", "sub_technique_count": 9, "group_count": 35, "software_count": 30, "mitigation_count": 4, "data_source_count": 4, "tactic": "defense-evasion", "wiki": "Anti-forensics", "platform": ["Windows", "macOS", "Linux", "Network"], "permission_required": "administrator"},
    "Modify Registry": {"type": "technique", "attack_id": "T1112", "sub_technique_count": 0, "group_count": 30, "software_count": 50, "mitigation_count": 2, "data_source_count": 2, "tactic": "defense-evasion", "wiki": "Windows Registry", "platform": ["Windows"], "permission_required": "user"},
    "Network Service Discovery": {"type": "technique", "attack_id": "T1046", "sub_technique_count": 0, "group_count": 25, "software_count": 20, "mitigation_count": 3, "data_source_count": 3, "tactic": "discovery", "wiki": "Port scanner", "platform": ["Windows", "macOS", "Linux"], "permission_required": "user"},
    "System Network Configuration Discovery": {"type": "technique", "attack_id": "T1016", "sub_technique_count": 2, "group_count": 35, "software_count": 30, "mitigation_count": 1, "data_source_count": 2, "tactic": "discovery", "wiki": "Computer network", "platform": ["Windows", "macOS", "Linux", "Network"], "permission_required": "user"},
    # --- Software/Malware ---
    "Mimikatz": {"type": "software", "attack_id": "S0002", "technique_count": 24, "group_count": 20, "wiki": "Mimikatz", "software_type": "tool", "platform": ["Windows"]},
    "Cobalt Strike": {"type": "software", "attack_id": "S0154", "technique_count": 42, "group_count": 35, "wiki": "Cobalt Strike", "software_type": "tool", "platform": ["Windows", "Linux", "macOS"]},
    "Metasploit": {"type": "software", "attack_id": "S0100", "technique_count": 30, "group_count": 15, "wiki": "Metasploit", "software_type": "tool", "platform": ["Windows", "Linux", "macOS"]},
    "PowerSploit": {"type": "software", "attack_id": "S0194", "technique_count": 18, "group_count": 12, "wiki": "PowerSploit", "software_type": "tool", "platform": ["Windows"]},
    "Empire": {"type": "software", "attack_id": "S0363", "technique_count": 28, "group_count": 10, "wiki": "Empire (hacking tool)", "software_type": "tool", "platform": ["Windows", "macOS", "Linux"]},
    "Emotet": {"type": "software", "attack_id": "S0367", "technique_count": 22, "group_count": 5, "wiki": "Emotet", "software_type": "malware", "platform": ["Windows"]},
    "TrickBot": {"type": "software", "attack_id": "S0266", "technique_count": 20, "group_count": 4, "wiki": "TrickBot", "software_type": "malware", "platform": ["Windows"]},
    "Remcos": {"type": "software", "attack_id": "S0332", "technique_count": 16, "group_count": 6, "wiki": "Remcos", "software_type": "malware", "platform": ["Windows"]},
    "njRAT": {"type": "software", "attack_id": "S0385", "technique_count": 18, "group_count": 8, "wiki": "NjRAT", "software_type": "malware", "platform": ["Windows"]},
    "PlugX": {"type": "software", "attack_id": "S0013", "technique_count": 20, "group_count": 14, "wiki": "PlugX", "software_type": "malware", "platform": ["Windows"]},
    "ShadowPad": {"type": "software", "attack_id": "S0596", "technique_count": 16, "group_count": 8, "wiki": "ShadowPad", "software_type": "malware", "platform": ["Windows"]},
    "Impacket": {"type": "software", "attack_id": "S0357", "technique_count": 14, "group_count": 10, "wiki": "Impacket", "software_type": "tool", "platform": ["Windows", "Linux"]},
    "BloodHound": {"type": "software", "attack_id": "S0521", "technique_count": 8, "group_count": 6, "wiki": "BloodHound (software)", "software_type": "tool", "platform": ["Windows"]},
    "Sliver": {"type": "software", "attack_id": "S0633", "technique_count": 20, "group_count": 7, "wiki": "Sliver (C2 framework)", "software_type": "tool", "platform": ["Windows", "Linux", "macOS"]},
    "AsyncRAT": {"type": "software", "attack_id": "S0640", "technique_count": 14, "group_count": 5, "wiki": "AsyncRAT", "software_type": "malware", "platform": ["Windows"]},
    # --- Ransomware Families ---
    "WannaCry": {"type": "ransomware", "attack_id": "S0366", "technique_count": 12, "group_count": 3, "wiki": "WannaCry ransomware attack", "software_type": "malware", "platform": ["Windows"]},
    "LockBit 3.0": {"type": "ransomware", "attack_id": "S1070", "technique_count": 18, "group_count": 1, "wiki": "LockBit", "software_type": "malware", "platform": ["Windows", "Linux", "VMware ESXi"]},
    "REvil": {"type": "ransomware", "attack_id": "S0496", "technique_count": 16, "group_count": 2, "wiki": "REvil", "software_type": "malware", "platform": ["Windows"]},
    "Conti": {"type": "ransomware", "attack_id": "S0575", "technique_count": 20, "group_count": 2, "wiki": "Conti (ransomware)", "software_type": "malware", "platform": ["Windows"]},
    "BlackCat": {"type": "ransomware", "attack_id": "S1068", "technique_count": 14, "group_count": 1, "wiki": "BlackCat (cyber gang)", "software_type": "malware", "platform": ["Windows", "Linux", "VMware ESXi"]},
    "Hive Ransomware": {"type": "ransomware", "attack_id": "S1060", "technique_count": 12, "group_count": 1, "wiki": "Hive (ransomware group)", "software_type": "malware", "platform": ["Windows", "Linux"]},
    "Maze": {"type": "ransomware", "attack_id": "S0449", "technique_count": 16, "group_count": 1, "wiki": "Maze (ransomware)", "software_type": "malware", "platform": ["Windows"]},
    "DoppelPaymer": {"type": "ransomware", "attack_id": "S0554", "technique_count": 14, "group_count": 1, "wiki": "DoppelPaymer", "software_type": "malware", "platform": ["Windows"]},
    "Clop": {"type": "ransomware", "attack_id": "S0611", "technique_count": 15, "group_count": 2, "wiki": "Clop (cyber gang)", "software_type": "malware", "platform": ["Windows", "Linux"]},
    "Ryuk": {"type": "ransomware", "attack_id": "S0446", "technique_count": 14, "group_count": 2, "wiki": "Ryuk (ransomware)", "software_type": "malware", "platform": ["Windows"]},
    "Black Basta": {"type": "ransomware", "attack_id": "S1070", "technique_count": 16, "group_count": 1, "wiki": "Black Basta", "software_type": "malware", "platform": ["Windows", "Linux", "VMware ESXi"]},
    "Royal Ransomware": {"type": "ransomware", "attack_id": "S1073", "technique_count": 12, "group_count": 1, "wiki": "Royal (ransomware)", "software_type": "malware", "platform": ["Windows", "Linux"]},
    "Akira Ransomware": {"type": "ransomware", "attack_id": "S1129", "technique_count": 10, "group_count": 1, "wiki": "Akira (ransomware)", "software_type": "malware", "platform": ["Windows", "Linux", "VMware ESXi"]},
    "Play Ransomware": {"type": "ransomware", "attack_id": "S1090", "technique_count": 12, "group_count": 1, "wiki": "Play (ransomware)", "software_type": "malware", "platform": ["Windows"]},
    "Ragnar Locker": {"type": "ransomware", "attack_id": "S0481", "technique_count": 10, "group_count": 1, "wiki": "Ragnar Locker", "software_type": "malware", "platform": ["Windows", "VMware ESXi"]},
    # --- Botnets ---
    "Mirai": {"type": "botnet", "attack_id": "S0400", "technique_count": 8, "group_count": 2, "wiki": "Mirai (malware)", "software_type": "malware", "platform": ["Linux", "IoT"]},
    "Zeus": {"type": "botnet", "attack_id": "S0412", "technique_count": 14, "group_count": 5, "wiki": "Zeus (malware)", "software_type": "malware", "platform": ["Windows"]},
    "Gameover ZeuS": {"type": "botnet", "attack_id": "S0413", "technique_count": 12, "group_count": 2, "wiki": "Gameover ZeuS", "software_type": "malware", "platform": ["Windows"]},
    "Necurs": {"type": "botnet", "attack_id": "S0470", "technique_count": 10, "group_count": 2, "wiki": "Necurs botnet", "software_type": "malware", "platform": ["Windows"]},
    "Srizbi": {"type": "botnet", "attack_id": "S0471", "technique_count": 6, "group_count": 1, "wiki": "Srizbi botnet", "software_type": "malware", "platform": ["Windows"]},
    "Cutwail": {"type": "botnet", "attack_id": "S0472", "technique_count": 8, "group_count": 2, "wiki": "Cutwail botnet", "software_type": "malware", "platform": ["Windows"]},
    "Storm Botnet": {"type": "botnet", "attack_id": "S0473", "technique_count": 10, "group_count": 1, "wiki": "Storm botnet", "software_type": "malware", "platform": ["Windows"]},
    "Conficker": {"type": "botnet", "attack_id": "S0474", "technique_count": 8, "group_count": 1, "wiki": "Conficker", "software_type": "malware", "platform": ["Windows"]},
    "Kelihos": {"type": "botnet", "attack_id": "S0475", "technique_count": 10, "group_count": 2, "wiki": "Kelihos botnet", "software_type": "malware", "platform": ["Windows"]},
    "Andromeda Botnet": {"type": "botnet", "attack_id": "S0476", "technique_count": 10, "group_count": 3, "wiki": "Andromeda (botnet)", "software_type": "malware", "platform": ["Windows"]},
    # --- Wipers (Destructive Malware) ---
    "Shamoon": {"type": "wiper", "attack_id": "S0140", "technique_count": 12, "group_count": 2, "wiki": "Shamoon", "software_type": "malware", "platform": ["Windows"]},
    "WhisperGate": {"type": "wiper", "attack_id": "S0689", "technique_count": 8, "group_count": 1, "wiki": "WhisperGate", "software_type": "malware", "platform": ["Windows"]},
    "HermeticWiper": {"type": "wiper", "attack_id": "S0697", "technique_count": 10, "group_count": 1, "wiki": "HermeticWiper", "software_type": "malware", "platform": ["Windows"]},
    "CaddyWiper": {"type": "wiper", "attack_id": "S0693", "technique_count": 8, "group_count": 1, "wiki": "CaddyWiper", "software_type": "malware", "platform": ["Windows"]},
    "Industroyer": {"type": "wiper", "attack_id": "S0604", "technique_count": 14, "group_count": 1, "wiki": "Industroyer", "software_type": "malware", "platform": ["Windows"]},
    "Industroyer2": {"type": "wiper", "attack_id": "S0694", "technique_count": 10, "group_count": 1, "wiki": "Industroyer", "software_type": "malware", "platform": ["Windows"]},
    "Olympic Destroyer": {"type": "wiper", "attack_id": "S0365", "technique_count": 12, "group_count": 1, "wiki": "Olympic Destroyer", "software_type": "malware", "platform": ["Windows"]},
    "ZeroCleare": {"type": "wiper", "attack_id": "S0660", "technique_count": 8, "group_count": 1, "wiki": "ZeroCleare", "software_type": "malware", "platform": ["Windows"]},
    "Destover": {"type": "wiper", "attack_id": "S0139", "technique_count": 10, "group_count": 1, "wiki": "Sony Pictures hack", "software_type": "malware", "platform": ["Windows"]},
    "Meteor Express": {"type": "wiper", "attack_id": "S0688", "technique_count": 6, "group_count": 1, "wiki": "Meteor Express (malware)", "software_type": "malware", "platform": ["Windows"]},
    # --- Info-Stealers ---
    "Raccoon Stealer": {"type": "stealer", "attack_id": "S0650", "technique_count": 12, "group_count": 3, "wiki": "Raccoon Stealer", "software_type": "malware", "platform": ["Windows"]},
    "Vidar": {"type": "stealer", "attack_id": "S0538", "technique_count": 10, "group_count": 2, "wiki": "Vidar (malware)", "software_type": "malware", "platform": ["Windows"]},
    "RedLine Stealer": {"type": "stealer", "attack_id": "S0655", "technique_count": 14, "group_count": 3, "wiki": "RedLine Stealer", "software_type": "malware", "platform": ["Windows"]},
    "FormBook": {"type": "stealer", "attack_id": "S0391", "technique_count": 16, "group_count": 4, "wiki": "FormBook", "software_type": "malware", "platform": ["Windows"]},
    "Agent Tesla": {"type": "stealer", "attack_id": "S0331", "technique_count": 18, "group_count": 5, "wiki": "Agent Tesla (malware)", "software_type": "malware", "platform": ["Windows"]},
    "AZORult": {"type": "stealer", "attack_id": "S0344", "technique_count": 10, "group_count": 3, "wiki": "AZORult", "software_type": "malware", "platform": ["Windows"]},
    "Predator the Thief": {"type": "stealer", "attack_id": "S0491", "technique_count": 8, "group_count": 1, "wiki": "Predator the Thief", "software_type": "malware", "platform": ["Windows"]},
    "Arkei": {"type": "stealer", "attack_id": "S0651", "technique_count": 8, "group_count": 1, "wiki": "Arkei (malware)", "software_type": "malware", "platform": ["Windows"]},
    "Lumma Stealer": {"type": "stealer", "attack_id": "S0652", "technique_count": 10, "group_count": 2, "wiki": "Lumma Stealer", "software_type": "malware", "platform": ["Windows"]},
    "StealC": {"type": "stealer", "attack_id": "S0653", "technique_count": 8, "group_count": 1, "wiki": "StealC", "software_type": "malware", "platform": ["Windows"]},
    # --- Malware Loaders/Droppers ---
    "QakBot": {"type": "loader", "attack_id": "S0650", "technique_count": 18, "group_count": 4, "wiki": "Qakbot", "software_type": "malware", "platform": ["Windows"]},
    "IcedID": {"type": "loader", "attack_id": "S0483", "technique_count": 16, "group_count": 3, "wiki": "IcedID", "software_type": "malware", "platform": ["Windows"]},
    "BazarLoader": {"type": "loader", "attack_id": "S0534", "technique_count": 14, "group_count": 2, "wiki": "BazarLoader", "software_type": "malware", "platform": ["Windows"]},
    "Bumblebee Loader": {"type": "loader", "attack_id": "S1039", "technique_count": 12, "group_count": 2, "wiki": "Bumblebee (malware)", "software_type": "malware", "platform": ["Windows"]},
    "Pikabot": {"type": "loader", "attack_id": "S1145", "technique_count": 10, "group_count": 1, "wiki": "Pikabot", "software_type": "malware", "platform": ["Windows"]},
    "SocGholish": {"type": "loader", "attack_id": "S0691", "technique_count": 12, "group_count": 2, "wiki": "SocGholish", "software_type": "malware", "platform": ["Windows"]},
    "Gootloader": {"type": "loader", "attack_id": "S0577", "technique_count": 10, "group_count": 1, "wiki": "Gootloader", "software_type": "malware", "platform": ["Windows"]},
    "Ursnif": {"type": "loader", "attack_id": "S0386", "technique_count": 14, "group_count": 3, "wiki": "Ursnif", "software_type": "malware", "platform": ["Windows"]},
    "Hancitor": {"type": "loader", "attack_id": "S0499", "technique_count": 10, "group_count": 2, "wiki": "Hancitor", "software_type": "malware", "platform": ["Windows"]},
    "SmokeLoader": {"type": "loader", "attack_id": "S0226", "technique_count": 12, "group_count": 3, "wiki": "SmokeLoader", "software_type": "malware", "platform": ["Windows"]},
}


# ---------------------------------------------------------------------------
# Entity Universe — maps category names to entity lists
# ---------------------------------------------------------------------------

ENTITY_UNIVERSE: Dict[str, List[Dict[str, Any]]] = {
    # Crypto categories
    "block_ciphers": [
        {"id": k, "name": k, "wiki": v["wiki"]}
        for k, v in CRYPTO_REFERENCE_DATA.items() if v["type"] == "block_cipher"
    ],
    "stream_ciphers": [
        {"id": k, "name": k, "wiki": v["wiki"]}
        for k, v in CRYPTO_REFERENCE_DATA.items() if v["type"] == "stream_cipher"
    ],
    "hash_functions": [
        {"id": k, "name": k, "wiki": v["wiki"]}
        for k, v in CRYPTO_REFERENCE_DATA.items() if v["type"] == "hash"
    ],
    "public_key": [
        {"id": k, "name": k, "wiki": v["wiki"]}
        for k, v in CRYPTO_REFERENCE_DATA.items() if v["type"] == "public_key"
    ],
    "key_exchange": [
        {"id": k, "name": k, "wiki": v["wiki"]}
        for k, v in CRYPTO_REFERENCE_DATA.items() if v["type"] == "key_exchange"
    ],
    "aead_ciphers": [
        {"id": k, "name": k, "wiki": v["wiki"]}
        for k, v in CRYPTO_REFERENCE_DATA.items() if v["type"] == "aead"
    ],
    "mac_algorithms": [
        {"id": k, "name": k, "wiki": v["wiki"]}
        for k, v in CRYPTO_REFERENCE_DATA.items() if v["type"] == "mac"
    ],
    # ATT&CK categories
    "attack_groups": [
        {"id": v["attack_id"], "name": k, "wiki": v["wiki"]}
        for k, v in ATTACK_REFERENCE_DATA.items() if v["type"] == "group"
    ],
    "attack_techniques": [
        {"id": v["attack_id"], "name": k, "wiki": v["wiki"]}
        for k, v in ATTACK_REFERENCE_DATA.items() if v["type"] == "technique"
    ],
    "attack_software": [
        {"id": v["attack_id"], "name": k, "wiki": v["wiki"]}
        for k, v in ATTACK_REFERENCE_DATA.items() if v["type"] == "software"
    ],
    # PQC categories
    "pqc_signatures": [
        {"id": k, "name": k, "wiki": v["wiki"]}
        for k, v in CRYPTO_REFERENCE_DATA.items() if v["type"] == "pqc_signature"
    ],
    "pqc_kem": [
        {"id": k, "name": k, "wiki": v["wiki"]}
        for k, v in CRYPTO_REFERENCE_DATA.items() if v["type"] == "pqc_kem"
    ],
    # Lightweight / IoT crypto
    "lightweight_ciphers": [
        {"id": k, "name": k, "wiki": v["wiki"]}
        for k, v in CRYPTO_REFERENCE_DATA.items() if v["type"] == "lightweight_cipher"
    ],
    # Elliptic curves
    "elliptic_curves": [
        {"id": k, "name": k, "wiki": v["wiki"]}
        for k, v in CRYPTO_REFERENCE_DATA.items() if v["type"] == "elliptic_curve"
    ],
    # XOF functions
    "xof_functions": [
        {"id": k, "name": k, "wiki": v["wiki"]}
        for k, v in CRYPTO_REFERENCE_DATA.items() if v["type"] == "xof"
    ],
    # ATT&CK malware sub-categories
    "ransomware_families": [
        {"id": v["attack_id"], "name": k, "wiki": v["wiki"]}
        for k, v in ATTACK_REFERENCE_DATA.items() if v["type"] == "ransomware"
    ],
    "botnets": [
        {"id": v["attack_id"], "name": k, "wiki": v["wiki"]}
        for k, v in ATTACK_REFERENCE_DATA.items() if v["type"] == "botnet"
    ],
    "wipers": [
        {"id": v["attack_id"], "name": k, "wiki": v["wiki"]}
        for k, v in ATTACK_REFERENCE_DATA.items() if v["type"] == "wiper"
    ],
    "stealers": [
        {"id": v["attack_id"], "name": k, "wiki": v["wiki"]}
        for k, v in ATTACK_REFERENCE_DATA.items() if v["type"] == "stealer"
    ],
    "loaders": [
        {"id": v["attack_id"], "name": k, "wiki": v["wiki"]}
        for k, v in ATTACK_REFERENCE_DATA.items() if v["type"] == "loader"
    ],
}


# ---------------------------------------------------------------------------
# Security Themes (categories → entity type mapping)
# ---------------------------------------------------------------------------

SECURITY_THEMES: Dict[str, Dict[str, Any]] = {
    "block_ciphers":     {"min_entities": 10, "entity_type": "algorithm"},
    "stream_ciphers":    {"min_entities": 5,  "entity_type": "algorithm"},
    "hash_functions":    {"min_entities": 10, "entity_type": "algorithm"},
    "public_key":        {"min_entities": 8,  "entity_type": "algorithm"},
    "key_exchange":      {"min_entities": 6,  "entity_type": "algorithm"},
    "aead_ciphers":      {"min_entities": 4,  "entity_type": "algorithm"},
    "mac_algorithms":    {"min_entities": 4,  "entity_type": "algorithm"},
    "attack_groups":     {"min_entities": 15, "entity_type": "attack_group"},
    "attack_techniques": {"min_entities": 15, "entity_type": "attack_technique"},
    "attack_software":   {"min_entities": 10, "entity_type": "attack_software"},
    # PQC categories
    "pqc_signatures":    {"min_entities": 6,  "entity_type": "algorithm"},
    "pqc_kem":           {"min_entities": 6,  "entity_type": "algorithm"},
    # Lightweight / IoT crypto
    "lightweight_ciphers": {"min_entities": 8, "entity_type": "algorithm"},
    # Elliptic curves
    "elliptic_curves":   {"min_entities": 8,  "entity_type": "algorithm"},
    # XOF functions
    "xof_functions":     {"min_entities": 6,  "entity_type": "algorithm"},
    # ATT&CK malware sub-categories
    "ransomware_families": {"min_entities": 10, "entity_type": "attack_software"},
    "botnets":           {"min_entities": 8,  "entity_type": "attack_software"},
    "wipers":            {"min_entities": 8,  "entity_type": "attack_software"},
    "stealers":          {"min_entities": 8,  "entity_type": "attack_software"},
    "loaders":           {"min_entities": 8,  "entity_type": "attack_software"},
}


# ---------------------------------------------------------------------------
# Lookup Functions (replace CVE API calls — simple dict lookups)
# ---------------------------------------------------------------------------

def get_crypto_algorithm_data(name: str) -> Optional[Dict[str, Any]]:
    """Look up crypto algorithm parameters from CRYPTO_REFERENCE_DATA.

    Args:
        name: Algorithm name (e.g., "AES-256", "SHA-256").

    Returns:
        Dict of algorithm parameters, or None if not found.
    """
    data = CRYPTO_REFERENCE_DATA.get(name)
    if data:
        return {"name": name, **data}
    # Case-insensitive fallback
    name_lower = name.lower()
    for k, v in CRYPTO_REFERENCE_DATA.items():
        if k.lower() == name_lower:
            return {"name": k, **v}
    return None


def get_attack_entity_data(name: str) -> Optional[Dict[str, Any]]:
    """Look up ATT&CK group/technique/software metrics from ATTACK_REFERENCE_DATA.

    Args:
        name: Entity name (e.g., "APT28") or ATT&CK ID (e.g., "G0007").

    Returns:
        Dict of entity metrics, or None if not found.
    """
    # Direct name lookup
    data = ATTACK_REFERENCE_DATA.get(name)
    if data:
        return {"name": name, **data}
    # Search by attack_id
    for k, v in ATTACK_REFERENCE_DATA.items():
        if v.get("attack_id") == name:
            return {"name": k, **v}
    # Case-insensitive fallback
    name_lower = name.lower()
    for k, v in ATTACK_REFERENCE_DATA.items():
        if k.lower() == name_lower:
            return {"name": k, **v}
    return None


def search_crypto_algorithms(query: str, algo_type: Optional[str] = None) -> List[Dict]:
    """Search crypto algorithms by name/family/type.

    Args:
        query: Search string (matched against name, family).
        algo_type: Optional type filter (e.g., "block_cipher", "hash").

    Returns:
        List of matching algorithm dicts.
    """
    results = []
    query_lower = query.lower()
    for name, data in CRYPTO_REFERENCE_DATA.items():
        if algo_type and data["type"] != algo_type:
            continue
        if (query_lower in name.lower() or
                query_lower in data.get("family", "").lower() or
                query_lower in data["type"].lower()):
            results.append({"name": name, **data})
    return results


def search_attack_entities(query: str, entity_type: Optional[str] = None) -> List[Dict]:
    """Search ATT&CK entities by name/ID/tactic.

    Args:
        query: Search string.
        entity_type: Optional type filter ("group", "technique", "software").

    Returns:
        List of matching entity dicts.
    """
    results = []
    query_lower = query.lower()
    for name, data in ATTACK_REFERENCE_DATA.items():
        if entity_type and data["type"] != entity_type:
            continue
        if (query_lower in name.lower() or
                query_lower in data.get("attack_id", "").lower() or
                query_lower in data.get("tactic", "").lower() or
                query_lower in data.get("wiki", "").lower()):
            results.append({"name": name, **data})
    return results


# ---------------------------------------------------------------------------
# Wikipedia Entity Clue Fetching (unchanged from original)
# ---------------------------------------------------------------------------

def fetch_entity_clues(entity_name: str, entity_type: str = "algorithm") -> List[dict]:
    """Fetch descriptive clue facts from Wikipedia for a security entity.

    Args:
        entity_name: Name or Wikipedia article title of the entity.
        entity_type: One of "algorithm", "attack_group", "attack_technique", "attack_software".

    Returns:
        List of fact dicts: {"fact_id", "fact", "topic", "source"}.
    """
    cache_key = f"{entity_type}|{entity_name}"
    cached = _kg_cache_get(SECURITY_CACHE_DIR, cache_key)
    if cached is not None:
        return cached

    search_url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": entity_name,
        "srlimit": 3,
        "utf8": 1,
    }
    headers = {
        "User-Agent": "DrBencher/1.0 (research@example.com)",
        "Accept": "application/json",
    }

    try:
        resp = requests.get(search_url, params=params, headers=headers, timeout=15)
        if resp.status_code != 200:
            return []
        search_data = resp.json()
        search_results = search_data.get("query", {}).get("search", [])
        if not search_results:
            return []
    except Exception:
        return []

    page_title = search_results[0].get("title", "")
    extract_url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "titles": page_title,
        "prop": "extracts",
        "exintro": True,
        "explaintext": True,
        "utf8": 1,
    }

    try:
        resp = requests.get(extract_url, params=params, headers=headers, timeout=15)
        if resp.status_code != 200:
            return []
        pages = resp.json().get("query", {}).get("pages", {})
        extract_text = ""
        for page in pages.values():
            extract_text = page.get("extract", "")
            break
    except Exception:
        return []

    if not extract_text:
        return []

    sentences = re.split(r'(?<=[.!?])\s+', extract_text.strip())
    clues = []

    topic_keywords = {
        "algorithm": {
            "design": ["designed", "proposed", "invented", "developed", "created", "published"],
            "structure": ["block", "key", "round", "cipher", "hash", "encryption", "decryption"],
            "security": ["broken", "attack", "collision", "weakness", "vulnerable", "secure"],
            "standard": ["NIST", "standard", "adopted", "approved", "recommended", "ISO"],
            "usage": ["used", "widely", "popular", "TLS", "SSL", "protocol", "deployed"],
        },
        "attack_group": {
            "attribution": ["attributed", "linked", "associated", "nation-state", "government", "sponsored"],
            "operations": ["targeted", "attacked", "compromised", "campaign", "operation"],
            "techniques": ["phishing", "exploit", "malware", "backdoor", "ransomware", "spear"],
            "targets": ["government", "military", "defense", "energy", "financial", "infrastructure"],
            "timeline": ["discovered", "active", "first", "since", "emerged", "identified"],
        },
        "attack_technique": {
            "description": ["technique", "method", "approach", "attack", "exploitation"],
            "execution": ["execute", "inject", "modify", "create", "download", "install"],
            "evasion": ["evade", "bypass", "hide", "obfuscate", "disable", "remove"],
            "impact": ["steal", "encrypt", "destroy", "deny", "disrupt", "exfiltrate"],
            "detection": ["detect", "monitor", "log", "alert", "signature", "indicator"],
        },
        "attack_software": {
            "description": ["tool", "framework", "malware", "trojan", "RAT", "backdoor"],
            "capability": ["capable", "feature", "module", "plugin", "function", "command"],
            "usage": ["used", "deployed", "distributed", "spread", "propagate"],
            "development": ["developed", "created", "written", "coded", "open-source"],
            "detection": ["detected", "identified", "analyzed", "signature", "YARA"],
        },
    }

    keywords = topic_keywords.get(entity_type, {})
    default_topics = ["description", "background", "details", "history", "context"]

    for i, sent in enumerate(sentences):
        sent = sent.strip()
        if not sent or len(sent) < 20:
            continue
        if len(clues) >= 10:
            break

        sent_lower = sent.lower()
        topic = default_topics[min(i, len(default_topics) - 1)]
        for topic_name, words in keywords.items():
            if any(w.lower() in sent_lower for w in words):
                topic = topic_name
                break

        clues.append({
            "fact_id": f"W{i + 1}",
            "fact": sent,
            "topic": topic,
            "source": f"Wikipedia:{page_title}",
        })

    if clues:
        _kg_cache_set(SECURITY_CACHE_DIR, cache_key, clues)
    return clues


# ---------------------------------------------------------------------------
# Category Helpers (unchanged)
# ---------------------------------------------------------------------------

def get_category_entities(category: str) -> List[Dict[str, Any]]:
    """Return list of entities in the entity universe for a category."""
    return ENTITY_UNIVERSE.get(category, [])


def resolve_security_wikidata_id(name: str, wiki_title: str = "") -> Optional[str]:
    """Resolve a security entity name to a Wikidata QID.

    Uses the wiki_title field from ENTITY_UNIVERSE as the search query.
    No P31 validation — security entities have diverse types.

    Args:
        name: Entity name (e.g., "AES-256", "APT28").
        wiki_title: Wikipedia article title for more accurate search.

    Returns:
        Wikidata QID string, or None if not found.
    """
    from .tool_util import search_wikidata_entities

    query = wiki_title or name
    results = search_wikidata_entities(query, limit=5)
    if not results:
        if wiki_title and wiki_title != name:
            results = search_wikidata_entities(name, limit=5)
        if not results:
            return None

    return results[0].get("id", None)
