# -*- coding: utf-8 -*-
"""
Image Byte Anonymizer for Benchmark Integrity
=============================================
Hashes raw image bytes to SHA-256 to ensure zero semantic filename leakage.
"""

import hashlib
import os
from typing import Tuple

def anonymize_image(image_path: str, staging_dir: str) -> Tuple[str, str]:
    """
    Hashes image bytes using SHA-256 and copies to staging_dir as sha256.jpg.
    Returns (anonymized_path, sha256_hash).
    """
    os.makedirs(staging_dir, exist_ok=True)
    with open(image_path, "rb") as f:
        data = f.read()
    img_hash = hashlib.sha256(data).hexdigest()
    dst_path = os.path.join(staging_dir, f"{img_hash}.jpg")
    if not os.path.exists(dst_path):
        with open(dst_path, "wb") as f:
            f.write(data)
    return dst_path, img_hash
