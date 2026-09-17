"""
Tests for Image Validators and Integrity Checking
"""

import io
from PIL import Image
from backend.app.utils.validators import ALLOWED_EXTENSIONS, MAX_FILE_SIZE, MAGIC_NUMBERS

def test_allowed_extensions():
    assert "jpg" in ALLOWED_EXTENSIONS
    assert "jpeg" in ALLOWED_EXTENSIONS
    assert "png" in ALLOWED_EXTENSIONS
    assert "webp" in ALLOWED_EXTENSIONS
    assert "exe" not in ALLOWED_EXTENSIONS
    assert "pdf" not in ALLOWED_EXTENSIONS

def test_file_size_limit():
    assert MAX_FILE_SIZE == 25 * 1024 * 1024  # 25 MB

def test_magic_numbers_present():
    assert b"\xff\xd8\xff" in MAGIC_NUMBERS
    assert b"\x89PNG\r\n\x1a\n" in MAGIC_NUMBERS
    assert b"RIFF" in MAGIC_NUMBERS
