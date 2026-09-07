# -*- coding: utf-8 -*-
"""
SANKETA API Verification Tests
backend/test_api.py

Tests all endpoints against a running Uvicorn server on localhost:8000.
Run from the backend/ directory:
  .\\venv\\Scripts\\python.exe test_api.py
"""
import os, sys
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import urllib.request
import urllib.error
import json
import io

BASE = "http://localhost:8000"
PASS = "[PASS]"
FAIL = "[FAIL]"
results = []

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get(path: str):
    url = f"{BASE}{path}"
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            # urllib lowercases header names
            return r.status, r.read().decode("ascii", errors="replace"), {k.lower(): v for k, v in r.headers.items()}
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("ascii", errors="replace"), {}


def post_file(path: str, field: str, filename: str, data: bytes, content_type: str):
    boundary = b"----SANKETA_BOUNDARY"
    body = (
        b"--" + boundary + b"\r\n"
        + f'Content-Disposition: form-data; name="{field}"; filename="{filename}"\r\n'.encode()
        + f"Content-Type: {content_type}\r\n\r\n".encode()
        + data
        + b"\r\n--" + boundary + b"--\r\n"
    )
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=body,
        method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary.decode()}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, r.read().decode("ascii", errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("ascii", errors="replace")


def check(label, condition, info=""):
    tag = PASS if condition else FAIL
    msg = f"{tag} {label}"
    if info:
        msg += f"\n       {info}"
    print(msg)
    results.append(condition)


# ---------------------------------------------------------------------------
# Valid white JPEG (320x240) — no hand present, used for soft-failure test.
# Generated with OpenCV so it is guaranteed decodable.
# ---------------------------------------------------------------------------
import cv2
import numpy as _np
_white = _np.ones((240, 320, 3), dtype=_np.uint8) * 255
_, _enc = cv2.imencode(".jpg", _white, [cv2.IMWRITE_JPEG_QUALITY, 95])
WHITE_PNG = _enc.tobytes()  # variable kept as WHITE_PNG for backwards compat


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

print()
print("=" * 60)
print("  SANKETA API Verification Tests")
print("=" * 60)
print()

# -- Test 1: Health
status, body, _ = get("/health")
data = json.loads(body) if body else {}
check(
    "Test 1 — GET /health",
    status == 200 and data.get("status") == "healthy",
    f"Status: {status}  Body: {body[:120]}",
)

# -- Test 2: Swagger UI
status, body, headers = get("/docs")
check(
    "Test 2 — GET /docs (Swagger UI)",
    status == 200 and "html" in headers.get("content-type", "").lower(),
    f"Status: {status}  Content-Type: {headers.get('content-type')}",
)

# -- Test 3: ReDoc
status, body, _ = get("/redoc")
check("Test 3 — GET /redoc", status == 200, f"Status: {status}")

# -- Test 4: Model info endpoint
status, body, _ = get("/model/info")
data = json.loads(body) if body else {}
check(
    "Test 4 — GET /model/info",
    status == 200
    and data.get("model_loaded") is True
    and data.get("input_features") == 63
    and data.get("supported_classes") == 8
    and len(data.get("classes", [])) == 8,
    f"Status: {status}  Body: {body[:200]}",
)

# -- Test 5: Image with no hand (white 1x1 PNG)
status, body = post_file("/translate/image", "file", "test.png", WHITE_PNG, "image/png")
data = json.loads(body) if body else {}
check(
    "Test 5 — POST /translate/image (no hand -> 200 success:false)",
    status == 200 and data.get("success") is False,
    f"Status: {status}  Body: {body[:120]}",
)

# -- Test 6: Garbage bytes -> 400
status, body = post_file("/translate/image", "file", "bad.jpg", b"\x00\x01\x02\x03", "image/jpeg")
check(
    "Test 6 — POST /translate/image (corrupt bytes -> 400)",
    status == 400,
    f"Status: {status}  Body: {body[:120]}",
)

# -- Test 7: Unsupported MIME type (video/mp4) -> 400
status, body = post_file("/translate/image", "file", "vid.mp4", b"\x00\x01\x02", "video/mp4")
check(
    "Test 7 — POST /translate/image (video/mp4 -> 400)",
    status == 400,
    f"Status: {status}  Body: {body[:120]}",
)

# -- Test 8: Empty file -> 400
status, body = post_file("/translate/image", "file", "empty.jpg", b"", "image/jpeg")
check(
    "Test 8 — POST /translate/image (empty file -> 400)",
    status == 400,
    f"Status: {body[:120]}",
)

# -- Test 9: Model info classes 8-word check
status, body, _ = get("/model/info")
data = json.loads(body) if body else {}
expected = ["HELLO", "HELP", "NO", "PLEASE", "SORRY", "THANK_YOU", "WATER", "YES"]
check(
    "Test 9 — GET /model/info classes == 8 ISL words",
    data.get("classes") == expected,
    f"Classes: {data.get('classes')}",
)

# -- Test 10: Dynamic sequence translation endpoint
import numpy as _np
dummy_seq = _np.ones((30, 63), dtype=_np.float32).tolist()
req_seq = urllib.request.Request(
    f"{BASE}/translate/sequence",
    data=json.dumps({"sequence": dummy_seq}).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
try:
    with urllib.request.urlopen(req_seq, timeout=10) as r:
        seq_status = r.status
        seq_data = json.loads(r.read().decode("utf-8"))
except Exception as e:
    seq_status = 500
    seq_data = {}

check(
    "Test 10 — POST /translate/sequence (30-frame sequence)",
    seq_status == 200 and seq_data.get("success") is True and seq_data.get("type") == "dynamic",
    f"Status: {seq_status} Data: {seq_data}",
)

# ---------------------------------------------------------------------------
print()
passed = sum(results)
total = len(results)
if passed == total:
    print(f"  ALL {total}/{total} TESTS PASSED")
else:
    print(f"  {passed}/{total} TESTS PASSED  ({total - passed} FAILED)")
print("=" * 60)
print()

sys.exit(0 if passed == total else 1)
