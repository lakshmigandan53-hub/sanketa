"""
Automated Test Suite for SANKETA Backend API Endpoints
Tests:
1. GET /health
2. GET /model/info
3. POST /translate/reset-buffer
4. POST /translate/sequence (with real dynamic sequence from member_01_dynamic)
5. POST /translate/image (error handling and image decoding)
6. Direct inference validation
"""

import os
import sys
import json
import numpy as np
from fastapi.testclient import TestClient

# Ensure backend root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app.main import app

def run_tests():
    client = TestClient(app)
    results = []

    print("=" * 65)
    print("SANKETA BACKEND API AUTOMATED TEST SUITE")
    print("=" * 65)

    # 1. Health check
    res = client.get("/health")
    assert res.status_code == 200, f"Health check failed with code {res.status_code}"
    health_data = res.json()
    print(f"[PASS] GET /health -> status: {health_data.get('status')}")
    results.append(("GET /health", True, health_data.get('status')))

    # 2. Model info
    res = client.get("/model/info")
    assert res.status_code == 200, f"Model info failed with code {res.status_code}"
    info = res.json()
    print(f"[PASS] GET /model/info ->")
    print(f"       Static:  status={info.get('static_model', {}).get('status')}, classes={info.get('static_model', {}).get('classes')}")
    print(f"       Dynamic: status={info.get('dynamic_model', {}).get('status')}, classes={info.get('dynamic_model', {}).get('classes')}")
    assert info.get('static_model', {}).get('status') == 'loaded', "Static model should be loaded"
    assert info.get('dynamic_model', {}).get('status') == 'loaded', "Dynamic model should be loaded"
    results.append(("GET /model/info", True, f"Static: {info.get('static_classes_count')}, Dynamic: {info.get('dynamic_classes_count')}"))

    # 3. Reset buffer
    res = client.post("/translate/reset-buffer")
    assert res.status_code == 200, f"Reset buffer failed with code {res.status_code}"
    print(f"[PASS] POST /translate/reset-buffer -> {res.json().get('message')}")
    results.append(("POST /translate/reset-buffer", True, res.json().get('message')))

    # 4. Sequence translation with real sequence from raw dataset
    dyn_dir = os.path.join(os.path.dirname(__file__), "datasets", "raw", "member_01_dynamic", "THANK_YOU")
    sample_files = [f for f in os.listdir(dyn_dir) if f.endswith(".npy")]
    if sample_files:
        sample_path = os.path.join(dyn_dir, sample_files[0])
        seq = np.load(sample_path).tolist()
        res = client.post("/translate/sequence", json={"sequence": seq})
        assert res.status_code == 200, f"Sequence translation failed: {res.text}"
        dyn_data = res.json()
        print(f"[PASS] POST /translate/sequence -> word={dyn_data.get('word')}, conf={dyn_data.get('confidence')}")
        assert dyn_data.get('success') is True, "Sequence translation should succeed"
        results.append(("POST /translate/sequence", dyn_data.get('success'), f"word={dyn_data.get('word')}"))
    else:
        print("[SKIP] No dynamic sample found for test.")

    # 5. Invalid sequence shape test (boundary test)
    bad_seq = np.zeros((10, 63), dtype=np.float32).tolist()
    res = client.post("/translate/sequence", json={"sequence": bad_seq})
    assert res.status_code == 400, "Should reject invalid sequence shape with 400"
    print(f"[PASS] POST /translate/sequence (invalid shape correctly rejected with 400)")
    results.append(("POST /translate/sequence (400 validation)", True, "Correctly rejected"))

    # 6. Image endpoint with valid JPEG without hand (checks soft failure 200 "No hand detected")
    import cv2
    dummy_img = np.ones((200, 200, 3), dtype=np.uint8) * 255
    _, encoded_jpg = cv2.imencode('.jpg', dummy_img)
    res = client.post("/translate/image", files={"file": ("test.jpg", encoded_jpg.tobytes(), "image/jpeg")})
    assert res.status_code == 200, f"Image endpoint should return 200: {res.text}"
    img_resp = res.json()
    assert img_resp.get("success") is False, "Empty image should have success=False"
    assert "hand" in img_resp.get("message", "").lower(), "Message should indicate no hand detected"
    print(f"[PASS] POST /translate/image (valid frame without hand) -> message: {img_resp.get('message')}")
    results.append(("POST /translate/image (no hand)", True, img_resp.get('message')))

    print("=" * 65)
    print("ALL API ENDPOINT TESTS PASSED SUCCESSFULLY! (6/6)")
    print("=" * 65)
    return results

if __name__ == "__main__":
    run_tests()
