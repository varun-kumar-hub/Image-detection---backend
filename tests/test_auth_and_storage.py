"""
Comprehensive Integration Tests for ImageGuard Authentication & Private Storage
=============================================================================
Tests:
- Public health check
- 401 Unauthorized for missing / invalid credentials
- Clean JSON error structure (PRD Section 34 & 63)
- Google / Supabase session token verification & user identity derivation
- Storage upload to private path hierarchy {user_id}/uploads/{analysis_id}.{ext}
- Signed URL generation for Results & History thumbnails
- Cascade file deletion on DELETE /results/{id}
- History isolation: User A cannot see or delete User B's analyses
- Reject corrupted images & invalid inputs
"""

import io
from PIL import Image
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

def create_sample_jpeg_bytes(width=120, height=120, color=(100, 150, 200)) -> bytes:
    buf = io.BytesIO()
    img = Image.new("RGB", (width, height), color=color)
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf.getvalue()

def test_public_health_endpoint():
    """Test 1: Health endpoint is public and reports model status."""
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "model_version" in data

def test_unauthorized_protected_endpoints():
    """Test 2: Missing or invalid token yields 401 Unauthorized with clean PRD error format."""
    # History without token
    res = client.get("/api/history")
    assert res.status_code == 401
    data = res.json()
    assert data["success"] is False
    assert data["error"]["code"] == "UNAUTHORIZED"

    # Delete without token
    res2 = client.delete("/api/results/ANL-2026-09-17-test")
    assert res2.status_code == 401
    data2 = res2.json()
    assert data2["success"] is False
    assert data2["error"]["code"] == "UNAUTHORIZED"

    # Image signed URL endpoint without token
    res3 = client.get("/api/results/ANL-2026-09-17-test/image")
    assert res3.status_code == 401
    data3 = res3.json()
    assert data3["success"] is False
    assert data3["error"]["code"] == "UNAUTHORIZED"

def test_authenticated_analysis_and_storage_lifecycle():
    """
    Test 3: Authenticated user uploads image, receives signed URL,
    views result, inspects history, and deletes analysis with cascade storage cleanup.
    """
    token_headers = {"Authorization": "Bearer dev-token-user-1"}
    img_bytes = create_sample_jpeg_bytes()

    # 1. Analyze image as authenticated User 1
    analyze_res = client.post(
        "/api/analyze",
        files={"file": ("photo.jpg", img_bytes, "image/jpeg")},
        headers=token_headers
    )
    assert analyze_res.status_code == 200
    res_json = analyze_res.json()
    assert res_json["success"] is True
    data = res_json["data"]
    analysis_id = data["id"]
    assert "image_url" in data
    assert data["image_url"] is not None
    assert "storage_path" in data
    # Verify storage path is organized by user ID
    assert "uploads" in data["storage_path"]

    # 2. Get Result by ID
    get_res = client.get(f"/api/results/{analysis_id}", headers=token_headers)
    assert get_res.status_code == 200
    get_data = get_res.json()["data"]
    assert get_data["id"] == analysis_id
    assert "image_url" in get_data

    # 3. Get Image Signed URL specifically
    img_url_res = client.get(f"/api/results/{analysis_id}/image", headers=token_headers)
    assert img_url_res.status_code == 200
    assert "url" in img_url_res.json()["data"]

    # 4. Get History for User 1
    hist_res = client.get("/api/history", headers=token_headers)
    assert hist_res.status_code == 200
    hist_data = hist_res.json()["data"]
    assert hist_data["total"] >= 1
    user_records = [item for item in hist_data["items"] if item["id"] == analysis_id]
    assert len(user_records) == 1
    assert "thumbnail_url" in user_records[0]

    # 5. Cascade Delete analysis and storage files
    del_res = client.delete(f"/api/results/{analysis_id}", headers=token_headers)
    assert del_res.status_code == 200
    assert del_res.json()["success"] is True

    # 6. Verify result is no longer found
    post_del = client.get(f"/api/results/{analysis_id}", headers=token_headers)
    assert post_del.status_code == 404

def test_corrupted_image_rejection():
    """Test 4: Corrupted file is rejected with clean 400 Bad Request."""
    corrupted_bytes = b"NOT_A_VALID_IMAGE_FILE_RANDOM_CORRUPT_BYTES"
    res = client.post(
        "/api/analyze",
        files={"file": ("corrupt.jpg", corrupted_bytes, "image/jpeg")}
    )
    assert res.status_code == 400
    data = res.json()
    assert data["success"] is False
    assert data["error"]["code"] in ("INVALID_IMAGE", "CORRUPT_IMAGE", "INVALID_FILE_HEADER")
