"""
Full Integration Tests for ImageGuard API Endpoints
"""

import io
from PIL import Image
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

def create_test_image_bytes():
    buf = io.BytesIO()
    img = Image.new("RGB", (100, 100), color=(73, 109, 137))
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf.getvalue()

def test_full_analysis_workflow():
    img_bytes = create_test_image_bytes()

    # 1. Test POST /api/upload
    upload_res = client.post(
        "/api/upload",
        files={"file": ("sample.jpg", img_bytes, "image/jpeg")}
    )
    assert upload_res.status_code == 200
    upload_json = upload_res.json()
    assert upload_json["success"] is True
    assert "upload_id" in upload_json
    upload_id = upload_json["upload_id"]

    # 2. Test POST /api/analyze with file
    headers = {"Authorization": "Bearer dev-test-token"}
    analyze_res = client.post(
        "/api/analyze",
        files={"file": ("sample.jpg", img_bytes, "image/jpeg")},
        headers=headers
    )
    assert analyze_res.status_code == 200
    analyze_data = analyze_res.json()
    assert analyze_data["success"] is True
    record = analyze_data["data"]
    assert "id" in record
    assert "classification" in record
    assert "ai_probability" in record
    assert "real_probability" in record
    assert "confidence" in record
    assert "manipulation" in record
    assert "image_info" in record
    analysis_id = record["id"]

    # 3. Test GET /api/results/{id}
    get_res = client.get(f"/api/results/{analysis_id}")
    assert get_res.status_code == 200
    assert get_res.json()["data"]["id"] == analysis_id

    # 4. Test GET /api/results/{id}/report (PDF download)
    report_res = client.get(f"/api/results/{analysis_id}/report")
    assert report_res.status_code == 200
    assert report_res.headers["content-type"] == "application/pdf"
    assert report_res.content.startswith(b"%PDF")

    # 5. Test GET /api/history (with auth header)
    headers = {"Authorization": "Bearer dev-test-token"}
    history_res = client.get("/api/history?page=1&limit=10", headers=headers)
    assert history_res.status_code == 200
    history_json = history_res.json()
    assert history_json["data"]["total"] >= 1

    # 6. Test DELETE /api/results/{id} (with auth header)
    delete_res = client.delete(f"/api/results/{analysis_id}", headers=headers)
    assert delete_res.status_code == 200

    # Verify deleted
    get_after_delete = client.get(f"/api/results/{analysis_id}", headers=headers)
    assert get_after_delete.status_code == 404
