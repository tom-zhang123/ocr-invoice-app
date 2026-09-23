import asyncio
from io import BytesIO
import importlib
import json
import os
import unittest
from unittest.mock import patch

from fastapi import HTTPException, UploadFile
from starlette.datastructures import Headers


JPEG_BYTES = b"\xff\xd8\xff\xe0" + b"test-image" + b"\xff\xd9"


class PackageMainTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["PACKAGE_TEXT_OCR_TOKEN"] = "test-token"
        import package_main

        cls.module = importlib.reload(package_main)

    def test_authenticated_request_returns_no_store_result(self):
        recognized = {
            "manufacture_at": "",
            "expire_at": "2028-01-03",
            "batch": "61840E316B",
            "batch_original": "618405316B",
            "batch_correction": "asn_batch_candidate",
            "raw_text": "03/01/2028\n618405316B",
            "missing_required_fields": [],
            "batch_review_positions": [5],
            "timing_ms": {"total": 400},
            "model": "PP-OCRv6-small",
        }
        with patch.object(
            self.module,
            "recognize_package_text",
            return_value=recognized,
        ):
            response = asyncio.run(self.module.package_text(
                file=self.upload("crop.jpg", JPEG_BYTES, "image/jpeg"),
                required_fields='["expire_at","batch"]',
                batch_candidates='["61840E316B"]',
                x_ocr_token="test-token",
            ))

        self.assertEqual(200, response.status_code)
        self.assertEqual("no-store", response.headers["cache-control"])
        payload = json.loads(response.body)
        self.assertEqual("61840E316B", payload["batch"])
        self.assertTrue(payload["request_id"])

    def test_missing_token_is_rejected_before_recognition(self):
        with self.assertRaises(HTTPException) as raised:
            asyncio.run(self.module.package_text(
                file=self.upload("crop.jpg", JPEG_BYTES, "image/jpeg"),
                required_fields='["batch"]',
                batch_candidates="[]",
                x_ocr_token="",
            ))

        self.assertEqual(401, raised.exception.status_code)

    def test_invalid_fields_and_content_type_are_rejected(self):
        with self.assertRaises(HTTPException) as invalid_type:
            asyncio.run(self.module.package_text(
                file=self.upload("crop.txt", b"text", "text/plain"),
                required_fields='["batch"]',
                batch_candidates="[]",
                x_ocr_token="test-token",
            ))
        with self.assertRaises(HTTPException) as invalid_fields:
            asyncio.run(self.module.package_text(
                file=self.upload("crop.jpg", JPEG_BYTES, "image/jpeg"),
                required_fields='["unknown"]',
                batch_candidates="[]",
                x_ocr_token="test-token",
            ))

        self.assertEqual(415, invalid_type.exception.status_code)
        self.assertEqual(400, invalid_fields.exception.status_code)

    def test_route_is_registered(self):
        paths = {route.path for route in self.module.app.routes}
        self.assertIn("/api/package-text", paths)
        self.assertIn("/health", paths)

    @staticmethod
    def upload(filename, content, content_type):
        return UploadFile(
            file=BytesIO(content),
            filename=filename,
            headers=Headers({"content-type": content_type}),
        )


if __name__ == "__main__":
    unittest.main()
