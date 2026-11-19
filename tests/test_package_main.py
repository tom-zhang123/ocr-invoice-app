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
        self.assertIn("/api/text", paths)

    def test_general_text_needs_no_business_fields(self):
        with patch.object(self.module, "recognize_text", return_value={"raw_text": "普通文字", "dates": []}) as recognizer:
            upload = self.upload("crop.jpg", JPEG_BYTES, "image/jpeg")
            response = asyncio.run(self.module.text_ocr(file=upload, date_order="auto", x_ocr_token="test-token"))
        recognizer.assert_called_once_with(JPEG_BYTES, "auto", text_mode="paragraph")
        self.assertEqual("普通文字", json.loads(response.body)["raw_text"])
        self.assertEqual("no-store", response.headers["cache-control"])
        self.assertTrue(upload.file.closed)

    def test_general_text_rejects_auth_size_and_invalid_order_before_inference(self):
        for token, order, content, status in [
            ("bad-token", "auto", JPEG_BYTES, 401),
            ("test-token", "unknown", JPEG_BYTES, 400),
            ("test-token", "auto", b"x" * (self.module.MAX_IMAGE_BYTES + 1), 413),
        ]:
            with self.subTest(status=status), patch.object(self.module, "recognize_text") as recognizer:
                with self.assertRaises(HTTPException) as raised:
                    asyncio.run(self.module.text_ocr(file=self.upload("crop.jpg", content, "image/jpeg"), date_order=order, x_ocr_token=token))
                self.assertEqual(status, raised.exception.status_code)
                recognizer.assert_not_called()

    def test_single_line_mode_is_forwarded_and_invalid_mode_is_rejected(self):
        for mode in ["identifier", "line"]:
            with patch.object(self.module, "recognize_text", return_value={"raw_text": "AS123", "dates": []}) as recognizer:
                asyncio.run(self.module.text_ocr(file=self.upload("crop.jpg", JPEG_BYTES, "image/jpeg"), date_order="auto", x_ocr_token="test-token", text_mode=mode))
                recognizer.assert_called_once_with(JPEG_BYTES, "auto", text_mode=mode)
        with patch.object(self.module, "recognize_text") as recognizer:
            with self.assertRaises(HTTPException) as raised:
                asyncio.run(self.module.text_ocr(file=self.upload("crop.jpg", JPEG_BYTES, "image/jpeg"), date_order="auto", x_ocr_token="test-token", text_mode="invalid"))
            self.assertEqual(400, raised.exception.status_code)
            recognizer.assert_not_called()

    @staticmethod
    def upload(filename, content, content_type):
        return UploadFile(
            file=BytesIO(content),
            filename=filename,
            headers=Headers({"content-type": content_type}),
        )


if __name__ == "__main__":
    unittest.main()
