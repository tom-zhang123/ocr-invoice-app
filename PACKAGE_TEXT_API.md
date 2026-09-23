# Package Text OCR API

This is a separate lightweight service for package production dates, expiry dates, and batch
numbers. It does not import the invoice table pipeline, spreadsheet export, or identifier model.

## Start

Create a local `.env` file from `package.env.example`, then run:

```bash
docker compose --env-file .env -f docker-compose.package.yml up -d --build
```

The container port is bound to host loopback only: `127.0.0.1:8005`. TWMS should call:

```text
POST http://127.0.0.1:8005/api/package-text
X-OCR-Token: <shared internal token>
```

Multipart fields:

- `file`: JPEG or PNG crop, at most 1 MiB by default.
- `required_fields`: JSON array containing `manufacture_at`, `expire_at`, and/or `batch`.
- `batch_candidates`: optional JSON array from the selected ASN lines.

Images are decoded in memory and are not written to disk. The first recognition uses the original
crop. A single contrast-enhanced retry runs only when a required field is missing.

## Response

```json
{
  "manufacture_at": "2026-04-01",
  "expire_at": "2028-03-21",
  "batch": "61840E316B",
  "batch_original": "618405316B",
  "batch_correction": "asn_batch_candidate",
  "raw_text": "...",
  "missing_required_fields": [],
  "batch_review_positions": [5],
  "timing_ms": {"decode": 5, "primary_ocr": 420, "enhanced_ocr": 0, "total": 426},
  "model": "PP-OCRv6-small",
  "request_id": "..."
}
```

TWMS remains the only PDA-facing endpoint. Do not expose this container port publicly.
