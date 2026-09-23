"""Dedicated package text OCR API. It does not load the invoice pipeline."""

import asyncio
from contextlib import asynccontextmanager
import json
import logging
import os
import secrets
import uuid

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from package_ocr import recognize_package_text
from package_ocr_models import warmup_package_model
from package_text_parser import parse_batch_candidates, parse_required_fields


LOGGER = logging.getLogger("package-text-ocr")
MAX_IMAGE_BYTES = int(os.getenv("PACKAGE_TEXT_OCR_MAX_IMAGE_BYTES", str(1024 * 1024)))
CONCURRENCY = max(1, int(os.getenv("PACKAGE_TEXT_OCR_CONCURRENCY", "1")))
INTERNAL_TOKEN = os.getenv("PACKAGE_TEXT_OCR_TOKEN", "").strip()
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png"}

@asynccontextmanager
async def lifespan(_app):
    await run_in_threadpool(warmup_package_model)
    yield


app = FastAPI(title="Package Text OCR", version="1.0", lifespan=lifespan)
_ocr_semaphore = asyncio.Semaphore(CONCURRENCY)


@app.get("/health")
def health():
    return {"status": "ok", "model": "PP-OCRv6-small"}


@app.post("/api/package-text")
async def package_text(
    file: UploadFile = File(...),
    required_fields: str = Form(...),
    batch_candidates: str = Form("[]"),
    x_ocr_token: str = Header("", alias="X-OCR-Token"),
):
    _verify_token(x_ocr_token)
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail="Only JPEG and PNG images are supported")

    request_id = str(uuid.uuid4())
    try:
        fields = parse_required_fields(required_fields)
        candidates = parse_batch_candidates(batch_candidates)
        raw = await file.read(MAX_IMAGE_BYTES + 1)
        if len(raw) > MAX_IMAGE_BYTES:
            raise HTTPException(status_code=413, detail="Image exceeds the configured size limit")
        async with _ocr_semaphore:
            result = await run_in_threadpool(
                recognize_package_text,
                raw,
                fields,
                candidates,
            )
        result["request_id"] = request_id
        return JSONResponse(result, headers={"Cache-Control": "no-store"})
    except HTTPException:
        raise
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except Exception:
        LOGGER.exception("Package OCR failed request_id=%s", request_id)
        raise HTTPException(status_code=500, detail="Package OCR failed")
    finally:
        await file.close()


def _verify_token(provided):
    if not INTERNAL_TOKEN:
        return
    if not provided or not secrets.compare_digest(INTERNAL_TOKEN, provided):
        raise HTTPException(status_code=401, detail="Invalid internal OCR token")


def serialize_form_list(values):
    """Shared helper for contract tests and proxy implementations."""
    return json.dumps(list(values), ensure_ascii=True, separators=(",", ":"))
