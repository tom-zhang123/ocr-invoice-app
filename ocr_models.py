"""OCR model selection and compatibility helpers for RapidOCR 3.x."""
import os
from threading import Lock

import numpy as np
from rapidocr import (
    EngineType,
    LangCls,
    LangDet,
    LangRec,
    ModelType,
    OCRVersion,
    RapidOCR,
)


_MODEL_ROOT = os.getenv("RAPIDOCR_MODEL_ROOT_DIR", "/opt/rapidocr-models")
_GENERAL_ENGINE = None
_HANDWRITING_ENGINE = None
_ENGINE_LOCK = Lock()


def _common_params():
    return {
        "Global.log_level": "warning",
        "Global.model_root_dir": _MODEL_ROOT,
        "Global.text_score": 0.0,
        "Det.engine_type": EngineType.ONNXRUNTIME,
        "Det.lang_type": LangDet.CH,
        "Det.model_type": ModelType.SMALL,
        "Det.ocr_version": OCRVersion.PPOCRV6,
        "Cls.engine_type": EngineType.ONNXRUNTIME,
        "Cls.lang_type": LangCls.CH,
        "Cls.model_type": ModelType.MOBILE,
        "Cls.ocr_version": OCRVersion.PPOCRV5,
    }


def get_general_engine():
    """PP-OCRv6 small model for whole-page and cropped-field recognition."""
    global _GENERAL_ENGINE
    if _GENERAL_ENGINE is None:
        with _ENGINE_LOCK:
            if _GENERAL_ENGINE is None:
                params = _common_params()
                params.update({
                    "Det.engine_type": EngineType.OPENVINO,
                    "Det.model_type": ModelType.SMALL,
                    "Det.ocr_version": OCRVersion.PPOCRV6,
                    "Cls.engine_type": EngineType.OPENVINO,
                    "Rec.engine_type": EngineType.OPENVINO,
                    "Rec.lang_type": LangRec.CH,
                    "Rec.model_type": ModelType.SMALL,
                    "Rec.ocr_version": OCRVersion.PPOCRV6,
                    "Rec.rec_batch_num": 16,
                })
                _GENERAL_ENGINE = RapidOCR(params=params)
    return _GENERAL_ENGINE


def get_handwriting_engine():
    """Reuse the v6 engine because OCR calls are serialized by the API."""
    return _HANDWRITING_ENGINE or get_general_engine()


def warmup_models():
    """Resolve model files during image build and initialize them during app startup."""
    get_general_engine()
    get_handwriting_engine()


def _recognize_region(image, engine):
    result = engine(image, use_det=True, use_cls=False, use_rec=True)
    if result is None or result.boxes is None:
        return []

    return [
        [np.asarray(box).tolist(), str(text), float(score)]
        for box, text, score in zip(result.boxes, result.txts, result.scores)
    ]


def recognize_page(image):
    return _recognize_region(image, get_general_engine())


def recognize_region(image, language="general"):
    if image is None or image.size == 0:
        return []

    if language == "handwriting":
        engine = get_handwriting_engine()
    else:
        engine = get_general_engine()
    return _recognize_region(image, engine)


def recognize_lines(images, language="handwriting"):
    """Recognize pre-cropped text lines as one batch and preserve input order."""
    if not images:
        return []
    engine = get_handwriting_engine() if language == "handwriting" else get_general_engine()
    result = engine.recognize_txt(images)
    return [
        (str(text).strip(), float(score))
        for text, score in zip(result.txts, result.scores)
    ]
