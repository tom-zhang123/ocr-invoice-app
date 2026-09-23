"""Single-engine PP-OCRv6 model used by the package text API."""

import os
from threading import Lock

import numpy as np
from rapidocr import EngineType, LangCls, LangDet, LangRec, ModelType, OCRVersion, RapidOCR


_MODEL_ROOT = os.getenv("RAPIDOCR_MODEL_ROOT_DIR", "/opt/rapidocr-models")
_ENGINE = None
_ENGINE_LOCK = Lock()


def get_package_engine():
    global _ENGINE
    if _ENGINE is None:
        with _ENGINE_LOCK:
            if _ENGINE is None:
                _ENGINE = RapidOCR(params={
                    "Global.log_level": "warning",
                    "Global.model_root_dir": _MODEL_ROOT,
                    "Global.text_score": 0.0,
                    "Global.use_cls": False,
                    "Det.engine_type": EngineType.OPENVINO,
                    "Det.lang_type": LangDet.CH,
                    "Det.model_type": ModelType.SMALL,
                    "Det.ocr_version": OCRVersion.PPOCRV6,
                    "Cls.engine_type": EngineType.OPENVINO,
                    "Cls.lang_type": LangCls.CH,
                    "Cls.model_type": ModelType.MOBILE,
                    "Cls.ocr_version": OCRVersion.PPOCRV5,
                    "Rec.engine_type": EngineType.OPENVINO,
                    "Rec.lang_type": LangRec.CH,
                    "Rec.model_type": ModelType.SMALL,
                    "Rec.ocr_version": OCRVersion.PPOCRV6,
                    "Rec.rec_batch_num": 8,
                })
    return _ENGINE


def warmup_package_model():
    get_package_engine()


def recognize_package_image(image):
    engine = get_package_engine()
    result = engine(image, use_det=True, use_cls=False, use_rec=True)
    if result is None or result.boxes is None:
        return []
    return [
        {
            "box": np.asarray(box).tolist(),
            "text": str(text).strip(),
            "confidence": float(score),
        }
        for box, text, score in zip(result.boxes, result.txts, result.scores)
        if str(text).strip()
    ]
