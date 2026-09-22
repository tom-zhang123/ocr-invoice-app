"""Compare an OCR model version against externally supplied quantity labels."""
import argparse
import json
import time

import ocr_models
from enhanced_ocr import recognize_enhanced
from rapidocr import EngineType, LangDet, LangRec, ModelType, OCRVersion, RapidOCR


def configure_model(version):
    if version == "v5":
        general_params = ocr_models._common_params()
        general_params.update({
            "Det.engine_type": EngineType.OPENVINO,
            "Det.lang_type": LangDet.CH,
            "Det.model_type": ModelType.MOBILE,
            "Det.ocr_version": OCRVersion.PPOCRV5,
            "Cls.engine_type": EngineType.OPENVINO,
            "Rec.engine_type": EngineType.OPENVINO,
            "Rec.lang_type": LangRec.EN,
            "Rec.model_type": ModelType.MOBILE,
            "Rec.ocr_version": OCRVersion.PPOCRV5,
            "Rec.rec_batch_num": 6,
        })
        handwriting_params = dict(general_params)
        handwriting_params.update({
            "Det.engine_type": EngineType.ONNXRUNTIME,
            "Det.model_type": ModelType.SERVER,
            "Rec.engine_type": EngineType.ONNXRUNTIME,
            "Rec.rec_batch_num": 32,
        })
        ocr_models._GENERAL_ENGINE = RapidOCR(params=general_params)
        ocr_models._HANDWRITING_ENGINE = RapidOCR(params=handwriting_params)
        return
    model_type = ModelType.MEDIUM if version == "v6-medium" else ModelType.SMALL
    params = ocr_models._common_params()
    params.update({
        "Det.engine_type": EngineType.OPENVINO,
        "Det.lang_type": LangDet.CH,
        "Det.model_type": model_type,
        "Det.ocr_version": OCRVersion.PPOCRV6,
        "Rec.engine_type": EngineType.OPENVINO,
        "Rec.lang_type": LangRec.CH,
        "Rec.model_type": model_type,
        "Rec.ocr_version": OCRVersion.PPOCRV6,
        "Rec.rec_batch_num": 16,
    })
    engine = RapidOCR(params=params)
    ocr_models._GENERAL_ENGINE = engine
    ocr_models._HANDWRITING_ENGINE = engine


def parse_case(value):
    image_path, separator, quantities = value.partition("=")
    if not separator or not image_path or not quantities:
        raise argparse.ArgumentTypeError("case must be IMAGE=QTY,QTY,...")
    try:
        expected = [int(value) for value in quantities.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("all quantities must be integers") from exc
    return image_path, expected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("v5", "v6-small", "v6-medium"), required=True)
    parser.add_argument("--case", action="append", type=parse_case, required=True)
    args = parser.parse_args()
    configure_model(args.model)

    results = []
    for image_path, expected in args.case:
        started_at = time.perf_counter()
        result = recognize_enhanced(image_path)
        elapsed = time.perf_counter() - started_at
        actual = [int(row.get("qty") or 0) for row in result["rows"]]
        pair_count = min(len(actual), len(expected))
        mismatches = [
            {"row": index + 1, "expected": expected[index], "actual": actual[index]}
            for index in range(pair_count)
            if actual[index] != expected[index]
        ]
        results.append({
            "image": image_path,
            "model": args.model,
            "elapsed_seconds": round(elapsed, 2),
            "layout": result.get("image", {}).get("table_layout"),
            "actual_rows": len(actual),
            "expected_rows": len(expected),
            "qty_exact_by_position": pair_count - len(mismatches),
            "missing_or_extra_rows": abs(len(actual) - len(expected)),
            "mismatches": mismatches,
        })
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
