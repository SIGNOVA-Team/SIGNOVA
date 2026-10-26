"""Make the frozen model's top-1 class available for research integration."""

from collections.abc import Mapping


def as_research_candidate(prediction):
    if not isinstance(prediction, Mapping) or prediction.get("status") != "uncertain":
        raise ValueError("Expected an uncertain KArSL-502 prediction.")
    ranked = prediction.get("top_k")
    if not isinstance(ranked, list) or not ranked or not isinstance(ranked[0], dict):
        raise ValueError("Expected at least one top_k candidate.")
    keys = ("class_id", "label", "label_arabic", "score")
    if any(key not in ranked[0] for key in keys):
        raise ValueError("Incomplete candidate.")
    return {
        "status": "research_candidate",
        "confirmed": False,
        "candidate": {key: ranked[0][key] for key in keys},
        "confidence": prediction["confidence"],
        "top_k": ranked,
        "model_status": "uncertain",
        "model_version": prediction["model_version"],
        "calibration_version": prediction["calibration_version"],
    }


def predict_for_preview(
    predictor,
    *,
    pose_bytes=None,
    pose_object=None,
    features=None,
    preprocessing_version=None,
    top_k=3,
):
    if sum(x is not None for x in (pose_bytes, pose_object, features)) != 1:
        raise ValueError("Provide exactly one of pose_bytes, pose_object, features.")
    if features is not None:
        if preprocessing_version is None:
            raise ValueError("Features require preprocessing_version.")
        raw = predictor.predict_features(features, preprocessing_version, top_k=top_k)
    elif pose_bytes is not None:
        raw = predictor.predict_pose_bytes(pose_bytes, top_k=top_k)
    else:
        raw = predictor.predict_pose_object(pose_object, top_k=top_k)
    return as_research_candidate(raw)
