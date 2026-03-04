import json
import re
from datetime import datetime
from typing import Any, Dict


OPEN_FIELD_KEYWORDS = (
    "details",
    "description",
    "reason",
    "remarks",
    "remark",
    "other_relevant_information",
    "additional_information",
    "abstract",
    "rationale",
    "objectives",
    "problems",
    "feasibility",
    "innovations",
    "expected_results",
    "justification",
    "notes",
    "attachments",
    "participants",
    "keywords",
    "signature",
)


def _flatten(data: Any, prefix: str = "") -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    if isinstance(data, dict):
        for key, value in data.items():
            new_prefix = f"{prefix}.{key}" if prefix else str(key)
            flat.update(_flatten(value, new_prefix))
        return flat
    flat[prefix] = data
    return flat


def _parse_date(text: str) -> str:
    s = text.strip()
    if not s:
        return ""
    if "T" in s:
        s = s.split("T", 1)[0]

    formats = (
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%Y.%m.%d",
        "%d-%m-%Y",
        "%d/%m/%Y",
        "%d %b %Y",
        "%d %B %Y",
    )
    for fmt in formats:
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return ""


def _normalize_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value).strip()
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)

    s = re.sub(r"\s+", " ", str(value).strip())
    if not s:
        return ""

    low = s.lower()
    if low in {"n/a", "na", "not applicable", "none", "not provided", "not available", "unknown"}:
        return "n/a"
    if low in {"yes", "y", "true", "1"}:
        return "yes"
    if low in {"no", "n", "false", "0", "not_required", "not required"}:
        return "no"
    if low in {"m", "male"}:
        return "male"
    if low in {"f", "female"}:
        return "female"

    d = _parse_date(s)
    if d:
        return f"date:{d}"

    if re.fullmatch(r"[+\d\-\s\(\)]+", s) and sum(ch.isdigit() for ch in s) >= 7:
        digits = "".join(ch for ch in s if ch.isdigit())
        return f"phone:{digits}"

    return low


def _is_open_field(field_key: str, ground_truth_value: Any) -> bool:
    key = field_key.lower()
    if any(token in key for token in OPEN_FIELD_KEYWORDS):
        return True
    if isinstance(ground_truth_value, (list, dict)):
        return True
    if isinstance(ground_truth_value, str) and "\n" in ground_truth_value:
        return True
    return False


def evaluate_form_state_against_ground_truth(form_state: Dict[str, Any], ground_truth_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare final form_state with ground truth and count correctly filled fields.
    Open-ended fields are skipped.
    """
    pred_flat = _flatten(form_state or {})
    gt_flat = _flatten(ground_truth_data or {})

    correct = 0
    compared = 0
    skipped_open = 0

    for key, gt_val in gt_flat.items():
        if not key:
            continue
        if _is_open_field(key, gt_val):
            skipped_open += 1
            continue
        compared += 1
        pred_val = pred_flat.get(key)
        if _normalize_value(pred_val) == _normalize_value(gt_val):
            correct += 1

    accuracy = round((correct / compared), 4) if compared > 0 else 0.0
    return {
        "correct_filled_fields": correct,
        "compared_fields": compared,
        "skipped_open_fields": skipped_open,
        "field_accuracy": accuracy,
    }

