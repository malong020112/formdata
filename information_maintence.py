import argparse
import concurrent.futures
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from config import API_KEY, BASE_URL, MODEL_NAME

# Configuration
DEFAULT_MAX_WORKERS = int(os.getenv("IM_MAX_WORKERS", "4"))

LLM = OpenAI(api_key=API_KEY, base_url=BASE_URL)


SYSTEM_PROMPT = """
You are a form data analysis assistant. There are two forms: Form A and Form B. 
Form A is the user's personal information table, and Form B is a specific type of form that the same user has filled out. 
You need to compare these two forms and extract the key personal information of the user that exists in Form B but not in Form A. 
Do NOT extract any entries in Form B whose values are "N/A". 

Note: If a certain item (itemA) in Form A and a certain item (itemB) in Form B have different keys but identical content (e.g., itemA is "phone": "123", itemB is "mobile number": "123"), such items do NOT need to be extracted. 

Return all the extracted items to me in JSON format.

Definition of key user personal information:
1. It refers to the user's identity information (or that of their family members), which can be reused when filling out other forms, not limited to the current form.
2. It does NOT include open-ended descriptive fields, scenario-specific fields, or fields that are prone to short-term changes.
3. It does NOT include honorifics, information about whether a person is alive, etc.

Note: You only need to extract the key personal information that exists in Form B but not in Form A.

Next, I will provide you with Form A and Form B.
"""


def _extract_json_from_text(content: str) -> Any:
    """Extract a JSON object from the model response."""
    if not content:
        raise ValueError("Empty response from model")

    content = content.strip()

    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content, re.DOTALL)
    if fence_match:
        content = fence_match.group(1).strip()

    first_curly = content.find("{")
    if first_curly != -1:
        last_curly = content.rfind("}")
        if last_curly > first_curly:
            content = content[first_curly : last_curly + 1]

    return json.loads(content)


def load_records(path: Path) -> Tuple[List[Dict[str, Any]], bool]:
    """Load records from JSON array or NDJSON; return records and is_array flag."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        raise ValueError(f"File is empty: {path}")
    if raw.startswith("["):
        data = json.loads(raw)
        if not isinstance(data, list):
            raise ValueError(f"Expected JSON array in: {path}")
        return data, True
    records = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records, False


def save_records(path: Path, records: List[Dict[str, Any]], as_array: bool) -> None:
    """Save records as JSON array or NDJSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if as_array:
        path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        return
    with path.open("w", encoding="utf-8") as f:
        for idx, record in enumerate(records):
            if idx:
                f.write("\n")
            f.write(json.dumps(record, ensure_ascii=False))


def build_prompt(form_a: Dict[str, Any], form_b: Dict[str, Any]) -> str:
    form_a_text = json.dumps(form_a, ensure_ascii=False, indent=2)
    form_b_text = json.dumps(form_b, ensure_ascii=False, indent=2)
    return f"Form A:\n{form_a_text}\n\nForm B:\n{form_b_text}"


def extract_details_update(
    form_a: Dict[str, Any],
    form_b: Dict[str, Any],
    max_retries: int = 3,
    retry_delay: float = 1.5,
) -> Dict[str, Any]:
    """Call the LLM to extract key personal info from Form B missing in Form A."""
    last_error: Optional[Exception] = None
    user_prompt = build_prompt(form_a, form_b)
    for attempt in range(1, max_retries + 1):
        try:
            response = LLM.responses.create(
                model=MODEL_NAME,
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                # temperature=0.2,
                text={"format": {"type": "json_object"}},
            )
            message_content = ""
            output_items = getattr(response, "output", None) or []
            for item in output_items:
                for c in getattr(item, "content", []) or []:
                    if getattr(c, "type", "") == "output_text":
                        message_content += c.text or ""
            if not message_content:
                message_content = getattr(response, "output_text", "") or ""
            parsed = _extract_json_from_text(message_content or "")
            if not isinstance(parsed, dict):
                raise ValueError("Model response is not a JSON object")
            return parsed
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < max_retries and retry_delay > 0:
                delay = retry_delay * (2 ** (attempt - 1))
                time.sleep(delay)

    raise RuntimeError(f"Failed to extract details after {max_retries} attempts: {last_error}")


def run_maintenance(
    form_a_path: Path,
    form_b_path: Path,
    target_form_type: str,
    max_workers: int = DEFAULT_MAX_WORKERS,
    allowed_uids: Optional[List[int]] = None,
) -> int:
    form_a_records, form_a_is_array = load_records(form_a_path)
    form_b_records, _ = load_records(form_b_path)

    form_a_by_uid: Dict[int, Dict[str, Any]] = {}
    for rec in form_a_records:
        try:
            uid_val = int(rec.get("uid"))
        except Exception:
            continue
        form_a_by_uid[uid_val] = rec

    work_items: List[Tuple[int, Dict[str, Any], Dict[str, Any]]] = []
    allowed_set = set(allowed_uids or [])
    for rec in form_b_records:
        try:
            uid_val = int(rec.get("uid"))
        except Exception:
            continue
        if allowed_set and uid_val not in allowed_set:
            continue
        form_a_rec = form_a_by_uid.get(uid_val)
        if not form_a_rec:
            continue
        details = form_a_rec.get("details")
        if not isinstance(details, dict):
            details = {}
        forms = rec.get("forms")
        if not isinstance(forms, list):
            continue
        matched = None
        for item in forms:
            if not isinstance(item, dict):
                continue
            if item.get("type") == target_form_type:
                matched = item
                break
        if not matched:
            continue
        fields = matched.get("data")
        if not isinstance(fields, dict):
            continue
        work_items.append((uid_val, form_a_rec, {"details": details, "fields": fields}))

    if not work_items:
        print("[warn] no matching uid records found between Form A and Form B.")
        return 0

    results: List[Tuple[int, Dict[str, Any]]] = []
    max_workers = max(1, min(max_workers, len(work_items)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(
                extract_details_update,
                item[2]["details"],
                item[2]["fields"],
            ): item[0]
            for item in work_items
        }
        for future in concurrent.futures.as_completed(future_map):
            uid_val = future_map[future]
            result = future.result()
            results.append((uid_val, result))
            print(f"[progress] updated uid {uid_val}")

    for uid_val, update in results:
        target = form_a_by_uid.get(uid_val)
        if not target:
            continue
        details = target.get("details")
        if not isinstance(details, dict):
            details = {}
        details.update(update)
        target["details"] = details

    save_records(form_a_path, form_a_records, form_a_is_array)
    print(f"[done] updated details for {len(results)} records -> {form_a_path}")
    return len(results)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--form-a-path", required=True, help="Path to Form A file (details field).")
    parser.add_argument("--form-b-path", required=True, help="Path to Form B file (forms list).")
    parser.add_argument("--form-type", required=True, help="Target form type to extract from Form B.")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help="Max concurrent workers (default from IM_MAX_WORKERS).",
    )
    args = parser.parse_args()

    run_maintenance(
        form_a_path=Path(args.form_a_path),
        form_b_path=Path(args.form_b_path),
        target_form_type=args.form_type.strip(),
        max_workers=args.max_workers,
    )


if __name__ == "__main__":
    main()
