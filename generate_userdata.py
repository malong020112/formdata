import argparse
import concurrent.futures
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from information_maintence import run_maintenance

from config import API_KEY, BASE_URL, MODEL_NAME

DEFAULT_RECORD_COUNT = int(os.getenv("FORM_RECORD_COUNT", os.getenv("SCHENGEN_USER_COUNT", "5")))

SCHENGEN_COUNTRIES = {
    "austria",
    "belgium",
    "croatia",
    "czech republic",
    "denmark",
    "estonia",
    "finland",
    "france",
    "germany",
    "greece",
    "hungary",
    "iceland",
    "italy",
    "latvia",
    "liechtenstein",
    "lithuania",
    "luxembourg",
    "malta",
    "netherlands",
    "norway",
    "poland",
    "portugal",
    "slovakia",
    "slovenia",
    "spain",
    "sweden",
    "switzerland",
}

SYSTEM_PROMPT = """
You are a multi-form synthetic data generation engine.

Task: Generate ONE realistic, internally consistent applicant filling ONE form instance (any form type: visa / passport / medical / customs / tax-finance / education / grant, etc.).
Return ONLY valid JSON with root keys:
- "type": form title
- "data": the filled form object (must follow the provided template exactly)

CORE RULES:
1) Identity consistency: Keep all personal/identity/contact details consistent with any existing forms for the same applicant (e.g., name, DOB, gender, nationality, ID/passport number, phone, email).
2) Cross-form coherence: Avoid conflicts across forms and events.
   - Travel-related forms: do not overlap trip dates with existing itineraries; arrival < departure; trips should be in the future unless explicitly historical.
   - Passport/ID-related forms: issue date < expiry date; passport validity coherent with travel dates.
   - Medical-related forms: admission date <= discharge date; treatment dates must be plausible.
   - Education-related forms: enrollment dates must match age/degree level (avoid unrealistic cases).
   - Tax/finance-related forms: amounts, income sources, and ownership details must be plausible and non-contradictory.
3) Realism: Use human-like, real-world values. Dates must be plausible (birth in the past; timelines logical). Addresses, institutions, employers, and relationships should look realistic.
4) Template fidelity: Do NOT add, remove, or rename any fields included in the provided template. Every field in the template must be fully completed; if a field is not applicable, fill it with "N/A".
5) Field constraints: Respect field-specific hints (enums, formats, length, ID patterns).Prefer ISO dates (YYYY-MM-DD) unless the template shows another format.
6) Conciseness for open-ended descriptive fields: For any open-ended declarative fields, the content must be concisely stated in 3-5 sentences only, without lengthy elaboration.
7) Output purity: Output must be pure JSON (no markdown, no comments, no extra text).

"""


LLM = OpenAI(api_key=API_KEY, base_url=BASE_URL)


FORM_META: Dict[str, Dict[str, str]] = {}


def _extract_json_from_text(content: str) -> Any:
    """Extract a JSON value (object) from the model response."""
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


def infer_form_meta_from_path(json_path: Path) -> Tuple[str, str]:
    stem = json_path.stem
    if stem.endswith("_form"):
        stem = stem[: -len("_form")]
    if stem.endswith("_visa"):
        return "visa", stem[: -len("_visa")].lower()
    if stem.endswith("_passport"):
        return "passport", stem[: -len("_passport")].lower()
    return "", ""


def load_form_templates(root: Path) -> Dict[str, Dict[str, Any]]:
    """Load all *_form.json templates under the form directory."""
    if not root.exists():
        raise FileNotFoundError(f"Form root not found: {root}")

    templates: Dict[str, Dict[str, Any]] = {}
    for json_path in sorted(root.rglob("*_form.json")):
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] failed to load template {json_path}: {exc}")
            continue

        form_name = str(data.get("name") or "").strip()
        if not form_name:
            form_name = json_path.stem.replace("_form", "")
        if form_name in templates:
            print(f"[warn] duplicate form name '{form_name}' from {json_path}")
        templates[form_name] = data
        form_type, form_country = infer_form_meta_from_path(json_path)
        if form_type:
            FORM_META[form_name] = {"type": form_type, "country": form_country}

    if not templates:
        raise ValueError(f"No form templates found under: {root}")

    return templates


def load_single_form_template(form_path: Path) -> Dict[str, Dict[str, Any]]:
    """Load one form template from a path and return {form_name: template}."""
    if not form_path.exists():
        raise FileNotFoundError(f"Form template not found: {form_path}")
    data = json.loads(form_path.read_text(encoding="utf-8"))
    form_name = str(data.get("name") or "").strip()
    if not form_name:
        form_name = form_path.stem.replace("_form", "")
    form_type, form_country = infer_form_meta_from_path(form_path)
    if form_type:
        FORM_META[form_name] = {"type": form_type, "country": form_country}
    return {form_name: data}


def load_user_profiles(profile_path: Path) -> List[Dict[str, Any]]:
    """Load user profiles from a JSON array or newline-delimited JSON file and align fields."""
    if not profile_path.exists():
        raise FileNotFoundError(f"Profile file not found: {profile_path}")

    raw = profile_path.read_text(encoding="utf-8").strip()
    if not raw:
        raise ValueError(f"Profile file is empty: {profile_path}")

    if raw.startswith("["):
        data = json.loads(raw)
        if not isinstance(data, list):
            raise ValueError("Profile file must contain a JSON array of objects")
        profiles = data
    else:
        profiles = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            profiles.append(json.loads(line))

    if not profiles:
        raise ValueError(f"No profiles loaded from: {profile_path}")

    return profiles


def load_existing_user_data(data_path: Path) -> List[Dict[str, Any]]:
    """Load existing user_data records (newline-delimited JSON)."""
    if not data_path.exists():
        return []
    records: List[Dict[str, Any]] = []
    with data_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                records.append(obj)
            except Exception:
                continue
    return records


def save_user_data(records: List[Dict[str, Any]], data_path: Path) -> None:
    """Write all user_data records as newline-delimited JSON."""
    data_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = data_path.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for idx, record in enumerate(records):
            if idx:
                f.write("\n")
            f.write(json.dumps(record, ensure_ascii=False))
    tmp_path.replace(data_path)


def extract_travel_windows(form_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract minimal travel windows from existing travel-related entries."""
    windows: List[Dict[str, Any]] = []
    for form_entry in form_list:
        if not isinstance(form_entry, dict):
            continue
        data = form_entry.get("data") or {}
        visit_plan = data.get("visit_plan") or {}
        intended = visit_plan.get("intended_visit_dates") or {}
        start = intended.get("from_date")
        end = intended.get("to_date")
        if start or end:
            windows.append(
                {
                    "destination": form_entry.get("type"),
                    "start_date": start,
                    "end_date": end,
                }
            )
    return windows


def normalize_country_name(value: str) -> str:
    return value.strip().lower()


def country_aliases(country_key: str) -> List[str]:
    base = normalize_country_name(country_key).replace("_", " ")
    aliases = {base}
    if base in {"usa", "us", "u.s.", "u.s.a.", "united states"}:
        aliases.update({"usa", "us", "united states", "u.s.", "u.s.a."})
    if base in {"uk", "united kingdom", "britain", "great britain", "england", "english"}:
        aliases.update({"uk", "united kingdom", "britain", "great britain", "england", "english"})
    if base in {"south korea", "republic of korea", "korea"}:
        aliases.update({"south korea", "republic of korea", "korea"})
    if base in {"china", "chinese"}:
        aliases.update({"china", "chinese"})
    if base in {"germany", "german"}:
        aliases.update({"germany", "german"})
    return list(aliases)


def is_own_country(profile: Dict[str, Any], country_key: str) -> bool:
    nat = normalize_country_name(str(profile.get("nationality", "")))
    res = normalize_country_name(str(profile.get("current_residence_country", "")))
    if country_key == "schengen":
        return nat in SCHENGEN_COUNTRIES or res in SCHENGEN_COUNTRIES
    aliases = set(country_aliases(country_key))
    return nat in aliases or res in aliases


def generate_user_data(
    profile: Optional[Dict[str, Any]],
    form_templates: Dict[str, Dict[str, Any]],
    selected_key: str,
    travel_windows: Optional[List[Dict[str, Any]]] = None,
    max_retries: int = 3,
    retry_delay: float = 1.5,
) -> Dict[str, Any]:
    """Call the LLM to generate one form (type + data) for a persona."""
    last_error: Optional[Exception] = None
    persona = profile or {}

    template = form_templates.get(selected_key)
    if not template:
        raise ValueError(f"Form template not found: {selected_key}")
    template_fields = template.get("fields") if isinstance(template, dict) else None
    template_for_prompt = template_fields if isinstance(template_fields, dict) else template

    persona_lines: List[str] = []
    full_name = persona.get("name")
    if full_name:
        persona_lines.append(f"Full name (can be adapted): {full_name}.")
    nationality = persona.get("nationality")
    residence = persona.get("current_residence_country")
    if nationality:
        persona_lines.append(f"Nationality: {nationality}.")
    if residence:
        persona_lines.append(f"Current residence country: {residence}.")
    age = persona.get("age")
    if age:
        persona_lines.append(f"Age: about {age} years old; choose date_of_birth accordingly (in the past).")
    sex = persona.get("sex")
    if sex:
        persona_lines.append(f"Sex: {sex}.")
    occupation = persona.get("occupation")
    if occupation:
        persona_lines.append(f"Occupation: {occupation}.")
    education_level = persona.get("education_level")
    if education_level:
        persona_lines.append(f"Education level: {education_level}.")
    brief_background = persona.get("brief_background")
    if brief_background:
        persona_lines.append(f"Background context: {brief_background}")
    details = persona.get("details")
    if isinstance(details, dict) and details:
        details_text = json.dumps(details, ensure_ascii=False, indent=2)
        persona_lines.append(
            "Existing personal details (must be followed strictly):\n" + details_text
        )

    template_prompt = json.dumps(template_for_prompt, ensure_ascii=False, indent=2)

    user_prompt = (
        f"Generate the JSON object now for form key `{selected_key}`. "
        'The output must be {"type": "<form_key>", "data": <filled form>} with the exact fields from the template.'
    )
    if persona_lines:
        user_prompt += "\nPersona constraints:\n" + "\n".join(f"- {line}" for line in persona_lines)
    if travel_windows:
        existing_block = json.dumps(travel_windows, ensure_ascii=False, indent=2)
        user_prompt += (
            "\n\nExisting travel windows for this applicant (avoid date conflicts with these trips):\n"
            f"{existing_block}"
        )
    user_prompt += f"\n\nForm template ({selected_key}):\n{template_prompt}"

    # Strict response schema: enforce {"type": str, "data": object} only.
    text_format = {
        "type": "json_schema",
        "name": "filled_form",
        "strict": False,
        "schema": {
            "type": "object",
            "properties": {
                "type": {"type": "string"},
                "data": {"type": "object", "additionalProperties": True},
            },
            "required": ["type", "data"],
            "additionalProperties": False,
        },
    }

    for attempt in range(1, max_retries + 1):
        try:
            response = LLM.responses.create(
                model=MODEL_NAME,
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                # temperature=0.8,
                text={"format": text_format},
            )
            # Extract text output from the Responses API
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

    raise RuntimeError(f"Failed to generate user data after {max_retries} attempts: {last_error}")


def main() -> None:
    base_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--form-path",
        dest="form_path",
        type=str,
        default=r"E:\project\主动提问\formdata\form\medical_form\New_Patient_Health_History_form\Health_history_form.json",
        help="Path to a single form template JSON (e.g. form/visa_form/India/India_visa_form.json).",
    )
    parser.add_argument(
        "--form-root",
        dest="form_root",
        type=str,
        default=os.getenv("FORM_ROOT", ""),
        help="Root directory containing form templates; used when --form-path is not provided.",
    )
    parser.add_argument(
        "--profile-path",
        dest="profile_path",
        type=str,
        default=r"E:\project\主动提问\formdata\data\user_data\senior_visit_or_medical_trip\userprofile.json",
        help="User profile file path (JSON array or NDJSON). Default: ./data/userprofile.json",
    )
    parser.add_argument(
        "--user-data-path",
        dest="user_data_path",
        type=str,
        default=r"E:\project\主动提问\formdata\data\user_data\senior_visit_or_medical_trip\user_data.json",
        help="Output path for generated user data (NDJSON). Default: ./data/user_data.json",
    )
    parser.add_argument(
        "--start-profile-index",
        dest="start_profile_index",
        type=int,
        default=int(os.getenv("SCHENGEN_PROFILE_START", "0")),
        help="Start index for iterating profiles (default 0).",
    )
    args = parser.parse_args()

    profile_path = Path(args.profile_path) if args.profile_path else base_dir / "data" / "userprofile.json"
    user_data_path = Path(args.user_data_path) if args.user_data_path else base_dir / "data" / "user_data.json"
    form_root = Path(args.form_root) if args.form_root else base_dir / "form"

    profiles = load_user_profiles(profile_path)

    if args.form_path:
        form_path = Path(args.form_path)
        if not form_path.is_absolute():
            form_path = base_dir / form_path
        form_templates = load_single_form_template(form_path)
    else:
        form_templates = load_form_templates(form_root)

    if not form_templates:
        raise ValueError("No templates loaded; please provide --form-path or ensure --form-root contains templates.")

    form_key_fixed = list(form_templates.keys())[0]
    form_meta = FORM_META.get(form_key_fixed, {})
    form_type = form_meta.get("type", "")
    form_country = form_meta.get("country", "")

    existing_records = load_existing_user_data(user_data_path)

    # Build uid -> record map
    uid_map: Dict[int, Dict[str, Any]] = {}
    other_records: List[Dict[str, Any]] = []
    for rec in existing_records:
        try:
            uid_val = int(rec.get("uid", 0))
            if uid_val > 0:
                uid_map[uid_val] = rec
            else:
                other_records.append(rec)
        except Exception:
            other_records.append(rec)

    profile_count = len(profiles)
    requested_count = DEFAULT_RECORD_COUNT
    total_slots = min(requested_count, profile_count)

    start_profile_index = int(args.start_profile_index) % profile_count
    ordered_indices = list(range(start_profile_index, profile_count)) + list(range(0, start_profile_index))

    # Build work items first to avoid duplicate generation and to preserve order.
    work_items: List[Tuple[int, int, Dict[str, Any], List[Dict[str, Any]]]] = []
    for idx in ordered_indices[:total_slots]:
        profile = profiles[idx]

        if form_type == "visa" and form_country:
            if is_own_country(profile, form_country):
                print(
                    f"[skip] profile_index {idx} applicant is from own country ({form_country}): "
                    f"{profile.get('nationality')} / {profile.get('current_residence_country')}"
                )
                continue
        if form_type == "passport" and form_country:
            if not is_own_country(profile, form_country):
                print(
                    f"[skip] profile_index {idx} applicant not from own country ({form_country}): "
                    f"{profile.get('nationality')} / {profile.get('current_residence_country')}"
                )
                continue

        uid = int(profile.get("uid") or idx + 1)
        existing = uid_map.get(uid, {"uid": uid, "forms": []})
        form_list = existing.get("forms")
        if not isinstance(form_list, list):
            form_list = existing.get("visa")
        if not isinstance(form_list, list):
            form_list = []

        existing_forms = form_list if isinstance(form_list, list) else []
        if any(isinstance(v, dict) and v.get("type") == form_key_fixed for v in existing_forms):
            print(f"[skip] profile_index {idx} duplicate form type {form_key_fixed} for uid {uid}")
            continue
        existing_windows = existing.get("travel_windows")
        if not isinstance(existing_windows, list):
            existing_windows = extract_travel_windows(existing_forms)

        work_items.append((idx, uid, profile, list(existing_windows)))

    results: List[Tuple[int, int, Dict[str, Any]]] = []
    total_appended = 0

    if work_items:
        max_workers = min(16, len(work_items))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(
                    generate_user_data,
                    profile=item[2],
                    form_templates=form_templates,
                    selected_key=form_key_fixed,
                    travel_windows=item[3],
                ): (item[0], item[1])
                for item in work_items
            }
            for future in concurrent.futures.as_completed(future_map):
                idx, uid = future_map[future]
                record = future.result()
                if not isinstance(record, dict) or "type" not in record or "data" not in record:
                    raise ValueError("Model response must contain 'type' and 'data'")
                results.append((idx, uid, record))
                # Progress signal for concurrent runs (write order preserved later).
                print(f"[progress] completed profile_index {idx}, uid {uid}")

        # Apply results in index order to keep deterministic data sequence.
        processed_uids: List[int] = []
        for idx, uid, record in sorted(results, key=lambda x: x[0]):
            existing = uid_map.get(uid, {"uid": uid, "forms": []})
            form_list = existing.get("forms")
            if not isinstance(form_list, list):
                form_list = existing.get("visa")
            if not isinstance(form_list, list):
                form_list = []
            form_list.append(record)
            existing.pop("visa", None)
            existing["forms"] = form_list
            uid_map[uid] = existing
            total_appended += 1
            processed_uids.append(uid)
            print(f"[ok] {total_appended}/{len(results)} processed (profile_index: {idx}, uid: {uid}, form: {form_key_fixed})")

        # Persist once after batch generation to avoid interleaving writes.
        merged_records = [uid_map[k] for k in sorted(uid_map)]
        merged_records.extend(other_records)
        save_user_data(merged_records, user_data_path)

    if total_appended == 0:
        raise RuntimeError("No user profiles available for generation")

    print(f"[done] total processed: {total_appended} -> {user_data_path}")

    run_maintenance(
        form_a_path=profile_path,
        form_b_path=user_data_path,
        target_form_type=form_key_fixed,
        allowed_uids=processed_uids if total_appended else [],
    )


if __name__ == "__main__":
    main()
