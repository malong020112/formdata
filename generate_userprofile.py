import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI

# Configuration
API_KEY = os.getenv("OPENAI_API_KEY", "sk-tEbmy3HeMVHfwSfw5a2BXZMOzc76PXd4OzoMkLUj6hYowDqE")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://zjuapi.com/v1")
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-5.2")
DEFAULT_RECORD_COUNT = int(os.getenv("PROFILE_COUNT", "300"))

LLM = OpenAI(api_key=API_KEY, base_url=BASE_URL)


SYSTEM_PROMPT = """
You are a persona generation engine.

Your task is to generate ONE realistic user profile (persona blueprint)

REQUIREMENTS:
- Output MUST be valid JSON.
- Output MUST be a single JSON object.
- No explanations, comments, or markdown.

OUTPUT SCHEMA (DO NOT ADD FIELDS):
{
  "name": string,
  "age": number,
  "sex": "male" | "female",
  "nationality": string,
  "current_residence_country": string,
  "occupation": string,
  "education_level": string,
  "brief_background": string
}

FIELD GUIDELINES:
- name: realistic full name, be diverse.
- age: use the provided constraint.
- sex: must be "male" or "female".
- nationality and current_residence_country: align with provided hints.
- occupation: realistic for the age; avoid military/security roles.
- education_level: consistent with age and occupation.
- brief_background: Introduce this character briefly based on this information .
"""

COUNTRIES = [
    # 申根区核心常见国家
    "Germany", "France", "Italy", "Spain", "Portugal", "Netherlands", "Belgium", 
    "Luxembourg", "Austria", "Switzerland", "Sweden", "Finland", "Denmark", "Norway", "Iceland",
    # 亚洲常见国家
    "China", "India", "Japan", "South Korea", "Indonesia", "Thailand", "Vietnam", "Philippines",
    "Malaysia", "Singapore", "Pakistan", "Bangladesh", "Sri Lanka", "Turkey", "Jordan", "Lebanon",
    "Israel", "United Arab Emirates", "Saudi Arabia", "Qatar", "Kuwait", "Oman",
    # 非洲常见国家（
    "Nigeria", "Ghana", "Kenya", "Ethiopia", "South Africa", "Egypt", "Morocco", "Tunisia",
    "Algeria", "Uganda", "Tanzania", "Rwanda", "Senegal", "Cameroon", "Ivory Coast",
    # 美洲常见国家
    "United States", "Canada", "Brazil", "Argentina", "Chile", "Peru", "Colombia", "Mexico",
    "Ecuador", "Uruguay", "Dominican Republic", "Costa Rica", "Panama", "Cuba", "Jamaica",
    # 欧洲非申根常见国家
    "United Kingdom", "Russia", "Poland", "Czech Republic", "Hungary", "Romania", "Bulgaria",
    "Greece", "Ireland", "Ukraine", "Serbia", "Montenegro", "Albania",
    # 大洋洲常见国家
    "Australia", "New Zealand"
]

PROFILE_TEMPLATES = [
    {
        "name": "student_young",
        "age_min": 18,
        "age_max": 25,
        "occupation_hint": "student or recent graduate",
        "weight": 20,
    },
    {
        "name": "early_career",
        "age_min": 22,
        "age_max": 30,
        "occupation_hint": "junior staff or specialist",
        "weight": 20,
    },
    {
        "name": "mid_career",
        "age_min": 28,
        "age_max": 45,
        "occupation_hint": "manager or engineer or consultant or unemployed",
        "weight": 25,
    },
    {
        "name": "executive",
        "age_min": 35,
        "age_max": 60,
        "occupation_hint": "founder or director or executive or unemployed",
        "weight": 15,
    },
    {
        "name": "family_visitor",
        "age_min": 25,
        "age_max": 65,
        "occupation_hint": "any (choose a realistic occupation)",
        "weight": 10,
    },
    {
        "name": "retired_senior",
        "age_min": 60,
        "age_max": 80,
        "occupation_hint": "retired",
        "weight": 5,
    },
    {
        "name": "cultural_sports",
        "age_min": 18,
        "age_max": 45,
        "occupation_hint": "artist or athlete or performer",
        "weight": 3,
    },
    {
        "name": "medical",
        "age_min": 30,
        "age_max": 75,
        "occupation_hint": "any (choose a realistic occupation)",
        "weight": 2,
    },
]


def _extract_json_from_text(content: str) -> Any:
    """Extract a JSON object from model text."""
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


def build_constraints(template: Dict[str, Any]) -> Dict[str, Any]:
    """Build constraints for a profile using a template."""
    return {
        "age": random.randint(template["age_min"], template["age_max"]),
        "sex": random.choice(["male", "female"]),
        "country": random.choice(COUNTRIES),
        "occupation_hint": template["occupation_hint"],
        "template_name": template["name"],
    }


def build_profile_plan(record_count: int) -> List[Dict[str, Any]]:
    """Create a template plan that follows the desired ratio as closely as possible."""
    templates = PROFILE_TEMPLATES[:]
    if record_count <= 0:
        return []

    total_weight = sum(t["weight"] for t in templates)

    if record_count >= len(templates):
        counts = {t["name"]: 1 for t in templates}
        remaining = record_count - len(templates)

        desired = []
        for t in templates:
            exact = (t["weight"] / total_weight) * remaining
            base = int(exact)
            counts[t["name"]] += base
            desired.append((t["name"], exact - base))

        remainder = remaining - sum(counts.values()) + len(templates)
        desired.sort(key=lambda item: item[1], reverse=True)
        for i in range(remainder):
            counts[desired[i % len(desired)][0]] += 1
    else:
        names = [t["name"] for t in templates]
        weights = [t["weight"] for t in templates]
        picks = random.choices(names, weights=weights, k=record_count)
        counts = {name: 0 for name in names}
        for name in picks:
            counts[name] += 1

    plan = []
    for t in templates:
        plan.extend([t] * counts[t["name"]])
    random.shuffle(plan)
    return plan


def generate_user_profile(
    constraints: Optional[Dict[str, Any]] = None,
    max_retries: int = 3,
    retry_delay: float = 1.5,
) -> Dict[str, Any]:
    """Call the LLM to generate one user profile with given constraints."""
    last_error: Optional[Exception] = None
    c = constraints or build_constraints(random.choice(PROFILE_TEMPLATES))

    occupation_hint = c.get("occupation_hint") or "any (choose a realistic occupation)"
    user_prompt = "\n".join(
        [
            "Generate the user profile now following these constraints:",
            f"- Age: {c.get('age', '18-65')}",
            f"- Sex: {c.get('sex', 'male/female')}",
            f"- Occupation example: {occupation_hint}",
            f"- Nationality/current residence should align with: {c.get('country')}",
            "- Occupation list just for reference, you can also choose an option outside the list.But make sure it's realistic.",
        ]
    )

    for attempt in range(1, max_retries + 1):
        try:
            response = LLM.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.8,
                response_format={"type": "json_object"},
            )
            message_content = response.choices[0].message.content
            parsed = _extract_json_from_text(message_content or "")
            if not isinstance(parsed, dict):
                raise ValueError("Model response is not a JSON object")
            return parsed
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < max_retries and retry_delay > 0:
                time.sleep(retry_delay)

    raise RuntimeError(f"Failed to generate user profile after {max_retries} attempts: {last_error}")


def append_profile(profile: Dict[str, Any], output_path: Path) -> None:
    """Append one JSON profile to the output file (newline-delimited JSON)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    has_content = output_path.exists() and output_path.stat().st_size > 0
    serialized = json.dumps(profile, ensure_ascii=False)
    with output_path.open("a", encoding="utf-8") as f:
        if has_content:
            f.write("\n")
        f.write(serialized)


def get_next_uid(output_path: Path) -> int:
    """Return the next uid by scanning existing records (uid starts from 1)."""
    if not output_path.exists():
        return 1

    max_uid = 0
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                uid_val = int(obj.get("uid", 0))
                if uid_val > max_uid:
                    max_uid = uid_val
            except Exception:
                continue

    return max_uid + 1


def main() -> None:
    output_path = Path(__file__).resolve().parent / "data" / "visa_data" / "userprofile.json"
    record_count = DEFAULT_RECORD_COUNT

    total = 0
    plan = build_profile_plan(record_count)
    next_uid = get_next_uid(output_path)

    for idx, template in enumerate(plan, start=1):
        constraints = build_constraints(template)
        profile = generate_user_profile(constraints=constraints)
        profile_with_uid = {"uid": next_uid + total, **profile}
        append_profile(profile_with_uid, output_path)
        total += 1
        print(
            f"[ok] {idx}/{record_count} appended to {output_path} "
            f"(uid: {profile_with_uid['uid']}, template: {constraints.get('template_name')})"
        )

    print(f"[done] total profiles appended: {total} -> {output_path}")


if __name__ == "__main__":
    main()
