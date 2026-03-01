import concurrent.futures
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI

from config import API_KEY, BASE_URL, MODEL_NAME

# Configuration
DEFAULT_RECORD_COUNT = int(os.getenv("PROFILE_COUNT", "5"))
PROFILE_TEMPLATE_NAME = os.getenv("PROFILE_TEMPLATE", "trade_logistics_customs").strip()

LLM = OpenAI(api_key=API_KEY, base_url=BASE_URL)


SYSTEM_PROMPT = """
You are a persona generation engine.

Your task is to generate ONE realistic user profile (persona blueprint)

REQUIREMENTS:
- Output MUST be valid JSON.
- Output MUST be a single JSON object.
- No explanations, comments, or markdown.

CORE RULES: user profile should be diverse as possible.

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
- name: realistic full name, be diverse as possible.
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

PASSPORT_COUNTRIES = ["China", "Germany", "United States", "Canada", "England"]
OTHER_COUNTRIES = [c for c in COUNTRIES if c not in PASSPORT_COUNTRIES]

# PROFILE_TEMPLATES = [
#     {
#         "name": "student_young",
#         "age_min": 18,
#         "age_max": 25,
#         "occupation_hint": "student or recent graduate",
#         "weight": 20,
#     },
#     {
#         "name": "early_career",
#         "age_min": 22,
#         "age_max": 30,
#         "occupation_hint": "junior staff or specialist",
#         "weight": 20,
#     },
#     {
#         "name": "mid_career",
#         "age_min": 28,
#         "age_max": 45,
#         "occupation_hint": "manager or engineer or consultant or unemployed",
#         "weight": 25,
#     },
#     {
#         "name": "executive",
#         "age_min": 35,
#         "age_max": 60,
#         "occupation_hint": "founder or director or executive or unemployed",
#         "weight": 15,
#     },
#     {
#         "name": "family_visitor",
#         "age_min": 25,
#         "age_max": 65,
#         "occupation_hint": "any (choose a realistic occupation)",
#         "weight": 10,
#     },
#     {
#         "name": "retired_senior",
#         "age_min": 60,
#         "age_max": 80,
#         "occupation_hint": "retired",
#         "weight": 5,
#     },
#     {
#         "name": "cultural_sports",
#         "age_min": 18,
#         "age_max": 45,
#         "occupation_hint": "artist or athlete or performer",
#         "weight": 3,
#     },
#     {
#         "name": "medical",
#         "age_min": 30,
#         "age_max": 75,
#         "occupation_hint": "any (choose a realistic occupation)",
#         "weight": 2,
#     },
# ]


PROFILE_TEMPLATES = [
    # 1) 18–24 国际升学/交换生：教育表覆盖主力 + 旅行证件 + 医疗/保险
    {
        "name": "intl_student_applicant",
        "age_min": 18,
        "age_max": 24,
        "weight": 18,
        "occupations": [
            "High school student",
            "Undergraduate student",
            "Gap year student",
            "Pre-university/Foundation student",
            "Language school student",
            "Student intern (part-time)",
            "Research assistant (student)",
            "Teaching assistant (student)"
        ],
        "occupation_hint": (
            "international student applicant /  gap year. "
            "Triggers: education (undergrad/intl admission/scholarship), visa, passport, customs, "
            "student insurance claim, basic health history."
        ),
        "family_status_hint": "single; may have guardian/parents as sponsors",
        "finance_hint": "parents sponsorship + scholarship possibility; proof of funds needed",
        "travel_hint": "first long-haul travel or first-time visa; clear itinerary",
    },

    # 2) 22–35 年轻白领出差/外派：签证+护照+报销+部分合规材料
    {
        "name": "young_professional_travel",
        "age_min": 22,
        "age_max": 35,
        "weight": 20,
        "occupations": [
            "Software engineer",
            "Product manager",
            "Data analyst",
            "Management consultant",
            "Sales specialist (B2B)",
            "Operations specialist",
            "Project coordinator",
            "Marketing specialist",
            "Account manager",
            "Field service engineer"
        ],
        "occupation_hint": (
            "junior/mid staff: engineer/consultant/sales/ops; frequent business travel or short assignment. "
            "Triggers: visa, passport renewal, customs, medical reimbursement (out-of-network), "
            "sometimes source-of-funds / bank KYC."
        ),
        "family_status_hint": "single or newly married; optional 0-1 child",
        "finance_hint": "salary income; employer-covered travel; reimbursement receipts",
        "travel_hint": "2-6 trips/year; short stays; invitation letter from partner/client",
    },

    # 3) 30–45 家庭迁居/子女教育：签证+护照（含未成年）+医疗+（可选）教育
    {
        "name": "family_relocation_or_kids_edu",
        "age_min": 30,
        "age_max": 45,
        "weight": 16,
        "occupations": [
            "HR manager",
            "Finance manager",
            "Civil engineer",
            "Nurse",
            "Teacher (K-12)",
            "Accountant",
            "Small business owner",
            "Government staff",
            "Customer success manager",
            "Freelancer (designer/writer)"
        ],
        "occupation_hint": (
            "parent relocating or traveling with kids; spouse employed or family visit/long stay. "
            "Triggers: family visas, passports for adults + minors, customs (family goods), "
            "health/vaccination forms, insurance claims; optional prep/intl school apps."
        ),
        "family_status_hint": "married; 1-2 kids (6-18)",
        "finance_hint": "dual-income or single-income; family savings; sponsor/relative invitation possible",
        "travel_hint": "family itinerary; longer stay; accommodation/relative address",
    },

    # 4) 28–55 跨境创业/投资/企业合规：税务金融主力 + 商务签证 + 项目扶持
    {
        "name": "crossborder_founder_compliance",
        "age_min": 28,
        "age_max": 55,
        "weight": 15,
        "occupations": [
            "Company founder / CEO",
            "Co-founder / Partner",
            "CFO / Finance director",
            "Cross-border e-commerce seller",
            "Import-export business owner",
            "Investment manager",
            "Corporate lawyer",
            "Tax consultant",
            "Compliance officer",
            "Venture partner"
        ],
        "occupation_hint": (
            "founder/partner/finance lead/cross-border e-commerce. "
            "Triggers: tax & finance (UBO, source-of-funds, corporate info, GST-like registration), "
            "business visas, customs declarations (samples/equipment/cash), "
            "optionally business grant/subsidy applications."
        ),
        "family_status_hint": "any; often married",
        "finance_hint": "business income + dividends; multi-account; ownership structure",
        "travel_hint": "business travel; may carry samples or higher cash",
    },

    # 5) 25–60 科研人员/高校：项目资助主力 + 学术签证 + 报销
    {
        "name": "researcher_grant_and_visit",
        "age_min": 25,
        "age_max": 60,
        "weight": 10,
        "occupations": [
            "PhD student",
            "Postdoctoral researcher",
            "University lecturer",
            "Associate professor",
            "Research scientist",
            "Lab engineer",
            "Research institute staff",
            "University administrator (research office)",
            "Clinical researcher",
            "R&D engineer"
        ],
        "occupation_hint": (
            "PhD/postdoc/faculty/research staff. "
            "Triggers: grant proposals (e.g., NSFC-style), academic visit/conference visas, "
            "customs for samples/instruments, travel reimbursement; optional scholarship funding."
        ),
        "family_status_hint": "any; often single/married without young kids",
        "finance_hint": "salary + grant funding; institution as sponsor",
        "travel_hint": "conference/visiting scholar; invitation letter from university/lab",
    },

    # 6) 50–75 退休/探亲/医疗旅游：医疗表主力 + 探亲签 + 护照换发
    {
        "name": "senior_visit_or_medical_trip",
        "age_min": 50,
        "age_max": 75,
        "weight": 12,
        "occupations": [
            "Retired",
            "Former teacher (retired)",
            "Former engineer (retired)",
            "Former government employee (retired)",
            "Former accountant (retired)",
            "Part-time consultant",
            "Small shop owner",
            "Self-employed (semi-retired)"
        ],
        "occupation_hint": (
            "retired or near-retirement; family visit or medical travel. "
            "Triggers: visit visas, passport renewal, medical intake/admission/discharge, "
            "insurance claims, medication lists; occasional cash/customs declarations."
        ),
        "family_status_hint": "married or widowed; adult children as inviter/sponsor",
        "finance_hint": "pension + family support; proof of relationship/sponsorship",
        "travel_hint": "longer stays; carries medications; needs accessible itinerary",
    },

    # 7) 25–50 外贸/物流/海关业务：海关申报主力 + 商务签证 + 公司合规
    {
        "name": "trade_logistics_customs",
        "age_min": 25,
        "age_max": 50,
        "weight": 9,
        "occupations": [
            "Customs broker",
            "Freight forwarder",
            "Logistics coordinator",
            "Supply chain manager",
            "Procurement specialist",
            "International trade specialist",
            "Warehouse manager",
            "Shipping documentation clerk",
            "Import/export operations manager",
            "Quality inspector (cross-border)"
        ],
        "occupation_hint": (
            "customs broker/freight forwarder/supply chain manager/trader. "
            "Triggers: CN23/international customs forms, business visas, corporate KYC materials, "
            "source-of-funds; occasional travel insurance reimbursement."
        ),
        "family_status_hint": "any",
        "finance_hint": "salary or business income; company documents available",
        "travel_hint": "frequent regional trips; shipments/samples documentation",
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
    if not OTHER_COUNTRIES:
        raise ValueError("OTHER_COUNTRIES is empty; check PASSPORT_COUNTRIES list.")
    if not PASSPORT_COUNTRIES:
        raise ValueError("PASSPORT_COUNTRIES is empty; check passport form countries.")
    if random.random() < 0.75:
        country = random.choice(PASSPORT_COUNTRIES)
    else:
        country = random.choice(OTHER_COUNTRIES)
    return {
        "age": random.randint(template["age_min"], template["age_max"]),
        "sex": random.choice(["male", "female"]),
        "country": country,
    }


def find_template(template_name: str) -> Dict[str, Any]:
    """Return a template by name."""
    if not template_name:
        available = ", ".join(t["name"] for t in PROFILE_TEMPLATES)
        raise ValueError(f"PROFILE_TEMPLATE is required. Available: {available}")
    for t in PROFILE_TEMPLATES:
        if t["name"] == template_name:
            return t
    available = ", ".join(t["name"] for t in PROFILE_TEMPLATES)
    raise ValueError(f"Unknown PROFILE_TEMPLATE '{template_name}'. Available: {available}")


def generate_user_profile(
    template_name: str,
    constraints: Optional[Dict[str, Any]] = None,
    max_retries: int = 3,
    retry_delay: float = 1.5,
) -> Dict[str, Any]:
    """Call the LLM to generate one user profile with given constraints."""
    last_error: Optional[Exception] = None
    template = find_template(template_name)
    base = build_constraints(template)
    c = {**base, **(constraints or {})}
    occupations = template.get("occupations") or []
    occupation_choice = random.choice(occupations) if occupations else ""

    user_prompt = "\n".join(
        [
            "Generate the user profile now following these constraints:",
            f"- Age: {c.get('age', '18-65')}",
            f"- Sex: {c.get('sex', 'male/female')}",
            f"- Occupation: {occupation_choice}" if occupation_choice else "",
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
                delay = retry_delay * (2 ** (attempt - 1))
                time.sleep(delay)

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
    output_path = Path(__file__).resolve().parent / "data" / "user_data" / PROFILE_TEMPLATE_NAME /"userprofile.json"
    record_count = DEFAULT_RECORD_COUNT

    total = 0
    next_uid = get_next_uid(output_path)

    max_workers = min(16, record_count) if record_count > 0 else 0
    results: Dict[int, Dict[str, Any]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(generate_user_profile, PROFILE_TEMPLATE_NAME): idx
            for idx in range(1, record_count + 1)
        }
        for future in concurrent.futures.as_completed(future_map):
            idx = future_map[future]
            results[idx] = future.result()
            print(f"[progress] completed {idx}/{record_count}")

    for idx in range(1, record_count + 1):
        profile = results[idx]
        profile_with_uid = {"uid": next_uid + total, **profile, "details": {}}
        append_profile(profile_with_uid, output_path)
        total += 1
        print(
            f"[ok] {idx}/{record_count} appended to {output_path} "
            f"(uid: {profile_with_uid['uid']}, template: {PROFILE_TEMPLATE_NAME})"
        )

    print(f"[done] total profiles appended: {total} -> {output_path}")


if __name__ == "__main__":
    main()
