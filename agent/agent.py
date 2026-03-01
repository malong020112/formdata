import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI
from config import API_KEY, BASE_URL, GPT4O_MINI_API_KEY, MODEL_NAME, SEARCH_MODEL_NAME

from agent.tool.tool_fill_form import FillForm

LLM = OpenAI(api_key=API_KEY, base_url=BASE_URL)
JUDGE_MODEL_NAME = "gpt-4o-mini"
JUDGE_LLM = OpenAI(api_key=GPT4O_MINI_API_KEY, base_url=BASE_URL)
API_RETRY_MAX_ATTEMPTS = 5
API_RETRY_BASE_DELAY_SEC = 1.0
API_RETRY_MAX_DELAY_SEC = 16.0

# Ensure stdout handles UTF-8 (including emojis) to avoid encoding errors when redirected on Windows.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")  # type: ignore[attr-defined]
except Exception:
    pass

AGENT_PROMPT = """
You are a "Multi-User Intelligent Form-Filling Agent" responsible for helping users complete form-filling tasks.
You can call the following tools: `add_memory`, `fill_form`, `ask_user`, `search_api`.
If you need to call a tool in one round of conversation, you can only call one tool at a time.
All operations on forms and memories can only be completed through these tools; you cannot directly modify forms or memories.

==================== I. Overall Objectives ====================
1. Help users complete the filling of the current form.
2. Write users' information into memory, supporting two types of long-term memory:
   - User Memory: Personalized long-term memory for a single user_id.
   - Global Memory: Universal long-term memory shared by all users.

==================== II. Form Information ====================
Users will provide the definition of the form to be filled (list of fields).
You must fill in the form strictly in accordance with these fields and must not fabricate non-existent fields.

==================== III. Tool Description ====================
The tools you can use are as follows (parameters and returns are guaranteed to be correct by the tool system; the semantics and usage strategies are described here):
1) `add_memory`
   - Function: Write a piece of information into long-term memory.
   - Input parameters:
     - `user_id`:
       - When `user_id` = "0", write to Global Memory (shared by all users);
       - Otherwise, write to the User Memory of the corresponding `user_id` (only available to this user).
     - `content`: A JSON object `{key: value}` representing the information to be stored.
   - Usage Strategy:
     - After the form-filling task is completed, uniformly write all valid information of the user into User Memory.
     - Do not write one-time information that is only related to the current conversation.
     - content must not be empty.

2) `fill_form`
   - Function: Fill field values into the current form.
   - Input parameters:
     - user_id: The ID of the current user.
     - content: A JSON object in the form of `{key1: value1, key2: value2, ...}`.
   - Usage Strategy:
     - When you obtain the value of a field from:
       - The user's current conversation content,
       - search_memory results,
       - The answer to ask_user,
       - Your own reasoning (e.g., inferring the birth year from age)
     you should call `fill_form`.
     - For fields that are not applicable to the current user, you need to fill in "N/A".

3) `ask_user`
   - Function: Ask the user a question to obtain the value of one or a type of field that you cannot determine on your own.
   - Input parameters:
     - user_id: The ID of the current user.
     - question: The natural language question you want to ask the user.
   - Usage Strategy:
     - You can only call `ask_user` when you have tried using `search_memory` (and `search_api` if necessary) but still cannot determine the field value.
     - The question must be concise, clear, and directly answerable, avoiding open-ended casual chat.

     - Default Rule: Ask only one/one type of the most blocking field in each `ask_user` call.

     - Exception Rule (Strongly correlated fields can be asked jointly):
       If multiple missing fields are strongly correlated with each other, and the user can naturally answer them together in the same context, to reduce the number of `ask_user` calls, you can ask multiple fields in one `ask_user` call.

       Criteria for judging "Strongly Correlated Fields" (meet any one of the following):
       1) Belong to the same entity or the same document/certificate information (e.g., passport number + issue date + validity period).
       2) Are components of the same structured composite field (e.g., address: country + city + street + postal code).
       3) Depend on the same condition trigger or decision (e.g., whether there is an inviter + his personal information).
       4) Must be provided together to avoid ambiguity or repeated questioning.

       Constraints for joint questioning:
       - Must list the questions using numbers or bullet points so that the user can answer them item by item;
       - Only allow "truly missing and strongly correlated" fields to be included in the same question;
       - It is strictly prohibited to mix irrelevant fields in one question.

     - After the user answers:
       - Parse all relevant fields in the answer;
       - Use `fill_form` to fill in all newly obtained field values at one time.

4) search_api
    - Function: Query objective, public, verifiable factual information.
    - Input parameters:
        - query: The natural language search question you construct.
    - Return parameters:
        - success: Boolean value.
        - response: Text or structured results of the search (guaranteed by the external system).
    - Usage Strategy:
        - When you encounter a factual question, and:
        - It does not belong to user memory (not the user's personal long-term information);
        - It is not suitable to be written into global memory, or it does not yet exist in global memory;
        - It should not be asked to the user (the user may not know); you should prioritize calling search_api.
        - You need to construct the search question yourself and cannot directly reuse the form field names.
    - Example:
        - Form field: Address = Fudan University Jiangwan Campus
        - Next field: Postal Code
        - You should construct the query as:
            -👉 "What is the postal code of Fudan University Jiangwan Campus?"
    - Common applicable scenarios include but are not limited to:
        - Address, postal code, phone number of schools/companies/institutions;
        - Public information of fixed campuses, parks, office locations;
        - Universal rules, numbers, standard names, etc.
    - After obtaining the results:
        - If the result is clear and credible, fill in the form directly using `fill_form`;
        - If the result is uncertain or conflicting, do not fill in the form, and instead conduct `ask_user` for confirmation in the next round of interaction.

==================== IV. Memory Rules (User & Global) ====================
1. User Memory (user_id!=0)
   - Stores information that is strongly related to a user_id and long-term stable, such as:
     - Name, gender, birthday, ID number, mobile phone number, email, company, position, city, etc.
   - Writing method:
     - When the user provides this information for the first time in the conversation, call `add_memory(user_id, content={...})`.
     - Do not write table items that are not applicable to the user into memory with "N/A".

2. Global Memory (user_id=0)
   - Stores information that is universal to all users and long-term valid.
   - Writing method:
     - When you find that the information is obviously shared by all users and stable, you can call `add_memory(user_id="0", content={...})` after the form-filling task is completed.

3. Do not forge memories:
   - Do not assume that a certain memory has been stored before;

==================== V. Form-Filling Strategy ====================
1. Understand user input independently:
   - You need to understand the user's natural language content by yourself and decide when to call `fill_form`.
   - When the user's reply contains multiple fillable fields, fill them in by calling `fill_form` multiple times through multi-round interactions, and only call the tool once per round of interaction.

2. Strategy:
   - First use the user information already provided in the current conversation:
     - Understand the user's natural language and fill in the determinable fields by calling `fill_form`;
   - For missing required fields:
     - First reason according to other information given by the user, and fill in directly if inferable;
     - If it cannot be obtained through reasoning:
         - If it is a factual question, call `search_api` to query;
         - If it is a user's personal information question, call `ask_user` to ask the user;
       - After getting the answer, fill in the form with `fill_form`.
   - You can only fill in "N/A" for a certain field in the following cases:
     - The field is logically inapplicable according to other confirmed answers (e.g., the legal guardian field for adults);
     - You have clearly confirmed the restrictive conditions (e.g., EU family member: No).
   - For open fields (e.g., personal profile, remarks, reasons, etc.):
     - You can automatically generate appropriate text based on the known user portrait (obtained through `search_memory`) and the current form context;
     - Generally call `fill_form` directly, and only call `ask_user` when it is really impossible to generate reasonably.
     
3. Before the task is completed, you can only interact by calling the above tools. After the form-filling task is completed: call `add_memory` to write the user's long-term information and globally valid information into Memory.
   
4. Minimize the number of `ask_user` calls:
   - After each user reply, as much as possible:
     - Identify multiple fields;
     - Fill them in by calling `fill_form`;
     - Only call the tool once in the same round.
   - When asking questions:
     - Prioritize asking the most critical and blocking fields for form filling;

5. Reasoning and completion:
   - If some fields can be deduced from other fields (e.g., age + current year → birth year), you can make reasonable inferences and then directly call `fill_form`.
   - For fields that can be uniquely determined by system context or time context (such as signature fields, filling date fields, automatic confirmation fields), on the premise that no subjective decision or additional confirmation from the user is required, you can directly infer and fill in based on known user information and the current date and time.
   - The values derived in this way can also be written into User Memory (if they belong to the user's long-term information).

==================== VI. Planning (Mandatory) ====================
Before calling any tool, you must first construct an internal Planning.

The plan should include at least the following content:
1. Based on the user's information, the purpose of the application form, and the form content, judge which fields are applicable to the user and which are not.
2. Which fields need to be filled but are still missing;
3. For each field that needs to be filled but is missing, judge its acquisition method:
   - a) Whether it has been provided in the current conversation;
   - b) Whether it is a factual question that needs to be queried through `search_api`;
   - c) Whether it can be obtained through reasonable reasoning;
   - d) Whether it must be obtained by asking the user through `ask_user`;
4. Which fields can be filled together in the next `fill_form` call;
5. Whether there are blocking fields that must be resolved first.

When deciding which tool to call next, you must strictly follow the plan to execute.

==================== VII. Gating Question Strategy ====================

Objective:
On the premise of not sacrificing accuracy, significantly reduce both:
1) ask_user call count (turns)
2) ask_user_question_total (total number of question items)
Constraint: Gating is used for "branch/rule/conflict resolution"; it is prohibited to disguise "bulk field collection checklists" as gating questions, thereby inflating the number of items.

-------------------- 7.0 Key Definitions (Mandatory) --------------------
【Gating Question】
A "single discriminant question" solely intended to determine a key branch/rule/authoritative source, and the answer must be able to:
- Directly fill an entire group of fields as N/A (pruning), or
- Determine which small group of fields should be collected subsequently (decomposition), or
- Resolve conflicts and confirm "which one to follow" (align calibers)

【Bulk Collection】
Multiple information requests listed to fill field values (e.g., 9 passport items, 13 financial items).
Prohibition: Using the gating template to output bulk collection checklists; gating questions must not require users to provide multiple field values.

-------------------- 7.1 When to Use Gating (Trigger Conditions) --------------------
Use gating first (instead of asking field by field) if any of the following situations occur:
1) Caliber/Rule Dependence: The same rule determines the value of multiple fields (unit caliber, name splitting caliber, etc.)
2) Yes/No Pruning: A single "yes/no" determines whether an entire module can be marked as N/A
3) Conflict Resolution: User input vs memory vs search_api
4) Pre-discrimination for High-Cost Collection: Before launching extensive collection, first narrow the scope with 1 discriminant (e.g., whether there is a spouse/children/UK contact/visa refusal history)

-------------------- 7.2 Construction Principles for Gating (Must Follow) --------------------
A. Rules First: Gating only asks about "selection points/calibers/existence/which one to follow", not specific field values.
B. Minimum Discrimination: Answers must be completable with A/B, Yes/No, or 1/2.
C. Pruning First: Prioritize gating questions that allow the most fields to be directly marked as N/A.
D. One Gating Question at a Time: Only 1 gating question (with only 1 selection point) can appear in a single ask_user call.
E. No Expansion After Gating: After asking a gating question, it is prohibited to append "Please also provide 1)...2)...3)..." in the same ask_user message.
F. Conflicts First: If conflicts exist, gating must first confirm the authoritative source before form filling (fill_form).
G. Rule Reuse: Once a caliber/rule is confirmed, it must be reused in subsequent steps; repeated gating on the same caliber is not allowed.

-------------------- 7.3 Item Count Control (Strict Constraints) --------------------
To reduce ask_user_question_total:
1) Total length of a gating question ≤ 4 lines (including conflict point/impact scope/options)
2) Numbered lists (1), 2)... ) are strictly prohibited in the gating template
3) If additional details are needed, questions must be raised in the next round in "Collection Mode";
4) Collect blocking fields first, and delay confirmation of non-blocking fields until the "Reconciliation Checklist" stage (confirm all at once)

-------------------- 7.4 Gating Question Output Format (Mandatory Template) --------------------
【Gating Confirmation】
Conflict/Selection Point: {Summarize in one sentence}
Will Affect: {Field groups/modules (≤2 lines)}
Please Choose: A) {Option A}  or  B) {Option B}

Allowed Optional Supplement (maximum 1 line):
- If uncertain: Please paste a screenshot (or original text) of the passport/form field
Prohibition: Requiring users to fill in multiple field values in the gating template.

-------------------- 7.5 Execution Rules After Gating (Mandatory) --------------------
1) Upon receiving the gating answer, immediately perform fill_form:
   - Modules eligible for pruning: Batch fill with No + N/A
   - Modules requiring expansion: Only fill the confirmed branch marker (e.g., has_children=Yes)
2) Do not immediately ask about non-blocking fields: Continue automatic filling (reasoning/default/search_api), and conduct unified reconciliation at the end.
3) If field values must be collected: Switch to "Collection Mode".

-------------------- 7.6 Collection Mode --------------------
When gating has confirmed the branch and field values truly need to be provided by the user, use:

【Information Collection (max 5 items)】
Please provide the following information in order (estimation/range is acceptable):
- Item 1
- Item 2
- Item 3
- Item 4
- Item 5

Note: Collection Mode is not called "Gating" and does not use the A/B template.

-------------------- 7.7 One-shot Example (Correct Separation of Gating vs Collection) --------------------
Gating Example:
【Gating Confirmation】
Conflict/Selection Point: Which party's information should be used as the caliber for filling in "Institution Contact Information" in the form?
Will Affect: cover_page.institution_phone, host_institution_information.contact_*, etc.
Please Choose: A) Applicant's Affiliated Institution  or  B) Host Institution

If Option A is selected after gating, collect in the next round:
【Information Collection (max 5 items)】
Please provide the contact information of the applicant's affiliated institution:
- Institutional phone number
- Institutional address
- Postal code
- Contact person's name (if needed)
- Contact person's email (if needed)

==================== VIII. Supplementary Explanation for Judging Factual Questions ==================
You should regard the following information as factual questions:
    Irrelevant to specific individuals;
    Does not depend on the user's subjective preferences;
    Can be found in public information;
    Does not require the user to "decide", only to "verify".
Examples:
    ❌「What is your home address」 (user information)
    ❌「Which campus do you prefer」 (subjective preference)
    ✅「What is the postal code of Fudan University Jiangwan Campus」
    ✅「Where is the headquarters of a certain company located」
    For factual questions, do not directly call `ask_user`, but prioritize `search_api`.

==================== IX. Behavioral Norms for Interacting with Tools ====================
1. Tool Calling:
   - When using the fill_form tool, the filled content must not be empty.
   - Do not fill in the content that has already been filled in repeatedly.
   - If you want to ask the user for information, you can only do so by calling "ask_user".
   
2. Tool Return:
   - For `ask_user`, the external system will append the user's answer as a new user message to you, and you need to continue to make the next decision based on the latest conversation.

3. State Awareness:
   - The external system will maintain the form state according to the `fill_form` call, and you can learn which fields have been filled from the system or tool return information.
   - When you think all required fields have been filled, you should stop calling tools, give a brief summary or confirmation, and end the task.

==================== X. Task Completion Criteria ====================
All fields have been filled in, including fields that are not applicable to the user (filled with "N/A").

Do not call any tools when ending.

Your core task:
Reasonably use `add_memory`, `fill_form`, `ask_user`, and `search_api` (only one tool can be called per round) to help users complete form filling efficiently and accurately.

"""


# ==================== Helper data structures ====================
class FormField(Dict[str, Any]):
    """Lightweight representation of a form field."""


class FormDefinition(Dict[str, Any]):
    """Simple form definition: {"name": str, "fields": List[FormField]}."""


def build_system_prompt(form_def: FormDefinition, user_id: str) -> str:
    """Attach form structure to the base agent prompt."""
    return (
        f"{AGENT_PROMPT}\n\n"
        f"Current user_id: {user_id}\n"
        f"Current form: {form_def.get('name', 'form')}\n"
    )


_FORM_TEMPLATE_MARKER = "[FORM_TEMPLATE_FIELDS]"


def build_user_prompt(first_user_message: str, form_def: FormDefinition) -> str:
    if _FORM_TEMPLATE_MARKER in first_user_message:
        return first_user_message
    template_fields = form_def.get("fields", form_def)
    template_text = json.dumps(template_fields, ensure_ascii=False)
    return (
        f"{first_user_message}\n\n"
        f"{_FORM_TEMPLATE_MARKER}\n"
        f"{template_text}"
    )


_FILL_FORM_TOOL: Optional[FillForm] = None


def tool_fill_form(user_id: str, content: Dict[str, Any]) -> Dict[str, Any]:
    """Adapter to call the FillForm tool instance bound to the current loop."""
    if _FILL_FORM_TOOL is None:
        return {
            "user_id": user_id,
            "success": False,
            "response": "fill_form tool is not initialized.",
        }
    return _FILL_FORM_TOOL.call({"user_id": user_id, "content": content})


def tool_ask_user(user_id: str, question: str) -> str:
    """Blocking CLI question to the user. Replace with front-end hook as needed."""
    print(f"[ask_user] To {user_id}: {question}")
    return input("> ").strip()


def tool_search_memory(user_id: str, query: str) -> Dict[str, Any]:
    from agent.tool.tool_memory_search import SearchMemory

    tool = SearchMemory()
    return tool.call({"userid": user_id, "query": query})


def tool_get_memory(user_id: str) -> Dict[str, Any]:
    from agent.tool.tool_get_memory import GetMemory

    tool = GetMemory()
    return tool.call({"userid": user_id})


def tool_add_memory(role: str, uid: str, content: Dict[str, Any]) -> Dict[str, Any]:
    """Adapter that writes each key/value pair using AddMemory."""
    from agent.tool.tool_add_memory import AddMemory

    tool = AddMemory()
    # Map global memory to user_id "0" by convention.
    target_uid = "0" if role == "global" else uid
    last_result = {"response": "no-op", "success": False}
    for k, v in content.items():
        last_result = tool.call({"userid": target_uid, "content": [k, v]})
    return last_result


def tool_search_api(query: str) -> Dict[str, Any]:
    """Adapter for Search tool; normalize output to {success, response}."""
    from agent.tool.tool_search_api import Search

    q = str(query or "").strip()
    if not q:
        return {"success": False, "response": "Empty query for search_api."}

    try:
        tool = Search()
        raw = tool.call({"query": q})
    except Exception as exc:
        return {"success": False, "response": f"search_api error: {exc}"}

    response_text = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)
    lower = response_text.lower()
    failed_markers = (
        "no results found",
        "timeout",
        "invalid request format",
        "please try again later",
    )
    success = bool(response_text.strip()) and not any(marker in lower for marker in failed_markers)
    return {"success": success, "response": response_text}


def _chat_completion_with_backoff(
    client: OpenAI,
    *,
    model: str,
    messages: List[Dict[str, Any]],
    temperature: Optional[float] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[str] = None,
    response_format: Optional[Dict[str, Any]] = None,
):
    """Call chat.completions.create with exponential backoff retries."""
    last_error: Optional[Exception] = None
    for attempt in range(1, API_RETRY_MAX_ATTEMPTS + 1):
        try:
            payload: Dict[str, Any] = {"model": model, "messages": messages}
            if temperature is not None:
                payload["temperature"] = temperature
            if tools is not None:
                payload["tools"] = tools
            if tool_choice is not None:
                payload["tool_choice"] = tool_choice
            if response_format is not None:
                payload["response_format"] = response_format
            return client.chat.completions.create(**payload)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt >= API_RETRY_MAX_ATTEMPTS:
                raise
            delay = min(API_RETRY_BASE_DELAY_SEC * (2 ** (attempt - 1)), API_RETRY_MAX_DELAY_SEC)
            print(f"[api retry] agent chat.completions attempt={attempt} failed: {exc}; retry in {delay:.1f}s")
            time.sleep(delay)
    if last_error:
        raise last_error


def _heuristic_question_count(question: str) -> int:
    text = str(question or "").strip()
    if not text:
        return 0
    numbered_items = re.findall(r"(?m)^\s*\d+[\)\.\:、]\s+", text)
    if numbered_items:
        return len(numbered_items)
    return 1


def judge_ask_user_question_count(question: str) -> int:
    """Use gpt-4o-mini to estimate how many atomic questions are asked in one ask_user prompt."""
    text = str(question or "").strip()
    if not text:
        return 0

    system_prompt = (
        "You count how many atomic questions a prompt asks the user to answer. "
        "Rules: each A/B choice question counts as 1; each numbered answer item counts as 1; "
        "explanations and context text do not count. "
        "Return JSON only: {\"count\": <non-negative integer>}."
    )
    user_prompt = f"Prompt:\n{text}"

    try:
        resp = _chat_completion_with_backoff(
            client=JUDGE_LLM,
            model=JUDGE_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        content = (resp.choices[0].message.content or "").strip()
        parsed = json.loads(content) if content else {}
        count = int(parsed.get("count", 0))
        if count < 0:
            raise ValueError("negative count")
        return count
    except Exception as exc:
        fallback = _heuristic_question_count(text)
        print(f"[ask_user judge fallback] reason={exc}; fallback_count={fallback}")
        return fallback


def is_form_completed(form_def: FormDefinition, form_state: Dict[str, Any]) -> bool:
    """Check whether all fields are present and non-empty in the current form_state."""
    for field in form_def.get("fields", []):
        key = field.get("key")
        if key not in form_state:
            return False
        val = form_state.get(key)
        if val is None or val == "":
            return False
    return True


def initialize_form_state(form_def: FormDefinition) -> Dict[str, Any]:
    """Pre-populate form_state with all keys set to None to expose progress."""
    state: Dict[str, Any] = {}
    for field in form_def.get("fields", []):
        key = field.get("key")
        if key is None:
            continue
        state[key] = None
    return state


def snapshot_form_state(form_def: FormDefinition, form_state: Dict[str, Any]) -> Dict[str, List[str]]:
    """Return a structured snapshot of filled/missing fields to help the model plan."""
    filled: List[str] = []
    missing: List[str] = []
    for field in form_def.get("fields", []):
        key = field.get("key")
        if key is None:
            continue
        val = form_state.get(key)
        has_val = val is not None and val != ""
        (filled if has_val else missing).append(key)
    return {
        "filled": filled,
        "missing": missing,
    }


def collect_metrics(metrics: Dict[str, Any], form_state: Dict[str, Any], form_def: FormDefinition) -> Dict[str, Any]:
    duration = time.time() - metrics.get("start_time", time.time())
    total_fields = len(form_def.get("fields", []))
    filled_fields = sum(1 for v in form_state.values() if v not in (None, ""))
    return {
        "duration_sec": round(duration, 2),
        "rounds": metrics.get("rounds", 0),
        "tool_calls": metrics.get("tool_calls", 0),
        "ask_user": metrics.get("ask_user", 0),
        "ask_user_question_total": metrics.get("ask_user_question_total", 0),
        "search_memory_total": metrics.get("search_memory_total", 0),
        "search_memory_success": metrics.get("search_memory_success", 0),
        "search_api_total": metrics.get("search_api_total", 0),
        "search_api_success": metrics.get("search_api_success", 0),
        "fill_form_calls": metrics.get("fill_form_calls", 0),
        "add_memory_calls": metrics.get("add_memory_calls", 0),
        "fields_filled": filled_fields,
        "fields_total": total_fields,
    }


def emit_metrics(metrics: Dict[str, Any], form_state: Dict[str, Any], form_def: FormDefinition) -> None:
    """Print a one-line summary of agent runtime metrics."""
    data = collect_metrics(metrics, form_state, form_def)
    print(
        "[agent metrics] "
        f"duration={data['duration_sec']:.2f}s, rounds={data['rounds']}, "
        f"tool_calls={data['tool_calls']}, ask_user={data['ask_user']}, "
        f"ask_user_question_total={data['ask_user_question_total']}, "
        f"search_memory={data['search_memory_total']}/{data['search_memory_success']}(total/hit), "
        f"search_api={data['search_api_total']}/{data['search_api_success']}(total/hit), "
        f"fill_form={data['fill_form_calls']}, add_memory={data['add_memory_calls']}, "
        f"fields_filled={data['fields_filled']}/{data['fields_total']}"
    )


# ==================== LLM driver ====================
def get_tool_specs() -> List[Dict[str, Any]]:
    """OpenAI tool schema definitions."""
    return [
        {
            "type": "function",
            "function": {
                "name": "fill_form",
                "description": "Fill values into the current form.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user_id": {"type": "string"},
                        "content": {"type": "object", "additionalProperties": True},
                    },
                    "required": ["user_id", "content"],
                },
            },
        },
        # search_memory is temporarily disabled in runtime workflow.
        # {
        #     "type": "function",
        #     "function": {
        #         "name": "search_memory",
        #         "description": "Retrieve information from long-term memory by user_id and query.",
        #         "parameters": {
        #             "type": "object",
        #             "properties": {
        #                 "user_id": {"type": "string"},
        #                 "query": {"type": "string"},
        #             },
        #             "required": ["user_id", "query"],
        #         },
        #     },
        # },
        {
            "type": "function",
            "function": {
                "name": "add_memory",
                "description": "Store key/value pairs into long-term memory.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "role": {"type": "string", "enum": ["user", "global"]},
                        "uid": {"type": "string"},
                        "content": {"type": "object", "additionalProperties": True},
                    },
                    "required": ["role", "uid", "content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "ask_user",
                "description": "Ask the end user a clarifying question.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user_id": {"type": "string"},
                        "question": {"type": "string"},
                    },
                    "required": ["user_id", "question"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_api",
                "description": "Look up public factual information.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        },
    ]


def llm_generate(messages: List[Dict[str, Any]], tool_specs: List[Dict[str, Any]]):
    """Call the LLM using OpenAI tool-call protocol."""
    response = _chat_completion_with_backoff(
        client=LLM,
        model=MODEL_NAME,
        messages=messages,
        tools=tool_specs,
        tool_choice="auto",
    )
    return response


# ==================== Agent loop ====================
def agent_loop(user_id: str, form_def: FormDefinition, first_user_message: str) -> Dict[str, Any]:
    """Main agent control loop aligned with OpenAI tool-call protocol."""
    fill_form_tool = FillForm(form_def)
    global _FILL_FORM_TOOL
    _FILL_FORM_TOOL = fill_form_tool
    form_state: Dict[str, Any] = fill_form_tool.get_state()
    initial_snapshot = snapshot_form_state(form_def, form_state)
    metrics = {
        "rounds": 0,  # total LLM decision rounds (includes tool and dialogue turns)
        "tool_calls": 0,
        "ask_user": 0,
        "ask_user_question_total": 0,
        "search_memory_total": 0,
        "search_memory_success": 0,
        "search_api_total": 0,
        "search_api_success": 0,
        "fill_form_calls": 0,
        "add_memory_calls": 0,
        "start_time": time.time(),
    }
    known_memory = tool_get_memory(user_id)
    known_memory_prompt = (
        "Known user information from memory (already known for this user; use when applicable): "
        + json.dumps(known_memory.get("memory", {}), ensure_ascii=False)
    )
    user_message = build_user_prompt(first_user_message, form_def)
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": build_system_prompt(form_def, user_id)},
        {"role": "assistant", "content": known_memory_prompt},
        {"role": "user", "content": user_message},
        {
            "role": "assistant",
            "content": f"Form state initialized. filled={initial_snapshot['filled']}, "
                       f"missing={initial_snapshot['missing']}",
        },
    ]

    tool_specs = get_tool_specs()

    while True:
        print("=============\n")
        metrics["rounds"] += 1
        response = llm_generate(messages, tool_specs=tool_specs)
        choice = response.choices[0].message
        print(f"[LLM output]: content={choice.content}, tool_calls={choice.tool_calls}\n")

        if choice.tool_calls:
            metrics["tool_calls"] += len(choice.tool_calls)
            # Persist the assistant message with tool_calls.
            messages.append(
                {
                    "role": "assistant",
                    "content": choice.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": tc.type,
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                        }
                        for tc in choice.tool_calls
                    ],
                }
            )

            for tc in choice.tool_calls:
                tool_name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}

                if tool_name == "fill_form":
                    metrics["fill_form_calls"] += 1
                    content = args.get("content", {})
                    result = tool_fill_form(args.get("user_id", user_id), content)
                    tool_payload = {"success": bool(result.get("success"))}
                    if not tool_payload["success"]:
                        failed_fields: List[str] = []
                        if isinstance(content, dict):
                            failed_fields = list(content.keys())
                        tool_payload["failed"] = failed_fields
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "name": tool_name,
                            "content": json.dumps(tool_payload, ensure_ascii=False),
                        }
                    )
                    skipped_keys = result.get("skipped", [])
                    if skipped_keys:
                        messages.append(
                            {
                                "role": "assistant",
                                "content": f"Fields already exist with the same value, skipped: {', '.join(skipped_keys)}",
                            }
                        )
                    if isinstance(result.get("form_state"), dict):
                        form_state = result["form_state"]
                    snapshot = result.get("snapshot") or snapshot_form_state(form_def, form_state)
                    messages.append(
                        {
                            "role": "assistant",
                            "content": f"Form state: filled={snapshot['filled']}, missing={snapshot['missing']}",
                        }
                    )
                    if is_form_completed(form_def, form_state):
                        messages.append(
                            {
                                "role": "assistant",
                                "content": "Hint: All required fields have been filled in, you can give the final confirmation.",
                            }
                        )

                elif tool_name == "search_memory":
                    metrics["search_memory_total"] += 1
                    # search_memory execution is intentionally disabled in current workflow.
                    # result = tool_search_memory(args.get("user_id", user_id), args.get("query", ""))
                    result = {
                        "success": False,
                        "response": "search_memory is disabled. Use known memory in system prompt.",
                    }
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "name": tool_name,
                            "content": json.dumps(result, ensure_ascii=False),
                        }
                    )

                elif tool_name == "add_memory":
                    metrics["add_memory_calls"] += 1
                    result = tool_add_memory(
                        role=args.get("role", "user"),
                        uid=args.get("uid", user_id),
                        content=args.get("content", {}),
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "name": tool_name,
                            "content": json.dumps(result, ensure_ascii=False),
                        }
                    )

                elif tool_name == "ask_user":
                    metrics["ask_user"] += 1
                    question = args.get("question", "")
                    question_count = judge_ask_user_question_count(question)
                    metrics["ask_user_question_total"] += question_count
                    print(f"[ask_user question_count]: {question_count}")
                    answer = tool_ask_user(args.get("user_id", user_id), question)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "name": tool_name,
                            "content": answer,
                        }
                    )

                elif tool_name == "search_api":
                    metrics["search_api_total"] += 1
                    result = tool_search_api(args.get("query", ""))
                    if result.get("success"):
                        metrics["search_api_success"] += 1
                    print(f"[tool search_api result]: {result}")
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "name": tool_name,
                            "content": json.dumps(result, ensure_ascii=False),
                        }
                    )
                else:
                    messages.append({"role": "assistant", "content": f"Unknown tool requested: {tool_name}."})

            # printable = [m for m in messages if m.get("role") != "system"]
            # print("messages so far (no system):", printable)
            continue

        
        # No tool calls: end only if all fields have values; otherwise keep looping.
        if is_form_completed(form_def, form_state):
            final_content = choice.content or ""
            messages.append({"role": "assistant", "content": final_content})
            emit_metrics(metrics, form_state, form_def)
            return {
                "form_state": form_state,
                "final_message": final_content,
                "metrics": collect_metrics(metrics, form_state, form_def),
                "messages": messages,
            }

        snapshot = snapshot_form_state(form_def, form_state)
        messages.append({"role": "assistant", "content": choice.content or ""})
        messages.append(
            {
                "role": "assistant",
                "content": f"Form progress: filled={snapshot['filled']}, missing={snapshot['missing']}. "
                           "Use tools (ask_user/search_memory/fill_form) to complete all fields.",
            }
        )
        continue

    emit_metrics(metrics, form_state, form_def)
    return {
        "form_state": form_state,
        "final_message": "",
        "metrics": collect_metrics(metrics, form_state, form_def),
        "messages": messages,
    }

if __name__ == "__main__":
    # Example usage with a toy form.
    demo_form = {
        "name": "Demo Form",
        "fields": [
            {"key": "name"},
            {"key": "phone"},
            {"key": "age"},
            {"key":"occupation"},
            {"key": "email"},
            {"key": "bio"},
        ],
    }
    first_msg = "I want to fill the form. Name is Zhang San, phone is 13800000000."
    result = agent_loop(user_id="1", form_def=demo_form, first_user_message=first_msg)
    # print("Final state:", result["form_state"])
    # print("Final message:", result["final_message"])
