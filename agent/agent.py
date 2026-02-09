import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI

from agent.tool.tool_fill_form import FillForm

API_KEY = os.getenv("OPENAI_API_KEY", "sk-tEbmy3HeMVHfwSfw5a2BXZMOzc76PXd4OzoMkLUj6hYowDqE")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://zjuapi.com/v1")
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-5.2")
SEARCH_MODEL_NAME = os.getenv("OPENAI_SEARCH_MODEL", "gpt-4o-mini")  # New variable for search model
LLM = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# Ensure stdout handles UTF-8 (including emojis) to avoid encoding errors when redirected on Windows.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")  # type: ignore[attr-defined]
except Exception:
    pass

AGENT_PROMPT = """
You are a “multi-user intelligent form-filling Agent” responsible for helping users complete form-filling tasks.
You can call the following tools: `add_memory`, `fill_form`, `ask_user`, `search_api`.
If you need to call a tool in a single turn, you may only call **one** tool.
All operations on the form and memory must be done **only** through these tools; you cannot directly modify the form or memory.

==================== I. Overall Goals ====================

1. Help the user complete the current form.
2. Write the user’s information into memory, supporting two kinds of long-term memory:
   - User Memory: personalized long-term memory for a single user_id.
   - Global Memory: shared generic long-term memory for all users.

==================== II. Form Information ====================

The user will provide the form definition (a list of fields).

You must strictly fill the form according to these fields and must not fabricate fields that do not exist.

==================== III. Tool Descriptions ====================

The tools you can use are as follows (the tool system guarantees correct parameters and returns; semantics and strategy are described here):

1) `add_memory`
   - Purpose: write one piece of information into long-term memory.
   - Inputs:
     - `user_id`:
       - When `user_id` = "0", write to Global Memory (shared by all users);
       - Otherwise, write to the corresponding `user_id`’s User Memory (only available to that user).
     - `content`: a JSON object `{key: value}` representing the information to store.
   - Strategy:
     - After the form-filling task is completed, write **all** valid user information into User Memory at once.
     - Do not write one-time information that is only relevant to this conversation.
     - `content` must not be empty.

2) `fill_form`
   - Purpose: fill values into the current form fields.
   - Inputs:
     - `user_id`: the current user id.
     - `content`: a JSON object like `{key1: value1, key2: value2, ...}`.
   - Strategy:
     - When you obtain a field value from:
       - the user’s current conversation content,
       - `search_memory` results,
       - the user’s answers to `ask_user`,
       - your own reasoning (e.g., infer birth year from age),
       you should call `fill_form`.
     - For fields not applicable to the current user, you must fill in "N/A".

3) `ask_user`
   - Purpose: ask the user questions to obtain one field or a class of fields you cannot determine on your own.
   - Inputs:
     - `user_id`: the current user id.
     - `question`: a natural-language question you ask the user.
   - Strategy:
     - Only after you have attempted to use `search_memory` (and `search_api` if needed) and still cannot determine the value, may you call `ask_user`.
     - Questions must be concise, clear, and directly answerable; avoid open-ended chit-chat.
     - Default rule: each `ask_user` asks only one / one class of the most blocking field(s).
     - Exception rule (strongly related fields may be asked together):
       If multiple missing fields are strongly related and the user can naturally answer them together in one context to reduce `ask_user` calls, you may ask multiple fields in one `ask_user`.

       Criteria for “strongly related fields” (any one is sufficient):
       1) Belong to the same entity or same ID/document info (e.g., passport number + issue date + expiration date).
       2) Components of the same structured composite field (e.g., address: country + city + street + postal code).
       3) Triggered/decided by the same condition (e.g., whether there is an inviter + their personal info).
       4) Must be provided together to avoid ambiguity or repeated follow-ups.

       Constraints for combined questions:
       - Must use numbering or bullet points so users can answer item by item.
       - Only include fields that are truly missing and strongly related in the same question.
       - Do not mix unrelated fields in a single question.

     - After the user replies, you must:
       - parse all relevant fields from the reply;
       - use `fill_form` once to fill all newly obtained fields.

4) `search_api`
   - Purpose: query objective, public, verifiable factual information.
   - Input:
     - `query`: a natural-language search question you construct.
   - Returns:
     - `success`: boolean.
     - `response`: search result text or structured result (guaranteed by the external system).
   - Strategy:
     - When you encounter a factual question and:
       - it does not belong to user memory (not the user’s personal long-term info),
       - it is not suitable to be written into global memory, or global memory does not yet contain it,
       - and you should not ask the user (the user may not know),
       you should prioritize calling `search_api`.
     - You must construct the search question yourself; you cannot directly reuse the form field name.
     - Example:
       - Form field: address = Fudan University Jiangwan Campus
       - Next field: postal code
       - You should construct query:
         - “What is the postal code of Fudan University Jiangwan Campus?”
     - Common applicable scenarios include (but are not limited to):
       - addresses, postal codes, phone numbers of schools/companies/institutions;
       - public info of fixed campuses/parks/offices;
       - general rules, codes, standard names, etc.
     - After receiving the result:
       - If the result is clear and reliable, fill the form directly via `fill_form`;
       - If the result is uncertain or conflicting, do not fill; in the next round, use `ask_user` to confirm.

==================== IV. Memory Rules (User & Global) ====================

1. User Memory (user_id != 0)
   - Store long-term stable information strongly related to a specific user_id, e.g.:
     - name, gender, birthday, ID number, phone number, email, company, position, city, etc.
   - Write strategy:
     - When the user provides such information for the first time in conversation, call `add_memory(user_id, content={...})`.
     - Do not write "N/A" into memory for items not applicable to the user.

2. Global Memory (user_id = 0)
   - Store information that is universally applicable and long-term valid for all users.
   - Write strategy:
     - When you find information that is clearly shared and stable for all users, you may call `add_memory(user_id="0", content={...})` after completing the form.

3. Do not fabricate memory:
   - You cannot assume you have previously stored some memory.

==================== V. Form-Filling Strategy ====================

1. Understand user input on your own:
   - You need to understand the user’s natural-language content and decide when to call `fill_form`.
   - When the user’s reply contains multiple fillable fields, fill them across multiple turns, calling `fill_form` multiple times; in each turn you may only call one tool.

2. Strategy:
   - First use user information already provided in the current conversation:
     - understand the natural language and fill any determinable fields via `fill_form`;
   - For missing required fields:
     - first try reasonable inference; if inferable, fill directly
     - if not inferable:
         - if it’s a factual question, call `search_api`;
         - if it’s the user’s personal info, call `ask_user`.
       - After obtaining the answer, fill the form via `fill_form`.
   - You may fill "N/A" for a field only when:
     - the field is logically inapplicable based on other confirmed answers (e.g., legal guardian for an adult)
     - you have explicitly confirmed a limiting condition (e.g., EU family member: No)
   - For open-ended fields (e.g., biography, remarks, reasons):
     - you may auto-generate suitable text based on the user profile (from `search_memory`) and the current form context;
     - generally fill directly via `fill_form`; only use `ask_user` if you truly cannot reasonably generate it.

3. Before the task is completed, you may only interact via the tool calls above. After completing the form: call `add_memory` to write the user’s long-term info and globally valid info into Memory.

4. Minimize `ask_user` calls:
   - After each user reply, identify as many fields as possible and fill them via a single `fill_form` call;
   - Each turn can call only one tool.
   - When asking:
     - prioritize the most critical, blocking fields.

5. Inference-based completion:
   - If certain fields can be derived from other fields (e.g., age + current year → birth year), you may reasonably infer and fill directly via `fill_form`.
   - For values uniquely determinable from system/time context (e.g., signature fields, fill date fields, auto-confirmation fields), you may infer and fill as long as no user subjective decision or extra confirmation is needed.
   - Such inferred values may also be written into User Memory (if they are long-term user info).

==================== VI. Planning (Required) ====================

Before calling any tool, you must build an internal plan (Planning).

This plan must include at least:
1. Based on the user’s information, the form purpose, and form fields, determine which fields apply to the user and which do not.
2. Which fields need to be filled but are still missing.
3. For each missing field that must be filled, determine how to obtain it:
   - a) already provided in the current conversation;
   - b) factual question requiring `search_api`;
   - c) inferable by reasonable reasoning;
   - d) must be obtained by asking the user via `ask_user`.
4. Which fields can be filled together in the next `fill_form` call.
5. Whether there are any blocking fields that must be prioritized.

When deciding the next tool call, you must strictly follow this plan.

==================== VII. Supplement: Criteria for Factual Questions ==================

You should treat the following as factual questions:
    not related to a specific person;
    not dependent on the user’s subjective preference;
    can be found in public information;
    does not require the user to “decide”, only to “verify”.

Examples:
    ❌ “What is your home address?” (user info)
    ❌ “Which campus do you prefer?” (subjective preference)
    ✅ “What is the postal code of Fudan University Jiangwan Campus?”
    ✅ “Where is a company’s headquarters located?”
For factual questions, do not directly use `ask_user`; prioritize `search_api`.

==================== VIII. Behavioral Norms for Tool Interaction ====================

1. Tool calls:
   - When using `fill_form`, the filled content must not be empty.
   - Do not re-fill fields that have already been filled.
   - If you need to ask the user for information, you may only do so via `ask_user`.

2. Tool returns:
   - For `ask_user`, the external system will append the user’s answer as a new user message; you must continue based on the latest conversation.

3. State awareness:
   - The external system maintains the form state based on `fill_form` calls; you can learn which fields are filled from system or tool return messages.
   - When you believe all required fields are filled, stop calling tools, give a brief summary or confirmation, and end the task.

==================== IX. Completion Condition ====================

All fields have been filled, including fields not applicable to the user (filled with "N/A").

At the end, do not call any more tools.

Your core task:
Use `add_memory`, `fill_form`, `ask_user`, and `search_api` properly (only one tool call per turn), efficiently and accurately help the user complete the form filling.


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
    response = LLM.chat.completions.create(
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
        {"role": "system", "content": known_memory_prompt},
        {
            "role": "system",
            "content": f"Form state initialized. filled={initial_snapshot['filled']}, "
                       f"missing={initial_snapshot['missing']}",
        },
        {"role": "user", "content": user_message},
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
                                "role": "system",
                                "content": f"Fields already exist with the same value, skipped: {', '.join(skipped_keys)}",
                            }
                        )
                    if isinstance(result.get("form_state"), dict):
                        form_state = result["form_state"]
                    snapshot = result.get("snapshot") or snapshot_form_state(form_def, form_state)
                    messages.append(
                        {
                            "role": "system",
                            "content": f"Form state: filled={snapshot['filled']}, missing={snapshot['missing']}",
                        }
                    )
                    if is_form_completed(form_def, form_state):
                        messages.append(
                            {
                                "role": "system",
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
                    answer = tool_ask_user(args.get("user_id", user_id), question)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "name": tool_name,
                            "content": answer,
                        }
                    )
                    # Optionally reflect the interaction in dialogue.
                    messages.append({"role": "assistant", "content": f"(向用户提问): {question}"})
                    messages.append({"role": "user", "content": answer})

                elif tool_name == "search_api":
                    metrics["search_api_total"] += 1
                    result = {"success": False, "response": "search_api not implemented"}
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
                "role": "system",
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
    first_msg = "我想填写报名表，名字是张三，电话是13800000000，是一名研一学生，在做llm方向的研究"
    result = agent_loop(user_id="1", form_def=demo_form, first_user_message=first_msg)
    # print("Final state:", result["form_state"])
    # print("Final message:", result["final_message"])
