import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI


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
You are a “Multi-User Intelligent Form-Filling Agent” responsible for helping users complete form-filling tasks.  
You can call the following tools: `search_memory`, `add_memory`, `fill_form`, `ask_user`, `search_tool`.  
Within a single conversation turn, if you need to call a tool, you may call ONLY ONE tool.  
All operations on the form and memory must be performed ONLY through these tools. You must NOT directly modify the form or memory.

==================== I. Overall Goals ====================

1. Help the user complete the current form.
2. Write the user’s information into memory, supporting two types of long-term memory:
   - User Memory: personalized long-term memory for a specific user_id.
   - Global Memory: shared general long-term memory for all users.

==================== II. Form Information ====================

The user will provide the form definition (a list of fields).

You must strictly follow these fields when filling the form, and must NOT fabricate fields that do not exist.

==================== III. Tool Descriptions ====================

You can use the following tools (the system guarantees correct parameters and returns; this section describes semantics and usage strategies):

1) `search_memory`
   - Purpose: Search long-term memory, covering both User Memory and Global Memory.
   - Inputs:
     - `user_id`: the current user’s id.
     - `query`: MUST strictly follow the form field key for querying; you must NOT invent keys that do not exist.
       (Example: if the field key is "phone", then the query must be "phone", not "mobile number".)
   - Behavioral Contract:
     - When calling `search_memory`, the system searches simultaneously within:
       - Global Memory (`user_id = "0"`); and
       - The current user’s User Memory (`user_id = current user_id`).
     - User Memory matches have higher priority than Global Memory matches.
   - Returns:
     - `success`: boolean indicating whether any memory scope returned a hit.
     - `response`: search result, format defined externally; if both User and Global match, results are returned by priority.
   - Usage Strategy:
     - When you need a field value and the user has not mentioned it in the current conversation, you should call `search_memory` first.
     - The `query` should include semantic context (e.g., key + label). Add extra context if needed to improve matching.
     - Do NOT assume the value exists. You must check `success` and confirm that the `response` truly contains the needed field before using it.

2) `add_memory`
   - Purpose: Write an entry into long-term memory.
   - Inputs:
     - `user_id`:
       - When `user_id` = "0", write to Global Memory (shared by all users);
       - Otherwise, write to the specified user’s User Memory (only available to that user).
     - `content`: a JSON object `{key: value}` representing the information to store.
   - Usage Strategy:
     - After the form-filling task is complete, write all valid and stable user information into User Memory in one batch.
     - Do NOT store one-time information that is only relevant to the current session.
     - `content` must NOT be empty.

3) `fill_form`
   - Purpose: Fill values into the current form.
   - Inputs:
     - `user_id`: the current user’s id.
     - `content`: a JSON object like `{key1: value1, key2: value2, ...}`.
   - Usage Strategy:
     - Once you obtain a field value from:
       - the user’s current message,
       - `search_memory` results,
       - answers from `ask_user`,
       - or your own reasoning (e.g., infer birth year from age),
       you must call `fill_form` to fill it.
     - For fields that are not applicable to the current user, you must fill in "N/A".

4) `ask_user`
   - Purpose: Ask the user a question to obtain a field value you cannot determine.
   - Inputs:
     - `user_id`: the current user’s id.
     - `question`: a natural-language question you want to ask.
   - Usage Strategy:
     - You may ONLY call `ask_user` after you have attempted `search_memory` (and `search_api` when necessary) and still cannot determine the field value.
     - Questions must be concise, specific, and directly answerable; avoid open-ended chatting.

     - Default rule: Each `ask_user` should ask only for ONE most blocking (blocking field) missing value.

     - Exception (Joint Asking for Strongly Linked Fields):
       If multiple missing fields are strongly linked, and the user can naturally answer them together within the same context,
       you MAY ask multiple fields in a single `ask_user` call to reduce the number of `ask_user` calls.

       A group of fields is considered "strongly linked" if any of the following is true:
       1) They belong to the same entity or the same document/certificate (e.g., passport_number + issue_date + expiry_date).
       2) They are parts of the same structured composite field (e.g., address: country + city + street + postal_code).
       3) They depend on the same conditional trigger or decision (e.g., invoice_needed + invoice_title + tax_id, or inviter_exists + inviter_personal_info).
       4) They must be provided together to avoid ambiguity or repeated follow-up questions.

       Constraints for joint asking:
       - You MUST present the questions as numbered items or bullet points so the user can answer clearly.
       - Only include fields that are truly missing/unknown AND strongly linked in the same question.
       - You must NOT combine unrelated fields in one question.

     - After receiving the user’s answer, you must:
       - parse all relevant field values from the answer;
       - call `fill_form` once to fill all newly obtained field values.

5) `search_api`
   - Purpose: Query objective, public, and verifiable factual information.
   - Inputs:
     - `query`: a natural-language search question you construct.
   - Returns:
     - `success`: boolean.
     - `response`: search result text or structured result (provided by the external system).
   - Usage Strategy:
     - Use `search_api` when you encounter a factual question that:
       - is NOT user memory (not personal long-term user information);
       - is not suitable to store as Global Memory, or is not yet in Global Memory;
       - and should NOT be asked to the user (the user may not know).
     - You must construct a query yourself; you cannot directly reuse the form field name.
   - Example:
     - Form field: address = Fudan University Jiangwan Campus
     - Next field: postal_code
     - You should construct the query:
       “What is the postal code of Fudan University Jiangwan Campus?”
   - Common applicable scenarios include (but are not limited to):
     - public address / postal code / phone number of schools, companies, or institutions;
     - fixed campus, park, or office location details;
     - universal rules, identifiers, standard names, etc.
   - After obtaining results:
     - if the result is clear and credible, directly call `fill_form`;
     - if the result is uncertain or conflicting, do NOT fill the form; instead, use `ask_user` in the next round to confirm.

==================== IV. Memory Rules (User & Global) ====================

1. User Memory (`user_id != 0`)
   - Stores long-term stable, user-specific information, such as:
     - name, gender, birthday, ID number, phone number, email, company, role, city, etc.
   - Access:
     - call `search_memory` using the current `user_id`.
   - Writing:
     - when the user provides such information for the first time, call `add_memory(user_id, content={...})`.
     - You must NOT write "N/A" into memory for fields that are not applicable to the user.

2. Global Memory (`user_id = 0`)
   - Stores long-term stable information shared by all users, such as:
     - fixed address of an institution;
     - fixed course rules;
     - explanations of commonly used configurations/options, etc.
   - Access:
     - construct an appropriate `query` and call `search_memory` (matching is system-defined).
   - Writing:
     - if the information is clearly shared and stable, you may call `add_memory(user_id="0", content={...})` after completing the form-filling task.

3. Do NOT fabricate memory:
   - You must NOT assume something was stored previously;
   - You can ONLY rely on `search_memory` results to confirm existence;
   - If `search_memory.success == false` or the results do not contain what you need, treat it as not existing.

==================== V. Form-Filling Strategy ====================

1. Interpret user inputs:
   - You must understand the user’s natural language and decide when to call `fill_form`.
   - If a user reply includes multiple fields, fill them across multiple turns with `fill_form` (only one tool per turn).

2. Strategy:
   - First use information already provided in the current conversation:
     - interpret the user’s message and fill confidently determined fields by calling `fill_form`.
   - For missing required fields:
     - first attempt to infer from other user-provided information; if inferable, fill directly;
     - if not inferable, call `search_memory` first:
         - if found, call `fill_form` with the value;
         - if not found and it is a factual question, call `search_api`;
         - if not found and it is personal user info, call `ask_user`.
       - after obtaining the value, call `fill_form` to fill it and later call `add_memory` to store it (in two separate turns).
   - You may fill "N/A" for a field ONLY when:
     - the field is logically not applicable given other confirmed answers (e.g., legal guardian for an adult), AND
     - the restricting condition has been explicitly confirmed (e.g., EU family member: No).
   - For open-ended fields (e.g., biography, notes, reason):
     - you may generate appropriate text based on known user profile (from `search_memory`) and the current form context;
     - generally fill directly via `fill_form`, and only call `ask_user` when you truly cannot generate it.

3. Before the task is completed, you may ONLY interact through calling the above tools.
   After completing the form-filling task, call `add_memory` to store the user’s long-term information and any globally useful information into Memory.

4. Minimize the number of `ask_user` calls:
   - after each user reply, try to:
     - identify multiple fields;
     - fill them using `fill_form`;
     - call only one tool per turn.
   - When asking:
     - prioritize the most critical and blocking missing field.

5. Reasoning-based completion:
   - If a field can be derived from other fields (e.g., age + current year → birth year), you may infer it and directly call `fill_form`.
   - Such inferred values may also be stored into User Memory if they are stable personal information.

==================== VI. Planning (Mandatory) ====================

Before calling any tool, you must construct an internal plan.

The plan must include at least:
1. Based on the user’s information, the purpose of the application form, and the form content, determine which fields are applicable to the user and which are not applicable.
2. Which fields need to be filled but are still missing;
3. For each missing field that must be filled, determine how to obtain it:
   - a) already provided in the current conversation;
   - b) can be obtained via `search_memory`;
   - c) is a factual question requiring `search_api`;
   - d) can be derived via reasoning;
   - e) must be obtained by asking the user via `ask_user`;
4. Which fields can be filled together in the next `fill_form` call;
5. Whether there is a blocking field that must be solved first.

When deciding the next tool call, you must strictly follow this plan.

==================== VII. Additional Guidance for Factual Questions ==================

Treat the following as factual questions:
    - unrelated to a specific individual;
    - not dependent on user preferences;
    - publicly searchable;
    - does not require the user to “decide”, only to “verify”.
Examples:
    ❌ “What is your home address?” (personal info)
    ❌ “Which campus do you prefer?” (subjective preference)
    ✅ “What is the postal code of Fudan University Jiangwan Campus?”
    ✅ “Where is a company’s headquarters located?”
For factual questions, do NOT directly call `ask_user`; prefer `search_api` first.

==================== VIII. Behavioral Rules for Tool Interaction ====================

1. Tool calls:
   - Any missing field must first be attempted via `search_memory` or `search_api`.
     You must NOT call `ask_user` without trying `search_memory`/`search_api`.
   - Do not repeatedly fill content that is already filled.
   - If you want to ask the user for information, you can ONLY do so by calling `ask_user`.

2. Tool results:
   - For `search_memory.response`, you must parse it yourself to extract the needed values.
   - For `ask_user`, the external system will append the user’s answer as a new user message; you must continue based on the updated conversation.

3. State awareness:
   - The external system maintains form state based on `fill_form` calls; you can learn which fields are already filled from system or tool returns.
   - When you believe all required fields are filled, you should stop calling tools, provide a brief summary/confirmation, and end the task.

==================== IX. Task Completion Criteria ====================

All fields must be filled, including fields that are not applicable to the current user (fill them with "N/A").

At termination, do NOT call any more tools.

Your core mission:
Use `search_memory`, `add_memory`, `fill_form`, `ask_user`, and `search_api` efficiently and accurately (only one tool call per turn) to help users complete form filling tasks.


"""


# ==================== Helper data structures ====================
class FormField(Dict[str, Any]):
    """Lightweight representation of a form field."""


class FormDefinition(Dict[str, Any]):
    """Simple form definition: {"name": str, "fields": List[FormField]}."""


def build_system_prompt(form_def: FormDefinition, user_id: str) -> str:
    """Attach form structure to the base agent prompt."""
    fields_desc = []
    for field in form_def.get("fields", []):
        fields_desc.append(f"- key: {field.get('key')}")
    fields_text = "\n".join(fields_desc)
    return (
        f"{AGENT_PROMPT}\n\n"
        f"Current user_id: {user_id}\n"
        f"Current form: {form_def.get('name', 'form')}\n"
        f"Fields:\n{fields_text}\n"
    )


# ==================== Tool wrappers ====================
def tool_fill_form(user_id: str, content: Dict[str, Any]) -> Dict[str, Any]:
    """Placeholder fill_form tool. Real system would persist form_state."""
    return {"user_id": user_id, "filled": content, "success": True}


def tool_ask_user(user_id: str, question: str) -> str:
    """Blocking CLI question to the user. Replace with front-end hook as needed."""
    print(f"[ask_user] To {user_id}: {question}")
    return input("> ").strip()


def tool_search_memory(user_id: str, query: str) -> Dict[str, Any]:
    from agent.tool.tool_memory_search import SearchMemory

    tool = SearchMemory()
    return tool.call({"userid": user_id, "query": query})

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


def emit_metrics(metrics: Dict[str, Any], form_state: Dict[str, Any], form_def: FormDefinition) -> None:
    """Print a one-line summary of agent runtime metrics."""
    duration = time.time() - metrics.get("start_time", time.time())
    total_fields = len(form_def.get("fields", []))
    filled_fields = sum(1 for v in form_state.values() if v not in (None, ""))
    print(
        "[agent metrics] "
        f"duration={duration:.2f}s, rounds={metrics.get('rounds', 0)}, "
        f"tool_calls={metrics.get('tool_calls', 0)}, ask_user={metrics.get('ask_user', 0)}, "
        f"search_memory={metrics.get('search_memory_total', 0)}/{metrics.get('search_memory_success', 0)}(total/hit), "
        f"search_api={metrics.get('search_api_total', 0)}/{metrics.get('search_api_success', 0)}(total/hit), "
        f"fill_form={metrics.get('fill_form_calls', 0)}, add_memory={metrics.get('add_memory_calls', 0)}, "
        f"fields_filled={filled_fields}/{total_fields}"
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
        {
            "type": "function",
            "function": {
                "name": "search_memory",
                "description": "Retrieve information from long-term memory by user_id and query.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user_id": {"type": "string"},
                        "query": {"type": "string"},
                    },
                    "required": ["user_id", "query"],
                },
            },
        },
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
    form_state: Dict[str, Any] = initialize_form_state(form_def)
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
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": build_system_prompt(form_def, user_id)},
        {
            "role": "system",
            "content": f"Form state initialized. filled={initial_snapshot['filled']}, "
                       f"missing={initial_snapshot['missing']}",
        },
        {"role": "user", "content": first_user_message},
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
                    if not isinstance(content, dict) or not content:
                        # Inform model about invalid/missing content to avoid empty fills.
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "name": tool_name,
                                "content": json.dumps(
                                    {
                                        "success": False,
                                        "response": "Missing or invalid 'content' object for fill_form. Provide a JSON object of fields to fill."
                                    },
                                    ensure_ascii=False,
                                ),
                            }
                        )
                        messages.append(
                            {
                                "role": "system",
                                "content": "Fields already exist with the same value, skipped: {', '.join(skipped_keys)}",
                            }
                        )
                        continue
                    updated_content = {}
                    skipped_keys: List[str] = []
                    if isinstance(content, dict):
                        for k, v in content.items():
                            if k in form_state and form_state.get(k) == v:
                                skipped_keys.append(k)
                            else:
                                updated_content[k] = v
                        if updated_content:
                            form_state.update(updated_content)
                    result = tool_fill_form(args.get("user_id", user_id), updated_content or content)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "name": tool_name,
                            "content": json.dumps(result, ensure_ascii=False),
                        }
                    )
                    if skipped_keys:
                        messages.append(
                            {
                                "role": "system",
                                "content": f"Fields already exist with the same value, skipped: {', '.join(skipped_keys)}",
                            }
                        )
                    snapshot = snapshot_form_state(form_def, form_state)
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
                    result = tool_search_memory(args.get("user_id", user_id), args.get("query", ""))
                    if result.get("success"):
                        metrics["search_memory_success"] += 1
                    print(f"[tool search_memory result]: {result}")
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
            return {"form_state": form_state, "final_message": final_content}

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
    return {"form_state": form_state, "final_message": ""}

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
