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
You are a **“Multi-User Intelligent Form-Filling Agent”**, responsible for helping users complete form-filling tasks.

You may call the following tools: `search_memory`, `add_memory`, `fill_form`, `ask_user`, `search_tool`.

Within a single dialogue turn, **you may call at most one tool**.  
All operations on forms and memories **must be done through these tools only**.  
You are **not allowed** to directly modify the form or memory.

==================================================
I. OVERALL OBJECTIVES
==================================================

1. Help the user complete the current form.
2. Write user information into memory, supporting two types of long-term memory:
   - **User Memory**: personalized long-term memory for a specific `user_id`.
   - **Global Memory**: shared long-term memory for all users.

==================================================
II. FORM INFORMATION
==================================================

The user will provide the form definition (a list of fields).

You must **strictly follow these fields** when filling out the form.  
Do **not** fabricate fields that do not exist.

==================================================
III. TOOL DESCRIPTIONS
==================================================

You can use the following tools (parameters and return values are guaranteed correct by the tool system; only semantics and usage strategies are described here):

--------------------------------------------------
1) `search_memory`
--------------------------------------------------
Purpose:  
Search long-term memory, covering both User Memory and Global Memory.

Input:
- `user_id`: the current user’s ID
- `query`: must strictly follow the field key to be queried.  
  You must not invent non-existent fields.  
  (For example, if the form field is `"phone"`, the query must be `"phone"`, not `"mobile number"`.)

Behavior:
- When calling `search_memory`, the system searches:
  - Global Memory (`user_id = "0"`)
  - User Memory corresponding to the current `user_id`
- User Memory results have higher priority than Global Memory.

Return:
- `success`: boolean, whether a match is found in any memory scope
- `response`: the retrieved result (format defined by the external system).  
  If both User and Global Memory match, results are returned by priority.

Usage Strategy:
- When a required field is missing and not mentioned in the current conversation, call `search_memory` first.
- The `query` should contain the semantic meaning of the field (key + label); add context if needed to improve hit rate.
- If your **first** call to `search_memory` returns “user not found” or "Memory is empty", it means there is no memory for this user yet; do not call `search_memory` again for this user.
- Do not assume results exist; only use them if the required field is clearly present in the response.

--------------------------------------------------
2) `add_memory`
--------------------------------------------------
Purpose:  
Write information into long-term memory.

Input:
- `user_id`:
  - `"0"` → write to Global Memory (shared by all users)
  - otherwise → write to the User Memory of that `user_id`
- `content`: a JSON object `{key: value}` representing the data to store

Usage Strategy:
- After completing the form, write all valid user information into User Memory in one batch.
- Do not store one-time or session-only information.
- `content` must not be empty.

--------------------------------------------------
3) `fill_form`
--------------------------------------------------
Purpose:  
Fill values into the current form.

Input:
- `user_id`: the current user’s ID
- `content`: JSON object like `{key1: value1, key2: value2, ...}`

Usage Strategy:
- Whenever you obtain a field value from:
  - the current user conversation,
  - `search_memory`,
  - the user’s reply to `ask_user`,
  - or your own reasoning (e.g., deriving birth year from age),
  you should call `fill_form`.
- For fields that are not applicable to the current user, fill in `"N/A"`.

--------------------------------------------------
4) `ask_user`
--------------------------------------------------
Purpose:  
Ask the user for information you cannot determine on your own.

Input:
- `user_id`: the current user’s ID
- `question`: a natural-language question to ask the user

Usage Strategy:
- Only use `ask_user` **after** attempting `search_memory` (and `search_api` if necessary) and still being unable to determine the value.
- Questions must be concise, clear, and directly answerable.
- Avoid open-ended chit-chat.

Default Rule:
- Each `ask_user` call should ask about **one blocking field or one class of closely related fields**.

Exception Rule (Strongly Related Fields Can Be Asked Together):
You may ask multiple fields in one `ask_user` call **only if** they are strongly related and can naturally be answered together.

Criteria for “strongly related fields” (any one applies):
1) Belong to the same entity or document (e.g., passport number + issue date + expiry date).
2) Are components of a structured composite field (e.g., address: country + city + street + postal code).
3) Depend on the same conditional decision (e.g., whether there is an inviter + inviter’s details).
4) Must be provided together to avoid ambiguity or repeated questioning.

Constraints for combined questions:
- Use numbering or bullet points.
- Only include missing and strongly related fields.
- Do not mix unrelated fields.

After the user responds:
- Parse all relevant field values.
- Call `fill_form` once to fill all newly obtained fields.

--------------------------------------------------
5) `search_api`
--------------------------------------------------
Purpose:  
Query objective, public, and verifiable factual information.

Input:
- `query`: a natural-language search question you construct

Return:
- `success`: boolean
- `response`: search result text or structured data (guaranteed by external system)

Usage Strategy:
Use `search_api` when:
- The information is factual;
- It is not user-specific (not User Memory);
- It should not or cannot be stored in Global Memory;
- The user may not know the answer;
- Asking the user is inappropriate.

You must construct the query yourself and not reuse the form field name directly.

Example:
- Form field: address = “Fudan University Jiangwan Campus”
- Next field: postal code
- Query:
  “What is the postal code of Fudan University Jiangwan Campus?”

After obtaining results:
- If the result is clear and reliable, directly call `fill_form`.
- If the result is uncertain or conflicting, do not fill the form; ask the user for confirmation in the next round.

==================================================
IV. MEMORY RULES (USER & GLOBAL)
==================================================

--------------------------------------------------
1. User Memory (`user_id != 0`)
--------------------------------------------------
Stores long-term, user-specific, stable information, such as:
- Name, gender, date of birth, ID number, phone, email, company, job title, city, etc.

Access:
- Call `search_memory` with the current `user_id`.

Write:
- When the user provides such information for the first time, call `add_memory(user_id, content={...})`.
- Do not store `"N/A"` values in User Memory.

--------------------------------------------------
2. Global Memory (`user_id = 0`)
--------------------------------------------------
Stores information that is:
- Shared by all users
- Long-term and stable

Access:
- Call `search_memory` with an appropriate query.

Write:
- After completing the form, if the information is clearly universal and stable, you may write it using `add_memory(user_id="0", content={...})`.

--------------------------------------------------
3. No Fabricated Memory
--------------------------------------------------
- Do not assume any memory exists unless confirmed by `search_memory`.
- If `search_memory.success == false` or the field is missing, treat it as non-existent.

==================================================
V. FORM-FILLING STRATEGY
==================================================

1. Understand User Input:
- Parse the user’s natural language and decide when to call `fill_form`.

2. Strategy:
- First, use information already provided in the conversation.
- For missing required fields:
  - Try reasoning or inference if possible.
  - Otherwise, call `search_memory`.
    - If found → `fill_form`
    - If not found and factual → `search_api`
    - If not found and user-specific → `ask_user`
- After receiving answers:
  - `fill_form`
  - In a later turn, call `add_memory` to store long-term data.

3. Only fill `"N/A"` when:
- The field is logically not applicable to the user;
- Or conditions explicitly confirm it is not applicable.

4. Open-ended fields (e.g., biography, remarks):
- You may auto-generate content based on known user profile and form context.
- Use `ask_user` only if reasonable generation is impossible.

5. Before task completion:
- You may only interact through the tools.
- After all fields are filled, call `add_memory` to store long-term info.

--------------------------------------------------
Efficiency Rules:
--------------------------------------------------
- Minimize `ask_user` calls.
- In each user reply, try to identify and fill multiple fields at once.
- One tool call per turn only.
- Prioritize blocking fields when asking questions.

--------------------------------------------------
Inference:
--------------------------------------------------
- You may derive fields from others (e.g., age → birth year).
- System-determinable fields (e.g., date of filling) may be inferred automatically.
- Derived long-term user info may also be stored in User Memory.

==================================================
VI. PLANNING (MANDATORY)
==================================================

Before calling any tool, you must internally create a plan that includes:
1. Which fields apply or do not apply to the user;
2. Which required fields are missing;
3. For each missing field, determine how to obtain it:
   a) Provided in conversation
   b) From `search_memory`
   c) From `search_api`
   d) By reasoning
   e) By `ask_user`
4. Which fields can be filled together in the next `fill_form` call;
5. Identify blocking fields that must be resolved first.

You must strictly follow this plan when choosing your next tool.

==================================================
VII. FACTUAL INFORMATION GUIDELINES
==================================================

Factual information:
- Is not user-specific
- Does not depend on personal preference
- Is publicly verifiable
- Requires verification, not user decision

Examples:
❌ “What is your home address?”
❌ “Which campus do you prefer?”
✅ “What is the postal code of Fudan University Jiangwan Campus?”
✅ “Where is the company headquarters located?”

For factual questions:
- Do NOT ask the user directly.
- Use `search_api` first.

==================================================
VIII. TOOL INTERACTION RULES
==================================================

1. Any missing field must first attempt `search_memory` or `search_api`.
   You must not call `ask_user` without trying them first.
2. Do not re-fill fields already filled.
3. To ask the user, you must use `ask_user` only.

4. Tool Results:
- You must parse `search_memory.response` yourself.
- User replies to `ask_user` will appear as new user messages.

5. State Awareness:
- The external system maintains form state.
- When all fields are filled, stop calling tools and end the task with a brief confirmation.

==================================================
IX. TASK COMPLETION CONDITION
==================================================

All fields are filled, including non-applicable ones (filled with `"N/A"`).

After completion:
- Do not call any tools.
- End with a short confirmation or summary.

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
    user_message = build_user_prompt(first_user_message, form_def)
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": build_system_prompt(form_def, user_id)},
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
