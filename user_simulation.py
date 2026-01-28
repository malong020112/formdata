import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

from openai import OpenAI

API_KEY = os.getenv("OPENAI_API_KEY", "sk-tEbmy3HeMVHfwSfw5a2BXZMOzc76PXd4OzoMkLUj6hYowDqE")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://zjuapi.com/v1")
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-5.2")
LLM = OpenAI(api_key=API_KEY, base_url=BASE_URL)


def load_json(path: Path) -> Any:
    """Load JSON; supports standard JSON or JSON-lines (one JSON per line)."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    items: List[Any] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if not items:
        raise ValueError(f"Failed to parse JSON from {path}")
    return items


USER_PROMPT_PREFIX = """
You will play the role of a user seeking help from a form-fill agent.
Your goal is to have the assistant successfully solve the initial task you provide.

Please follow these interaction rules carefully:
1. You are acting purely as the user — not as the assistant.
2. If the agent asks about your information, always respond truthfully based on the provided data.
3. You only need to answer the agent's questions as a user, and you don't need to ask questions in return.You should answer questions directly and briefly.
4. Do not repeat the assistant's questions back to it.

Below are the attributes and background of the problem you want help with:
"""
USER_PROMPT_SUFFIX = """
The above user information is entirely fictional and does not correspond to any real person. You need to simulate such a user based on this information. When the assistant asks you about this information, you must provide it.
"""

def build_user_system_prompt(user_data: Dict[str, Any]) -> str:
    return f"{USER_PROMPT_PREFIX}{json.dumps(user_data, ensure_ascii=False)}{USER_PROMPT_SUFFIX}"


def user_llm_reply(user_data: Dict[str, Any], history: List[Dict[str, str]], prompt: str) -> str:
    sys_prompt = build_user_system_prompt(user_data)
    messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": prompt}]
    resp = LLM.chat.completions.create(model=MODEL_NAME, messages=messages, temperature=0)
    return resp.choices[0].message.content.strip()


def get_first_prompt(user_data: Dict[str, Any], form_def: Dict[str, Any]) -> str:
    """Let the LLM craft a brief self-intro + form-fill request based on user_data."""
    form_name = form_def.get("name", "Schengen Visa Application Form")
    sys_prompt = (
        "You are a user profile transcriber. Please generate a conversational self-introduction based on the provided user data."
        "And made a request to fill out the target form. No need to include all the fields, just mention the important information in 3-4 sentences."
        "And indicate the name of the form that needs to be filled out. Just provide it in a casual, user-friendly way."
    )
    messages = [
        {"role": "system", "content": sys_prompt},
        {
            "role": "user",
            "content": f"form name: {form_name}\nuser data: {json.dumps(user_data, ensure_ascii=False)}",
        },
    ]
    resp = LLM.chat.completions.create(model=MODEL_NAME, messages=messages, temperature=0.3)
    return resp.choices[0].message.content.strip()


def simulate(form_path: Path, profile_path: Path, count: int) -> None:
    import agent.agent as ag

    def _load_trace(path: Path) -> List[Dict[str, Any]]:
        if not path.exists():
            return []
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            return []
        data = json.loads(text)
        return data if isinstance(data, list) else []

    def _save_trace(path: Path, payload: List[Dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    form_def = load_json(form_path)
    # Normalize nested fields to a flat list of dotted keys for agent compatibility.
    def _flatten_fields(node: Any, prefix: str = "") -> List[Dict[str, str]]:
        flat: List[Dict[str, str]] = []
        if isinstance(node, dict):
            for k, v in node.items():
                new_prefix = f"{prefix}.{k}" if prefix else k
                if isinstance(v, dict):
                    flat.extend(_flatten_fields(v, new_prefix))
                else:
                    flat.append({"key": new_prefix})
        else:
            if prefix:
                flat.append({"key": prefix})
        return flat

    if isinstance(form_def, dict) and isinstance(form_def.get("fields"), dict):
        form_def["fields"] = _flatten_fields(form_def["fields"])

    print(f"form_def: {form_def}\n")
    print("=== Starting user simulation ===\n")
    profiles = load_json(profile_path)
    profile_list: List[Dict[str, Any]]
    if isinstance(profiles, list):
        profile_list = profiles
    else:
        profile_list = [profiles]
    if count > 0:
        profile_list = profile_list[:count]

    original_ask_user = ag.tool_ask_user
    original_fill_form = ag.tool_fill_form
    original_search_memory = ag.tool_search_memory
    original_add_memory = ag.tool_add_memory
    original_search_api = ag.tool_search_api if hasattr(ag, "tool_search_api") else None
    try:
        for idx, user_data in enumerate(profile_list):
            user_history: List[Dict[str, str]] = []

            def user_answer(question: str) -> str:
                user_history.append({"role": "assistant", "content": question})
                answer = user_llm_reply(
                    user_data,
                    user_history,
                    prompt=f"Question:\n{question}\n\nAnswer as the user using the provided data.",
                )
                user_history.append({"role": "user", "content": answer})
                print(f"[User reply]: {answer}")
                return answer

            # 首条用户发言由大模型根据用户数据生成自我介绍 + 填表请求
            first_message = get_first_prompt(user_data, form_def)
            first_message = ag.build_user_prompt(first_message, form_def)
            user_history.append({"role": "user", "content": first_message})
            print(f"[User first message]: {first_message}")

            def patched_ask_user(uid: str, question: str) -> str:
                return user_answer(question)

            def patched_fill_form(uid: str, content: Dict[str, Any]) -> Dict[str, Any]:
                return original_fill_form(uid, content)

            def patched_search_memory(uid: str, query: str) -> Dict[str, Any]:
                return original_search_memory(uid, query)

            def patched_add_memory(role: str, uid: str, content: Dict[str, Any]) -> Dict[str, Any]:
                return original_add_memory(role=role, uid=uid, content=content)

            def patched_search_api(query: str) -> Dict[str, Any]:
                return original_search_api(query) if original_search_api else {"success": False, "response": "N/A"}

            ag.tool_ask_user = patched_ask_user
            ag.tool_fill_form = patched_fill_form
            ag.tool_search_memory = patched_search_memory
            ag.tool_add_memory = patched_add_memory
            if original_search_api:
                ag.tool_search_api = patched_search_api

            current_user_id = str(user_data.get("uid", idx + 1))
            result = ag.agent_loop(user_id=current_user_id, form_def=form_def, first_user_message=first_message)

            final_message = ""
            metrics = {}
            trace_messages: List[Dict[str, Any]] = []
            if result is not None:
                final_message = result.get("final_message") or ""
                metrics = result.get("metrics") or {}
                raw_messages = result.get("messages") or []
                if isinstance(raw_messages, list):
                    for msg in raw_messages:
                        if not isinstance(msg, dict):
                            continue
                        role = msg.get("role") or ""
                        content = msg.get("content") or ""
                        tool_calls = msg.get("tool_calls") if role == "assistant" else None
                        tool_name = msg.get("name") if role == "tool" else ""
                        trace_messages.append(
                            {
                                "role": role,
                                "content": content,
                                "tool_calls": tool_calls,
                                "tool_name": tool_name or "",
                            }
                        )

            trace_entry = {
                "name": current_user_id,
                "data": [
                    {
                        "type": user_data.get("type") or form_def.get("name", ""),
                        "metric": metrics,
                        "messages": trace_messages,
                    }
                ],
            }
            trace_path = profile_path.parent / "trace.json"
            existing = _load_trace(trace_path)
            target = None
            for item in existing:
                if isinstance(item, dict) and str(item.get("name")) == str(current_user_id):
                    target = item
                    break
            if target:
                target.setdefault("data", [])
                target["data"].extend(trace_entry["data"])
            else:
                existing.append(trace_entry)
            _save_trace(trace_path, existing)
    finally:
        ag.tool_ask_user = original_ask_user
        ag.tool_fill_form = original_fill_form
        ag.tool_search_memory = original_search_memory
        ag.tool_add_memory = original_add_memory
        if original_search_api:
            ag.tool_search_api = original_search_api


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate a user filling a form via the agent.")
    parser.add_argument(
        "--form",
        required=False,
        default=Path(r"E:\project\主动提问\formdata\visa_form\Schengen\Schengen_visa_form.json"),
        type=Path,
        help="Path to form definition JSON file.",
    )
    parser.add_argument(
        "--profile",
        required=False,
        default=Path(r"E:\project\主动提问\formdata\visa_data\user_data.json"),
        type=Path,
        help="Path to user data JSON file (completed forms).",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=2,
        help="Number of user records to simulate (from the top of the profile file).",
    )
    args = parser.parse_args()

    # simulate(form_path=args.form, profile_path=args.profile, count=args.count)
    # simulate(form_path=args.form, profile_path=args.profile, count=1)
    # simulate(form_path=Path(r"E:\project\主动提问\formdata\visa_form\Japan\Japan_visa_form.json"), profile_path=args.profile, count=1)

if __name__ == "__main__":
    main()
