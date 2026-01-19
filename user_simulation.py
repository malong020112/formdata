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
    messages = [{"role": "system", "content": sys_prompt}] + history + [{"role": "user", "content": prompt}]
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
    try:
        for idx, user_data in enumerate(profile_list):
            user_history: List[Dict[str, str]] = []

            def user_answer(question: str) -> str:
                user_history.append({"role": "assistant", "content": question})
                answer = user_llm_reply(user_data, user_history, prompt="Now, please answer the question as a user.")
                user_history.append({"role": "user", "content": answer})
                print(f"[User reply]: {answer}")
                return answer

            # 首条用户发言由大模型根据用户数据生成自我介绍 + 填表请求
            first_message = get_first_prompt(user_data, form_def)
            user_history.append({"role": "user", "content": first_message})
            print(f"[User first message]: {first_message}")

            def patched_ask_user(uid: str, question: str) -> str:
                return user_answer(question)

            ag.tool_ask_user = patched_ask_user

            current_user_id = str(user_data.get("uid", idx + 1))
            result = ag.agent_loop(user_id=current_user_id, form_def=form_def, first_user_message=first_message)

            print(f"=== Simulation finished for user {current_user_id} ===")
            if result is not None:
                print("Final state:", result.get("form_state"))
                print("Final message:", result.get("final_message"))
            print("\n")
    finally:
        ag.tool_ask_user = original_ask_user


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
    simulate(form_path=args.form, profile_path=args.profile, count=1)
    simulate(form_path=Path(r"E:\project\主动提问\formdata\visa_form\Japan\Japan_visa_form.json"), profile_path=args.profile, count=1)

if __name__ == "__main__":
    main()
