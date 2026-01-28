import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from openai import OpenAI

from qwen_agent.tools.base import BaseTool

API_KEY = os.getenv("OPENAI_API_KEY", "sk-tEbmy3HeMVHfwSfw5a2BXZMOzc76PXd4OzoMkLUj6hYowDqE")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://zjuapi.com/v1")
SEARCH_MODEL_NAME = os.getenv("OPENAI_SEARCH_MODEL", "gpt-4o-mini")  # New variable for search model
LLM = OpenAI(api_key=API_KEY, base_url=BASE_URL)

class SearchMemory(BaseTool):
    name = "searchmemory"
    description = (
        "Look up a value by key for a given user_id in memory.json. "
        "Returns {'response': <value or message>, 'success': <bool>}."
    )
    parameters = {
        "type": "object",
        "properties": {
            "userid": {
                "type": "string",
                "description": "The user_id whose memory should be searched.",
            },
            "query": {
                "type": "string",
                "description": "The key to search for in the user's information list.",
            },
        },
        "required": ["userid", "query"],
    }

    def __init__(self, cfg: Optional[dict] = None, memory_path: Optional[Path] = None):
        super().__init__(cfg)
        if memory_path is None:
            memory_path = Path(__file__).resolve().parent.parent / "memory.json"
        self.memory_path = Path(memory_path)

    def _load_memory(self) -> List[Dict[str, Any]]:
        """Safely load memory records from disk."""
        if not self.memory_path.exists():
            return []

        raw = self.memory_path.read_text(encoding="utf-8").strip()
        if not raw:
            return []

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return []

        if isinstance(data, dict):
            return [data]
        if isinstance(data, list):
            return data
        return []

    @staticmethod
    def _find_user_record(memory: List[Dict[str, Any]], user_id: str) -> Optional[Dict[str, Any]]:
        for record in memory:
            if str(record.get("user_id", "")) == str(user_id):
                return record
        return None

    @staticmethod
    def _lookup_value(info_list: Any, query: str) -> Optional[Any]:
        if not isinstance(info_list, list):
            return None

        q_lower = str(query).lower()
        for item in info_list:
            if not isinstance(item, dict):
                continue
            key = str(item.get("key", "")).lower()
            if key == q_lower:
                return item.get("value")
        return None

    def call(self, params: Union[str, dict], **kwargs) -> Dict[str, Any]:
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except json.JSONDecodeError:
                return {"response": "Invalid params: expected JSON with userid and query.", "success": False}

        if not isinstance(params, dict):
            return {"response": "Params must be a JSON object with userid and query.", "success": False}

        user_id = params.get("userid")
        query = params.get("query")
        if user_id is None or query is None:
            return {"response": "Missing userid or query.", "success": False}

        memory_records = self._load_memory()
        if not memory_records:
            return {"response": "Memory is empty", "success": False}

        # Search the user's memory; fall back to global memory with user_id '0' if needed.
        record = self._find_user_record(memory_records, user_id) or self._find_user_record(memory_records, "0")
        if record is None:
            return {"response": f"No memory found for user_id.", "success": False}

        information = record.get("information", [])
        if not information:
            return {"response": f"No information stored for user_id {record.get('user_id')}.", "success": False}

        # Let the LLM pick the best-matching value even if the query wording differs.
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a retrieval helper. Given a query and a list of memory entries "
                    "(each has 'key' and 'value'), return the best matching value even when "
                    "the query wording is different (e.g., 'tele number' vs 'mobile number'). "
                    "Respond ONLY with JSON: {\"success\": true/false, \"response\": <value or short message>}."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {"query": query, "memory": information},
                    ensure_ascii=False,
                ),
            },
        ]

        try:
            llm_resp = LLM.chat.completions.create(
                model=SEARCH_MODEL_NAME,
                messages=messages,
                temperature=0,
            )
            print(llm_resp)
            content = llm_resp.choices[0].message.content or ""
        except Exception as exc:  # pragma: no cover - defensive path
            return {"response": f"LLM search failed: {exc}", "success": False}

        try:
            parsed = json.loads(content)
            return {
                "response": parsed.get("response", content),
                "success": bool(parsed.get("success")),
            }
        except json.JSONDecodeError:
            # Fall back to raw content if JSON parsing fails.
            return {"response": content, "success": bool(content.strip())}
