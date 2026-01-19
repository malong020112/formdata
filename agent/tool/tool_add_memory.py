import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from qwen_agent.tools.base import BaseTool


class AddMemory(BaseTool):
    name = "addmemory"
    description = (
        "Add or update a key/value pair in memory.json for a given user_id. "
        "Params: userid (str), content: [key, value]. Returns response and success."
    )
    parameters = {
        "type": "object",
        "properties": {
            "userid": {
                "type": "string",
                "description": "The user_id whose memory should be updated.",
            },
            "content": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 2,
                "maxItems": 2,
                "description": "Two-element array: [key, value].",
            },
        },
        "required": ["userid", "content"],
    }

    def __init__(self, cfg: Optional[dict] = None, memory_path: Optional[Path] = None):
        super().__init__(cfg)
        if memory_path is None:
            memory_path = Path(__file__).resolve().parent.parent / "memory.json"
        self.memory_path = Path(memory_path)

    def _load_memory(self) -> List[Dict[str, Any]]:
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
    def _find_or_create_record(memory: List[Dict[str, Any]], user_id: str) -> Dict[str, Any]:
        for record in memory:
            if str(record.get("user_id", "")) == str(user_id):
                if "information" not in record or not isinstance(record["information"], list):
                    record["information"] = []
                return record

        new_record = {"user_id": str(user_id), "information": []}
        memory.append(new_record)
        return new_record

    @staticmethod
    def _parse_content(content: Any) -> Optional[Dict[str, str]]:
        if isinstance(content, list) and len(content) >= 2:
            return {"key": str(content[0]), "value": content[1]}
        return None

    @staticmethod
    def _upsert_information(record: Dict[str, Any], key: str, value: Any) -> None:
        info_list = record.get("information")
        if not isinstance(info_list, list):
            info_list = []
            record["information"] = info_list

        key_lower = key.lower()
        for item in info_list:
            if isinstance(item, dict) and str(item.get("key", "")).lower() == key_lower:
                item["value"] = value
                return
        info_list.append({"key": key, "value": value})

    def _save_memory(self, memory: List[Dict[str, Any]]) -> bool:
        try:
            self.memory_path.write_text(json.dumps(memory, ensure_ascii=False, indent=2), encoding="utf-8")
            return True
        except Exception:
            return False

    def call(self, params: Union[str, dict], **kwargs) -> Dict[str, Any]:
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except json.JSONDecodeError:
                return {"response": "Invalid params: expected JSON with userid and content.", "success": False}

        if not isinstance(params, dict):
            return {"response": "Params must be a JSON object with userid and content.", "success": False}

        user_id = params.get("userid")
        content = params.get("content")
        parsed = self._parse_content(content)
        if user_id is None or parsed is None:
            return {"response": "Missing userid or invalid content; expected [key, value].", "success": False}

        memory_records = self._load_memory()
        record = self._find_or_create_record(memory_records, user_id)

        self._upsert_information(record, parsed["key"], parsed["value"])

        if not self._save_memory(memory_records):
            return {"response": "Failed to save memory.", "success": False}

        return {"response": f"Stored '{parsed['key']}' for user_id {record.get('user_id')}.", "success": True}
