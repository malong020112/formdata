import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from qwen_agent.tools.base import BaseTool


class GetMemory(BaseTool):
    name = "getmemory"
    description = (
        "Get all memory fields for a given user_id from memory.json. "
        "Returns a flattened object mapping keys to values."
    )
    parameters = {
        "type": "object",
        "properties": {
            "userid": {
                "type": "string",
                "description": "The user_id whose memory should be returned.",
            }
        },
        "required": ["userid"],
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
    def _find_user_record(memory: List[Dict[str, Any]], user_id: str) -> Optional[Dict[str, Any]]:
        for record in memory:
            if str(record.get("user_id", "")) == str(user_id):
                return record
        return None

    @staticmethod
    def _to_flat_map(info_list: Any) -> Dict[str, Any]:
        if not isinstance(info_list, list):
            return {}
        out: Dict[str, Any] = {}
        for item in info_list:
            if not isinstance(item, dict):
                continue
            key = item.get("key")
            if key is None:
                continue
            out[str(key)] = item.get("value")
        return out

    def call(self, params: Union[str, dict], **kwargs) -> Dict[str, Any]:
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except json.JSONDecodeError:
                return {"success": False, "response": "Invalid params: expected JSON with userid.", "memory": {}}

        if not isinstance(params, dict):
            return {"success": False, "response": "Params must be a JSON object with userid.", "memory": {}}

        user_id = params.get("userid")
        if user_id is None:
            return {"success": False, "response": "Missing userid.", "memory": {}}

        memory_records = self._load_memory()
        if not memory_records:
            return {"success": False, "response": "Memory is empty.", "memory": {}}

        user_record = self._find_user_record(memory_records, str(user_id))
        if user_record is None:
            return {"success": False, "response": f"No memory found for user_id {user_id}.", "memory": {}}

        memory_map = self._to_flat_map(user_record.get("information", []))
        return {
            "success": True,
            "response": f"Loaded {len(memory_map)} memory fields for user_id {user_id}.",
            "memory": memory_map,
        }
