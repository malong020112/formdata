import json
from typing import Any, Dict, List, Optional, Union


class FillForm:
    def __init__(self, form_def: Dict[str, Any]):
        self.form_def = form_def or {}
        self.form_state = self._initialize_form_state()

    def _initialize_form_state(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {}
        for field in self.form_def.get("fields", []):
            key = field.get("key")
            if key is None:
                continue
            state[key] = None
        return state

    def _snapshot_form_state(self) -> Dict[str, List[str]]:
        filled: List[str] = []
        missing: List[str] = []
        for field in self.form_def.get("fields", []):
            key = field.get("key")
            if key is None:
                continue
            val = self.form_state.get(key)
            has_val = val is not None and val != ""
            (filled if has_val else missing).append(key)
        return {"filled": filled, "missing": missing}

    @staticmethod
    def _set_nested_value(target: Dict[str, Any], dotted_key: str, value: Any) -> None:
        parts = [p for p in str(dotted_key).split(".") if p]
        if not parts:
            return
        node = target
        for part in parts[:-1]:
            child = node.get(part)
            if not isinstance(child, dict):
                child = {}
                node[part] = child
            node = child
        node[parts[-1]] = value

    def _form_state_view(self) -> Dict[str, Any]:
        """Return form state in original nested form-style structure."""
        view: Dict[str, Any] = {}
        for field in self.form_def.get("fields", []):
            key = field.get("key")
            if key is None:
                continue
            self._set_nested_value(view, str(key), self.form_state.get(key))
        return view

    def get_state(self) -> Dict[str, Any]:
        return dict(self.form_state)

    def call(self, params: Union[str, dict], **kwargs) -> Dict[str, Any]:
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except json.JSONDecodeError:
                return {
                    "success": False,
                    "response": "Invalid params: expected JSON with user_id and content.",
                    "form_state": self.get_state(),
                    "form_state_view": self._form_state_view(),
                    "snapshot": self._snapshot_form_state(),
                }

        if not isinstance(params, dict):
            return {
                "success": False,
                "response": "Params must be a JSON object with user_id and content.",
                "form_state": self.get_state(),
                "form_state_view": self._form_state_view(),
                "snapshot": self._snapshot_form_state(),
            }

        user_id = params.get("user_id")
        content = params.get("content")
        if not isinstance(content, dict) or not content:
            return {
                "user_id": user_id,
                "success": False,
                "response": "Missing or invalid 'content' object for fill_form. Provide a JSON object of fields to fill.",
                "filled": {},
                "form_state": self.get_state(),
                "form_state_view": self._form_state_view(),
                "snapshot": self._snapshot_form_state(),
            }

        updated_content: Dict[str, Any] = {}
        skipped_keys: List[str] = []
        for k, v in content.items():
            if k in self.form_state and self.form_state.get(k) == v:
                skipped_keys.append(k)
            else:
                updated_content[k] = v

        if updated_content:
            self.form_state.update(updated_content)

        return {
            "user_id": user_id,
            "filled": updated_content or content,
            "skipped": skipped_keys,
            "success": True,
            "form_state": self.get_state(),
            "form_state_view": self._form_state_view(),
            "snapshot": self._snapshot_form_state(),
        }
