import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import user_simulation


def load_ndjson(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not path.exists():
        raise FileNotFoundError(f"User data not found: {path}")
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def build_form_index(form_root: Path) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    for p in form_root.rglob("*.json"):
        data = json.loads(p.read_text(encoding="utf-8"))
        name = str(data.get("name") or "").strip()
        if not name:
            name = p.stem.replace("_form", "")
        index[name] = p
    return index


def verify_form_types(form_root: Path, form_types: List[str]) -> None:
    """Check whether each form type matches a template's name field."""
    if not form_types:
        return
    name_set = set()
    for p in form_root.rglob("*.json"):
        data = json.loads(p.read_text(encoding="utf-8"))
        name = str(data.get("name") or "").strip()
        if not name:
            name = p.stem.replace("_form", "")
        name_set.add(name)
    for t in form_types:
        if t not in name_set:
            print(f"[warn] form type not found by name in form templates: {t}")


def write_temp_profile(path: Path, uid: int, form_type: str, form_data: Dict[str, Any]) -> None:
    record = {"uid": uid, "type": form_type, "data": form_data}
    path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")

def load_trace_index(path: Path) -> Dict[str, set]:
    if not path.exists():
        return {}
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, list):
        return {}
    index: Dict[str, set] = {}
    for entry in data:
        if not isinstance(entry, dict):
            continue
        uid = str(entry.get("name") or "")
        if not uid:
            continue
        items = entry.get("data")
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            form_type = item.get("type")
            if not form_type:
                continue
            index.setdefault(uid, set()).add(str(form_type))
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch run user_simulation over user_data forms.")
    parser.add_argument(
        "--user-class",
        required=False,
        default="",
        help="Folder name under data/user_data. If empty, run the first record of every class.",
    )
    parser.add_argument(
        "--user-data-root",
        default="data/user_data",
        help="Root folder for user data directories.",
    )
    parser.add_argument(
        "--form-root",
        default="form",
        help="Root folder containing form templates.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional limit for number of user records to process (0 = all).",
    )
    parser.add_argument(
        "--user-count",
        type=int,
        default=5,
        help="Number of user records to process per class (0 = all).",
    )
    args = parser.parse_args()

    form_root = Path(args.form_root)
    form_index = build_form_index(form_root)

    if args.user_class:
        class_dirs = [Path(args.user_data_root) / args.user_class]
    else:
        class_dirs = [p for p in Path(args.user_data_root).iterdir() if p.is_dir()]
        class_dirs.sort(key=lambda p: p.name.lower())

    for class_dir in class_dirs:
        user_data_path = class_dir / "user_data.json"
        if not user_data_path.exists():
            continue
        trace_index = load_trace_index(class_dir / "trace.json")
        records = load_ndjson(user_data_path)
        if not records:
            continue
        if args.user_count > 0:
            records = records[: args.user_count]

        for rec in records:
            uid = int(rec.get("uid") or 0)
            forms = rec.get("forms")
            if not isinstance(forms, list):
                continue
            verify_form_types(form_root, [f.get("type") for f in forms if isinstance(f, dict)])
            for idx, form in enumerate(forms, start=1):
                if not isinstance(form, dict):
                    continue
                form_type = form.get("type")
                form_data = form.get("data")
                if not form_type or not isinstance(form_data, dict):
                    continue
                if str(uid) in trace_index and str(form_type) in trace_index[str(uid)]:
                    print(f"[skip] uid {uid} already has trace for form type: {form_type}")
                    continue
                form_path = form_index.get(form_type)
                if not form_path:
                    print(f"[warn] no template for form type: {form_type}")
                    continue
                temp_profile = user_data_path.parent / f"_tmp_user_{uid}_{idx}.json"
                write_temp_profile(temp_profile, uid, form_type, form_data)
                try:
                    user_simulation.simulate(form_path=form_path, profile_path=temp_profile, count=1)
                finally:
                    if temp_profile.exists():
                        temp_profile.unlink()


if __name__ == "__main__":
    main()
