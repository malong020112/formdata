import json
from pathlib import Path

from tool.tool_add_memory import AddMemory
from tool.tool_memory_search import SearchMemory


def run_tests() -> None:
    """Simple sanity checks for AddMemory and SearchMemory tools."""
    base_dir = Path(__file__).resolve().parent
    temp_memory = base_dir / "memory_test.json"

    # Start fresh for repeatable runs.
    if temp_memory.exists():
        temp_memory.unlink()

    add_tool = AddMemory(memory_path=temp_memory)
    search_tool = SearchMemory(memory_path=temp_memory)

    print("Test 1: Add new key/value for user 1")
    res1 = add_tool.call({"userid": "1", "content": ["name", "Alice"]})
    print(res1)

    print("Test 2: Search just-added key for user 1")
    res2 = search_tool.call({"userid": "1", "query": "name"})
    print(res2)

    print("Test 3: Update existing key for user 1")
    res3 = add_tool.call({"userid": "1", "content": ["name", "Alice Smith"]})
    print(res3)

    print("Test 4: Search updated key for user 1")
    res4 = search_tool.call({"userid": "1", "query": "name"})
    print(res4)

    print("Test 5: Search missing key for user 1")
    res5 = search_tool.call({"userid": "1", "query": "phone"})
    print(res5)

    print("Test 6: Add second user and search")
    res6 = add_tool.call({"userid": "2", "content": ["phone", "12345678"]})
    res7 = search_tool.call({"userid": "2", "query": "mobile number"})
    print(res6)
    print(res7)

    print("Final memory file contents:")
    print(json.loads(temp_memory.read_text(encoding="utf-8")))


if __name__ == "__main__":
    run_tests()
