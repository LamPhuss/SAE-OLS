"""
Utility functions for SAE-OLS.
"""

import json
from typing import List, Dict, Any, Optional


def read_jsonl(file_path: str, start: int = 0, end: Optional[int] = None) -> List[Dict[str, Any]]:
    """Read a JSONL file and return a list of dictionaries."""
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    lines = [line.strip() for line in lines if line.strip()]
    if end is not None:
        lines = lines[start:end]
    else:
        lines = lines[start:]

    return [json.loads(line) for line in lines]


def write_jsonl(file_path: str, data: List[Dict[str, Any]], append: bool = False):
    """Write a list of dictionaries to a JSONL file."""
    mode = "a" if append else "w"
    with open(file_path, mode, encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def append_jsonl_line(file_path: str, item: Dict[str, Any]):
    """Append a single JSON line to a JSONL file."""
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
