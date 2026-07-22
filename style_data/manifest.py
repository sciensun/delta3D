"""JSON and JSONL helpers for collections of style task records."""
import json
from typing import Iterable, List

from .schema import StyleTaskRecord


def save_manifest(records: Iterable[StyleTaskRecord], path: str) -> None:
    values = [record.to_dict() for record in records]
    if path.endswith(".jsonl"):
        with open(path, "w", encoding="utf-8") as handle:
            for value in values:
                handle.write(json.dumps(value, ensure_ascii=False) + "\n")
    else:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(values, handle, indent=2, ensure_ascii=False)


def load_manifest(path: str) -> List[StyleTaskRecord]:
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as handle:
            values = [json.loads(line) for line in handle if line.strip()]
    else:
        with open(path, "r", encoding="utf-8") as handle:
            values = json.load(handle)
    if not isinstance(values, list):
        raise ValueError("style manifest must contain a list of records")
    return [StyleTaskRecord.from_dict(value) for value in values]
