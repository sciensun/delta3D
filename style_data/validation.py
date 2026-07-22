"""Validation helpers for style-task manifests."""
from typing import Iterable, Tuple

from .schema import StyleTaskRecord


def validate_records(records: Iterable[StyleTaskRecord]) -> Tuple[bool, str]:
    records = list(records)
    if not records:
        return False, "style manifest is empty"
    keys = set()
    for record in records:
        record.validate()
        key = (record.object_id, record.style_family, float(record.intensity), record.repeat_id)
        if key in keys:
            return False, "duplicate task key: {}".format(key)
        keys.add(key)
    return True, "valid"
