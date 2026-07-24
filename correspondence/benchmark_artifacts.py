"""Deterministic benchmark record keys and replacement semantics."""

DEFAULT_KEY_FIELDS = ("teacher", "mode", "fraction", "views_per_track",
                      "baseline_policy", "seed", "noise", "outlier_rate",
                      "confidence_mode", "robust_kernel", "reject_threshold",
                      "solver", "control_count", "track_consensus",
                      "consensus_method", "fallback", "confidence_cap",
                      "oracle_outliers")


def record_key(record, fields=DEFAULT_KEY_FIELDS):
    return tuple(record.get(name) for name in fields)


def upsert_records(existing, new_records, fields=DEFAULT_KEY_FIELDS):
    merged = {record_key(r, fields): r for r in existing}
    for record in new_records:
        merged[record_key(record, fields)] = record
    return list(merged.values())


def validate_records(records, expected_keys=None, fields=DEFAULT_KEY_FIELDS):
    keys = [record_key(r, fields) for r in records]
    if len(keys) != len(set(keys)):
        raise ValueError("benchmark contains duplicate deterministic record keys")
    if expected_keys is not None:
        missing = set(expected_keys) - set(keys)
        if missing:
            raise ValueError("benchmark is incomplete: %d records missing" % len(missing))
    return {"records": len(records), "unique": len(set(keys)), "complete": expected_keys is None or set(expected_keys) <= set(keys)}
