"""Strict, lightweight records for one object/style/intensity/repeat task."""
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class StyleTaskRecord:
    object_id: str
    object_category: str
    source_glb: str
    source_3dgs: str
    style_family: str
    source_attributes: Dict[str, Any]
    target_attributes: Dict[str, Any]
    intensity: float
    repeat_id: str
    camera_names: List[str]
    source_image_root: str
    target_image_root: str
    generation_prompt: str
    affected_parts: List[str] = field(default_factory=list)
    preserved_parts: List[str] = field(default_factory=list)
    quality_control: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> "StyleTaskRecord":
        required_strings = {
            "object_id": self.object_id,
            "object_category": self.object_category,
            "style_family": self.style_family,
            "repeat_id": self.repeat_id,
            "source_glb": self.source_glb,
            "source_3dgs": self.source_3dgs,
            "source_image_root": self.source_image_root,
            "target_image_root": self.target_image_root,
            "generation_prompt": self.generation_prompt,
        }
        for name, value in required_strings.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError("{} must be a non-empty string".format(name))
        if not isinstance(self.intensity, (int, float)) or not math.isfinite(float(self.intensity)):
            raise ValueError("intensity must be finite")
        if float(self.intensity) < 0:
            raise ValueError("intensity must be non-negative")
        if not isinstance(self.camera_names, list) or not self.camera_names:
            raise ValueError("camera_names must be a non-empty list")
        if any(not isinstance(x, str) or not x.strip() for x in self.camera_names):
            raise ValueError("camera_names must contain non-empty strings")
        for name in ("source_attributes", "target_attributes", "quality_control", "metadata"):
            if not isinstance(getattr(self, name), dict):
                raise ValueError("{} must be a dictionary".format(name))
        for name in ("affected_parts", "preserved_parts"):
            values = getattr(self, name)
            if not isinstance(values, list) or any(not isinstance(x, str) for x in values):
                raise ValueError("{} must be a list of strings".format(name))
        return self

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "StyleTaskRecord":
        if not isinstance(payload, dict):
            raise TypeError("style task record must be a dictionary")
        known = {field_name for field_name in cls.__dataclass_fields__}
        unknown = sorted(set(payload) - known)
        if unknown:
            raise ValueError("unknown StyleTaskRecord fields: {}".format(unknown))
        record = cls(**payload)
        return record.validate()

    def save_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, ensure_ascii=False)

    @classmethod
    def load_json(cls, path: str) -> "StyleTaskRecord":
        with open(path, "r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))
