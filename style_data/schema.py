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


@dataclass
class TargetTemplateRecord:
    """One conditional target-template sample, not an exact object pair.

    A template may vary in appearance and moderate within-category geometry.
    These fields make that nuisance explicit so factorization does not silently
    promote template-specific variation to a style delta.
    """
    target_template_id: str
    target_style_family: str
    style_operation: str
    style_intensity: float
    template_variant_id: str
    target_style_attributes: Dict[str, Any] = field(default_factory=dict)
    template_nuisance_attributes: Dict[str, Any] = field(default_factory=dict)
    appearance_nuisance_attributes: Dict[str, Any] = field(default_factory=dict)
    geometry_nuisance_attributes: Dict[str, Any] = field(default_factory=dict)
    required_invariants: List[str] = field(default_factory=list)
    allowed_variations: List[str] = field(default_factory=list)
    forbidden_changes: List[str] = field(default_factory=list)
    view_relation: Dict[str, Any] = field(default_factory=dict)
    semantic_part_requirements: List[str] = field(default_factory=list)
    generation_seed: Optional[int] = None
    generation_run: Optional[str] = None
    quality_metadata: Dict[str, Any] = field(default_factory=dict)
    object_id: Optional[str] = None

    def validate(self) -> "TargetTemplateRecord":
        for name in ("target_template_id", "target_style_family", "style_operation", "template_variant_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError("{} must be a non-empty string".format(name))
        if not isinstance(self.style_intensity, (int, float)) or not math.isfinite(float(self.style_intensity)):
            raise ValueError("style_intensity must be finite")
        if float(self.style_intensity) < 0:
            raise ValueError("style_intensity must be non-negative")
        for name in ("target_style_attributes", "template_nuisance_attributes",
                     "appearance_nuisance_attributes", "geometry_nuisance_attributes",
                     "view_relation", "quality_metadata"):
            if not isinstance(getattr(self, name), dict):
                raise ValueError("{} must be a dictionary".format(name))
        for name in ("required_invariants", "allowed_variations", "forbidden_changes",
                     "semantic_part_requirements"):
            value = getattr(self, name)
            if not isinstance(value, list) or any(not isinstance(x, str) for x in value):
                raise ValueError("{} must be a list of strings".format(name))
        if self.generation_seed is not None and not isinstance(self.generation_seed, int):
            raise ValueError("generation_seed must be an integer or null")
        return self

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self.validate())

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "TargetTemplateRecord":
        if not isinstance(payload, dict):
            raise TypeError("target template record must be a dictionary")
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(payload) - known)
        if unknown:
            raise ValueError("unknown TargetTemplateRecord fields: {}".format(unknown))
        return cls(**payload).validate()

    def save_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, ensure_ascii=False)

    @classmethod
    def load_json(cls, path: str) -> "TargetTemplateRecord":
        with open(path, "r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    @classmethod
    def from_style_task(cls, record: StyleTaskRecord, template_variant_id: str) -> "TargetTemplateRecord":
        record.validate()
        return cls(
            target_template_id="{}:{}".format(record.object_id, template_variant_id),
            target_style_family=record.style_family,
            style_operation=record.style_family,
            style_intensity=float(record.intensity),
            template_variant_id=template_variant_id,
            target_style_attributes=dict(record.target_attributes),
            required_invariants=list(record.preserved_parts),
            semantic_part_requirements=list(record.affected_parts),
            object_id=record.object_id,
            generation_run=record.repeat_id,
        )
