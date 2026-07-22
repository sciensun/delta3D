"""Minimal Stage 2 contracts; no model or training implementation is included."""
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch


@dataclass
class DeltaComponent:
    name: str
    delta_xyz: torch.Tensor
    part_mask: Optional[torch.Tensor] = None
    local_frame: Optional[torch.Tensor] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class StyleCondition:
    style_family: str
    intensity: float
    attributes: Dict = field(default_factory=dict)


@dataclass
class StyleModelInput:
    features: torch.Tensor
    condition: StyleCondition
    source_xyz: torch.Tensor
    part_mask: Optional[torch.Tensor] = None


@dataclass
class StyleModelOutput:
    delta_xyz: torch.Tensor
    components: Dict[str, DeltaComponent] = field(default_factory=dict)
    confidence: Optional[torch.Tensor] = None
