"""
Registry mapping expert id -> Expert object.

The generator, teacher, compute_experts_correct and the standalone eval all look
up config.active_experts here rather than hardcoding expert names. Each expert
module registers itself on import, and experts/__init__.py imports the builtin ones.

To add an expert: write a module that calls register(MyExpert()) and add its id
to config.active_experts. To drop one, just remove its id from active_experts.
"""
from __future__ import annotations

from typing import Dict, List

EXPERT_REGISTRY: Dict[str, "object"] = {}


def register(expert):
    """Register an expert under expert.id (replaces it if the module is re-imported)."""
    if not getattr(expert, "id", ""):
        raise ValueError(f"Expert {expert!r} has no id.")
    EXPERT_REGISTRY[expert.id] = expert
    return expert


def get(expert_id: str):
    if expert_id not in EXPERT_REGISTRY:
        raise KeyError(
            f"Unknown expert id {expert_id!r}. Registered: "
            f"{sorted(EXPERT_REGISTRY)}"
        )
    return EXPERT_REGISTRY[expert_id]


def get_active(config) -> List["object"]:
    """Turn config.active_experts (list of ids) into Expert objects, in order.
    Raises KeyError if an id isn't registered."""
    return [get(eid) for eid in config.active_experts]


def all_experts() -> Dict[str, "object"]:
    return dict(EXPERT_REGISTRY)
