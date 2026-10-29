"""A byte-exact fingerprint of everything the engine concludes at one moment."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from drawoffer.memory.assemble import State
from drawoffer.memory.catalog import Catalog
from drawoffer.memory.compiler.items import propositions_of


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def conclusions(st: State, cat: Catalog) -> list[dict[str, Any]]:
    """Every conclusion the product would show, with the observations each rests on."""
    return sorted(({"key": p.key, "predicate": p.predicate, "object": p.object,
                    "evidence": p.qualifiers.get("evidence"), "support": sorted(p.support)}
                   for p in propositions_of(st, cat, include_closed=True)), key=lambda p: p["key"])


def fingerprint(st: State, cat: Catalog) -> dict[str, str]:
    post = st.posterior
    parts = {
        "posterior": hashlib.sha256(post.mean.tobytes() + post.cov.tobytes()).hexdigest(),
        "issues": hashlib.sha256(_json([(i.id, i.kind, i.entities, i.state, i.severity, i.history)
                                        for i in st.issues]).encode()).hexdigest(),
        "conclusions": hashlib.sha256(_json(conclusions(st, cat)).encode()).hexdigest(),
    }
    parts["all"] = hashlib.sha256(_json(parts).encode()).hexdigest()
    return parts
