"""Thin adapter over the DrawOffer memory engine (``drawoffer.memory`` in LinkCV, installed editable).

Everything that touches engine internals lives here, so the benchmarks read like a spec. Model labels
(the observers M1–M7) come from a JSON-lines cache keyed by request: a rerun with the same cache makes
no model call, which is what makes the deterministic benchmarks (stability, traceability, timing) cheap.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from drawoffer.memory.assemble import ModelLabels, State
from drawoffer.memory.catalog import Catalog, load_valid_catalog
from drawoffer.memory.gold import DEFAULT_ROOT
from drawoffer.memory.gold.checkpoints import Keys, PersonaState, label_persona, state_at
from drawoffer.memory.gold.personas import Persona, evaluate, load_personas
from drawoffer.memory.gold.simulate import Replayable, read_sources, replayable, sampled
from drawoffer.memory.observers.models.offline import offline_structured

GOLD_ROOT = Path(os.environ.get("DREVAL_GOLD_ROOT") or DEFAULT_ROOT)
ENGINE_MODEL = os.environ.get("DREVAL_ENGINE_MODEL", "deepseek-v4.1-flash")


@dataclass
class Case:
    persona: Persona
    replay: Replayable


def catalog() -> Catalog:
    return load_valid_catalog()


def personas(split: str = "dev", only: list[str] | None = None) -> list[Persona]:
    chosen = [p for p in load_personas(GOLD_ROOT) if split == "all" or p.split == split]
    return [p for p in chosen if not only or p.id in only]


def case(persona: Persona, run: int = 1) -> Case:
    generated = read_sources(GOLD_ROOT, persona, run, "")
    if generated is None:
        raise FileNotFoundError(f"{persona.id}: no simulated sources for run {run}")
    return Case(persona, replayable(persona, generated))


def labeller(cache: Path, cache_only: bool = False):
    return offline_structured(ENGINE_MODEL, cache, None, cache_only)


async def labels_for(item: Case, cat: Catalog, structured: Any, sample: int = 1) -> ModelLabels:
    """The observers' labels; ``sample`` > 1 draws fresh model outputs (a separate cache slot)."""
    return await label_persona(item.persona, item.replay, cat, sampled(structured, sample))


def label_all(items: list[Case], cat: Catalog, structured: Any, sample: int = 1) -> list[ModelLabels]:
    return label_samples(items, cat, structured, [sample])[0]


def label_samples(items: list[Case], cat: Catalog, structured: Any, samples: list[int]) -> list[list[ModelLabels]]:
    """Labels for several samples in one event loop: the model client's queue is bound to the loop it first ran in."""
    async def go():
        return [list(await asyncio.gather(*(labels_for(i, cat, structured, s) for i in items))) for s in samples]
    return asyncio.run(go())


def state(item: Case, cat: Catalog, labels: ModelLabels, day: str) -> State:
    return state_at(item.replay, cat, labels, day)


def verdicts(item: Case, cat: Catalog, labels: ModelLabels):
    """Every scored checkpoint against the engine, with the gold set's own judge."""
    keys = Keys.of(item.persona)
    found, states = [], {}
    for cp in item.persona.checkpoints:
        if "m8" in (cp.params.get("requires") or []):
            continue
        day = cp.at.isoformat()
        if day not in states:
            states[day] = PersonaState(state(item, cat, labels, day), keys, cat)
        found.append(evaluate(item.persona, cp, states[day]))
    return found, states
