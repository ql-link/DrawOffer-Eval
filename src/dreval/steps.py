"""The prediction task for AUC: before each knowledge question is absorbed, P(the student passes depth L).

Taken from the engine's own replay (``infer``'s ``on_evidence``): every depth step of every direct knowledge
question, with the engine's prediction made just before it and the outcome the interview grader gave. The
baselines are asked about exactly these steps, seeing only the record that came before the question.
"""

from __future__ import annotations

from dataclasses import dataclass

from drawoffer.memory.inference.engine import infer
from drawoffer.memory.replay import DEPTH_PARTS

from dreval import engine


@dataclass(frozen=True)
class Step:
    persona: str
    episode: str
    at: str
    nodes: tuple[str, ...]
    level: int
    y: float
    engine_p: float


def steps(item: engine.Case, cat, labels) -> list[Step]:
    day = max(c.at.isoformat() for c in item.persona.checkpoints)
    st = engine.state(item, cat, labels, day)
    found: list[Step] = []

    def record(evidence, p: float) -> None:
        if evidence.part in DEPTH_PARTS and evidence.level is not None:
            ep = evidence.order[0]
            found.append(Step(item.persona.id, ep, st.prepared.times.get(ep, ""), tuple(evidence.u),
                              int(evidence.level), float(evidence.y), float(p)))

    infer(cat, st.prepared.observations, st.prepared.times, st.now, on_evidence=record)
    return found
