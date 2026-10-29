"""The published personas (``data/personas``) are byte-identical to the ones the benchmarks read."""

from __future__ import annotations

import sys
from pathlib import Path

from dreval.engine import GOLD_ROOT

DATA = Path(__file__).resolve().parents[2] / "data" / "personas"


def main() -> int:
    published = {p.parent.name: p.read_bytes() for p in DATA.glob("*/persona.json")}
    used = {p.parent.name: p.read_bytes() for p in (GOLD_ROOT / "personas").glob("*/persona.json")}
    problems = [f"{pid}: 只在{'本仓库' if pid in published else 'DrawOffer'}" for pid in sorted(set(published) ^ set(used))]
    problems += [f"{pid}: 内容不同" for pid in sorted(set(published) & set(used)) if published[pid] != used[pid]]
    for line in problems:
        print(line)
    print("一致" if not problems else f"{len(problems)} 处不一致", f"（{len(published)} 份剧本）")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
