"""ASSISTments 2009–2010 skill builder (corrected), read into one sequence of interactions per student.

The file has one row per (interaction, skill): a problem tagged with two skills appears twice with the same
``order_id``. An interaction here is one ``order_id`` with the set of its skills; rows without a skill are dropped,
as in the usual preprocessing. The file has no timestamps, so ``order_id`` gives the order within a student.
"""

from __future__ import annotations

import csv
import hashlib
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

URL = "http://base.ustc.edu.cn/data/ASSISTment/2009_skill_builder_data_corrected.zip"
SHA256 = "10c4fa76ae9410d8577cf9604cf21f8eed029552717a4b281d64945cf44ebe46"
CSV = "skill_builder_data_corrected.csv"


@dataclass(frozen=True)
class Interaction:
    order: int
    problem: str
    skills: tuple[str, ...]
    correct: int


@dataclass(frozen=True)
class Student:
    id: str
    items: tuple[Interaction, ...]


def fetch(root: Path) -> Path:
    """The CSV under ``root``, downloaded from the mirror and checked against its hash when missing."""
    path = root / CSV
    if path.exists():
        return path
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "assist09.zip"
    if not archive.exists():
        urllib.request.urlretrieve(URL, archive)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != SHA256:
        raise ValueError(f"{archive}: sha256 {digest}, expected {SHA256}")
    with zipfile.ZipFile(archive) as z:
        z.extract(CSV, root)
    return path


def load(path: Path, min_items: int = 2) -> list[Student]:
    rows: dict[str, dict[int, dict]] = {}
    with path.open(encoding="latin-1", newline="") as f:
        for row in csv.DictReader(f):
            if not row["skill_id"]:
                continue
            order = int(row["order_id"])
            item = rows.setdefault(row["user_id"], {}).setdefault(
                order, {"problem": row["problem_id"], "skills": [], "correct": int(row["correct"])})
            if row["skill_id"] not in item["skills"]:
                item["skills"].append(row["skill_id"])
    students = []
    for user in sorted(rows, key=int):
        items = tuple(Interaction(order, v["problem"], tuple(sorted(v["skills"], key=int)), v["correct"])
                      for order, v in sorted(rows[user].items()))
        if len(items) >= min_items:
            students.append(Student(user, items))
    return students


def folds(students: list[Student], k: int, seed: int = 0) -> list[list[int]]:
    """Student indices of each of ``k`` folds, at random with a fixed seed."""
    order = np.random.default_rng(seed).permutation(len(students))
    return [sorted(int(i) for i in order[f::k]) for f in range(k)]


def skills_of(students: list[Student]) -> list[str]:
    return sorted({s for st in students for it in st.items for s in it.skills}, key=int)
