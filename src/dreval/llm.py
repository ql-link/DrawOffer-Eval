"""One model for every baseline: Codex CLI (``codex exec``), no tools, cached by request.

The request is (system, user, JSON schema or none, sample). ``sample`` > 0 asks for an independent draw of
the same request, so five runs are five samples, not one sample read five times. Answers are appended to a
JSON-lines cache; a rerun reads them back and calls nothing. ``DREVAL_CACHE_ONLY=1`` makes a miss an error.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from drawoffer.memory.observers.models.cli_answer import CODEX_OFF, prune, strict

MODEL = os.environ.get("DREVAL_BASELINE_MODEL", "gpt-6.1-terra")
EFFORT = os.environ.get("DREVAL_BASELINE_EFFORT", "medium")
TIMEOUT = int(os.environ.get("DREVAL_BASELINE_TIMEOUT", "600"))


class Codex:
    def __init__(self, cache: Path, model: str = MODEL, effort: str = EFFORT) -> None:
        self.cache_path, self.model, self.effort = cache, model, effort
        self.lock = threading.Lock()
        self.cache: dict[str, str] = {}
        self.calls = self.hits = self.failures = 0
        self.seconds = 0.0
        self.tokens = 0
        if cache.exists():
            for line in cache.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    self.cache[row["key"]] = row["content"]

    def key(self, system: str, user: str, schema: dict | None, sample: int) -> str:
        blob = json.dumps([self.model, self.effort, system, user, schema, sample], ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def __call__(self, system: str, user: str, schema: dict | None = None, sample: int = 0) -> Any:
        key = self.key(system, user, schema, sample)
        if key in self.cache:
            self.hits += 1
            return self._parse(self.cache[key], schema)
        if os.environ.get("DREVAL_CACHE_ONLY") == "1":
            raise RuntimeError("cache miss with DREVAL_CACHE_ONLY=1")
        for attempt in range(3):
            try:
                content = self._run(system, user, schema, sample)
                value = self._parse(content, schema)
            except Exception:  # noqa: BLE001 — a failed attempt is retried, then counted
                if attempt == 2:
                    with self.lock:
                        self.failures += 1
                    raise
                time.sleep(2 * (attempt + 1))
                continue
            with self.lock:
                self.cache[key] = content
                with self.cache_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"key": key, "content": content}, ensure_ascii=False) + "\n")
            return value

    @staticmethod
    def _parse(content: str, schema: dict | None) -> Any:
        if schema is None:
            return content
        value = json.loads(content)
        return prune(value, schema, schema.get("$defs", {}))

    def _run(self, system: str, user: str, schema: dict | None, sample: int) -> str:
        with tempfile.TemporaryDirectory(prefix="dreval-codex-") as root:
            work = Path(root) / "work"
            work.mkdir()
            last = Path(root) / "last.txt"
            command = ["codex", "exec", "--model", self.model, "--sandbox", "read-only", "--ephemeral",
                       "--ignore-user-config", "--ignore-rules", "--skip-git-repo-check", "--color", "never",
                       "-C", str(work), "-c", 'approval_policy="never"', "-c", 'web_search="disabled"',
                       "-c", f'model_reasoning_effort="{self.effort}"', "--json", "-o", str(last)]
            for feature in CODEX_OFF:
                command += ["--disable", feature]
            if schema is not None:
                schema_file = Path(root) / "schema.json"
                schema_file.write_text(json.dumps(strict(schema), ensure_ascii=False), encoding="utf-8")
                command += ["--output-schema", str(schema_file)]
            # The sample number is part of the prompt only as a nonce, so independent draws are not deduplicated.
            nonce = f"\n\n（请求编号 {sample}，与回答无关）" if sample else ""
            started = time.perf_counter()
            done = subprocess.run([*command, "-"], input=f"{system}\n\n{user}{nonce}", capture_output=True, text=True,
                                  timeout=TIMEOUT, cwd=str(work))
            elapsed = time.perf_counter() - started
            if done.returncode != 0 or not last.exists():
                raise RuntimeError(f"codex exit {done.returncode}: {done.stderr.strip()[-300:]}")
            used = 0
            for line in done.stdout.splitlines():
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                usage = (event.get("usage") or (event.get("msg") or {}).get("info", {}) or {})
                if isinstance(usage, dict):
                    used = max(used, int(usage.get("total_tokens") or usage.get("input_tokens", 0) or 0))
            with self.lock:
                self.calls += 1
                self.seconds += elapsed
                self.tokens += used
            return last.read_text(encoding="utf-8").strip()

    def summary(self) -> dict[str, Any]:
        return {"model": self.model, "effort": self.effort, "calls": self.calls, "cached": self.hits,
                "failures": self.failures, "seconds": round(self.seconds, 1), "tokens": self.tokens}
