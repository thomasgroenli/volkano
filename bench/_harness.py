"""Timing harness + result model shared by the benchmark modules."""
from __future__ import annotations

import statistics
import sys
import time
from dataclasses import dataclass, field


def bench(fn, *, iters: int = 100_000, reps: int = 5, warmup: int = 10_000) -> float:
    """Median of *reps* runs of *iters* calls; returns nanoseconds per call.

    Median-of-reps rejects the occasional GC pause / scheduler hiccup
    without discarding the steady state the way best-of would.
    """
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(reps):
        t = time.perf_counter()
        for _ in range(iters):
            fn()
        samples.append((time.perf_counter() - t) / iters)
    return statistics.median(samples) * 1e9


@dataclass
class Row:
    label: str
    ns: float
    note: str = ""


@dataclass
class Section:
    title: str
    rows: list[Row] = field(default_factory=list)
    footnotes: list[str] = field(default_factory=list)

    def add(self, label: str, ns: float, note: str = "") -> Row:
        row = Row(label, ns, note)
        self.rows.append(row)
        return row

    def footnote(self, text: str) -> None:
        self.footnotes.append(text)


def render(sections: list[Section], *, file=sys.stdout) -> None:
    width = max(
        (len(r.label) for s in sections for r in s.rows),
        default=40,
    )
    width = max(width, 40)
    for s in sections:
        print(f"\n{s.title}", file=file)
        print("-" * (width + 26), file=file)
        for r in s.rows:
            line = f"{r.label:<{width}} {r.ns:9.1f} ns  ({r.ns / 1000:7.3f} us)"
            if r.note:
                line += f"   {r.note}"
            print(line, file=file)
        for fn in s.footnotes:
            print(f"  * {fn}", file=file)
