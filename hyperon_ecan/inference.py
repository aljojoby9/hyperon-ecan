"""Attention-gated inheritance chaining.

Iklé et al. 2009, last section: pick the next inference step with probability
proportional to the STI of the atoms it uses, then pay those atoms wages.
Classic OpenCog described this loop; Hyperon never shipped a working one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .network import ECAN


@dataclass(frozen=True)
class Triple:
    rel: str
    src: str
    dst: str

    def atoms(self) -> tuple[str, str, str]:
        return (self.rel, self.src, self.dst)

    def as_tuple(self) -> tuple[str, str, str]:
        return (self.rel, self.src, self.dst)


class AttentionChainer:
    """Forward inheritance chainer that only expands high-STI atoms."""

    def __init__(self, ecan: ECAN, facts: Iterable[Triple] | None = None) -> None:
        self.ecan = ecan
        self.facts: set[Triple] = set()
        if facts:
            for fact in facts:
                self.assert_fact(fact)

    def assert_fact(self, fact: Triple) -> None:
        self.facts.add(fact)
        for name in fact.atoms():
            self.ecan.add(name)
        if fact.rel == "Inheritance":
            self.ecan.link(fact.src, fact.dst, strength=0.55)
        else:
            self.ecan.link(fact.src, fact.dst, strength=0.35)

    def candidates(self, require_focus: bool = True) -> list[tuple[Triple, float]]:
        """Possible Inheritance transitivity steps, scored by mean STI."""
        focus = {a.name for a in self.ecan.attentional_focus()} if require_focus else None
        inherit = [f for f in self.facts if f.rel == "Inheritance"]
        out: list[tuple[Triple, float]] = []
        for left in inherit:
            for right in inherit:
                if left.dst != right.src:
                    continue
                inferred = Triple("Inheritance", left.src, right.dst)
                if inferred in self.facts or inferred.src == inferred.dst:
                    continue
                involved = {left.src, left.dst, right.dst}
                if focus is not None and involved.isdisjoint(focus):
                    continue
                score = sum(self.ecan.atoms[n].sti for n in involved) / 3.0
                out.append((inferred, score))
        out.sort(key=lambda item: item[1], reverse=True)
        return out

    def step(self, require_focus: bool = True) -> Triple | None:
        options = self.candidates(require_focus=require_focus)
        if not options:
            return None
        inferred, _score = options[0]
        self.assert_fact(inferred)
        self.ecan.stimulate(inferred.atoms())
        return inferred

    def prove(self, goal: Triple, steps: int = 16, require_focus: bool = True) -> bool:
        for _ in range(steps):
            if goal in self.facts:
                return True
            if self.step(require_focus=require_focus) is None:
                break
        return goal in self.facts
