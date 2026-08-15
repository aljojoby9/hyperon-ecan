"""Perceive → attend → infer → update.

This is the four-step PLN/ECAN interlock described at the end of
Iklé, Pitt, Goertzel & Sellman (2009). It did not exist as running
Hyperon code.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .inference import AttentionChainer, Triple
from .network import ECAN, CycleStats


@dataclass
class Tick:
    percepts: list[str]
    inferred: Triple | None
    ecan: CycleStats
    focus: list[str] = field(default_factory=list)
    proved: bool = False


class CognitiveCycle:
    def __init__(self, ecan: ECAN, chainer: AttentionChainer) -> None:
        self.ecan = ecan
        self.chainer = chainer

    def tick(self, percepts: list[str] | None = None, goal: Triple | None = None) -> Tick:
        if percepts:
            self.ecan.stimulate(percepts)
        inferred = self.chainer.step(require_focus=True)
        stats = self.ecan.cycle()
        proved = goal in self.chainer.facts if goal is not None else False
        return Tick(
            percepts=list(percepts or []),
            inferred=inferred,
            ecan=stats,
            focus=stats.af,
            proved=proved,
        )

    def run(
        self,
        percepts: list[str],
        goal: Triple,
        ticks: int = 12,
        restimulate_every: int = 3,
    ) -> list[Tick]:
        log: list[Tick] = []
        for i in range(ticks):
            pulse = percepts if i == 0 or (restimulate_every and i % restimulate_every == 0) else None
            log.append(self.tick(pulse, goal=goal))
            if log[-1].proved:
                break
        return log
