"""ECAN Variant 1 (Iklé et al. 2009) plus implicit neural Hebbian links.

STI and LTI are treated as two currencies held by a central bank. Stimulus
moves currency from the bank onto atoms. Rent moves it back. Diffusion
redistributes STI along a left-stochastic matrix built from explicit Hebbian
weights mixed with cosine similarities of atom embeddings.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Mapping

import numpy as np

from .neural import cosine, hashed_embedding
from .params import ECANParams


@dataclass
class AtomState:
    name: str
    sti: float = 0.0
    lti: float = 1.0
    vlti: bool = False
    embedding: np.ndarray | None = None


@dataclass
class CycleStats:
    rent_collected: float = 0.0
    wages_paid: float = 0.0
    sti_diffused: float = 0.0
    forgotten: list[str] = field(default_factory=list)
    af: list[str] = field(default_factory=list)


class ECAN:
    """Working-memory / associative-memory dynamics over a named atom graph."""

    def __init__(self, params: ECANParams | None = None) -> None:
        self.p = params or ECANParams()
        self.atoms: dict[str, AtomState] = {}
        self.hebbian: dict[tuple[str, str], float] = {}
        self.bank_sti = self.p.bank_sti
        self.bank_lti = self.p.bank_lti
        self.history: list[CycleStats] = []

    def add(self, name: str, embedding: np.ndarray | None = None, lti: float = 1.0) -> AtomState:
        if name not in self.atoms:
            vec = embedding if embedding is not None else hashed_embedding(name, self.p.embedding_dim)
            self.atoms[name] = AtomState(name=name, lti=lti, embedding=vec)
        elif embedding is not None:
            self.atoms[name].embedding = embedding
        return self.atoms[name]

    def add_many(self, names: Iterable[str]) -> None:
        for name in names:
            self.add(name)

    def link(self, a: str, b: str, strength: float = 0.5) -> None:
        """Undirected symbolic Hebbian / Inheritance-style association."""
        if a == b:
            return
        self.add(a)
        self.add(b)
        key = _edge(a, b)
        self.hebbian[key] = max(self.hebbian.get(key, 0.0), float(strength))

    def set_embedding(self, name: str, vector: np.ndarray) -> None:
        atom = self.add(name)
        vec = np.asarray(vector, dtype=np.float64).reshape(-1)
        norm = np.linalg.norm(vec)
        atom.embedding = vec / norm if norm > 0 else vec

    def stimulate(self, names: Mapping[str, float] | Iterable[str], amount: float | None = None) -> None:
        """Pay wages from the bank. Cue atoms get `amount` (default <Wage>)."""
        if isinstance(names, Mapping):
            payload = {n: float(v) for n, v in names.items()}
        else:
            wage = self.p.wage if amount is None else float(amount)
            payload = {n: wage for n in names}
        for name, value in payload.items():
            self.add(name)
            paid_sti = min(value, self.bank_sti)
            paid_lti = min(value * (self.p.lti_wage / max(self.p.wage, 1e-9)), self.bank_lti)
            self.atoms[name].sti += paid_sti
            self.atoms[name].lti += paid_lti
            self.bank_sti -= paid_sti
            self.bank_lti -= paid_lti

    def attentional_focus(self) -> list[AtomState]:
        """Atoms whose STI is at or above <FocusBoundary>, highest first."""
        focused = [a for a in self.atoms.values() if a.sti >= self.p.focus_boundary]
        focused.sort(key=lambda a: a.sti, reverse=True)
        return focused

    def query_similar(self, name: str, k: int = 5) -> list[tuple[str, float]]:
        self.add(name)
        probe = self.atoms[name].embedding
        scored = []
        for other in self.atoms.values():
            if other.name == name or other.embedding is None:
                continue
            scored.append((other.name, cosine(probe, other.embedding)))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:k]

    def cycle(self, forget: bool = False) -> CycleStats:
        """One ImportanceUpdating + Hebbian + diffusion step."""
        stats = CycleStats()
        stats.rent_collected = self._collect_rent()
        self._update_hebbian()
        stats.sti_diffused = self._diffuse()
        if forget:
            stats.forgotten = self._forget()
        self._refresh_recent_max()
        stats.af = [a.name for a in self.attentional_focus()]
        self.history.append(stats)
        return stats

    def run(self, steps: int, forget: bool = False) -> list[CycleStats]:
        return [self.cycle(forget=forget) for _ in range(steps)]

    def total_sti(self) -> float:
        return self.bank_sti + sum(a.sti for a in self.atoms.values())

    def snapshot(self) -> dict[str, dict[str, float]]:
        return {
            name: {"sti": atom.sti, "lti": atom.lti}
            for name, atom in sorted(self.atoms.items())
        }

    def _collect_rent(self) -> float:
        collected = 0.0
        recent_max = max(self.p.recent_max_sti, 1e-9)
        for atom in self.atoms.values():
            if atom.sti > 0:
                ratio = 20.0 * atom.sti / recent_max
                rent = self.p.rent * max(0.0, math.log(ratio) / 2.0) if ratio > 0 else 0.0
                rent = min(rent, atom.sti)
                atom.sti -= rent
                self.bank_sti += rent
                collected += rent
            lti_rent = min(self.p.lti_rent, max(0.0, atom.lti))
            atom.lti -= lti_rent
            self.bank_lti += lti_rent
        return collected

    def _norm_sti(self, sti: float) -> float:
        if sti >= 0:
            return sti / max(self.p.recent_max_sti, 1e-9)
        denom = self.p.recent_min_sti if self.p.recent_min_sti != 0 else -1.0
        return sti / denom

    def _update_hebbian(self) -> None:
        names = list(self.atoms)
        decay = self.p.conj_decay
        for i, a in enumerate(names):
            na = self._norm_sti(self.atoms[a].sti)
            for b in names[i + 1 :]:
                conj = na * self._norm_sti(self.atoms[b].sti)
                key = _edge(a, b)
                old = self.hebbian.get(key, 0.0)
                updated = decay * conj + (1.0 - conj) * old
                if updated >= 0:
                    self.hebbian[key] = float(min(1.0, updated))

    def _hybrid_strength(self, a: str, b: str) -> float:
        symbolic = self.hebbian.get(_edge(a, b), 0.0)
        ea = self.atoms[a].embedding
        eb = self.atoms[b].embedding
        neural = 0.0
        if ea is not None and eb is not None:
            neural = max(0.0, cosine(ea, eb) - self.p.neural_threshold)
        mix = self.p.neural_mix
        return (1.0 - mix) * symbolic + mix * neural

    def _diffuse(self) -> float:
        names = list(self.atoms)
        n = len(names)
        if n < 2:
            return 0.0
        index = {name: i for i, name in enumerate(names)}
        d = np.zeros((n, n), dtype=np.float64)
        for i, a in enumerate(names):
            for j, b in enumerate(names):
                if i == j:
                    continue
                d[j, i] = self._hybrid_strength(a, b)
        cap = self.p.max_spread
        for col in range(n):
            off = d[:, col].copy()
            off[col] = 0.0
            total = float(off.sum())
            if total > cap and total > 0:
                off *= cap / total
                total = cap
            d[:, col] = off
            d[col, col] = 1.0 - total
        sti = np.array([self.atoms[name].sti for name in names], dtype=np.float64)
        lo = float(sti.min())
        hi = float(sti.max())
        if math.isclose(hi, lo):
            return 0.0
        v = (sti - lo) / (hi - lo)
        v_next = d @ v
        sti_next = lo + v_next * (hi - lo)
        sti_next = np.clip(sti_next, self.p.sti_min, self.p.sti_max)
        moved = 0.0
        for name, new_sti in zip(names, sti_next):
            moved += abs(new_sti - self.atoms[name].sti)
            self.atoms[name].sti = float(new_sti)
        _ = index
        return moved / 2.0

    def _forget(self) -> list[str]:
        candidates = [
            a
            for a in self.atoms.values()
            if not a.vlti and a.sti < self.p.focus_boundary and a.lti < self.p.lti_forget_floor
        ]
        if not candidates:
            return []
        candidates.sort(key=lambda a: a.lti)
        drop_n = max(1, int(len(self.atoms) * self.p.lti_forget_fraction))
        dropped = []
        for atom in candidates[:drop_n]:
            self.bank_sti += atom.sti
            self.bank_lti += atom.lti
            del self.atoms[atom.name]
            self.hebbian = {k: v for k, v in self.hebbian.items() if atom.name not in k}
            dropped.append(atom.name)
        return dropped

    def _refresh_recent_max(self) -> None:
        if not self.atoms:
            return
        peak = max(a.sti for a in self.atoms.values())
        self.p.recent_max_sti = max(peak, 1.0)
        self.p.recent_min_sti = min(a.sti for a in self.atoms.values())


def _edge(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a < b else (b, a)
