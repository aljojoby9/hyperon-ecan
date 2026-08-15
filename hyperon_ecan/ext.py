"""Optional Hyperon / MeTTa bindings.

Install extra: `pip install hyperon-ecan[hyperon]`
Then from MeTTa:

    !(import! &self hyperon_ecan)
    !(ecan-new)
    !(ecan-add dog)
    !(ecan-link dog mammal)
    !(ecan-stimulate dog)
    !(ecan-cycle)
    !(ecan-focus)
    !(neural-similar dog)
"""

from __future__ import annotations

from .cycle import CognitiveCycle
from .inference import AttentionChainer, Triple
from .network import ECAN
from .params import ECANParams

try:
    from hyperon import OperationAtom, ValueAtom
    from hyperon.ext import register_atoms
except ImportError:  # pragma: no cover - hyperon is optional
    OperationAtom = None
    ValueAtom = None
    register_atoms = None


_STATE: dict[str, object] = {}


def _ecan() -> ECAN:
    ecan = _STATE.get("ecan")
    if not isinstance(ecan, ECAN):
        ecan = ECAN(ECANParams())
        _STATE["ecan"] = ecan
        _STATE["chainer"] = AttentionChainer(ecan)
    return ecan


def _chainer() -> AttentionChainer:
    _ecan()
    chainer = _STATE.get("chainer")
    if not isinstance(chainer, AttentionChainer):
        chainer = AttentionChainer(_ecan())
        _STATE["chainer"] = chainer
    return chainer


def ecan_new() -> str:
    _STATE["ecan"] = ECAN(ECANParams())
    _STATE["chainer"] = AttentionChainer(_STATE["ecan"])  # type: ignore[arg-type]
    return "ok"


def ecan_add(name: str) -> str:
    _ecan().add(str(name))
    return str(name)


def ecan_link(a: str, b: str, strength: float = 0.5) -> str:
    _ecan().link(str(a), str(b), float(strength))
    return f"{a}-{b}"


def ecan_stimulate(name: str, amount: float = 1.0) -> float:
    _ecan().stimulate({str(name): float(amount)})
    return _ecan().atoms[str(name)].sti


def ecan_cycle(forget: bool = False) -> int:
    stats = _ecan().cycle(forget=bool(forget))
    return len(stats.af)


def ecan_focus() -> list[str]:
    return [a.name for a in _ecan().attentional_focus()]


def ecan_sti(name: str) -> float:
    atom = _ecan().atoms.get(str(name))
    return 0.0 if atom is None else atom.sti


def neural_similar(name: str, k: int = 5) -> list[str]:
    return [n for n, _score in _ecan().query_similar(str(name), k=int(k))]


def ecan_fact(rel: str, src: str, dst: str) -> str:
    _chainer().assert_fact(Triple(str(rel), str(src), str(dst)))
    return f"({rel} {src} {dst})"


def ecan_infer() -> str:
    inferred = _chainer().step(require_focus=True)
    return "none" if inferred is None else f"({inferred.rel} {inferred.src} {inferred.dst})"


def ecan_tick(percept: str) -> str:
    cycle = CognitiveCycle(_ecan(), _chainer())
    tick = cycle.tick([str(percept)])
    focus = ",".join(tick.focus) if tick.focus else "-"
    inferred = "none" if tick.inferred is None else f"{tick.inferred.src}->{tick.inferred.dst}"
    return f"focus={focus}; inferred={inferred}"


if register_atoms is not None:  # pragma: no cover

    @register_atoms
    def ecan_atoms():
        return {
            "ecan-new": OperationAtom("ecan-new", ecan_new),
            "ecan-add": OperationAtom("ecan-add", ecan_add),
            "ecan-link": OperationAtom("ecan-link", ecan_link),
            "ecan-stimulate": OperationAtom("ecan-stimulate", ecan_stimulate),
            "ecan-cycle": OperationAtom("ecan-cycle", ecan_cycle),
            "ecan-focus": OperationAtom("ecan-focus", ecan_focus),
            "ecan-sti": OperationAtom("ecan-sti", ecan_sti),
            "neural-similar": OperationAtom("neural-similar", neural_similar),
            "ecan-fact": OperationAtom("ecan-fact", ecan_fact),
            "ecan-infer": OperationAtom("ecan-infer", ecan_infer),
            "ecan-tick": OperationAtom("ecan-tick", ecan_tick),
        }
