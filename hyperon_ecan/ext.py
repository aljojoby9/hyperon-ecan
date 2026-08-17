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
from .semantic import attach_clusters

try:
    from hyperon import E, ExpressionAtom, GroundedAtom, OperationAtom, S, SymbolAtom, ValueAtom
    from hyperon.ext import register_atoms
except ImportError:  # pragma: no cover - hyperon is optional
    E = None
    ExpressionAtom = None
    GroundedAtom = None
    OperationAtom = None
    S = None
    SymbolAtom = None
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


def _name(atom) -> str:
    if SymbolAtom is not None and isinstance(atom, SymbolAtom):
        return atom.get_name()
    if GroundedAtom is not None and isinstance(atom, GroundedAtom):
        obj = atom.get_object()
        for attr in ("value", "content"):
            if hasattr(obj, attr):
                return str(getattr(obj, attr))
        return str(obj)
    return str(atom)


def _num(atom, default: float) -> float:
    if GroundedAtom is not None and isinstance(atom, GroundedAtom):
        obj = atom.get_object()
        val = getattr(obj, "value", getattr(obj, "content", None))
        if isinstance(val, (int, float)):
            return float(val)
    try:
        return float(_name(atom))
    except ValueError:
        return default


def ecan_new(*_args):
    net = ECAN(ECANParams())
    _STATE["ecan"] = net
    _STATE["chainer"] = AttentionChainer(net)
    return [ValueAtom("ok")]


def ecan_add(name, *_args):
    label = _name(name)
    _ecan().add(label)
    return [S(label)]


def ecan_link(a, b, *rest):
    strength = _num(rest[0], 0.5) if rest else 0.5
    left, right = _name(a), _name(b)
    _ecan().link(left, right, strength)
    return [E(S(left), S(right))]


def ecan_stimulate(name, *rest):
    label = _name(name)
    amount = _num(rest[0], 1.0) if rest else 1.0
    _ecan().stimulate({label: amount})
    return [ValueAtom(_ecan().atoms[label].sti)]


def ecan_cycle(*rest):
    forget = bool(rest) and _name(rest[0]).lower() == "true"
    stats = _ecan().cycle(forget=forget)
    return [ValueAtom(len(stats.af))]


def ecan_focus(*_args):
    names = [a.name for a in _ecan().attentional_focus()]
    return [E(*[S(n) for n in names])] if names else [E()]


def ecan_sti(name, *_args):
    atom = _ecan().atoms.get(_name(name))
    return [ValueAtom(0.0 if atom is None else atom.sti)]


def neural_similar(name, *rest):
    k = int(_num(rest[0], 5.0)) if rest else 5
    hits = [n for n, _score in _ecan().query_similar(_name(name), k=k)]
    return [E(*[S(n) for n in hits])] if hits else [E()]


def ecan_fact(rel, src, dst, *_args):
    r, s, d = _name(rel), _name(src), _name(dst)
    _chainer().assert_fact(Triple(r, s, d))
    return [E(S(r), S(s), S(d))]


def ecan_infer(*_args):
    inferred = _chainer().step(require_focus=True)
    if inferred is None:
        return [S("none")]
    return [E(S(inferred.rel), S(inferred.src), S(inferred.dst))]


def _names_from(atom) -> list[str]:
    if ExpressionAtom is not None and isinstance(atom, ExpressionAtom):
        return [_name(child) for child in atom.get_children()]
    return [_name(atom)]


def ecan_report(*names):
    net = _ecan()
    keys = [_name(n) for n in names] if names else sorted(net.atoms)
    focus = {a.name for a in net.attentional_focus()}
    lines = []
    for key in keys:
        atom = net.atoms.get(key)
        if atom is None:
            lines.append(f"{key}: missing")
            continue
        flag = "focus" if key in focus else "idle"
        lines.append(f"{key}: sti={atom.sti:.2f} {flag}")
    return [ValueAtom(" | ".join(lines) if lines else "empty")]


def ecan_cluster(*groups):
    """!(ecan-cluster (dog wolf) (oak tree)) — shared embedding per group."""
    clusters = [_names_from(group) for group in groups]
    attach_clusters(_ecan(), clusters)
    return [E(*[E(*[S(n) for n in cluster]) for cluster in clusters])]


def ecan_tick(percept, *_args):
    cycle = CognitiveCycle(_ecan(), _chainer())
    tick = cycle.tick([_name(percept)])
    focus = ",".join(tick.focus) if tick.focus else "-"
    inferred = "none" if tick.inferred is None else f"{tick.inferred.src}->{tick.inferred.dst}"
    return [ValueAtom(f"focus={focus}; inferred={inferred}")]


def ecan_atoms():
    """Exposed on the hyperon_ecan package so `!(import! &self hyperon_ecan)` finds it."""
    if OperationAtom is None:
        return {}
    return {
        "ecan-new": OperationAtom("ecan-new", ecan_new, unwrap=False),
        "ecan-add": OperationAtom("ecan-add", ecan_add, unwrap=False),
        "ecan-link": OperationAtom("ecan-link", ecan_link, unwrap=False),
        "ecan-stimulate": OperationAtom("ecan-stimulate", ecan_stimulate, unwrap=False),
        "ecan-cycle": OperationAtom("ecan-cycle", ecan_cycle, unwrap=False),
        "ecan-focus": OperationAtom("ecan-focus", ecan_focus, unwrap=False),
        "ecan-sti": OperationAtom("ecan-sti", ecan_sti, unwrap=False),
        "neural-similar": OperationAtom("neural-similar", neural_similar, unwrap=False),
        "ecan-fact": OperationAtom("ecan-fact", ecan_fact, unwrap=False),
        "ecan-cluster": OperationAtom("ecan-cluster", ecan_cluster, unwrap=False),
        "ecan-report": OperationAtom("ecan-report", ecan_report, unwrap=False),
        "ecan-infer": OperationAtom("ecan-infer", ecan_infer, unwrap=False),
        "ecan-tick": OperationAtom("ecan-tick", ecan_tick, unwrap=False),
    }


if register_atoms is not None:  # pragma: no cover
    ecan_atoms = register_atoms(ecan_atoms)
