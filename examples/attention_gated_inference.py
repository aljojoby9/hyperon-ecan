"""Attention-gated inheritance: prove (Inheritance dog animal) without
walking the whole graph.

This is the PLN/ECAN interlock from the last section of Iklé et al. 2009,
running as Hyperon-oriented Python. The neural mix also pulls `wolf` into
focus from `dog` even though no Inheritance edge was asserted.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hyperon_ecan import (
    AttentionChainer,
    CognitiveCycle,
    ECAN,
    ECANParams,
    Triple,
    attach_clusters,
)


def run_demo() -> dict[str, object]:
    net = ECAN(
        ECANParams(
            focus_boundary=5.0,
            wage=10.0,
            rent=0.45,
            neural_mix=0.45,
            neural_threshold=0.35,
            max_spread=0.45,
        )
    )
    chainer = AttentionChainer(net)
    facts = [
        Triple("Inheritance", "dog", "mammal"),
        Triple("Inheritance", "mammal", "animal"),
        Triple("Inheritance", "cat", "mammal"),
        Triple("Inheritance", "oak", "tree"),
        Triple("Inheritance", "tree", "plant"),
        Triple("Inheritance", "invoice", "document"),
        Triple("Evaluation", "bark", "dog"),
    ]
    for fact in facts:
        chainer.assert_fact(fact)
    net.add("wolf")  # no symbolic Inheritance; enters via the neural cluster
    attach_clusters(
        net,
        [
            ["dog", "wolf", "bark", "mammal"],
            ["oak", "tree", "plant"],
            ["invoice", "document"],
        ],
    )

    goal = Triple("Inheritance", "dog", "animal")
    cycle = CognitiveCycle(net, chainer)
    log = cycle.run(["dog", "bark"], goal=goal, ticks=10, restimulate_every=3)

    return {
        "proved": any(tick.proved for tick in log),
        "inferences": [
            f"{t.inferred.src}->{t.inferred.dst}" for t in log if t.inferred is not None
        ],
        "final_focus": log[-1].focus if log else [],
        "wolf_sti": round(net.atoms["wolf"].sti, 2),
        "oak_sti": round(net.atoms["oak"].sti, 2),
        "ticks": len(log),
    }


if __name__ == "__main__":
    result = run_demo()
    print("Proved (Inheritance dog animal):", result["proved"])
    print("Inferences:", result["inferences"])
    print("Final focus:", result["final_focus"])
    print("wolf STI (neural neighbor of dog):", result["wolf_sti"])
    print("oak STI (unrelated distractor):   ", result["oak_sti"])
    print("Ticks used:", result["ticks"])
