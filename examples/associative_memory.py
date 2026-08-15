"""Replicate the associative-memory experiment from Iklé et al. 2009.

Imprint a sparse pattern, then cue a subset and see whether the rest of
the pattern re-enters attentional focus via Hebbian + neural spreading.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hyperon_ecan import ECAN, ECANParams


def imprint_and_retrieve() -> dict[str, object]:
    net = ECAN(
        ECANParams(
            focus_boundary=6.0,
            rent=0.5,
            wage=12.0,
            max_spread=0.5,
            neural_mix=0.25,
            neural_threshold=0.5,
        )
    )
    # Two stored "memories" plus distractors.
    pattern_a = ["dog", "bark", "leash", "park"]
    pattern_b = ["cat", "meow", "yarn", "sofa"]
    distractors = ["quantum", "invoice", "volcano", "algebra"]
    net.add_many(pattern_a + pattern_b + distractors)
    for group in (pattern_a, pattern_b):
        for i, src in enumerate(group):
            for dst in group[i + 1 :]:
                net.link(src, dst, strength=0.8)

    # Imprint memory A, then let it settle.
    net.stimulate(pattern_a, amount=18.0)
    net.run(4)

    # Clear working memory by collecting rent without new stimulus.
    for _ in range(8):
        net.cycle()

    # Cue with a noisy subset of A.
    net.stimulate(["dog", "bark"], amount=16.0)
    net.run(6)

    focus = {a.name for a in net.attentional_focus()}
    return {
        "focus": sorted(focus, key=lambda n: net.atoms[n].sti, reverse=True),
        "recovered_a": sorted(focus & set(pattern_a)),
        "leaked_b": sorted(focus & set(pattern_b)),
        "leaked_distractors": sorted(focus & set(distractors)),
        "sti": {name: round(net.atoms[name].sti, 2) for name in pattern_a + pattern_b},
    }


if __name__ == "__main__":
    result = imprint_and_retrieve()
    print("Attentional focus:", result["focus"])
    print("Recovered pattern A:", result["recovered_a"])
    print("Leaked pattern B:   ", result["leaked_b"])
    print("Leaked distractors: ", result["leaked_distractors"])
    print("STI snapshot:")
    for name, sti in result["sti"].items():
        print(f"  {name:10s} {sti:6.2f}")
