from hyperon_ecan import (
    AttentionChainer,
    CognitiveCycle,
    ECAN,
    ECANParams,
    Triple,
    attach_clusters,
)
from examples.associative_memory import imprint_and_retrieve
from examples.attention_gated_inference import run_demo


def test_associative_memory_recovers_imprinted_pattern():
    result = imprint_and_retrieve()
    recovered = set(result["recovered_a"])
    assert "dog" in recovered
    assert "bark" in recovered
    assert len(result["leaked_distractors"]) == 0
    assert len(recovered) >= 3


def test_attention_gated_inference_proves_goal():
    result = run_demo()
    assert result["proved"] is True
    assert "dog->animal" in result["inferences"]
    assert result["wolf_sti"] > result["oak_sti"]


def test_walkthrough_stimulus_keeps_dog_in_focus():
    net = ECAN(ECANParams())
    chainer = AttentionChainer(net)
    chainer.assert_fact(Triple("Inheritance", "dog", "mammal"))
    chainer.assert_fact(Triple("Inheritance", "mammal", "animal"))
    chainer.assert_fact(Triple("Inheritance", "oak", "tree"))
    net.add("wolf")
    attach_clusters(net, [["dog", "wolf", "mammal"], ["oak", "tree"]])
    net.stimulate({"dog": 20.0})
    net.cycle()
    focus = {a.name for a in net.attentional_focus()}
    assert "dog" in focus
    assert net.atoms["wolf"].sti > net.atoms["oak"].sti
    assert chainer.step() == Triple("Inheritance", "dog", "animal")


def test_unfocused_subgraph_is_not_expanded():
    net = ECAN(ECANParams(focus_boundary=8.0, wage=12.0, neural_mix=0.0))
    chainer = AttentionChainer(net)
    chainer.assert_fact(Triple("Inheritance", "dog", "mammal"))
    chainer.assert_fact(Triple("Inheritance", "mammal", "animal"))
    chainer.assert_fact(Triple("Inheritance", "oak", "tree"))
    chainer.assert_fact(Triple("Inheritance", "tree", "plant"))
    net.stimulate(["dog"], amount=16.0)
    net.cycle()
    cycle = CognitiveCycle(net, chainer)
    for _ in range(6):
        cycle.tick(["dog"])
    inferred = {f.as_tuple() for f in chainer.facts}
    assert ("Inheritance", "dog", "animal") in inferred
    assert ("Inheritance", "oak", "plant") not in inferred
