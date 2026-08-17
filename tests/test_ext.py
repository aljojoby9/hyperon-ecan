import hyperon_ecan
from hyperon_ecan.ext import _name, ecan_atoms, ecan_fact, ecan_new


def test_ecan_atoms_lives_on_the_package():
    assert hasattr(hyperon_ecan, "ecan_atoms")
    assert callable(hyperon_ecan.ecan_atoms)


def test_name_reads_plain_values():
    assert _name("dog") == "dog"


def test_fact_returns_expression_when_hyperon_is_present():
    table = ecan_atoms()
    if not table:
        return
    from hyperon import E, MeTTa, S

    ecan_new()
    result = ecan_fact(S("Inheritance"), S("dog"), S("mammal"))
    assert result == [E(S("Inheritance"), S("dog"), S("mammal"))]

    metta = MeTTa()
    metta.run("!(import! &self hyperon_ecan)")
    out = metta.run("!(ecan-new)\n!(ecan-fact Inheritance dog mammal)")
    assert str(out[-1][0]) == "(Inheritance dog mammal)"
