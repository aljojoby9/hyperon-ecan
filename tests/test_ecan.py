import math

from hyperon_ecan import ECAN, ECANParams


def test_rent_zero_below_threshold():
    p = ECANParams(recent_max_sti=40.0, rent=1.0)
    net = ECAN(p)
    net.add("quiet")
    net.atoms["quiet"].sti = 1.0  # 20 * 1 / 40 = 0.5; log(0.5) < 0 → rent 0
    before = net.atoms["quiet"].sti
    net._collect_rent()
    assert net.atoms["quiet"].sti == before


def test_rent_positive_above_threshold():
    p = ECANParams(recent_max_sti=40.0, rent=1.0)
    net = ECAN(p)
    net.add("loud")
    net.atoms["loud"].sti = 20.0  # 20 * 20 / 40 = 10; log(10)/2 > 0
    net._collect_rent()
    assert net.atoms["loud"].sti < 20.0


def test_stimulus_moves_currency_from_bank():
    net = ECAN(ECANParams(bank_sti=100.0, wage=10.0))
    before = net.total_sti()
    net.stimulate(["dog"], amount=10.0)
    assert math.isclose(net.total_sti(), before)
    assert net.atoms["dog"].sti == 10.0
    assert net.bank_sti == 90.0


def test_hebbian_grows_for_coattended_atoms():
    net = ECAN(ECANParams(conj_decay=0.5, recent_max_sti=10.0))
    net.add_many(["a", "b"])
    net.atoms["a"].sti = 10.0
    net.atoms["b"].sti = 10.0
    net._update_hebbian()
    assert net.hebbian[("a", "b")] > 0.4


def test_forgetting_drops_low_lti():
    net = ECAN(ECANParams(lti_forget_floor=0.2, lti_forget_fraction=1.0, focus_boundary=8.0))
    net.add("keep", lti=5.0)
    net.add("drop", lti=0.01)
    net.atoms["keep"].sti = 12.0
    forgotten = net._forget()
    assert "drop" in forgotten
    assert "keep" in net.atoms
