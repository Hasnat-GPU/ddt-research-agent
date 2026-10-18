from ddt import selfmodel

T = "eu ai act"


def test_default_bar_is_1():
    assert selfmodel.get(T)["evidence_bar"] == 1


def test_raise_at_2_of_5():
    line = selfmodel.update(T, [True, True, False, False, False], 3, "m")
    assert selfmodel.get(T)["evidence_bar"] == 2
    assert line == 'evidence bar for "eu ai act": 1 -> 2 (2 of 5 claims passed)'


def test_keep_at_4_of_5():
    selfmodel.update(T, [True, True, False, False, False], 3, "m")
    selfmodel.update(T, [True, True, True, True, False], 3, "m")
    assert selfmodel.get(T)["evidence_bar"] == 2
    selfmodel.update("other", [True, True, True, True, False], 3, "m")
    assert selfmodel.get("other")["evidence_bar"] == 1


def test_lower_at_5_of_5():
    selfmodel.update(T, [False, False, False], 3, "m")
    assert selfmodel.get(T)["evidence_bar"] == 2
    line = selfmodel.update(T, [True] * 5, 3, "m")
    assert selfmodel.get(T)["evidence_bar"] == 1
    assert line.endswith("2 -> 1 (5 of 5 claims passed)")
    assert selfmodel.get(T)["runs"] == 2 and selfmodel.get(T)["checked"] == 8 and selfmodel.get(T)["passed"] == 5
