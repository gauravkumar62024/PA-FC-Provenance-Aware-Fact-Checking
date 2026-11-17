"""LIAR-RAW 6-way -> 3-way. Must be applied identically to gold and predictions."""
SIX_TO_THREE = {
    "pants-fire": "FALSE", "false": "FALSE", "barely-true": "FALSE",
    "half-true": "HALF-TRUE",
    "mostly-true": "TRUE", "true": "TRUE",
}


def test_three_classes_only():
    assert sorted(set(SIX_TO_THREE.values())) == ["FALSE", "HALF-TRUE", "TRUE"]


def test_all_six_source_labels_covered():
    assert len(SIX_TO_THREE) == 6


def test_boundary_assignments():
    assert SIX_TO_THREE["barely-true"] == "FALSE"      # not HALF-TRUE
    assert SIX_TO_THREE["mostly-true"] == "TRUE"
    assert SIX_TO_THREE["half-true"] == "HALF-TRUE"
