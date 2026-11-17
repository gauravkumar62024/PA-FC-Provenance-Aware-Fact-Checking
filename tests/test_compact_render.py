"""The compact metadata line is part of the trained token sequence.

FLAGS order and the field layout are load-bearing: the verifier was fine-tuned
on exactly this string shape. Reordering flags, renaming one, or changing the
separator produces prompts that no longer match training, and the resulting
numbers are not comparable to the reported ones. These tests pin the format.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pafc.compact import FLAG_FIELDS, flags_for, line_for  # noqa: E402

# The published order. Do not sort, do not regroup.
EXPECTED_FLAG_ORDER = [
    "leakage_sensitive",
    "archive_copy",
    "fact_check",
    "primary_source",
    "low_credibility",
    "satire",
    "reference_data",
    "multimedia",
]


def test_flag_order_is_pinned():
    assert [label for _, label in FLAG_FIELDS] == EXPECTED_FLAG_ORDER


def test_flag_keys_are_pinned():
    assert [key for key, _ in FLAG_FIELDS] == [
        "f2_leakage_sensitive",
        "f4_is_archive_or_copy",
        "f4_is_fact_check_article",
        "f4_is_primary_or_authoritative",
        "f4_is_low_credibility_or_abuse",
        "f4_is_satire_or_humor",
        "f4_is_reference_or_data",
        "f4_is_multimedia",
    ]


def test_flags_render_in_declaration_order_not_insertion_order():
    # fact_check set before leakage_sensitive in the dict; output must still
    # put leakage_sensitive first.
    evidence = {"f4_is_fact_check_article": True, "f2_leakage_sensitive": True}
    assert flags_for(evidence) == "leakage_sensitive, fact_check"


def test_no_flags_renders_as_dash():
    assert flags_for({}) == "-"


def test_line_shape_and_nli_passthrough():
    evidence = {
        "f2_source_type": "fact_checker",
        "f3_trust_level": "high",
        "f4_provenance_level": "weak",
        "f2_leakage_sensitive": True,
        "f4_is_fact_check_article": True,
    }
    original = "NLI: refute (0.94) | SRC: x | TRUST: y | PROV: z | FLAGS: -"
    assert line_for(evidence, original) == (
        "NLI: refute (0.94) | SRC: fact_checker | TRUST: high"
        " | PROV: weak | FLAGS: leakage_sensitive, fact_check"
    )


def test_f1_is_never_recomputed_only_carried_through():
    # F1 is computed once and copied verbatim; the renderer must not touch it
    # even when every other field changes.
    original = "NLI: support (0.81) | SRC: a | TRUST: b | PROV: c | FLAGS: satire"
    out = line_for({"f2_source_type": "blog_cms"}, original)
    assert out.startswith("NLI: support (0.81) |")


def test_unknown_fields_fall_back_to_unknown_not_none():
    out = line_for({}, "NLI: nei (0.50) | SRC: a | TRUST: b | PROV: c | FLAGS: -")
    assert "SRC: unknown" in out and "TRUST: unknown" in out and "PROV: unknown" in out
    assert "None" not in out
