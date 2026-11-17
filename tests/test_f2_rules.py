"""F2 inventory integrity. These guard the failure that produced a false
30.22% annotation-change figure: silently using the wrong inventory."""
import glob, os, collections, yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CANON = os.path.join(ROOT, "configs", "f2")


def _entries(d):
    out = []
    for f in glob.glob(os.path.join(d, "*.yml")):
        if os.path.basename(f) == "matching_config.yml":
            continue
        out += (yaml.safe_load(open(f)) or {}).get("entries") or []
    return out


def test_canonical_inventory_is_1097():
    assert len(_entries(CANON)) == 1097


def test_forty_corpus_informed_rules_present_and_composed_correctly():
    per = collections.Counter()
    for f in glob.glob(os.path.join(CANON, "*.yml")):
        for e in (yaml.safe_load(open(f)) or {}).get("entries") or []:
            if str(e.get("source_basis")) == "averitec_dataset_audit":
                per[os.path.basename(f)] += 1
    assert sum(per.values()) == 40
    assert per["established_news.yml"] == 21
    assert per["fact_checker.yml"] == 9
    assert per["aggregator.yml"] == 7
    assert per["low_credibility_news.yml"] == 3


def test_every_entry_has_required_fields():
    for e in _entries(CANON):
        for k in ("pattern", "match_type", "bucket", "source_basis"):
            assert k in e, f"entry missing {k}: {e}"
