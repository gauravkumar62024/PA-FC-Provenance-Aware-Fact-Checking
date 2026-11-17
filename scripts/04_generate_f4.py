import os as _os, sys as _sys
_PAFC_SRC = _os.path.join(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__))), "src")
if _PAFC_SRC not in _sys.path:
    _sys.path.insert(0, _PAFC_SRC)

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


TOP_KEYS = ("top_10", "top_20", "top_30", "top_k", "evidence")


TWO_PART_SUFFIXES = {
    "co.uk", "ac.uk", "gov.uk", "org.uk", "nhs.uk",
    "com.au", "gov.au", "edu.au", "org.au",
    "co.nz", "govt.nz", "ac.nz", "org.nz",
    "co.in", "gov.in", "nic.in", "ac.in", "edu.in", "org.in",
    "com.ng", "gov.ng", "edu.ng", "org.ng",
    "com.ph", "gov.ph", "edu.ph", "org.ph",
    "com.pk", "gov.pk", "edu.pk", "org.pk",
    "co.ke", "go.ke", "ac.ke", "or.ke",
    "com.br", "gov.br", "org.br",
    "com.mx", "gob.mx", "org.mx",
    "com.tr", "gov.tr", "edu.tr",
    "com.sg", "gov.sg", "edu.sg",
    "co.za", "gov.za", "ac.za", "org.za",
    "or.jp", "co.jp", "go.jp", "ne.jp",
    "com.my", "gov.my", "edu.my",
    "com.bd", "gov.bd", "edu.bd",
    "com.hk", "gov.hk", "edu.hk",
    "com.cn", "gov.cn", "edu.cn",
    "com.tw", "gov.tw", "edu.tw",
}


ARCHIVE_DOMAINS = {
    "archive.org",
    "web.archive.org",
    "archive.today",
    "archive.ph",
    "archive.is",
    "archive.vn",
    "perma.cc",
}


SATIRE_DOMAINS = {
    "scoopertino.com",
    "theonion.com",
    "babylonbee.com",
    "clickhole.com",
    "duffelblog.com",
    "dailycurrant.com",
    "nationalreport.net",
    "worldnewsdailyreport.com",
    "empirenews.net",
    "waterfordwhispersnews.com",
    "newsthump.com",
    "newsbiscuit.com",
    "thebeaverton.com",
    "satirewire.com",
}


REFERENCE_DOMAINS = {
    "wikipedia.org",
    "wikidata.org",
    "britannica.com",
    "encyclopedia.com",
    "worldatlas.com",
    "history.com",
    "biography.com",
}


DATA_REFERENCE_DOMAINS = {
    "worldometers.info",
    "worldpopulationreview.com",
    "macrotrends.net",
    "statista.com",
    "tradingeconomics.com",
    "ourworldindata.org",
    "populationpyramid.net",
    "data.worldbank.org",
    "data.gov",
    "census.gov",
    "ons.gov.uk",
    "bls.gov",
    "bea.gov",
    "eia.gov",
    "fred.stlouisfed.org",
    "knoema.com",
    "indexmundi.com",
}


HEALTH_REFERENCE_DOMAINS = {
    "healthline.com",
    "medicalnewstoday.com",
    "mayoclinic.org",
    "webmd.com",
    "clevelandclinic.org",
    "hopkinsmedicine.org",
    "cdc.gov",
    "who.int",
    "nih.gov",
    "ncbi.nlm.nih.gov",
    "pubmed.ncbi.nlm.nih.gov",
    "fda.gov",
    "nhs.uk",
    "kff.org",
    "health.gov",
}


LEGAL_CIVIC_REFERENCE_DOMAINS = {
    "ballotpedia.org",
    "brennancenter.org",
    "law.cornell.edu",
    "oyez.org",
    "supreme.justia.com",
    "justia.com",
    "supremecourt.gov",
    "congress.gov",
    "federalregister.gov",
    "ecfr.gov",
    "legislation.gov.uk",
    "parliament.uk",
    "loc.gov",
    "courtlistener.com",
    "openstates.org",
}


TRANSCRIPT_DOMAINS = {
    "rev.com",
    "factba.se",
    "c-span.org",
    "presidency.ucsb.edu",
    "whitehouse.gov",
    "transcripts.cnn.com",
    "debates.org",
}


NGO_THINKTANK_POLICY_DOMAINS = {
    "cfr.org",
    "pewresearch.org",
    "brookings.edu",
    "rand.org",
    "heritage.org",
    "migrationpolicy.org",
    "amnesty.org",
    "hrw.org",
    "transparency.org",
    "prb.org",
    "fas.org",
    "refworld.org",
    "ifrc.org",
    "icrc.org",
    "unhcr.org",
    "urban.org",
    "cbpp.org",
    "epi.org",
    "csis.org",
    "carnegieendowment.org",
    "chathamhouse.org",
    "atlanticcouncil.org",
}


INTERGOVERNMENTAL_OR_OFFICIAL_ORG_DOMAINS = {
    "un.org",
    "who.int",
    "worldbank.org",
    "imf.org",
    "oecd.org",
    "europa.eu",
    "nato.int",
    "wto.org",
    "unesco.org",
    "unicef.org",
    "ilo.org",
    "fao.org",
    "reliefweb.int",
}


ACADEMIC_OR_SCIENCE_DOMAINS = {
    "arxiv.org",
    "doi.org",
    "nature.com",
    "science.org",
    "sciencedirect.com",
    "springer.com",
    "wiley.com",
    "tandfonline.com",
    "jstor.org",
    "researchgate.net",
    "semanticscholar.org",
    "scholar.google.com",
    "plos.org",
    "bmj.com",
    "thelancet.com",
    "nejm.org",
    "pubmed.ncbi.nlm.nih.gov",
    "ncbi.nlm.nih.gov",
}


FACT_CHECK_DOMAINS = {
    "snopes.com",
    "politifact.com",
    "factcheck.org",
    "fullfact.org",
    "reuters.com",
    "apnews.com",
    "afp.com",
    "factcheck.afp.com",
    "leadstories.com",
    "checkyourfact.com",
    "boomlive.in",
    "altnews.in",
    "factly.in",
    "facthunt.in",
    "newsmeter.in",
    "aap.com.au",
    "truthorfiction.com",
    "healthfeedback.org",
    "climatefeedback.org",
}


NEWS_OR_MEDIA_DOMAINS = {
    "bbc.co.uk",
    "bbc.com",
    "reuters.com",
    "apnews.com",
    "nytimes.com",
    "washingtonpost.com",
    "theguardian.com",
    "npr.org",
    "pbs.org",
    "voanews.com",
    "dw.com",
    "france24.com",
    "rfi.fr",
    "aljazeera.com",
    "cnn.com",
    "nbcnews.com",
    "cbsnews.com",
    "abcnews.go.com",
    "foxnews.com",
    "usatoday.com",
    "latimes.com",
    "bloomberg.com",
    "businessinsider.com",
    "cnet.com",
    "telegraph.co.uk",
    "dailymail.co.uk",
    "nypost.com",
    "economictimes.com",
    "timesofindia.com",
    "indiatimes.com",
    "indiatoday.in",
    "newindianexpress.com",
    "thehindu.com",
    "hindustantimes.com",
    "dawn.com",
    "tribune.com.pk",
    "rappler.com",
    "channelstv.com",
    "dailypost.ng",
    "thecable.ng",
    "statesman.com",
    "qz.com",
}


LOW_CREDIBILITY_OR_HIGH_CAUTION_DOMAINS = {
    "realrawnews.com",
    "naturalnews.com",
    "beforeitsnews.com",
    "infowars.com",
    "thegatewaypundit.com",
    "zerohedge.com",
    "childrenshealthdefense.org",
    "worldtruth.tv",
    "yournewswire.com",
    "newswars.com",
}


SOCIAL_DOMAINS = {
    "twitter.com",
    "x.com",
    "facebook.com",
    "instagram.com",
    "tiktok.com",
    "youtube.com",
    "youtu.be",
    "telegram.org",
    "telegram.me",
    "reddit.com",
    "pinterest.com",
    "linkedin.com",
    "threads.net",
}


FORUM_COMMUNITY_DOMAINS = {
    "reddit.com",
    "quora.com",
    "stackexchange.com",
    "stackoverflow.com",
    "hackernews.com",
    "news.ycombinator.com",
    "nairaland.com",
    "philmug.ph",
}


BLOG_CMS_DOMAINS = {
    "medium.com",
    "substack.com",
    "wordpress.com",
    "blogspot.com",
    "tumblr.com",
    "typepad.com",
}


AGGREGATOR_OR_INDEX_DOMAINS = {
    "google.com",
    "news.google.com",
    "yahoo.com",
    "finance.yahoo.com",
    "money.yahoo.com",
    "msn.com",
    "apple.news",
    "flipboard.com",
    "newsbreak.com",
    "allafrica.com",
    "newsnow.co.uk",
}


GOVERNMENT_SUFFIXES = {
    ".gov",
    ".mil",
    ".gov.uk",
    ".gov.au",
    ".gov.in",
    ".gov.ng",
    ".gov.ph",
    ".gov.pk",
    ".gov.sg",
    ".gov.za",
    ".gob.mx",
    ".go.ke",
    ".go.jp",
    ".gov.br",
    ".gov.tr",
    ".gov.bd",
    ".gov.hk",
    ".gov.cn",
    ".gov.tw",
}


ACADEMIC_SUFFIXES = {
    ".edu",
    ".ac.uk",
    ".edu.au",
    ".ac.in",
    ".edu.in",
    ".edu.ng",
    ".edu.ph",
    ".edu.pk",
    ".ac.ke",
    ".edu.sg",
    ".ac.za",
    ".edu.tr",
    ".edu.bd",
    ".edu.hk",
    ".edu.cn",
    ".edu.tw",
}


FORUM_HINTS = (
    "/forum",
    "/forums",
    "/community",
    "/thread",
    "/threads",
    "/discussion",
    "/discussions",
    "/topic",
    "/posts",
    "/comment",
)


BLOG_HINTS = (
    "blog.",
    "/blog/",
    "wordpress",
    "blogspot",
    "medium.com",
    "substack.com",
    "/opinion/",
    "/column/",
)


PDF_HINTS = (".pdf",)

VIDEO_HINTS = (
    "youtube.com",
    "youtu.be",
    "vimeo.com",
    "tiktok.com",
    "/video/",
    "/videos/",
)

IMAGE_HINTS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".webp",
    ".svg",
)


FACT_CHECK_PATH_HINTS = (
    "/fact-check",
    "/factcheck",
    "/fact-checks",
    "/fact_checks",
    "/debunk",
    "/verify",
    "/hoax",
)


def read_json_or_jsonl(path: Path) -> tuple[list[dict[str, Any]], str]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return [], "jsonl"

    if text[0] == "[":
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"Expected JSON array in {path}")
        return data, "json_array"

    records = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue

        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at line {line_no} in {path}: {exc}") from exc

        if not isinstance(obj, dict):
            raise ValueError(f"Expected JSON object at line {line_no} in {path}")

        records.append(obj)

    return records, "jsonl"


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def find_evidence_key(record: dict[str, Any]) -> str | None:
    for key in TOP_KEYS:
        if isinstance(record.get(key), list):
            return key
    return None


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        if math.isnan(result) or math.isinf(result):
            return default
        return result
    except Exception:
        return default


def normalize_domain(value: Any) -> str:
    value = "" if value is None else str(value).strip().lower()
    if not value:
        return ""

    if "://" in value:
        parsed = urlparse(value)
        host = parsed.netloc.lower()
    else:
        host = value.split("/")[0].lower()

    host = host.split("@")[-1].split(":")[0].strip(".")
    if host.startswith("www."):
        host = host[4:]

    if not host:
        return ""

    try:
        import tldextract

        ext = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=False)(host)
        if ext.domain and ext.suffix:
            return f"{ext.domain}.{ext.suffix}".lower()
    except Exception:
        pass

    parts = host.split(".")
    if len(parts) <= 2:
        return host

    suffix2 = ".".join(parts[-2:])
    if suffix2 in TWO_PART_SUFFIXES and len(parts) >= 3:
        return ".".join(parts[-3:])

    return ".".join(parts[-2:])


def domain_matches(domain: str, patterns: set[str]) -> bool:
    domain = (domain or "").lower().strip(".")
    if not domain:
        return False

    for pattern in patterns:
        pattern = pattern.lower().strip(".")
        if domain == pattern or domain.endswith("." + pattern):
            return True

    return False


def domain_has_suffix(domain: str, suffixes: set[str]) -> bool:
    domain = (domain or "").lower().strip(".")
    if not domain:
        return False

    for suffix in suffixes:
        suffix = suffix.lower()
        bare = suffix.lstrip(".")
        if domain == bare or domain.endswith(suffix):
            return True

    return False


def url_contains_any(url: str, hints: tuple[str, ...]) -> bool:
    text = (url or "").lower()
    return any(hint.lower() in text for hint in hints)


def text_contains_any(ev: dict[str, Any], needles: tuple[str, ...]) -> bool:
    text = " ".join([
        str(ev.get("sentence") or ""),
        str(ev.get("url") or ""),
        str(ev.get("canonical_source_url") or ""),
        str(ev.get("domain") or ""),
    ]).lower()
    return any(needle.lower() in text for needle in needles)


def add_label(labels: list[str], label: str) -> None:
    if label not in labels:
        labels.append(label)


def get_domain(ev: dict[str, Any]) -> str:
    for key in (
        "domain",
        "f4_domain",
        "f3_domain",
        "registered_domain",
        "canonical_source_url",
        "retrieved_url",
        "url",
        "original_domain",
    ):
        value = ev.get(key)
        if value:
            domain = normalize_domain(value)
            if domain:
                return domain
    return ""


def get_url(ev: dict[str, Any]) -> str:
    return str(
        ev.get("canonical_source_url")
        or ev.get("retrieved_url")
        or ev.get("url")
        or ""
    ).strip()


def get_original_url(ev: dict[str, Any]) -> str:
    return str(ev.get("retrieved_url") or ev.get("url") or "").strip()


def get_path(url: str) -> str:
    try:
        return urlparse(url).path.lower()
    except Exception:
        return ""


def get_f2_source_type(ev: dict[str, Any]) -> str:
    if ev.get("f2_source_type"):
        return str(ev["f2_source_type"]).strip()

    pafc = ev.get("pafc")
    if isinstance(pafc, dict):
        f2 = pafc.get("f2")
        if isinstance(f2, dict) and f2.get("source_type"):
            return str(f2["source_type"]).strip()

    return "unknown"


def get_f3_trust_score(ev: dict[str, Any]) -> float:
    if ev.get("f3_trust_score") is not None:
        return safe_float(ev.get("f3_trust_score"), 0.5)

    pafc = ev.get("pafc")
    if isinstance(pafc, dict):
        f3 = pafc.get("f3")
        if isinstance(f3, dict):
            return safe_float(f3.get("trust_score"), 0.5)

    return 0.5


def apply_source_medium_labels(ev: dict[str, Any], labels: list[str], reasons: list[str]) -> None:
    raw = str(ev.get("source_medium_raw") or "").lower().strip()

    if not raw:
        return

    if "pdf" in raw:
        add_label(labels, "document_or_pdf")
        reasons.append("source_medium_pdf")

    if "table" in raw:
        add_label(labels, "table_or_structured_data")
        reasons.append("source_medium_table")

    if "image" in raw or "graphic" in raw:
        add_label(labels, "image_or_graphic_source")
        reasons.append("source_medium_image_or_graphic")

    if "video" in raw:
        add_label(labels, "multimedia_or_video_source")
        reasons.append("source_medium_video")

    if "metadata" in raw:
        add_label(labels, "metadata_source")
        reasons.append("source_medium_metadata")


def apply_f2_labels(ev: dict[str, Any], labels: list[str], reasons: list[str]) -> None:
    f2_source_type = get_f2_source_type(ev)

    if f2_source_type in {"gov", "government_official"}:
        add_label(labels, "official_record")
        add_label(labels, "primary_or_authoritative_source")
        reasons.append("f2_government_official")

    elif f2_source_type in {"academic", "academic_scientific"}:
        add_label(labels, "academic_or_scientific_source")
        add_label(labels, "primary_or_expert_source")
        reasons.append("f2_academic_scientific")

    elif f2_source_type == "fact_checker":
        add_label(labels, "fact_check_article")
        add_label(labels, "secondary_verification_source")
        reasons.append("f2_fact_checker")
        if bool(ev.get("f2_leakage_sensitive")):
            add_label(labels, "leakage_sensitive_source")
            reasons.append("f2_leakage_sensitive")

    elif f2_source_type in {"est_news", "established_news"}:
        add_label(labels, "established_secondary_report")
        reasons.append("f2_established_news")

    elif f2_source_type == "aggregator":
        add_label(labels, "aggregator_or_index")
        reasons.append("f2_aggregator")

    elif f2_source_type in {"blog", "blog_cms"}:
        add_label(labels, "self_published_or_blog")
        add_label(labels, "secondary_or_commentary")
        reasons.append("f2_blog_cms")

    elif f2_source_type in {"forum", "forum_community"}:
        add_label(labels, "forum_or_community_discussion")
        reasons.append("f2_forum_community")

    elif f2_source_type in {"social", "social_platform"}:
        add_label(labels, "social_platform_post")
        reasons.append("f2_social_platform")

    elif f2_source_type == "low_credibility_news":
        add_label(labels, "low_credibility_watchlist")
        add_label(labels, "secondary_or_commentary")
        reasons.append("f2_low_credibility_news")

    elif f2_source_type == "cyber_abuse_or_fraud_signal":
        add_label(labels, "cyber_abuse_or_fraud_signal")
        reasons.append("f2_cyber_abuse_or_fraud_signal")

    else:
        add_label(labels, "source_type_unknown")
        reasons.append("f2_unknown")


def apply_domain_labels(domain: str, url: str, labels: list[str], reasons: list[str]) -> None:
    if domain_matches(domain, SATIRE_DOMAINS):
        add_label(labels, "satire_or_humor_source")
        add_label(labels, "not_independent_factual_source")
        reasons.append(f"satire_domain:{domain}")

    if domain_matches(domain, REFERENCE_DOMAINS):
        add_label(labels, "reference_encyclopedia")
        reasons.append(f"reference_domain:{domain}")

    if domain_matches(domain, DATA_REFERENCE_DOMAINS):
        add_label(labels, "statistical_or_data_reference")
        reasons.append(f"data_reference_domain:{domain}")

    if domain_matches(domain, HEALTH_REFERENCE_DOMAINS):
        add_label(labels, "medical_or_health_reference")
        reasons.append(f"health_reference_domain:{domain}")

    if domain_matches(domain, LEGAL_CIVIC_REFERENCE_DOMAINS):
        add_label(labels, "legal_or_civic_reference")
        reasons.append(f"legal_civic_domain:{domain}")

    if domain_matches(domain, TRANSCRIPT_DOMAINS):
        add_label(labels, "transcript_or_primary_speech_database")
        reasons.append(f"transcript_domain:{domain}")

    if domain_matches(domain, NGO_THINKTANK_POLICY_DOMAINS):
        add_label(labels, "ngo_thinktank_policy_source")
        add_label(labels, "expert_or_institutional_source")
        reasons.append(f"ngo_thinktank_domain:{domain}")

    if domain_matches(domain, INTERGOVERNMENTAL_OR_OFFICIAL_ORG_DOMAINS):
        add_label(labels, "intergovernmental_or_official_org_source")
        add_label(labels, "primary_or_authoritative_source")
        reasons.append(f"intergovernmental_domain:{domain}")

    if domain_matches(domain, ACADEMIC_OR_SCIENCE_DOMAINS):
        add_label(labels, "academic_or_scientific_source")
        add_label(labels, "primary_or_expert_source")
        reasons.append(f"academic_science_domain:{domain}")

    if domain_matches(domain, FACT_CHECK_DOMAINS) or url_contains_any(url, FACT_CHECK_PATH_HINTS):
        add_label(labels, "fact_check_article")
        add_label(labels, "secondary_verification_source")
        add_label(labels, "leakage_sensitive_source")
        reasons.append(f"fact_check_domain_or_path:{domain}")

    if domain_matches(domain, NEWS_OR_MEDIA_DOMAINS):
        add_label(labels, "established_secondary_report")
        reasons.append(f"news_media_domain:{domain}")

    if domain_matches(domain, LOW_CREDIBILITY_OR_HIGH_CAUTION_DOMAINS):
        add_label(labels, "low_credibility_watchlist")
        reasons.append(f"low_credibility_domain:{domain}")

    if domain_matches(domain, SOCIAL_DOMAINS):
        add_label(labels, "social_platform_post")
        reasons.append(f"social_domain:{domain}")

    if domain_matches(domain, FORUM_COMMUNITY_DOMAINS):
        add_label(labels, "forum_or_community_discussion")
        reasons.append(f"forum_domain:{domain}")

    if domain_matches(domain, BLOG_CMS_DOMAINS):
        add_label(labels, "self_published_or_blog")
        add_label(labels, "secondary_or_commentary")
        reasons.append(f"blog_cms_domain:{domain}")

    if domain_matches(domain, AGGREGATOR_OR_INDEX_DOMAINS):
        add_label(labels, "aggregator_or_index")
        reasons.append(f"aggregator_domain:{domain}")

    if domain_has_suffix(domain, GOVERNMENT_SUFFIXES):
        add_label(labels, "official_record")
        add_label(labels, "primary_or_authoritative_source")
        reasons.append(f"government_suffix:{domain}")

    if domain_has_suffix(domain, ACADEMIC_SUFFIXES):
        add_label(labels, "academic_or_scientific_source")
        add_label(labels, "primary_or_expert_source")
        reasons.append(f"academic_suffix:{domain}")


def apply_url_and_text_hints(ev: dict[str, Any], labels: list[str], reasons: list[str]) -> None:
    url = get_url(ev)
    original_url = get_original_url(ev)
    url_text = f"{url} {original_url}".lower()
    path = get_path(url)

    if url_contains_any(url_text, FORUM_HINTS):
        add_label(labels, "forum_or_thread_page")
        reasons.append("url_forum_thread_hint")

    if url_contains_any(url_text, BLOG_HINTS):
        add_label(labels, "blog_or_cms_page")
        reasons.append("url_blog_hint")

    if any(path.endswith(ext) or original_url.lower().endswith(ext) for ext in PDF_HINTS):
        add_label(labels, "document_or_pdf")
        reasons.append("url_pdf_hint")

    if url_contains_any(url_text, VIDEO_HINTS):
        add_label(labels, "multimedia_or_video_source")
        reasons.append("url_video_hint")

    if any(original_url.lower().endswith(ext) or url.lower().endswith(ext) for ext in IMAGE_HINTS):
        add_label(labels, "image_or_graphic_source")
        reasons.append("url_image_hint")

    if text_contains_any(ev, ("satire", "satirical", "parody", "fake news", "fabricate", "humor site", "hoax")):
        add_label(labels, "satire_or_humor_source")
        reasons.append("text_satire_or_humor_hint")

    if text_contains_any(ev, ("press release", "official statement", "official release")):
        add_label(labels, "official_statement")
        reasons.append("text_official_statement_hint")

    if text_contains_any(ev, ("dataset", "statistics", "census", "survey", "poll", "report", "study")):
        add_label(labels, "statistical_or_data_reference")
        reasons.append("text_data_reference_hint")

    if text_contains_any(ev, ("transcript", "remarks", "speech", "interview")):
        add_label(labels, "transcript_or_primary_speech_database")
        reasons.append("text_transcript_or_speech_hint")


def provenance_level(score: float) -> str:
    if score >= 0.80:
        return "strong"
    if score >= 0.60:
        return "moderate"
    if score >= 0.40:
        return "weak"
    return "very_weak"


def compute_provenance_score(labels: list[str], trust: float) -> float:
    score = 0.50

    positive_weights = {
        "primary_or_authoritative_source": 0.25,
        "official_record": 0.15,
        "intergovernmental_or_official_org_source": 0.18,
        "academic_or_scientific_source": 0.20,
        "primary_or_expert_source": 0.12,
        "expert_or_institutional_source": 0.10,
        "ngo_thinktank_policy_source": 0.08,
        "established_secondary_report": 0.12,
        "fact_check_article": 0.10,
        "secondary_verification_source": 0.05,
        "reference_encyclopedia": 0.05,
        "statistical_or_data_reference": 0.08,
        "medical_or_health_reference": 0.08,
        "legal_or_civic_reference": 0.08,
        "transcript_or_primary_speech_database": 0.08,
        "official_statement": 0.08,
        "document_or_pdf": 0.03,
        "table_or_structured_data": 0.03,
    }

    negative_weights = {
        "archive_or_copy": -0.04,
        "aggregator_or_index": -0.08,
        "self_published_or_blog": -0.10,
        "blog_or_cms_page": -0.06,
        "forum_or_community_discussion": -0.15,
        "forum_or_thread_page": -0.10,
        "social_platform_post": -0.18,
        "source_type_unknown": -0.06,
        "satire_or_humor_source": -0.30,
        "not_independent_factual_source": -0.15,
        "low_credibility_watchlist": -0.35,
        "cyber_abuse_or_fraud_signal": -0.45,
        "missing_url": -0.25,
        "malformed_or_unparsed_url": -0.25,
        "image_or_graphic_source": -0.04,
        "multimedia_or_video_source": -0.03,
    }

    for label in labels:
        score += positive_weights.get(label, 0.0)
        score += negative_weights.get(label, 0.0)

    score = (0.75 * score) + (0.25 * trust)
    return max(0.0, min(1.0, score))


def compute_f4(ev: dict[str, Any]) -> dict[str, Any]:
    labels: list[str] = []
    reasons: list[str] = []

    url = get_url(ev)
    original_url = get_original_url(ev)
    domain = get_domain(ev)
    original_domain = normalize_domain(ev.get("original_domain") or original_url)
    trust = get_f3_trust_score(ev)

    is_archive_url = bool(ev.get("is_archive_url")) or domain_matches(original_domain, ARCHIVE_DOMAINS)
    archive_url = str(ev.get("archive_url") or "")

    if not url:
        add_label(labels, "missing_url")
        reasons.append("missing_url")
    else:
        add_label(labels, "has_url")

    if url and not domain:
        add_label(labels, "malformed_or_unparsed_url")
        reasons.append("url_present_but_domain_unparsed")

    if is_archive_url or archive_url:
        add_label(labels, "archive_or_copy")
        reasons.append("archive_wrapper_or_archive_url_present")

    if domain_matches(domain, ARCHIVE_DOMAINS):
        add_label(labels, "archive_or_copy")
        reasons.append(f"archive_domain:{domain}")

    apply_source_medium_labels(ev, labels, reasons)
    apply_f2_labels(ev, labels, reasons)
    apply_domain_labels(domain, url, labels, reasons)
    apply_url_and_text_hints(ev, labels, reasons)

    score = compute_provenance_score(labels, trust)

    return {
        "domain": domain,
        "provenance_labels": labels,
        "provenance_score": round(score, 6),
        "provenance_level": provenance_level(score),
        "provenance_reasons": reasons,
        "is_archive_or_copy": "archive_or_copy" in labels,
        "is_fact_check_article": "fact_check_article" in labels,
        "is_primary_or_authoritative": "primary_or_authoritative_source" in labels,
        "is_low_credibility_or_abuse": (
            "low_credibility_watchlist" in labels
            or "cyber_abuse_or_fraud_signal" in labels
        ),
        "is_satire_or_humor": "satire_or_humor_source" in labels,
        "is_reference_or_data": (
            "reference_encyclopedia" in labels
            or "statistical_or_data_reference" in labels
        ),
        "is_multimedia": (
            "multimedia_or_video_source" in labels
            or "image_or_graphic_source" in labels
        ),
    }


def attach_f4(records: list[dict[str, Any]]) -> int:
    row_count = 0

    for claim_rec in records:
        evidence_key = find_evidence_key(claim_rec)
        if evidence_key is None:
            continue

        evidences = claim_rec.get(evidence_key)
        if not isinstance(evidences, list):
            continue

        for ev in evidences:
            if not isinstance(ev, dict):
                continue

            f4 = compute_f4(ev)

            ev["f4_domain"] = f4["domain"]
            ev["f4_provenance_labels"] = f4["provenance_labels"]
            ev["f4_provenance_score"] = f4["provenance_score"]
            ev["f4_provenance_level"] = f4["provenance_level"]
            ev["f4_provenance_reasons"] = f4["provenance_reasons"]
            ev["f4_is_archive_or_copy"] = f4["is_archive_or_copy"]
            ev["f4_is_fact_check_article"] = f4["is_fact_check_article"]
            ev["f4_is_primary_or_authoritative"] = f4["is_primary_or_authoritative"]
            ev["f4_is_low_credibility_or_abuse"] = f4["is_low_credibility_or_abuse"]
            ev["f4_is_satire_or_humor"] = f4["is_satire_or_humor"]
            ev["f4_is_reference_or_data"] = f4["is_reference_or_data"]
            ev["f4_is_multimedia"] = f4["is_multimedia"]

            pafc = ev.get("pafc")
            if not isinstance(pafc, dict):
                pafc = {}

            pafc["f4"] = {
                "domain": f4["domain"],
                "provenance_labels": f4["provenance_labels"],
                "provenance_score": f4["provenance_score"],
                "provenance_level": f4["provenance_level"],
                "provenance_reasons": f4["provenance_reasons"],
                "is_archive_or_copy": f4["is_archive_or_copy"],
                "is_fact_check_article": f4["is_fact_check_article"],
                "is_primary_or_authoritative": f4["is_primary_or_authoritative"],
                "is_low_credibility_or_abuse": f4["is_low_credibility_or_abuse"],
                "is_satire_or_humor": f4["is_satire_or_humor"],
                "is_reference_or_data": f4["is_reference_or_data"],
                "is_multimedia": f4["is_multimedia"],
            }

            ev["pafc"] = pafc
            row_count += 1

    return row_count


def write_audit(output_dir: Path, input_stem: str, records: list[dict[str, Any]]) -> None:
    rows = []

    for claim_rec in records:
        claim_id = claim_rec.get("claim_id", claim_rec.get("id", ""))
        claim = claim_rec.get("claim", "")
        evidence_key = find_evidence_key(claim_rec)

        if evidence_key is None:
            continue

        for ev in claim_rec.get(evidence_key, []):
            if not isinstance(ev, dict):
                continue

            rows.append({
                "claim_id": claim_id,
                "claim": claim,
                "rank": ev.get("rank"),
                "source_doc_idx": ev.get("source_doc_idx"),
                "retrieval_score": ev.get("retrieval_score"),
                "url": ev.get("url"),
                "canonical_source_url": ev.get("canonical_source_url"),
                "domain": ev.get("domain"),
                "f1_relation_label": ev.get("f1_relation_label"),
                "f1_confidence": ev.get("f1_confidence"),
                "f2_source_type": ev.get("f2_source_type"),
                "f2_match_source": ev.get("f2_match_source"),
                "f2_match_confidence": ev.get("f2_match_confidence"),
                "f3_trust_score": ev.get("f3_trust_score"),
                "f3_trust_level": ev.get("f3_trust_level"),
                "f4_domain": ev.get("f4_domain"),
                "f4_provenance_labels": "|".join(ev.get("f4_provenance_labels", [])),
                "f4_provenance_score": ev.get("f4_provenance_score"),
                "f4_provenance_level": ev.get("f4_provenance_level"),
                "f4_is_archive_or_copy": ev.get("f4_is_archive_or_copy"),
                "f4_is_fact_check_article": ev.get("f4_is_fact_check_article"),
                "f4_is_primary_or_authoritative": ev.get("f4_is_primary_or_authoritative"),
                "f4_is_low_credibility_or_abuse": ev.get("f4_is_low_credibility_or_abuse"),
                "f4_is_satire_or_humor": ev.get("f4_is_satire_or_humor"),
                "f4_is_reference_or_data": ev.get("f4_is_reference_or_data"),
                "f4_is_multimedia": ev.get("f4_is_multimedia"),
                "sentence": ev.get("sentence", ""),
            })

    if not rows:
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    audit_path = output_dir / f"{input_stem}_f4_audit.csv"
    with audit_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    label_counter = Counter()
    domain_counter = Counter()
    level_counter = Counter()

    for row in rows:
        level_counter[str(row["f4_provenance_level"])] += 1
        domain_counter[(row["domain"], row["f2_source_type"], row["f4_provenance_level"])] += 1

        for label in str(row["f4_provenance_labels"] or "").split("|"):
            if label:
                label_counter[label] += 1

    label_summary_path = output_dir / f"{input_stem}_f4_label_summary.csv"
    with label_summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "count"])
        writer.writeheader()
        for label, count in label_counter.most_common():
            writer.writerow({"label": label, "count": count})

    domain_summary_path = output_dir / f"{input_stem}_f4_domain_summary.csv"
    with domain_summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["domain", "f2_source_type", "f4_provenance_level", "count"],
        )
        writer.writeheader()
        for (domain, f2_type, level), count in domain_counter.most_common():
            writer.writerow({
                "domain": domain,
                "f2_source_type": f2_type,
                "f4_provenance_level": level,
                "count": count,
            })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="f3_feature_data/retrieval_top_k_with_idx_f1_f2_f3.json",
        help="Input file that already contains F1, F2, and F3 features.",
    )
    parser.add_argument(
        "--output-dir",
        default="f4_feature_data",
        help="Folder where F4 output will be saved.",
    )
    parser.add_argument(
        "--output-name",
        default="retrieval_top_k_with_idx_f1_f2_f3_f4.json",
        help="Output filename.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_path = output_dir / args.output_name

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"Reading F1+F2+F3 file: {input_path}")

    records, input_format = read_json_or_jsonl(input_path)
    if not records:
        raise ValueError(f"No records found in {input_path}")

    evidence_rows = attach_f4(records)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, records)
    write_audit(output_dir, input_path.stem, records)

    level_counts = Counter()
    label_counts = Counter()

    for claim_rec in records:
        evidence_key = find_evidence_key(claim_rec)
        if evidence_key is None:
            continue

        for ev in claim_rec.get(evidence_key, []):
            if not isinstance(ev, dict):
                continue

            level_counts[str(ev.get("f4_provenance_level"))] += 1
            for label in ev.get("f4_provenance_labels", []):
                label_counts[str(label)] += 1

    print("\nDone.")
    print(f"Input format detected: {input_format}")
    print(f"Claim rows: {len(records)}")
    print(f"Evidence rows updated with F4: {evidence_rows}")
    print(f"F4 output file: {output_path}")
    print(f"F4 audit CSV: {output_dir / (input_path.stem + '_f4_audit.csv')}")
    print(f"F4 label summary CSV: {output_dir / (input_path.stem + '_f4_label_summary.csv')}")
    print(f"F4 domain summary CSV: {output_dir / (input_path.stem + '_f4_domain_summary.csv')}")

    print("\nF4 provenance-level counts:")
    for label, count in level_counts.most_common():
        print(f"  {label}: {count}")

    print("\nTop F4 labels:")
    for label, count in label_counts.most_common(30):
        print(f"  {label}: {count}")


if __name__ == "__main__":
    main()
