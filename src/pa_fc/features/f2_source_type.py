from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import pandas as pd

from pa_fc.data.urls import canonicalize_source_url
from pa_fc.utils.logging import get_logger

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


logger = get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_LIST_DIR = REPO_ROOT / "configs" / "source_lists" / "f2"

LONG_TO_SCHEMA = {
    "government_official": "gov",
    "academic_scientific": "academic",
    "fact_checker": "fact_checker",
    "established_news": "est_news",
    "low_credibility_news": "low_credibility_news",
    "social_platform": "social",
    "forum_community": "forum",
    "blog_cms": "blog",
    "aggregator": "aggregator",
    "unknown": "unknown",
    "cyber_abuse_or_fraud_signal": "cyber_abuse_or_fraud_signal",
}

BUCKET_PRIORITY = {
    "fact_checker": 0,
    "gov": 1,
    "academic": 2,
    "blog": 3,
    "forum": 4,
    "social": 5,
    "est_news": 6,
    "aggregator": 7,
    "low_credibility_news": 8,
    "cyber_abuse_or_fraud_signal": 9,
    "unknown": 99,
}

MEDIUM_KEYWORDS = (
    ("social", ("twitter", "x.com", "facebook", "instagram", "tiktok", "youtube", "telegram")),
    ("forum", ("reddit", "forum", "quora", "stackexchange", "stackoverflow", "hacker news")),
    ("blog", ("blog", "wordpress", "blogspot", "medium", "substack", "newsletter")),
)


@dataclass(frozen=True)
class SourceRule:
    pattern: str
    match_type: str
    bucket: str
    confidence: str
    source_file: str
    leakage_sensitive: bool = False


def _normalize_url(url: object) -> tuple[str, str, str]:
    raw = "" if url is None or pd.isna(url) else str(url).strip()
    if not raw:
        return "", "", "/"
    if "://" not in raw:
        raw = "https://" + raw
    parsed = urlparse(raw)
    host = (parsed.netloc or "").lower().split("@")[-1].split(":")[0].strip(".")
    path = parsed.path or "/"
    query = []
    for key, value in parse_qsl(parsed.query, keep_blank_values=True):
        key_l = key.lower()
        if key_l.startswith(("utm_", "fbclid", "gclid", "mc_cid", "mc_eid", "igshid", "ref")):
            continue
        query.append((key, value))
    normalized = urlunparse((parsed.scheme.lower() or "https", host, path, "", urlencode(query), ""))
    return normalized, host.removeprefix("www."), path


def _registered_domain(host: str) -> str:
    host = (host or "").lower().strip(".")
    if not host:
        return ""
    try:
        import tldextract

        extracted = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=False)(host)
        if extracted.domain and extracted.suffix:
            return f"{extracted.domain}.{extracted.suffix}".lower()
    except Exception:
        pass
    parts = host.split(".")
    if len(parts) <= 2:
        return host
    two_part_suffixes = {"co.uk", "ac.uk", "gov.uk", "com.au", "gov.au", "edu.au", "co.in", "gov.in"}
    suffix2 = ".".join(parts[-2:])
    if suffix2 in two_part_suffixes:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])


def _schema_bucket(bucket: str) -> str:
    return LONG_TO_SCHEMA.get((bucket or "unknown").strip(), bucket or "unknown")


def get_source_lists_dir() -> Path:
    override = os.environ.get("PA_FC_F2_SOURCE_LIST_DIR")
    return Path(override).expanduser().resolve() if override else DEFAULT_SOURCE_LIST_DIR


def _load_yaml_rules(source_lists_path: Path | None = None) -> list[SourceRule]:
    source_lists_path = source_lists_path or get_source_lists_dir()
    if yaml is None or not source_lists_path.exists():
        return []
    rules: list[SourceRule] = []
    for path in sorted(source_lists_path.glob("*.yml")):
        if path.name.startswith("matching_config"):
            continue
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception:
            continue
        for raw in data.get("entries", []):
            pattern = str(raw.get("pattern", "")).strip().lower()
            if not pattern:
                continue
            rules.append(
                SourceRule(
                    pattern=pattern,
                    match_type=str(raw.get("match_type", "registered_domain")).strip().lower(),
                    bucket=_schema_bucket(str(raw.get("bucket", "unknown")).strip()),
                    confidence=str(raw.get("confidence", "medium")).strip(),
                    source_file=path.name,
                    leakage_sensitive=bool(raw.get("leakage_sensitive", False)),
                )
            )
    return sorted(rules, key=lambda r: (_match_priority(r.match_type), BUCKET_PRIORITY.get(r.bucket, 50), -len(r.pattern)))


def source_list_runtime_info() -> dict[str, object]:
    path = get_source_lists_dir()
    rules = _load_yaml_rules(path)
    return {
        "source_list_dir": str(path),
        "source_list_rule_count": len(rules),
        "source_list_files": len(list(path.glob("*.yml"))) if path.exists() else 0,
        "using_fallback_rules": not bool(rules),
    }


def _fallback_rules() -> list[SourceRule]:
    return [
        SourceRule(".gov", "suffix", "gov", "high", "builtin"),
        SourceRule(".mil", "suffix", "gov", "high", "builtin"),
        SourceRule(".edu", "suffix", "academic", "high", "builtin"),
        SourceRule("doi.org", "registered_domain", "academic", "high", "builtin"),
        SourceRule("arxiv.org", "registered_domain", "academic", "high", "builtin"),
        SourceRule("reuters.com/fact-check", "path_prefix", "fact_checker", "high", "builtin", True),
        SourceRule("apnews.com/hub/ap-fact-check", "path_prefix", "fact_checker", "high", "builtin", True),
    ]


def _match_priority(match_type: str) -> int:
    return {"path_prefix": 0, "registered_domain": 1, "suffix": 2, "fallback": 9}.get(match_type, 8)


def _matches(rule: SourceRule, normalized_url: str, host: str, registered_domain: str, path: str) -> bool:
    pattern = rule.pattern.lower().strip()
    if rule.match_type == "path_prefix":
        key_host = f"{host}{path}".lower().lstrip("/")
        key_reg = f"{registered_domain}{path}".lower().lstrip("/")
        return key_host.startswith(pattern.lstrip("/")) or key_reg.startswith(pattern.lstrip("/"))
    if rule.match_type == "registered_domain":
        pat = pattern.strip(".")
        return bool(host == pat or host.endswith("." + pat) or registered_domain == pat)
    if rule.match_type == "suffix":
        pat = pattern if pattern.startswith(".") else "." + pattern
        return host.endswith(pat) or registered_domain.endswith(pat)
    if rule.match_type == "fallback":
        return True
    return bool(pattern and pattern in normalized_url.lower())


def _medium_bucket(source_medium: object) -> Optional[str]:
    text = "" if source_medium is None or pd.isna(source_medium) else str(source_medium).lower()
    for bucket, keywords in MEDIUM_KEYWORDS:
        if any(keyword in text for keyword in keywords):
            return bucket
    return None


def _classify_row(row: object, rules: Iterable[SourceRule]) -> tuple[str, str, str, bool]:
    raw_type = getattr(row, "source_type_raw", "unknown")
    raw_type = "unknown" if raw_type is None or pd.isna(raw_type) else str(raw_type).strip()
    source_url = getattr(row, "canonical_source_url", "") or getattr(row, "source_url", "")
    canonical_url, _, _ = canonicalize_source_url(source_url)
    normalized_url, host, path = _normalize_url(canonical_url)
    registered_domain = (getattr(row, "registered_domain", "") or "").strip().lower()
    if not registered_domain:
        registered_domain = _registered_domain(host)

    fallback_rules = []
    for rule in rules:
        if rule.match_type == "fallback":
            fallback_rules.append(rule)
            continue
        if _matches(rule, normalized_url, host, registered_domain, path):
            return rule.bucket, rule.source_file, rule.confidence, rule.leakage_sensitive

    medium_bucket = _medium_bucket(getattr(row, "source_medium_raw", ""))
    if medium_bucket:
        return medium_bucket, "source_medium", "medium", False

    if raw_type in BUCKET_PRIORITY and raw_type != "unknown":
        return raw_type, "source_type_raw", "low", False

    for rule in fallback_rules:
        if _matches(rule, normalized_url, host, registered_domain, path):
            return rule.bucket, rule.source_file, rule.confidence, rule.leakage_sensitive

    return raw_type if raw_type in BUCKET_PRIORITY else "unknown", "source_type_raw", "low", False


def compute_f2(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    runtime = source_list_runtime_info()
    rules = _load_yaml_rules(get_source_lists_dir()) or _fallback_rules()
    logger.info(
        "F2 source lists: path=%s rules=%s fallback=%s",
        runtime["source_list_dir"],
        runtime["source_list_rule_count"],
        runtime["using_fallback_rules"],
    )
    classified = [_classify_row(row, rules) for row in df.itertuples(index=False)]
    df["f2_source_type"] = [item[0] for item in classified]
    df["f2_match_source"] = [item[1] for item in classified]
    df["f2_match_confidence"] = [item[2] for item in classified]
    df["f2_leakage_sensitive"] = pd.Series([item[3] for item in classified], index=df.index, dtype=bool)
    return df
