from functools import lru_cache
import re
from urllib.parse import unquote, urlparse

import tldextract

from pa_fc.data.schema import IFCN_DOMAINS


MEDIUM_TO_TYPE = {
    "Twitter": "social",
    "Facebook": "social",
    "Reddit": "forum",
    "YouTube": "social",
    "TV": "est_news",
    "Radio": "est_news",
    "PDF": "unknown",
    "Image": "unknown",
    "Other": "unknown",
}

GOV_DOMAINS = {".gov", ".mil", ".gc.ca", ".gov.uk", ".gov.au", ".europa.eu"}
ACADEMIC_DOMAINS = set()
ESTABLISHED_NEWS_DOMAINS = {
    "reuters.com",
    "apnews.com",
    "bbc.com",
    "bbc.co.uk",
    "nytimes.com",
    "washingtonpost.com",
    "theguardian.com",
    "wsj.com",
    "bloomberg.com",
    "nbcnews.com",
    "cbsnews.com",
    "abcnews.go.com",
    "cnn.com",
    "foxnews.com",
    "npr.org",
    "theverge.com",
    "axios.com",
    "theatlantic.com",
    "time.com",
    "usatoday.com",
    "latimes.com",
    "chicagotribune.com",
    "sfchronicle.com",
    "bostonglobe.com",
    "politico.com",
    "thehill.com",
    "rollcall.com",
    "propublica.org",
    "vox.com",
    "slate.com",
    "salon.com",
}
SOCIAL_DOMAINS = {
    "twitter.com",
    "x.com",
    "facebook.com",
    "instagram.com",
    "tiktok.com",
    "youtube.com",
    "t.co",
}
FORUM_DOMAINS = {
    "reddit.com",
    "quora.com",
    "stackexchange.com",
    "stackoverflow.com",
}
BLOG_SUFFIXES = {
    "wordpress.com",
    "blogspot.com",
    "medium.com",
    "substack.com",
    "tumblr.com",
    "ghost.io",
    "weebly.com",
    "wix.com",
}

ARCHIVE_WRAPPER_HOSTS = {
    "web.archive.org",
    "archive.is",
    "archive.ph",
    "archive.today",
    "archive.vn",
    "perma.cc",
}


@lru_cache(maxsize=1)
def _extractor() -> tldextract.TLDExtract:
    return tldextract.TLDExtract(suffix_list_urls=(), cache_dir=False)


def canonicalize_source_url(source_url: str) -> tuple[str, str, bool]:
    """
    Return (canonical_source_url, archive_url, is_archive_copy).

    Wayback URLs expose their original target and are unwrapped. Other archive
    wrappers remain canonicalized to themselves because the target is not
    reliably encoded in the URL.
    """
    raw = (source_url or "").strip()
    if not raw:
        return "", "", False

    parsed = urlparse(raw if "://" in raw else f"https://{raw}")
    host = parsed.netloc.lower().removeprefix("www.")
    if host == "web.archive.org":
        match = re.match(r"^/web/(?:\d{1,14}[a-z_]*|save)/(?:id_/)?(.+)$", parsed.path)
        if match:
            target = unquote(match.group(1)).strip()
            if target:
                return target, raw, True
        return raw, raw, True

    if host in ARCHIVE_WRAPPER_HOSTS:
        return raw, raw, True
    return raw, "", False


def registered_domain(source_url: str) -> str:
    """
    Return the registered domain for a URL, or "" for missing/malformed values.
    Uses tldextract's bundled suffix snapshot and does not fetch suffix lists.
    """
    canonical_url, _, _ = canonicalize_source_url(source_url)
    if not canonical_url:
        return ""

    try:
        extracted = _extractor()(canonical_url)
    except Exception:
        return ""

    return extracted.registered_domain or ""


def extract_registered_domain(source_url: str) -> str:
    """Backward-compatible alias for code that uses the spec pseudocode name."""
    return registered_domain(source_url)


def normalize_source_type(
    source_url: str, source_medium_raw: str, registered_domain: str
) -> str:
    canonical_url, _, _ = canonicalize_source_url(source_url)
    medium = (source_medium_raw or "").strip()
    if medium in MEDIUM_TO_TYPE and MEDIUM_TO_TYPE[medium] != "unknown":
        return MEDIUM_TO_TYPE[medium]

    domain = (registered_domain or "").strip().lower()
    if not domain:
        return "unknown"

    try:
        extracted = _extractor()(canonical_url or domain)
    except Exception:
        extracted = None

    suffix = f".{extracted.suffix.lower()}" if extracted and extracted.suffix else ""
    fqdn = ".".join(part for part in [getattr(extracted, "subdomain", ""), domain] if part)
    domain_candidates = {domain}
    if fqdn:
        domain_candidates.add(fqdn.lower())

    if (
        suffix in {".gov", ".mil", ".edu"}
        or any(fqdn.endswith(pattern.lstrip(".")) for pattern in GOV_DOMAINS)
        or any(domain.endswith(pattern.lstrip(".")) for pattern in GOV_DOMAINS)
    ):
        return "academic" if suffix == ".edu" else "gov"

    if domain in ACADEMIC_DOMAINS:
        return "academic"

    if domain_candidates & IFCN_DOMAINS or _matches_path_domain(canonical_url, IFCN_DOMAINS):
        return "fact_checker"

    if domain_candidates & ESTABLISHED_NEWS_DOMAINS:
        return "est_news"

    if domain_candidates & SOCIAL_DOMAINS:
        return "social"

    if domain_candidates & FORUM_DOMAINS:
        return "forum"

    if any(domain == suffix_domain or domain.endswith(f".{suffix_domain}") for suffix_domain in BLOG_SUFFIXES):
        return "blog"

    if MEDIUM_TO_TYPE.get(medium) == "unknown":
        return "unknown"

    return "unknown"


def _matches_path_domain(source_url: str, patterns: set[str]) -> bool:
    if not source_url:
        return False

    parsed = urlparse(source_url if "://" in source_url else f"https://{source_url}")
    host = (parsed.netloc or "").lower()
    path = (parsed.path or "").rstrip("/").lower()
    host_path = f"{host}{path}"

    return any("/" in pattern and host_path.startswith(pattern.lower()) for pattern in patterns)
