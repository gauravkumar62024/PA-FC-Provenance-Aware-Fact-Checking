from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Optional

import pandas as pd

from pa_fc.utils.logging import get_logger


logger = get_logger(__name__)

REFERENCE_DIR = Path("data/reference")
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRUST_DIR = REPO_ROOT / "configs" / "source_lists" / "f3"
IFFY_PATH = REFERENCE_DIR / "iffy_index.json"
NELA_PATH = REFERENCE_DIR / "nela_gt_labels.csv"
IFFY_URLS = [
    "https://opensheet.elk.sh/1ck1_FZC-97uDLIlvRJDTrGqBk0FuDe9yHkluROgpGS8/Iffy-news",
    "https://iffy.news/iffy-index.json",
]
NELA_DATASET_API = (
    "https://dataverse.harvard.edu/api/datasets/:persistentId"
    "?persistentId=doi:10.7910/DVN/AMCV2H"
)
NELA_DATAONE_METADATA_URL = (
    "https://cn.dataone.org/cn/v2/object/"
    "sha256:ec4ffab73bef75702948ba8cb005437f235733e2cc210674ab13d7099fd28d76"
)
NELA_SCORE_COLUMNS = [
    "mbfc.factual_reporting",
    "mbfc_factual_reporting",
    "factual_reporting",
    "factuality",
    "mbfc_factuality",
]

MBFC_TO_TRUST = {
    5: 1.00,
    4: 0.85,
    3: 0.70,
    2: 0.40,
    1: 0.15,
    0: 0.05,
}

SPECIAL_TRUST = {
    "gov": 0.95,
    "academic": 0.90,
    "fact_checker": 0.80,
    "est_news": 0.75,
    "think_tank": 0.65,
    "blog": 0.45,
    "social": 0.25,
    "forum": 0.30,
    "aggregator": 0.40,
    "low_credibility_news": 0.15,
    "cyber_abuse_or_fraud_signal": 0.10,
    "iffy_listed": 0.10,
    "unknown": 0.50,
}

LONG_TO_SCHEMA = {
    "government_official": "gov",
    "academic_scientific": "academic",
    "established_news": "est_news",
    "blog_cms": "blog",
    "low_credibility_news": "low_credibility_news",
    "social_platform": "social",
    "forum_community": "forum",
    "aggregator": "aggregator",
    "cyber_abuse_or_fraud_signal": "cyber_abuse_or_fraud_signal",
}


def _download_if_missing(urls: list[str], path: Path) -> None:
    if path.exists():
        return
    if os.environ.get("PA_FC_DOWNLOAD_REFERENCES") != "1":
        logger.info("Reference download skipped for %s; set PA_FC_DOWNLOAD_REFERENCES=1 to enable.", path)
        return
    try:
        import requests
    except Exception as exc:
        logger.warning("requests unavailable; cannot download %s: %s", path, exc)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    for url in urls:
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            path.write_bytes(response.content)
            logger.info("Downloaded reference data from %s to %s", url, path)
            return
        except Exception as exc:
            logger.warning("Could not download %s: %s", url, exc)


def load_iffy_domains(reference_dir: Path = REFERENCE_DIR) -> set[str]:
    path = reference_dir / IFFY_PATH.name
    _download_if_missing(IFFY_URLS, path)
    if not path.exists():
        return set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        domains = set()
        for item in data:
            domain = (
                item.get("domain")
                or item.get("Domain")
                or item.get("url")
                or item.get("URL")
                or item.get("Site")
                or ""
            ).strip().lower()
            if domain:
                domains.add(domain.removeprefix("www."))
        return domains
    except Exception as exc:
        logger.warning("Could not load Iffy Index from %s: %s", path, exc)
        return set()


def _coerce_mbfc_score(value: object) -> Optional[int]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    label_map = {
        "very high": 5,
        "high": 4,
        "mostly factual": 3,
        "mixed": 2,
        "low": 1,
        "very low": 0,
    }
    lowered = text.lower()
    if lowered in label_map:
        return label_map[lowered]
    try:
        score = int(float(text))
    except ValueError:
        return None
    return score if score in MBFC_TO_TRUST else None


def load_nela_lookup(reference_dir: Path = REFERENCE_DIR) -> dict[str, int]:
    path = reference_dir / NELA_PATH.name
    _download_nela_if_missing(path)
    if not path.exists():
        logger.warning(
            "NELA labels not found at %s; F3 will use source-type priors where needed.",
            path,
        )
        return {}
    lookup: dict[str, int] = {}
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            sample = handle.read(4096)
            handle.seek(0)
            delimiter = "\t" if sample.count("\t") > sample.count(",") else ","
            reader = csv.DictReader(handle, delimiter=delimiter)
            for row in reader:
                domain = (
                    row.get("url")
                    or row.get("source_url")
                    or row.get("domain")
                    or row.get("source")
                    or ""
                ).strip().lower()
                domain = domain.removeprefix("http://").removeprefix("https://")
                domain = domain.removeprefix("www.").split("/")[0]
                score = None
                for score_col in NELA_SCORE_COLUMNS:
                    score = _coerce_mbfc_score(row.get(score_col))
                    if score is not None:
                        break
                if domain and score is not None:
                    lookup[domain] = score
    except Exception as exc:
        logger.warning("Could not load NELA labels from %s: %s", path, exc)
    return lookup


def _download_nela_if_missing(path: Path) -> None:
    if path.exists():
        return
    if os.environ.get("PA_FC_DOWNLOAD_REFERENCES") != "1":
        logger.info("NELA download skipped; set PA_FC_DOWNLOAD_REFERENCES=1 to enable.")
        return
    try:
        import requests

        response = requests.get(NELA_DATASET_API, timeout=30)
        response.raise_for_status()
        dataset_data = response.json().get("data", {})
        version = dataset_data.get("latestVersion") or dataset_data.get("draftVersion")
        if version is None:
            dataset_id = dataset_data.get("id")
            logger.warning(
                "NELA Dataverse metadata for dataset id %s does not expose a public latestVersion file list.",
                dataset_id,
            )
            _download_nela_via_dataone_metadata(path)
            return
        files = version.get("files", [])
        labels_file = None
        for item in files:
            data_file = item.get("dataFile", {})
            filename = data_file.get("filename", "")
            directory = item.get("directoryLabel", "")
            candidate = f"{directory}/{filename}".lower()
            if filename.lower() == "labels.csv" or candidate.endswith("/labels.csv"):
                labels_file = data_file
                break
        if not labels_file:
            logger.warning("Could not find labels.csv in NELA Dataverse metadata.")
            return
        file_id = labels_file["id"]
        file_url = f"https://dataverse.harvard.edu/api/access/datafile/{file_id}"
        _download_nela_file(file_url, path)
    except Exception as exc:
        logger.warning("Could not download NELA labels via Dataverse API: %s", exc)
        _download_nela_via_dataone_metadata(path)


def _download_nela_via_dataone_metadata(path: Path) -> None:
    if path.exists():
        return
    try:
        import requests

        response = requests.get(NELA_DATAONE_METADATA_URL, timeout=30)
        response.raise_for_status()
        distributions = response.json().get("distribution", [])
        labels_url = None
        for item in distributions:
            name = (item.get("name") or "").lower()
            if name in {"labels_all_2022.tab", "labels.csv"}:
                labels_url = item.get("contentUrl")
                break
        if not labels_url:
            logger.warning("Could not find NELA labels distribution in DataONE metadata.")
            return
        _download_nela_file(labels_url, path)
    except Exception as exc:
        logger.warning(
            "Could not download NELA labels via DataONE metadata: %s. "
            "Place labels_all_2022.tab or labels.csv at %s to enable NELA trust scores.",
            exc,
            path,
        )


def _download_nela_file(url: str, path: Path) -> None:
    import requests

    response = requests.get(url, timeout=60)
    try:
        response.raise_for_status()
    except Exception as exc:
        message = response.text[:300].replace("\n", " ")
        raise RuntimeError(f"{url} returned {response.status_code}: {message}") from exc

    text = response.content.decode("utf-8", errors="replace")
    header = text.splitlines()[0].lower() if text.splitlines() else ""
    if not any(column in header for column in NELA_SCORE_COLUMNS):
        raise RuntimeError(
            f"{url} did not look like NELA labels; first row was {header[:200]!r}"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    logger.info("Downloaded NELA labels to %s", path)


def get_trust_score(
    registered_domain: str,
    source_type: str,
    iffy_domains: set[str] | None = None,
    nela_lookup: dict[str, int] | None = None,
    curated_lookup: dict[str, tuple[float, str]] | None = None,
) -> tuple[float, str]:
    domain = (registered_domain or "").strip().lower()
    source = (source_type or "").strip()
    iffy_domains = iffy_domains or set()
    nela_lookup = nela_lookup or {}
    curated_lookup = curated_lookup or {}

    if not domain:
        return 0.50, "no_url"
    if domain in iffy_domains:
        return 0.10, "iffy"
    curated = _lookup_curated_prior(domain, curated_lookup)
    if curated is not None:
        return curated
    if source == "gov":
        return 0.95, "gov"
    if source == "academic":
        return 0.90, "edu"
    if source == "fact_checker":
        return 0.80, "ifcn"
    if source == "social":
        return 0.25, "social_prior"
    if source == "forum":
        return 0.30, "forum_prior"
    if source == "blog":
        return 0.45, "blog_prior"
    if source == "est_news":
        return 0.75, "established_news_prior"
    if source == "think_tank":
        return 0.65, "think_tank_prior"
    if source == "aggregator":
        return 0.40, "aggregator_prior"
    if source == "low_credibility_news":
        return 0.15, "low_credibility_prior"
    if source == "cyber_abuse_or_fraud_signal":
        return 0.10, "cyber_abuse_prior"
    nela_score = nela_lookup.get(domain)
    if nela_score is not None:
        return MBFC_TO_TRUST[nela_score], "nela_gt"
    return 0.50, "unknown_prior"


def load_curated_domain_priors(
    trust_dir: Path | None = None,
) -> dict[str, tuple[float, str]]:
    trust_dir = trust_dir or Path(
        os.environ.get("PA_FC_F3_TRUST_LIST_DIR", DEFAULT_TRUST_DIR)
    ).expanduser().resolve()
    path = trust_dir / "f3_domain_trust_priors.yml"
    if not path.exists():
        return {}
    try:
        import yaml
    except Exception:
        logger.warning("PyYAML unavailable; curated F3 trust priors are skipped.")
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        logger.warning("Could not load curated F3 trust priors from %s: %s", path, exc)
        return {}
    lookup: dict[str, tuple[float, str]] = {}
    for raw in data.get("entries", []):
        pattern = str(raw.get("pattern", "")).strip().lower().removeprefix("www.")
        if not pattern:
            continue
        bucket = LONG_TO_SCHEMA.get(str(raw.get("f2_bucket", raw.get("bucket", ""))).strip(), "")
        score = max(0.0, min(1.0, float(raw.get("trust_prior", SPECIAL_TRUST.get(bucket, 0.50)))))
        match_type = str(raw.get("match_type", "registered_domain")).strip()
        lookup[pattern] = (score, f"curated_{match_type}")
    return lookup


def _lookup_curated_prior(
    domain: str,
    curated_lookup: dict[str, tuple[float, str]],
) -> tuple[float, str] | None:
    if not domain:
        return None
    if domain in curated_lookup:
        return curated_lookup[domain]
    parts = domain.split(".")
    for idx in range(1, max(len(parts) - 1, 1)):
        candidate = ".".join(parts[idx:])
        if candidate in curated_lookup:
            return curated_lookup[candidate]
    return None


def compute_f3(df: pd.DataFrame, reference_dir: Path = REFERENCE_DIR) -> pd.DataFrame:
    df = df.copy()
    iffy_domains = load_iffy_domains(reference_dir)
    nela_lookup = load_nela_lookup(reference_dir)
    curated_lookup = load_curated_domain_priors()
    scores: list[float] = []
    sources: list[str] = []
    for row in df.itertuples(index=False):
        score, source = get_trust_score(
            getattr(row, "registered_domain", ""),
            getattr(row, "f2_source_type", getattr(row, "source_type_raw", "")),
            iffy_domains=iffy_domains,
            nela_lookup=nela_lookup,
            curated_lookup=curated_lookup,
        )
        scores.append(score)
        sources.append(source)
    df["f3_trust_score"] = scores
    df["f3_trust_source"] = sources
    return df
