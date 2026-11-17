from dataclasses import dataclass
from datetime import date
from typing import List, Optional


VALID_LABELS = {
    "Supported",
    "Refuted",
    "Not Enough Evidence",
    "Conflicting Evidence/Cherry-picking",
}

LABEL_ALIASES = {
    "Conflicting Evidence/Cherrypicking": "Conflicting Evidence/Cherry-picking",
}

VALID_ANSWER_TYPES = {
    "abstractive",
    "extractive",
    "boolean",
    "unanswerable",
}

VALID_SOURCE_TYPES = {
    "gov",
    "academic",
    "est_news",
    "fact_checker",
    "think_tank",
    "social",
    "blog",
    "forum",
    "aggregator",
    "low_credibility_news",
    "cyber_abuse_or_fraud_signal",
    "unknown",
}

IFCN_DOMAINS = {
    "politifact.com",
    "factcheck.org",
    "snopes.com",
    "fullfact.org",
    "africacheck.org",
    "chequeado.com",
    "boomlive.in",
    "factly.in",
    "altnews.in",
    "factcheckni.org",
    "vishvasnews.com",
    "factcrescendo.com",
    "factcheck.afp.com",
    "reuters.com/fact-check",
    "apnews.com/hub/ap-fact-check",
    "usafacts.org",
    "leadstories.com",
    "pesacheck.org",
    "mafindo.or.id",
    "dubawa.org",
    "correctiv.org",
    "dpa-factchecking.com",
    "mimikama.at",
    "demagog.org.pl",
}


@dataclass
class EvidenceRecord:
    # --- Identity ---
    evidence_id: str
    claim_id: int
    split: str

    # --- Claim fields ---
    claim_text: str
    claim_date_raw: str
    claim_date: Optional[date]
    gold_label: Optional[str]
    speaker: Optional[str]
    reporting_source: Optional[str]
    location_iso: Optional[str]
    original_claim_url: Optional[str]
    fact_checking_article: Optional[str]
    claim_types: List[str]
    fact_checking_strategies: List[str]

    # --- Question/Answer fields ---
    question_idx: int
    question_text: str
    answer_idx: int
    answer_text: str
    answer_type: Optional[str]

    # --- Source fields (raw) ---
    source_url: str
    cached_source_url: str
    source_medium_raw: str

    # --- Derived fields ---
    registered_domain: str
    source_type_raw: str
    canonical_source_url: str
    archive_url: str
    is_archive_copy: bool
