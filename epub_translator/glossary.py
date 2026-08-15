"""Term glossary: normalization, fuzzy dedup, thread-safe store, prompt blocks."""

import difflib
import json
import os
import re
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

CATEGORY_ZH = {
    "person": "人名",
    "place": "地名",
    "organization": "机构",
    "term": "术语",
    "other": "其他",
}


def normalize_term_key(en: str) -> str:
    """Normalize an English term for dedup keys."""
    s = re.sub(r"\s+", " ", (en or "").strip()).strip("'\"“”‘’.,;:!?()[]{}")
    return s.lower()


def normalize_category(category: Optional[str]) -> str:
    cat = str(category or "other").lower()
    return cat if cat in CATEGORY_ZH else "other"


def _similar(a: str, b: str) -> bool:
    if a == b:
        return True
    if min(len(a), len(b)) < 4:
        return False
    return difflib.SequenceMatcher(None, a, b).ratio() >= 0.86


def _pick_canonical(variants: List[str], key: str) -> str:
    """Prefer a variant that matches the normalized key, else the first one."""
    for variant in variants:
        if normalize_term_key(variant) == key:
            return variant
    return variants[0]


def group_candidates(candidates: List[dict]) -> List[dict]:
    """Merge raw candidate terms: exact (case-insensitive) + fuzzy dedup, frequency sums.

    Returns a list of grouped dicts: {"key", "en", "variants", "zh", "category", "frequency"}.
    """
    buckets: Dict[str, dict] = {}
    for candidate in candidates:
        en = str(candidate.get("en", "")).strip()
        if not en:
            continue
        key = normalize_term_key(en)
        if key not in buckets:
            buckets[key] = {
                "key": key,
                "variants": [en],
                "zh": str(candidate.get("zh", "")).strip(),
                "category": normalize_category(candidate.get("category")),
                "frequency": max(1, int(candidate.get("frequency") or 1)),
            }
        else:
            group = buckets[key]
            if en not in group["variants"]:
                group["variants"].append(en)
            if not group["zh"] and candidate.get("zh"):
                group["zh"] = str(candidate["zh"]).strip()
            if group["category"] == "other":
                group["category"] = normalize_category(candidate.get("category"))
            group["frequency"] += max(1, int(candidate.get("frequency") or 1))

    groups = list(buckets.values())
    merged: List[dict] = []
    used = set()
    for i, group in enumerate(groups):
        if id(group) in used:
            continue
        for other in groups[i + 1:]:
            if id(other) in used:
                continue
            if _similar(group["key"], other["key"]):
                group["variants"].extend(v for v in other["variants"] if v not in group["variants"])
                if not group["zh"] and other["zh"]:
                    group["zh"] = other["zh"]
                if group["category"] == "other" and other["category"] != "other":
                    group["category"] = other["category"]
                group["frequency"] += other["frequency"]
                used.add(id(other))
        used.add(id(group))
        merged.append(group)

    for group in merged:
        group["en"] = _pick_canonical(group["variants"], group["key"])
        group.pop("key", None)
    merged.sort(key=lambda g: -g["frequency"])
    return merged


class Glossary:
    """Thread-safe term store persisted as JSON.

    Structure: {"terms": {normalized_key: {"en", "zh", "category", "frequency"}}}.
    """

    def __init__(self, json_path: Optional[Path] = None):
        self.path = Path(json_path) if json_path else None
        self.terms: Dict[str, dict] = {}
        self._lock = threading.Lock()
        self.load()

    def __len__(self) -> int:
        return len(self.terms)

    def load(self):
        if not self.path or not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.terms = data.get("terms", {})
        except Exception:
            self.terms = {}

    def save(self):
        if not self.path:
            return
        with self._lock:
            payload = {"terms": self.terms}
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp, self.path)

    def replace_terms(self, terms: Dict[str, dict]):
        """Replace the whole glossary (used after pre-analysis consolidation)."""
        with self._lock:
            cleaned: Dict[str, dict] = {}
            for key, entry in (terms or {}).items():
                normalized = normalize_term_key(key)
                cleaned[normalized] = {
                    "en": str(entry.get("en") or key).strip(),
                    "zh": str(entry.get("zh", "")).strip(),
                    "category": normalize_category(entry.get("category")),
                    "frequency": max(1, int(entry.get("frequency") or 1)),
                }
            self.terms = cleaned

    def add_terms(self, terms: List[dict]) -> Tuple[int, int]:
        """Merge new terms in. Returns (added, merged) counts."""
        added = 0
        merged = 0
        with self._lock:
            for term in terms:
                en = str(term.get("en", "")).strip()
                if not en:
                    continue
                key = normalize_term_key(en)
                if key in self.terms:
                    entry = self.terms[key]
                    entry["frequency"] += max(1, int(term.get("frequency") or 1))
                    if not entry.get("zh") and term.get("zh"):
                        entry["zh"] = str(term["zh"]).strip()
                    if entry.get("category") in (None, "other") and normalize_category(term.get("category")) != "other":
                        entry["category"] = normalize_category(term.get("category"))
                    merged += 1
                else:
                    self.terms[key] = {
                        "en": en,
                        "zh": str(term.get("zh", "")).strip(),
                        "category": normalize_category(term.get("category")),
                        "frequency": max(1, int(term.get("frequency") or 1)),
                    }
                    added += 1
        return added, merged

    def keys(self) -> List[str]:
        with self._lock:
            return list(self.terms.keys())

    def to_prompt_block(self, max_terms: int = 150) -> str:
        """Render the top-frequency terms as a prompt block."""
        with self._lock:
            items = sorted(self.terms.items(), key=lambda kv: -kv[1].get("frequency", 0))[:max_terms]
            lines = []
            for key, entry in items:
                en = entry.get("en") or key
                zh = entry.get("zh") or "（待定）"
                category = CATEGORY_ZH.get(entry.get("category", "other"), "其他")
                lines.append(f"- {en} → {zh}（{category}）")
        return "\n".join(lines) if lines else "（暂无）"
