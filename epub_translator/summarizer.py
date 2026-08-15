"""Pre-analysis phase: chapter summaries, book summary, consolidated glossary.

Runs on the cheaper analysis model before translation starts.
"""

from typing import Dict, List

from . import extractor as extractor_mod
from .glossary import group_candidates, normalize_category, normalize_term_key
from .llm_client import extract_json_block
from .prompts import (
    build_book_summary_prompt,
    build_chapter_summary_prompt,
    build_glossary_consolidation_prompt,
)


class BookAnalyzer:
    """Generates chapter summaries, a book summary and a unified glossary."""

    def __init__(self, llm, config: Dict):
        self.llm = llm
        pre = config.get("pre_analysis", {})
        self.sample_chars = pre.get("chapter_sample_chars", 4000)
        self.batch_size = pre.get("chapters_per_summary_batch", 4)
        self.max_terms_total = pre.get("glossary_max_terms_total", 500)
        self.temperature = pre.get("temperature", 0.2)
        self.max_tokens = pre.get("max_tokens", 4096)

    def _chat(self, system: str, user: str):
        return self.llm.chat(
            system, user, analysis=True, temperature=self.temperature, max_tokens=self.max_tokens
        )

    def analyze(self, chapters, extractor) -> Dict:
        """Return {"book_summary": {...}, "chapter_summaries": {...}, "glossary_terms": {...}}."""
        # 1. Sample each chapter (head/middle/tail) for analysis
        infos = []
        for chapter in chapters:
            try:
                html = chapter.path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            text = extractor_mod.html_body_text(html)
            if len(text) < 80:
                continue
            infos.append((chapter.path.name, extractor_mod.sample_text(text, self.sample_chars)))

        # 2. Batched chapter summaries + candidate terms
        chapter_summaries: Dict[str, str] = {}
        candidates: List[dict] = []
        batch_count = (len(infos) + self.batch_size - 1) // self.batch_size
        for batch_index in range(batch_count):
            batch = infos[batch_index * self.batch_size:(batch_index + 1) * self.batch_size]
            print(f"  [{batch_index + 1}/{batch_count}] 生成章节摘要（{len(batch)} 章）...")
            system, user = build_chapter_summary_prompt(batch)
            data = extract_json_block(self._chat(system, user))
            valid_names = {name for name, _ in batch}
            for item in data.get("chapters", []):
                name = item.get("chapter")
                if name not in valid_names:
                    continue
                chapter_summaries[name] = str(item.get("summary", "")).strip()
                for term in item.get("terms", []):
                    if term.get("en"):
                        candidates.append(
                            {
                                "en": str(term["en"]).strip(),
                                "zh": str(term.get("zh", "")).strip(),
                                "category": normalize_category(term.get("category")),
                                "frequency": max(1, int(term.get("frequency") or 1)),
                            }
                        )

        # 3. Book summary aggregated from chapter summaries
        meta = extractor.metadata
        book_summary: Dict = {}
        if chapter_summaries:
            print("  生成全书概要...")
            system, user = build_book_summary_prompt(
                chapter_summaries, meta.get("title", ""), meta.get("creator", "")
            )
            try:
                book_summary = extract_json_block(self._chat(system, user))
            except Exception as exc:
                print(f"  ⚠ 全书概要生成失败：{exc}")

        # 4. Consolidate candidates into the final glossary
        glossary_terms: Dict[str, dict] = {}
        if candidates:
            print(f"  整合术语表（{len(candidates)} 个候选）...")
            grouped = group_candidates(candidates)
            system, user = build_glossary_consolidation_prompt(grouped, self.max_terms_total)
            try:
                data = extract_json_block(self._chat(system, user))
                for term in data.get("terms", []):
                    en = str(term.get("en", "")).strip()
                    if not en:
                        continue
                    key = normalize_term_key(en)
                    glossary_terms[key] = {
                        "en": en,
                        "zh": str(term.get("zh", "")).strip(),
                        "category": normalize_category(term.get("category")),
                        "frequency": max(1, int(term.get("frequency") or 1)),
                    }
            except Exception as exc:
                print(f"  ⚠ 术语表整合失败（将直接使用去重后的候选词）：{exc}")
                for group in grouped:
                    key = normalize_term_key(group["en"])
                    glossary_terms[key] = {
                        "en": group["en"],
                        "zh": group.get("zh", ""),
                        "category": normalize_category(group.get("category")),
                        "frequency": group["frequency"],
                    }

        return {
            "book_summary": book_summary,
            "chapter_summaries": chapter_summaries,
            "glossary_terms": glossary_terms,
        }
