"""Translation pipeline: parallel chapters, sequential chunks, glossary injection.

- Chapters are translated in parallel (shared, thread-safe glossary).
- Chunks inside a chapter are translated strictly in order, carrying the
  previous chunk's tail as style continuity context.
- After a chapter finishes, new terms are extracted and merged into the
  glossary (incremental update, visible to subsequently started chunks).
- Checkpointing is per (file, chunk); translated fragments are stored on disk
  so a crash never loses paid-for output.
"""

import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

from . import extractor as extractor_mod
from .glossary import Glossary
from .llm_client import extract_json_block
from .prompts import (
    build_new_terms_prompt,
    build_translation_system_prompt,
    build_translation_user_prompt,
    format_book_summary,
)


class TranslationError(Exception):
    """Raised when a chunk cannot be translated into valid output."""


def estimate_cost(provider: str, model: str, input_tokens: int, output_tokens: int) -> float:
    """Rough USD cost estimate. Prices are approximate and may change."""
    name = (model or "").lower()
    if provider == "anthropic":
        if "opus" in name:
            input_price, output_price = 15.0, 75.0
        elif "haiku" in name:
            input_price, output_price = 1.0, 5.0
        else:
            input_price, output_price = 3.0, 15.0
    else:
        if "128k" in name:
            input_price = output_price = 60.0
        elif "32k" in name:
            input_price = output_price = 24.0
        else:
            input_price = output_price = 12.0
        input_price /= 7.2  # approximate CNY -> USD
        output_price /= 7.2
    return input_tokens / 1e6 * input_price + output_tokens / 1e6 * output_price


def _tail_text(fragment: str, length: int = 300) -> str:
    text = re.sub(r"<[^>]+>", " ", fragment or "")
    text = re.sub(r"\s+", " ", text).strip()
    return text[-length:]


def clean_translated_output(text: str) -> str:
    """Strip code fences and prose around the HTML fragment returned by the model."""
    text = (text or "").strip()
    fence = re.match(r"^```(?:html|xml)?\s*\n?(.*?)\n?```$", text, re.S)
    if fence:
        text = fence.group(1).strip()
    first = text.find("<")
    if first > 0:
        head = text[:first].strip()
        if head and not re.search(r"[<>]", head):
            text = text[first:]
    last = text.rfind(">")
    if 0 <= last < len(text) - 1:
        tail = text[last + 1:].strip()
        if tail and not re.search(r"[<>]", tail):
            text = text[:last + 1]
    return text.strip()


class TranslationPipeline:
    """Orchestrates the translation phase over extracted chapters."""

    def __init__(
        self,
        llm,
        config: Dict,
        extractor,
        glossary: Glossary,
        summary: Dict,
        work_path: Path,
        checkpoint_file: Path,
    ):
        self.llm = llm
        self.config = config
        self.extractor = extractor
        self.glossary = glossary
        self.summary = summary or {}
        self.work_path = Path(work_path)
        self.checkpoint_file = Path(checkpoint_file)

        tcfg = config.get("translation", {})
        precfg = config.get("pre_analysis", {})
        pcfg = config.get("processing", {})
        self.max_chunk_chars = tcfg.get("max_chunk_chars", 15000)
        self.min_chunk_chars = tcfg.get("min_chunk_chars", 200)
        self.skip_short = tcfg.get("skip_files_shorter_than", pcfg.get("skip_files_shorter_than", 50))
        self.delay = tcfg.get("delay_between_requests", tcfg.get("delay_between_chapters", 1.0))
        self.glossary_max_terms = tcfg.get("glossary_max_terms_in_prompt", 150)
        self.incremental = bool(precfg.get("incremental_glossary", True))
        self.analysis_sample_chars = precfg.get("chapter_sample_chars", 4000)
        self.analysis_temperature = precfg.get("temperature", 0.2)
        self.analysis_max_tokens = precfg.get("max_tokens", 4096)
        self.max_validation_attempts = config.get("max_retries", 5)

        self.book_summary_text = format_book_summary(
            self.summary.get("book", {}),
            self.extractor.metadata.get("title", ""),
            self.extractor.metadata.get("creator", ""),
        )

        self._lock = threading.Lock()
        self.completed: Dict[str, Optional[set]] = self._load_checkpoint()
        self.failed_files: List[str] = []
        self.translated_chunks = 0

    # ------------------------------------------------------------------
    # Checkpointing (per file + chunk index; legacy file-level supported)
    # ------------------------------------------------------------------

    def _load_checkpoint(self) -> Dict[str, Optional[set]]:
        if not self.checkpoint_file.exists():
            return {}
        try:
            data = json.loads(self.checkpoint_file.read_text(encoding="utf-8"))
        except Exception:
            return {}
        raw = data.get("completed", {})
        if isinstance(raw, list):  # legacy: list of fully-translated file names
            return {name: None for name in raw}
        result: Dict[str, Optional[set]] = {}
        for name, value in (raw or {}).items():
            if value is None:
                result[name] = None
            else:
                result[name] = {int(index) for index in value}
        return result

    def _save_checkpoint(self, filename: str, indices: List[int]):
        with self._lock:
            current = self.completed.setdefault(filename, set())
            if current is not None:
                current.update(indices)
            payload = {
                "completed": {
                    name: (None if value is None else sorted(value))
                    for name, value in self.completed.items()
                }
            }
            tmp = self.checkpoint_file.with_suffix(self.checkpoint_file.suffix + ".tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp, self.checkpoint_file)

    # ------------------------------------------------------------------
    # Chunk translation with format validation
    # ------------------------------------------------------------------

    def _translate_chunk(self, fragment: str, system_prompt: str, user_prompt: str) -> str:
        extra_note = ""
        last_message = ""
        for _ in range(self.max_validation_attempts):
            raw = self.llm.chat(system_prompt, user_prompt + extra_note)
            cleaned = clean_translated_output(raw)
            ok, message = extractor_mod.validate_fragment(fragment, cleaned)
            if ok:
                return cleaned
            last_message = message
            extra_note = (
                f"\n\n[格式警告] 上一次输出的 HTML 标签结构与输入不一致（{message}）。"
                "请严格保留输入的全部 HTML 标签，只翻译标签之间的文本，不要增删或改变任何标签。"
            )
        raise TranslationError(
            f"chunk format validation failed after {self.max_validation_attempts} attempts: {last_message}"
        )

    # ------------------------------------------------------------------
    # Incremental glossary update (timely term induction)
    # ------------------------------------------------------------------

    def _update_glossary(self, chapter):
        try:
            html = chapter.path.read_text(encoding="utf-8", errors="replace")
            text = extractor_mod.html_body_text(html)
            sample = extractor_mod.sample_text(text, self.analysis_sample_chars)
            system_prompt, user_prompt = build_new_terms_prompt(sample, self.glossary.keys())
            raw = self.llm.chat(
                system_prompt,
                user_prompt,
                analysis=True,
                temperature=self.analysis_temperature,
                max_tokens=self.analysis_max_tokens,
            )
            data = extract_json_block(raw)
            terms = [term for term in data.get("terms", []) if term.get("en")]
            if terms:
                added, updated = self.glossary.add_terms(terms)
                self.glossary.save()
                tqdm.write(f"  ◆ 术语表已更新：新增 {added} 条、合并 {updated} 条（共 {len(self.glossary)} 条）")
        except Exception as exc:
            tqdm.write(f"  ⚠ 术语增量更新失败（不影响译文）：{type(exc).__name__}: {exc}")

    # ------------------------------------------------------------------
    # Chapter processing (sequential chunks)
    # ------------------------------------------------------------------

    def _translate_chapter(self, chapter, pbar, progress: Dict) -> Dict:
        name = chapter.path.name
        with self._lock:
            state = self.completed.get(name)

        if name in self.completed:
            if state is None:
                return {"name": name, "status": "done"}
            if len(state) == 0:
                return {"name": name, "status": "skipped"}

        html = chapter.path.read_text(encoding="utf-8", errors="replace")
        text = extractor_mod.html_body_text(html)
        if len(text) < self.skip_short:
            self._save_checkpoint(name, [])
            return {"name": name, "status": "skipped"}

        chunks = extractor_mod.split_html_chunks(html, self.max_chunk_chars, self.min_chunk_chars)
        progress["total"] = len(chunks)
        if not chunks:
            self._save_checkpoint(name, [])
            return {"name": name, "status": "skipped"}

        done_set = state or set()

        translated_dir = self.work_path / "translated" / chapter.path.stem
        translated_dir.mkdir(parents=True, exist_ok=True)

        def fragment_file(index: int) -> Path:
            return translated_dir / f"{index:05d}.html"

        # Chunks are re-translated if their saved fragment is missing (never fall back to English)
        missing_fragments = [i for i in done_set if i < len(chunks) and not fragment_file(i).exists()]
        already = sorted(i for i in done_set if i < len(chunks) and fragment_file(i).exists())
        todo = sorted(set(range(len(chunks))) - done_set | set(missing_fragments))
        for _ in already:
            progress["done"] += 1
            pbar.update(1)
        if not todo:
            return {"name": name, "status": "done"}

        chapter_summary = self.summary.get("chapters", {}).get(name, "")
        glossary_block = self.glossary.to_prompt_block(self.glossary_max_terms)
        system_prompt = build_translation_system_prompt(
            self.config, self.book_summary_text, chapter_summary, glossary_block
        )

        prev_source = None
        prev_translation = None
        for index in todo:
            fragment_path = fragment_file(index)
            if fragment_path.exists():  # translated earlier but checkpoint was lost
                progress["done"] += 1
                pbar.update(1)
                continue
            user_prompt = build_translation_user_prompt(chunks[index], prev_source, prev_translation)
            translated = self._translate_chunk(chunks[index], system_prompt, user_prompt)
            fragment_path.write_text(translated, encoding="utf-8")
            self._save_checkpoint(name, [index])
            prev_source = _tail_text(chunks[index])
            prev_translation = _tail_text(translated)
            with self._lock:
                self.translated_chunks += 1
            progress["done"] += 1
            pbar.update(1)
            if self.delay > 0:
                time.sleep(self.delay)

        # Reassemble the file from stored fragments (string surgery on body only)
        fragments = []
        for index in range(len(chunks)):
            path = fragment_file(index)
            fragments.append(path.read_text(encoding="utf-8") if path.exists() else chunks[index])
        new_html = extractor_mod.reassemble_html(html, fragments)
        chapter.path.write_text(new_html, encoding="utf-8")

        if self.incremental:
            self._update_glossary(chapter)
        return {"name": name, "status": "translated"}

    def _worker(self, chapter, pbar) -> Dict:
        progress = {"done": 0, "total": 0}
        try:
            result = self._translate_chapter(chapter, pbar, progress)
            result["units_done"] = progress["done"]
            result["units_total"] = progress["total"]
            return result
        except Exception as exc:
            return {
                "name": chapter.path.name,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "units_done": progress["done"],
            }

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def translate(self, chapters, workers: int = 3) -> Dict:
        print(f"Translating {len(chapters)} chapter(s) with {workers} worker(s)...\n")

        # Pre-scan chunk counts for the progress bar
        plan: Dict[str, int] = {}
        for chapter in chapters:
            name = chapter.path.name
            with self._lock:
                state = self.completed.get(name)
            if name in self.completed and (state is None or len(state) == 0):
                plan[name] = 0
                continue
            if chapter.char_count < self.skip_short:
                plan[name] = 0
                continue
            try:
                html = chapter.path.read_text(encoding="utf-8", errors="replace")
                plan[name] = len(extractor_mod.split_html_chunks(html, self.max_chunk_chars, self.min_chunk_chars))
            except Exception:
                plan[name] = 0
        total_units = sum(plan.values())

        results = []
        with tqdm(total=total_units, desc="Translation Progress", unit="chunk") as pbar:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(self._worker, chapter, pbar): chapter for chapter in chapters}
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    if result["status"] == "failed":
                        gap = max(0, plan.get(result["name"], 0) - result.get("units_done", 0))
                        if gap:
                            pbar.update(gap)
                        tqdm.write(f"  ✗ Failed: {result['name']}: {result.get('error')}")
                        with self._lock:
                            self.failed_files.append(result["name"])

        translated = sum(1 for r in results if r["status"] == "translated")
        skipped = sum(1 for r in results if r["status"] in ("skipped", "done"))
        print()
        return {
            "chapters_total": len(chapters),
            "translated": translated,
            "skipped": skipped,
            "failed": self.failed_files,
            "chunks_translated": self.translated_chunks,
        }
