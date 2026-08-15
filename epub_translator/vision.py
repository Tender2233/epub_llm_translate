"""Optional multimodal pass: OCR + translate text inside images via a vision model.

Runs after the text translation phase. For every <img> referenced by a chapter,
the vision model extracts English text from the image and translates it. The
translation is written back into the HTML right after the image tag (append
mode) or into its alt attribute (alt mode). Results are checkpointed per image
(keyed by relative path + sha256) so re-runs resume cheaply and the same image
shared by several chapters is only OCR'd once.
"""

import base64
import hashlib
import html as html_mod
import json
import os
import re
import threading
import time
import urllib.parse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

from . import extractor as extractor_mod
from .llm_client import extract_json_block
from .prompts import build_image_ocr_prompt

IMG_TAG_RE = re.compile(r"<img\b[^>]*>", re.I)
SRC_ATTR_RE = re.compile(r"""src\s*=\s*["']([^"']+)["']""", re.I)
ALT_ATTR_RE = re.compile(r"""alt\s*=\s*["'][^"']*["']""", re.I)


class ImageTranslator:
    """Discovers chapter images, OCR-translates them and injects the results into HTML."""

    def __init__(self, llm, config: Dict, work_path: Path, extractor):
        self.llm = llm
        self.config = config
        self.work_path = Path(work_path)
        self.extractor = extractor
        mm = config.get("multimodal", {}) or {}
        self.output_mode = mm.get("output_mode", "append")  # "append" | "alt"
        self.max_image_mb = mm.get("max_image_mb", 2)
        self.delay = mm.get("delay_between_requests", 1.0)
        self.temperature = mm.get("temperature", 0.2)
        self.max_tokens = mm.get("max_tokens", 4096)
        self.checkpoint_file = self.work_path / "image_checkpoint.json"
        self._lock = threading.Lock()
        self.checkpoint = self._load_checkpoint()
        self.images_translated = 0
        self.images_skipped = 0

    # ------------------------------------------------------------------
    # Checkpoint: {"images": {rel_key: {"digest", "items"}}, "injected": [name...]}
    # ------------------------------------------------------------------

    def _load_checkpoint(self) -> Dict:
        if not self.checkpoint_file.exists():
            return {"images": {}, "injected": []}
        try:
            data = json.loads(self.checkpoint_file.read_text(encoding="utf-8"))
        except Exception:
            return {"images": {}, "injected": []}
        return {
            "images": dict(data.get("images", {}) or {}),
            "injected": [str(name) for name in (data.get("injected", []) or [])],
        }

    def _save_checkpoint(self):
        tmp = self.checkpoint_file.with_suffix(self.checkpoint_file.suffix + ".tmp")
        tmp.write_text(json.dumps(self.checkpoint, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, self.checkpoint_file)

    def _mark_injected(self, name: str):
        with self._lock:
            if name not in self.checkpoint["injected"]:
                self.checkpoint["injected"].append(name)
                self._save_checkpoint()

    # ------------------------------------------------------------------
    # Image resolution helpers
    # ------------------------------------------------------------------

    def _rel_key(self, path: Path) -> str:
        try:
            return path.relative_to(self.work_path).as_posix()
        except ValueError:
            return str(path)

    def _resolve_key(self, chapter_path: Path, src: str) -> Optional[str]:
        """Resolve an <img src> to the canonical relative key used by the checkpoint."""
        src = urllib.parse.unquote((src or "").strip()).split("#")[0].strip()
        if not src or src.lower().startswith(("data:", "http:", "https:", "//")):
            return None
        path = (chapter_path.parent / src).resolve()
        try:
            path.relative_to(self.work_path)
        except ValueError:
            return None
        if not path.is_file() or extractor_mod.image_media_type(path) is None:
            return None
        return self._rel_key(path)

    @staticmethod
    def _digest(path: Path) -> str:
        hasher = hashlib.sha256()
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(65536), b""):
                hasher.update(block)
        return hasher.hexdigest()

    # ------------------------------------------------------------------
    # OCR + translation
    # ------------------------------------------------------------------

    def _items_for(self, path: Path) -> Tuple[List[dict], str]:
        """Return (items, digest). Reuses cached items when the digest matches."""
        digest = self._digest(path)
        key = self._rel_key(path)
        cached = self.checkpoint["images"].get(key)
        if cached and cached.get("digest") == digest:
            return cached.get("items") or [], digest

        mime = extractor_mod.image_media_type(path)
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        system, user = build_image_ocr_prompt()
        raw = self.llm.chat_vision(
            system, user, [(mime, data)], temperature=self.temperature, max_tokens=self.max_tokens
        )
        parsed = extract_json_block(raw)
        items = []
        for item in parsed.get("items", []):
            text = str(item.get("text", "")).strip()
            translation = str(item.get("translation", "")).strip()
            if text and translation:
                items.append({"text": text, "translation": translation})

        with self._lock:
            self.checkpoint["images"][key] = {"digest": digest, "items": items}
            self._save_checkpoint()
        return items, digest

    # ------------------------------------------------------------------
    # HTML write-back
    # ------------------------------------------------------------------

    @staticmethod
    def _build_translation_paragraph(items: List[dict]) -> str:
        lines = []
        for item in items:
            src = html_mod.escape(str(item["text"]), quote=True)
            dst = html_mod.escape(str(item["translation"]), quote=True)
            lines.append(f"{src} → {dst}")
        return '<p class="llm-img-translation">' + "<br/>".join(lines) + "</p>"

    @staticmethod
    def _set_alt(tag: str, items: List[dict]) -> str:
        alt_text = "；".join(f'{item["text"]} → {item["translation"]}' for item in items)
        alt_text = html_mod.escape(alt_text, quote=True)
        if ALT_ATTR_RE.search(tag):
            return ALT_ATTR_RE.sub(f'alt="{alt_text}"', tag, count=1)
        attr = f'alt="{alt_text}"'
        if tag.endswith("/>"):
            return tag[:-2] + " " + attr + " />"
        if tag.endswith(">"):
            return tag[:-1] + " " + attr + ">"
        return tag

    def _inject(self, chapter_html: str, chapter_path: Path, results: Dict[str, List[dict]]) -> str:
        """Splice translated text after each <img> (append) or into alt (alt mode)."""
        pieces: List[str] = []
        last = 0
        for match in IMG_TAG_RE.finditer(chapter_html):
            pieces.append(chapter_html[last:match.start()])
            tag = match.group(0)
            src_match = SRC_ATTR_RE.search(tag)
            key = self._resolve_key(chapter_path, src_match.group(1)) if src_match else None
            items = (results.get(key) or []) if key else []
            if items:
                if self.output_mode == "alt":
                    tag = self._set_alt(tag, items)
                pieces.append(tag)
                if self.output_mode == "append":
                    pieces.append(self._build_translation_paragraph(items))
            else:
                pieces.append(tag)
            last = match.end()
        pieces.append(chapter_html[last:])
        return "".join(pieces)

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def translate(self, chapters) -> Dict:
        """OCR-translate all chapter images, then inject results into the HTML files."""
        size_limit = int(self.max_image_mb * 1024 * 1024)
        jobs: List[Path] = []
        job_keys = set()
        chapter_keys: Dict[str, List[str]] = {}

        for chapter in chapters:
            name = chapter.path.name
            keys: List[str] = []
            for path in self.extractor.chapter_images(chapter):
                key = self._rel_key(path)
                keys.append(key)
                cached = self.checkpoint["images"].get(key)
                if cached and cached.get("digest") == self._digest(path):
                    continue  # already OCR'd
                if path.stat().st_size > size_limit:
                    tqdm.write(
                        f"  ⚠ Skipping oversized image "
                        f"({path.stat().st_size / 1048576:.1f}MB > {self.max_image_mb}MB): {key}"
                    )
                    self.images_skipped += 1
                    continue
                if key not in job_keys:
                    job_keys.add(key)
                    jobs.append(path)
            if keys:
                chapter_keys[name] = keys

        if not jobs:
            print(f"  No new images to translate "
                  f"(checkpoint already covers {len(self.checkpoint['images'])} image(s))")
        else:
            print(f"  Found {len(jobs)} unique image(s) to process ...")
            with tqdm(total=len(jobs), desc="Image Translation", unit="img") as pbar:
                for path in jobs:
                    key = self._rel_key(path)
                    try:
                        items, _digest = self._items_for(path)
                    except Exception as exc:
                        tqdm.write(f"  ⚠ Image OCR failed ({key}): {type(exc).__name__}: {exc}")
                        pbar.update(1)
                        continue
                    if items:
                        self.images_translated += 1
                    else:
                        tqdm.write(f"  ◇ No text found in image: {key}")
                    pbar.update(1)
                    if self.delay > 0:
                        time.sleep(self.delay)

        # Inject results into chapters whose images are all accounted for.
        updated = 0
        for chapter in chapters:
            name = chapter.path.name
            keys = chapter_keys.get(name) or []
            if not keys or name in self.checkpoint["injected"]:
                continue
            if not all(key in self.checkpoint["images"] for key in keys):
                continue  # some image failed or was skipped; re-run can complete it
            results = {
                key: self.checkpoint["images"][key].get("items") or [] for key in keys
            }
            if not any(results.values()):
                self._mark_injected(name)
                continue
            try:
                chapter_html = chapter.path.read_text(encoding="utf-8", errors="replace")
                new_html = self._inject(chapter_html, chapter.path, results)
                if new_html != chapter_html:
                    chapter.path.write_text(new_html, encoding="utf-8")
                    updated += 1
            except Exception as exc:
                tqdm.write(f"  ⚠ Failed to inject translations into {name}: {type(exc).__name__}: {exc}")
                continue
            self._mark_injected(name)

        print(
            f"  ✓ Image pass done: {self.images_translated} with text, "
            f"{self.images_skipped} skipped, {updated} chapter(s) updated"
        )
        return {
            "images_translated": self.images_translated,
            "images_skipped": self.images_skipped,
            "chapters_updated": updated,
        }
