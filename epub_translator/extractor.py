"""EPUB extraction: spine parsing, HTML cleaning, block-level chunking, reassembly."""

import re
import shutil
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from bs4 import BeautifulSoup, NavigableString, Tag

HTML_MEDIA_TYPES = {"application/xhtml+xml", "text/html", "application/html+xml", "application/xml"}
HTML_EXTENSIONS = (".html", ".xhtml", ".htm")


@dataclass
class ChapterUnit:
    """One translatable chapter file in reading order."""

    path: Path
    spine_index: int
    title: str = ""
    char_count: int = 0


# ---------------------------------------------------------------------------
# Pure HTML helpers
# ---------------------------------------------------------------------------

def html_body_inner(html: str) -> str:
    """Return the inner HTML of <body> (falls back to whole content)."""
    match = re.search(r"<body\b[^>]*>(.*)</body>", html, re.I | re.S)
    if match:
        return match.group(1)
    soup = BeautifulSoup(html, "lxml")
    if soup.body is not None:
        return "".join(str(child) for child in soup.body.children)
    return html


def html_body_text(html: str) -> str:
    """Return the normalized visible text of the body."""
    inner = html_body_inner(html)
    soup = BeautifulSoup(f"<body>{inner}</body>", "lxml")
    return re.sub(r"\s+", " ", soup.get_text(" ")).strip()


def sample_text(text: str, chars: int) -> str:
    """Sample head + middle + tail so analysis sees terms from the whole chapter."""
    if len(text) <= chars:
        return text
    head = int(chars * 0.6)
    mid = int(chars * 0.2)
    tail = chars - head - mid
    middle_start = len(text) // 2
    return (
        text[:head]
        + "\n…[中间部分省略]…\n"
        + text[middle_start:middle_start + mid]
        + "\n…[结尾部分省略]…\n"
        + text[-tail:]
    )


def _piece_len(piece) -> int:
    """Visible-text length of a parse-tree node."""
    if isinstance(piece, NavigableString):
        return len(str(piece).strip())
    return len(piece.get_text(strip=True))


def _new_tag_like(tag: Tag) -> Tag:
    new_tag = BeautifulSoup("", "lxml").new_tag(tag.name)
    for key, value in tag.attrs.items():
        new_tag[key] = value
    return new_tag


def _wrap_text(text: str, tag: Tag) -> str:
    new_tag = _new_tag_like(tag)
    new_tag.string = text
    return str(new_tag)


def _split_long_text(text: str, tag: Tag, max_chars: int) -> List[str]:
    """Split a too-long leaf text at sentence/word boundaries, keeping tags balanced."""
    sentences = re.split(r"(?<=[.!?;])\s+(?=[A-Z])", text)
    if not sentences or max(len(s) for s in sentences) > max_chars:
        sentences = []
        rest = text
        while len(rest) > max_chars:
            cut = rest.rfind(" ", 0, max_chars)
            if cut <= 0:
                cut = max_chars
            sentences.append(rest[:cut])
            rest = rest[cut:]
        sentences.append(rest)
    pieces: List[str] = []
    buf = ""
    for sentence in sentences:
        if buf and len(buf) + len(sentence) > max_chars:
            pieces.append(_wrap_text(buf, tag))
            buf = ""
        buf += sentence
    if buf:
        pieces.append(_wrap_text(buf, tag))
    return pieces


def _split_tag(tag: Tag, max_chars: int) -> List[str]:
    """Split an oversized block element into balanced sub-fragments."""
    children = list(tag.children)
    if not children:
        return [str(tag)]
    if len(children) == 1 and isinstance(children[0], NavigableString) and len(str(children[0])) > max_chars:
        return _split_long_text(str(children[0]), tag, max_chars)

    pieces: List[str] = []
    current: List = []
    current_len = 0
    has_text = False

    def flush():
        nonlocal current, current_len, has_text
        if has_text:
            wrapper = _new_tag_like(tag)
            for child in current:
                wrapper.append(child)
            pieces.append(str(wrapper))
        current = []
        current_len = 0
        has_text = False

    for child in children:
        child_len = _piece_len(child)
        if child_len > max_chars:
            flush()
            if isinstance(child, Tag):
                pieces.extend(_split_tag(child, max_chars))
            else:
                pieces.extend(_split_long_text(str(child), tag, max_chars))
            continue
        if current_len + child_len > max_chars and has_text and child_len > 0:
            flush()
        current.append(child)
        current_len += child_len
        if child_len > 0:
            has_text = True
    flush()
    return pieces or [str(tag)]


def _merge_small_chunks(chunks: List[str], max_chars: int, min_chars: int) -> List[str]:
    """Merge tiny chunks into neighbours to reduce request count."""
    merged: List[str] = []
    for chunk in chunks:
        if merged and len(merged[-1]) + len(chunk) <= max_chars and len(merged[-1]) < min_chars:
            merged[-1] += chunk
        else:
            merged.append(chunk)
    return merged


def split_html_chunks(html: str, max_chars: int = 15000, min_chars: int = 200) -> List[str]:
    """Split body content into balanced HTML fragments, each within max_chars."""
    inner = html_body_inner(html)
    soup = BeautifulSoup(f"<body>{inner}</body>", "lxml")
    body = soup.body
    if body is None:
        return [inner] if inner.strip() else []

    chunks: List[str] = []
    current: List = []
    current_len = 0
    has_text = False

    def flush():
        nonlocal current, current_len, has_text
        if has_text:
            chunks.append("".join(str(child) for child in current))
        current = []
        current_len = 0
        has_text = False

    for child in list(body.children):
        child_len = _piece_len(child)
        if child_len > max_chars and isinstance(child, Tag):
            flush()
            chunks.extend(_split_tag(child, max_chars))
            continue
        if current_len + child_len > max_chars and has_text and child_len > 0:
            flush()
        current.append(child)
        current_len += child_len
        if child_len > 0:
            has_text = True
    flush()
    return _merge_small_chunks(chunks, max_chars, min_chars)


def reassemble_html(html: str, fragments: List[str]) -> str:
    """String surgery: replace only the body inner HTML, keeping the rest byte-identical."""
    match = re.search(r"<body\b[^>]*>", html, re.I)
    end = html.lower().rfind("</body>")
    if match and end > match.end():
        return html[:match.end()] + "".join(fragments) + html[end:]
    return "".join(fragments)


def _tag_counts(fragment: str) -> Counter:
    soup = BeautifulSoup(f"<body>{fragment}</body>", "lxml")
    return Counter(tag.name for tag in soup.body.find_all(True))


def validate_fragment(original: str, translated: str) -> Tuple[bool, str]:
    """Check that the translated fragment keeps the exact tag structure of the input."""
    text = (translated or "").strip()
    if not text:
        return False, "输出为空"
    if "<" not in text:
        return False, "输出不包含 HTML"
    original_counts = _tag_counts(original)
    translated_counts = _tag_counts(text)
    if original_counts == translated_counts:
        return True, ""
    missing = original_counts - translated_counts
    extra = translated_counts - original_counts
    problems = []
    if missing:
        sample = ", ".join(f"{name}×{count}" for name, count in list(missing.items())[:5])
        problems.append(f"缺失标签: {sample}")
    if extra:
        sample = ", ".join(f"{name}×{count}" for name, count in list(extra.items())[:5])
        problems.append(f"多余标签: {sample}")
    return False, "；".join(problems) if problems else "标签结构不一致"


# ---------------------------------------------------------------------------
# EPUB container handling
# ---------------------------------------------------------------------------

class EPubExtractor:
    """Parses the EPUB package (opf) to get spine order, metadata and chapters."""

    def __init__(self, extract_path: Path):
        self.extract_path = Path(extract_path)
        self.opf_path: Optional[Path] = None
        self._manifest: Dict[str, dict] = {}
        self._spine: List[dict] = []
        self._title = ""
        self._creator = ""
        self._parse_package()

    @staticmethod
    def extract_epub(epub_path: str, work_dir: str) -> Path:
        print(f"Extracting EPUB: {epub_path}")
        extract_path = Path(work_dir)
        extract_path.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(epub_path, "r") as zip_ref:
            zip_ref.extractall(extract_path)
        print(f"✓ Extracted to: {extract_path}")
        return extract_path

    def _find_opf(self) -> Optional[Path]:
        container = self.extract_path / "META-INF" / "container.xml"
        if container.exists():
            try:
                soup = BeautifulSoup(container.read_text(encoding="utf-8", errors="replace"), "xml")
                rootfile = soup.find("rootfile")
                if rootfile and rootfile.get("full-path"):
                    candidate = (self.extract_path / rootfile["full-path"]).resolve()
                    if candidate.exists():
                        return candidate
            except Exception:
                pass
        opf_files = sorted(self.extract_path.glob("**/*.opf"))
        return opf_files[0] if opf_files else None

    def _parse_package(self):
        self.opf_path = self._find_opf()
        if self.opf_path is None:
            return
        try:
            soup = BeautifulSoup(self.opf_path.read_text(encoding="utf-8", errors="replace"), "xml")
            for item in soup.find_all("item"):
                item_id = item.get("id")
                href = item.get("href")
                if item_id and href:
                    self._manifest[item_id] = {
                        "href": href,
                        "media_type": (item.get("media-type") or "").lower(),
                        "properties": (item.get("properties") or "").lower(),
                    }
            for itemref in soup.find_all("itemref"):
                self._spine.append(
                    {"idref": itemref.get("idref"), "linear": (itemref.get("linear") or "yes").lower()}
                )
            title = soup.find("dc:title")
            creator = soup.find("dc:creator")
            self._title = title.get_text(strip=True) if title else ""
            self._creator = creator.get_text(strip=True) if creator else ""
        except Exception as exc:
            print(f"  ⚠ Failed to parse package file: {exc}")

    @property
    def metadata(self) -> Dict[str, str]:
        return {"title": self._title, "creator": self._creator}

    def _build_unit(self, item_id: str, spine_index: int) -> Optional[ChapterUnit]:
        item = self._manifest.get(item_id)
        if not item or self.opf_path is None:
            return None
        if "nav" in item["properties"].split():
            return None
        href = item["href"].split("#")[0]
        media_type = item["media_type"]
        is_html = href.lower().endswith(HTML_EXTENSIONS)
        if not is_html:
            return None
        if media_type and media_type not in HTML_MEDIA_TYPES:
            return None
        path = (self.opf_path.parent / href).resolve()
        if not path.exists() or str(path) in self._seen_paths:
            return None
        self._seen_paths.add(str(path))
        try:
            html = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return None
        return ChapterUnit(
            path=path,
            spine_index=spine_index,
            title=self._chapter_title(html),
            char_count=len(html_body_text(html)),
        )

    def chapters(self, include_outside_spine: bool = False) -> List[ChapterUnit]:
        """Return chapter files in spine order (reading order)."""
        self._seen_paths = set()
        units: List[ChapterUnit] = []

        for index, itemref in enumerate(self._spine):
            unit = self._build_unit(itemref.get("idref"), index)
            if unit:
                units.append(unit)

        if include_outside_spine:
            spine_ids = {itemref.get("idref") for itemref in self._spine}
            for item_id in sorted(self._manifest.keys()):
                if item_id in spine_ids:
                    continue
                unit = self._build_unit(item_id, -1)
                if unit:
                    units.append(unit)

        if not units and self.opf_path is None:
            # Malformed EPUB without a package file: fall back to globbing.
            html_files = set()
            for pattern in ("**/*.html", "**/*.xhtml", "**/*.htm"):
                html_files.update(str(f) for f in self.extract_path.glob(pattern))
            html_files = sorted(
                f for f in html_files
                if not any(x in Path(f).name.lower() for x in ("toc", "nav", "cover"))
            )
            for index, html_file in enumerate(html_files):
                path = Path(html_file)
                try:
                    html = path.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    continue
                units.append(
                    ChapterUnit(
                        path=path,
                        spine_index=index,
                        title=self._chapter_title(html),
                        char_count=len(html_body_text(html)),
                    )
                )
        return units

    @staticmethod
    def _chapter_title(html: str) -> str:
        match = re.search(r"<title[^>]*>(.*?)</title>", html, re.I | re.S)
        return re.sub(r"\s+", " ", match.group(1)).strip() if match else ""


def update_metadata(extract_path: Path, target_lang: str = "zh"):
    """Update EPUB metadata language declarations."""
    print("Updating metadata...")
    opf_files = list(Path(extract_path).glob("**/*.opf"))
    for opf_file in opf_files:
        content = opf_file.read_text(encoding="utf-8")
        content = content.replace('lang="en"', f'lang="{target_lang}"')
        content = content.replace("<dc:language>en</dc:language>", f"<dc:language>{target_lang}</dc:language>")
        content = content.replace("<dc:language>en-US</dc:language>", f"<dc:language>{target_lang}</dc:language>")
        content = content.replace("<dc:language>en-GB</dc:language>", f"<dc:language>{target_lang}</dc:language>")
        opf_file.write_text(content, encoding="utf-8")
        print(f"  ✓ Updated: {opf_file.name}")


def rebuild_epub(extract_path: Path, output_path: str) -> Path:
    """Rebuild a valid EPUB (zip) from the extracted directory."""
    print(f"Rebuilding EPUB: {output_path}")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as epub_zip:
        mimetype_path = Path(extract_path) / "mimetype"
        if mimetype_path.exists():
            epub_zip.write(mimetype_path, "mimetype", compress_type=zipfile.ZIP_STORED)
        for file_path in Path(extract_path).rglob("*"):
            if file_path.is_file() and file_path.name != "mimetype":
                epub_zip.write(file_path, file_path.relative_to(extract_path))

    print(f"✓ Created: {output}")
    return output
