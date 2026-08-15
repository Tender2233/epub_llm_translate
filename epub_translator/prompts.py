"""Centralized prompt builders for translation and pre-analysis phases."""

from typing import Dict, List, Tuple

STYLE_MAP = {
    "literary": "文学性强，注重意境、韵律与文字美感，贴近原作的文学表达",
    "colloquial": "口语化、轻松自然，贴近日常表达习惯",
    "academic": "严谨准确，符合学术著作的规范表达",
    "technical": "专业准确，使用业界通行的标准术语",
    "simple": "简明易懂，面向大众读者",
}

FORMALITY_MAP = {
    "formal": "正式书面语",
    "moderate": "适度正式，兼顾可读性",
    "casual": "轻松随意",
}


# ---------------------------------------------------------------------------
# Translation prompts
# ---------------------------------------------------------------------------

def format_book_summary(book: Dict, title: str = "", creator: str = "") -> str:
    parts = []
    if title:
        parts.append(f"书名：{title}")
    if creator:
        parts.append(f"作者：{creator}")
    if book.get("genre"):
        parts.append(f"体裁背景：{book['genre']}")
    if book.get("style"):
        parts.append(f"语言风格：{book['style']}")
    if book.get("summary"):
        parts.append(f"内容概要：{book['summary']}")
    return "\n".join(parts) if parts else "（暂无）"


def build_translation_system_prompt(
    config: Dict, book_summary_text: str, chapter_summary: str, glossary_block: str
) -> str:
    """Stable system instructions: role, style, format rules, book/chapter/glossary context."""
    tcfg = config.get("translation", {})
    prompt_cfg = config.get("prompt_customization", {})
    target = tcfg.get("target_language_name", "简体中文")

    style = STYLE_MAP.get(
        prompt_cfg.get("translation_style", "literary"), prompt_cfg.get("translation_style", "文学性强")
    )
    formality = FORMALITY_MAP.get(
        prompt_cfg.get("formality", "moderate"), prompt_cfg.get("formality", "适度正式")
    )
    if prompt_cfg.get("preserve_names", True):
        name_policy = (
            "人名、地名等专有名词必须严格遵循术语表译名；术语表未收录的人名地名按通用音译处理，"
            "并尽量保持全书一致"
        )
    else:
        name_policy = "专有名词可适当意译，但同一实体的译名应保持一致"
    if prompt_cfg.get("cultural_adaptation", True):
        culture_policy = "在保留原意的前提下进行适度文化适配，使中文读者易于理解"
    else:
        culture_policy = "尽量贴近原文，不做过度归化"

    return f"""你是一位专业的文学译者，精通英语与中文，长期从事图书翻译工作，擅长将英文作品译为{target}。

# 翻译任务
将用户提供的英文 HTML 片段翻译成{target}，只输出翻译后的 HTML 片段本身。

# 翻译要求
- 译文准确、流畅、自然，符合中文表达习惯；保持原文的语气、情感与文学性
- 翻译风格：{style}
- 正式程度：{formality}
- 专有名词处理：{name_policy}
- 文化处理：{culture_policy}
- 忠实原文，不增删内容，不添加解释、评论或注脚

# 格式要求（必须严格遵守）
1. 输入是一段 HTML 片段；输出必须是标签结构完全相同的翻译后 HTML 片段
2. 完整保留所有 HTML 标签、属性与嵌套层级，只翻译标签之间的文本
3. 不得新增、删除、合并或拆分任何标签；不要输出代码围栏（```）
4. 不要翻译代码、URL、邮箱地址、文件名、编号、度量单位等非自然语言内容
5. 只输出翻译后的 HTML 片段，禁止任何前言、说明或后记

# 本书概要
{book_summary_text}

# 本章摘要
{chapter_summary or "（暂无摘要）"}

# 统一术语表（必须严格遵守，不得改用其他译法）
{glossary_block}"""


def build_translation_user_prompt(fragment: str, prev_source: str = None, prev_translation: str = None) -> str:
    """User message: previous chunk tails for continuity + the fragment to translate."""
    prev = ""
    if prev_source:
        prev = (
            f"## 前文衔接（仅作风格参考，无需翻译）\n"
            f"原文结尾：{prev_source}\n"
            f"参考译文结尾：{prev_translation or ''}\n\n"
        )
    return f"""{prev}## 待翻译 HTML 片段
{fragment}

请只输出翻译后的 HTML 片段："""


# ---------------------------------------------------------------------------
# Pre-analysis prompts (run with the analysis model)
# ---------------------------------------------------------------------------

def build_chapter_summary_prompt(batch: List[Tuple[str, str]]) -> Tuple[str, str]:
    """One request summarizes several chapters and extracts candidate terms."""
    system = (
        "你是图书编辑助理，负责为英文图书逐章生成内容摘要与关键术语表，供后续机器翻译统一术语使用。"
        "只输出 JSON，不要输出任何其他文字、解释或代码围栏。"
    )
    parts = []
    for name, text in batch:
        parts.append(f"### {name}\n{text}\n")
    user = f"""请阅读以下 {len(batch)} 个章节（每章为节选文本），为每章完成两项工作：
1. 用简体中文写 80~120 字的内容摘要（概述本章发生的事件或论述要点，不含评价）
2. 提取本章重要的人名、地名、机构名、专有术语（最多 15 个），给出建议的简体中文译名与类别

类别只能是：person（人名）、place（地名）、organization（机构）、term（专有术语）

严格按如下 JSON 格式输出：
{{"chapters": [{{"chapter": "文件名", "summary": "中文摘要", "terms": [{{"en": "英文原文", "zh": "中文译名", "category": "person"}}]}}]}}

章节内容：
{"".join(parts)}"""
    return system, user


def build_book_summary_prompt(chapter_summaries: Dict[str, str], title: str, creator: str) -> Tuple[str, str]:
    """Aggregate chapter summaries into a book-level summary."""
    system = "你是图书编辑，擅长撰写全书概要。只输出 JSON，不要输出任何其他文字或代码围栏。"
    lines = [f"{i + 1}. （{name}）{summary}" for i, (name, summary) in enumerate(chapter_summaries.items())]
    joined = "\n".join(lines)[:12000]
    user = f"""以下是《{title}》（作者：{creator or '未知'}）各章节的中文摘要。

请基于这些摘要生成：
- summary：全书内容概要（200~400 字，概括主线、主题与主要人物）
- style：语言风格与写作特点（如叙事口吻、时代背景、文体特征）
- genre：体裁与题材（如奇幻小说、历史传记、技术专著等）

严格按如下 JSON 格式输出：
{{"summary": "...", "style": "...", "genre": "..."}}

章节摘要：
{joined}"""
    return system, user


def build_glossary_consolidation_prompt(grouped: List[dict], max_terms: int) -> Tuple[str, str]:
    """Merge deduplicated candidate terms into the final book-wide glossary."""
    system = (
        "你是术语管理专家，负责把候选术语合并去重，整理为全书统一的术语表。\n"
        "规则：\n"
        "1. 同一实体的不同写法（大小写、缩写、全称、单复数、变体）合并为一条，en 采用最常见写法\n"
        "2. 译名采用最标准、最常见的简体中文译法，全书统一\n"
        "3. 只保留人名、地名、机构名、专有术语等有统一译名价值的词；普通词汇不要收录\n"
        "4. category 只能是 person/place/organization/term/other\n"
        "只输出 JSON，不要输出任何其他文字或代码围栏。"
    )
    lines = []
    for group in grouped:
        variants = "；".join(group["variants"][:4])
        more = "…" if len(group["variants"]) > 4 else ""
        zh = group.get("zh") or "未定"
        lines.append(f'- {group["en"]}（变体：{variants}{more}）×{group["frequency"]} → {zh} [{group["category"]}]')
        if sum(len(line) for line in lines) > 12000:
            break
    user = f"""候选术语如下（含出现频率与建议译名，按频率降序排列）：

{chr(10).join(lines)}

请整合输出最终术语表（最多 {max_terms} 条，按重要性排序），严格按如下 JSON 格式：
{{"terms": [{{"en": "标准英文", "zh": "统一中文译名", "category": "person", "frequency": 12}}]}}"""
    return system, user


def build_new_terms_prompt(chapter_text: str, existing_keys: List[str]) -> Tuple[str, str]:
    """Find terms not yet in the glossary (used for incremental updates)."""
    system = (
        "你是术语管理专家。从给定章节文本中，找出尚未收录的重要专有名词"
        "（人名、地名、机构名、专有术语），供全书术语统一使用。\n"
        "只找确实需要统一译名的专有名词，普通词汇、已收录术语都不要输出。"
        "只输出 JSON，不要输出任何其他文字或代码围栏。"
    )
    keys_block = "\n".join(f"- {key}" for key in existing_keys[:300]) or "（暂无）"
    user = f"""已收录术语（请勿重复输出）：
{keys_block}

本章文本节选：
{chapter_text}

找出文中新的专有名词，给出建议的简体中文译名与类别（person/place/organization/term），
严格按如下 JSON 格式输出；若没有新术语则输出空数组：
{{"terms": [{{"en": "英文原文", "zh": "中文译名", "category": "person", "frequency": 3}}]}}"""
    return system, user


# ---------------------------------------------------------------------------
# Multimodal prompts (image OCR + translation)
# ---------------------------------------------------------------------------

def build_image_ocr_prompt(target_language: str = "简体中文") -> Tuple[str, str]:
    """Ask the vision model to extract and translate all English text inside an image."""
    system = (
        "你是图像文字识别与翻译助手。仔细观察图片，找出图片中出现的所有英文文字"
        "（包括标题、标注、对话气泡、图表文字、地图地名等），并翻译成简体中文。\n"
        "要求：\n"
        "1. 按图中文字出现顺序（大致从上到下、从左到右）逐条输出\n"
        "2. 忠实原意，不解释、不评论；专有名词按通用译法处理\n"
        "3. 只输出 JSON，不要输出任何其他文字、解释或代码围栏\n"
        "4. 若图中没有任何文字，输出空数组"
    )
    user = f"""请把图片中的英文文字翻译成{target_language}，严格按如下 JSON 格式输出：
{{"items": [{{"text": "图片中的英文原文", "translation": "简体中文译文"}}]}}"""
    return system, user
