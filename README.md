# EPUB Translator: English to Chinese

自用 AI 编程工具辅助开发的 epub 电子书翻译器。
Automatically translate English EPUB books to Chinese using AI (any OpenAI-compatible endpoint or Claude) while preserving formatting, structure, and metadata.

## ✨ Key Features

- **两阶段流水线**：先用便宜的分析模型通读全书（章节摘要 → 全书概要 → 统一术语表），再并行翻译
- **通用模型接入**：任意 OpenAI-compatible 网关（OpenAI / Kimi / GLM / Qwen / DeepSeek …）与 Anthropic Messages API（含 Claude-like 网关，支持自定义 base_url）
- **可选多模态翻译**：用 vision 模型识别并翻译图片内的文字（漫画气泡、图表、地图标注），译文自动写回图片之后；vision 模型可单独配置
- **术语统一**：人名 / 地名 / 机构名 / 专有术语在预分析阶段自动归纳成术语表，翻译中还会**增量补充**新术语，保证全书译名一致
- **上下文充分利用**：按块级元素切分章节（默认每请求约 15k 字符），system prompt 注入全书概要 + 本章摘要 + 术语表，上一块译文尾部作为衔接上下文
- **格式保真**：只翻译 `<body>` 内文本，标签结构与校验（标签缺失自动重试），HTML/CSS/图片原样保留
- **Prompt 优化**：system/user 消息分离、`prompt_customization` 全部生效（风格/正式度/人名策略/文化适配）
- **断点续译**：chunk 粒度 checkpoint，崩溃后重跑不重复计费
- **成本统计**：翻译与分析阶段 token 分开计价

## Prerequisites

1. **Python 3.8+**
2. **API Key** from one of these providers:
   - **OpenAI 或任意 OpenAI-compatible 网关**（推荐，性价比高）：如 [OpenAI](https://platform.openai.com)、[Kimi](https://platform.moonshot.cn)、GLM、Qwen、DeepSeek 等，填对应 `base_url` 即可
   - **Claude（Anthropic）**：在 [console.anthropic.com](https://console.anthropic.com) 注册；使用第三方 Claude-like 网关时填写 `base_url`

### Kimi 老用户迁移

`kimi` provider 已重命名为 `openai`（不再保留别名）。在 `config.json` 中改：

```json
{
  "api_provider": "openai",
  "openai": {
    "api_key": "your-kimi-api-key-here",
    "base_url": "https://api.moonshot.cn/v1",
    "model": "moonshot-v1-128k",
    "analysis_model": "moonshot-v1-8k"
  }
}
```

## Installation

### 1. Clone or download this repository

```bash
git clone <repository-url>
cd epub-translator
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API settings

```bash
cp config.template.json config.json
```

Then edit `config.json` and add your API key:

```json
{
  "api_provider": "openai",
  "openai": {
    "api_key": "your-api-key-here",
    "base_url": "https://api.openai.com/v1",
    "model": "gpt-4o",
    "analysis_model": "gpt-4o-mini"
  }
}
```

`analysis_model` 用于预分析（摘要/术语表）与翻译中的术语增量更新，选便宜的模型即可。也可以省略，默认与翻译模型相同。

## Usage

### Basic Usage

```bash
python translate_epub.py input.epub output_chinese.epub
```

### Full Command Reference

```bash
python translate_epub.py [-h] [--config CONFIG] [--provider {openai,anthropic}]
                         [--model MODEL] [--analysis-model ANALYSIS_MODEL]
                         [--vision-model VISION_MODEL] [--api-key API_KEY]
                         [--work-dir WORK_DIR] [--workers N] [--keep-work-dir]
                         [--translate-images] [--skip-analysis] [--reanalyze]
                         input_epub output_epub
```

| Option | Description |
|--------|-------------|
| `--provider` | `openai` (any OpenAI-compatible endpoint) or `anthropic` (overrides config) |
| `--model` | Translation model (overrides config) |
| `--analysis-model` | Model used for pre-analysis & incremental glossary |
| `--vision-model` | Vision model for the image pass (overrides config; wins over all config sources) |
| `--api-key` | API key (overrides config) |
| `--work-dir` | Working directory (default: `./epub_work`) |
| `--workers` | Number of parallel chapter workers (default: 3) |
| `--keep-work-dir` | Keep working directory (contains summary/glossary/checkpoint) |
| `--translate-images` | Enable the multimodal pass: OCR + translate text inside images |
| `--skip-analysis` | Skip pre-analysis; reuse saved `summary.json`/`glossary.json` |
| `--reanalyze` | Force re-running pre-analysis |

### Multimodal Image Translation

在 `config.json` 的 `multimodal` 节启用（`"enabled": true`），或加 `--translate-images` 参数。
vision 模型解析顺序：`--vision-model` > `multimodal.vision_model` > provider 节 `vision_model` > `analysis_model` > `model`。

- 图片内英文文字（漫画气泡、图表标注、地图地名等）会被识别并译成简体中文
- 译文默认以 `<p class="llm-img-translation">` 段落追加在图片之后；`output_mode: "alt"` 则写入图片的 `alt` 属性
- 每张图按内容哈希缓存（`image_checkpoint.json`），同一张图多章引用只识别一次，重跑断点续传
- 超过 `max_image_mb` 的图片会跳过并警告；SVG、外部 URL、data: URI 图片暂不支持

### Example Workflow

```bash
# 1. Set up config.json with your API key (one time)
cp config.template.json config.json
# Edit config.json and add your OpenAI-compatible or Claude API key

# 2. Translate your book
python translate_epub.py my_book.epub my_book_chinese.epub

# 3. Check the output
# - my_book_chinese.epub          (translated book)
# - my_book_chinese.epub.stats.json (translation statistics)
```

### Example Output

```
============================================================
EPUB Translation: my_book.epub
Provider: openai | Model: gpt-4o | Analysis model: gpt-4o-mini
Workers: 3
============================================================

Extracting EPUB: my_book.epub
✓ Extracted to: epub_work
✓ Found 25 chapter file(s) in spine order
Running pre-analysis: chapter summaries, book summary, unified glossary ...
  [1/7] 生成章节摘要（4 章）...
  ...
  生成全书概要...
  整合术语表（316 个候选）...
✓ Pre-analysis done: 25 chapter summaries, 187 glossary terms

Translating 25 chapter(s) with 3 worker(s)...
Translation Progress: 100%|██████████| 87/87 [05:23<00:00]

Updating metadata...
  ✓ Updated: content.opf
Rebuilding EPUB: my_book_chinese.epub
✓ Created: my_book_chinese.epub

============================================================
TRANSLATION COMPLETE
============================================================
Chunks translated: 87
Glossary terms: 187
Translation tokens: 321,000 in / 540,000 out
Analysis tokens: 48,000 in / 21,000 out
Estimated cost: $0.31 (translation) + $0.03 (analysis) = $0.34 USD
Duration: 6.2 minutes
============================================================
```

## How It Works

1. **提取 (Extraction)**: 解压 EPUB，解析 `content.opf` 得到 **spine 阅读顺序**与书名/作者，清理 `<head>`/`<style>`/`<script>` 等非正文内容
2. **预分析 (Pre-analysis)**: 每章采样（头/中/尾）批量生成章节摘要 → 汇总全书概要（风格/体裁）→ 候选术语模糊去重合并 → LLM 整合为统一术语表（`glossary.json`）
3. **分块 (Chunking)**: 章节按块级元素切分为约 `max_chunk_chars` 大小的片段，标签始终平衡
4. **翻译 (Translation)**: 章节并行、章内分块顺序；system prompt 注入全书概要 + 本章摘要 + 术语表；user 消息带上一块译文尾部作为衔接
5. **校验 (Validation)**: 逐块校验输出标签结构与输入一致，不一致自动带警告重试
6. **增量术语 (Incremental glossary)**: 每章译完用分析模型抽取新术语，线程安全合并到术语表，后续分块生效
7. **多模态 (可选)**: vision 模型识别图片内英文文字并翻译，译文写回图片之后（`--translate-images`）
8. **重建 (Rebuild)**: 字符串手术只替换 `<body>` 内层，更新语言元数据，重新打包 EPUB；chunk 粒度 checkpoint 支持断点续译

工作目录（`--keep-work-dir` 可保留）包含：
- `summary.json` — 全书概要 + 章节摘要
- `glossary.json` — 统一术语表
- `.translation_checkpoint.json` — 正文断点（文件 + chunk 粒度）
- `image_checkpoint.json` — 图片识别断点（图片哈希 + 注入标记）
- `translated/` — 各分块译文

## Configuration Reference

见 `config.template.json`，主要配置项：

| Section | Key | Meaning |
|---------|-----|---------|
| `openai` / `anthropic` | `base_url` | API 端点（openai-compat 与 Claude-like 网关均可自定义；留空用官方端点） |
| | `model` | 翻译模型 |
| | `analysis_model` | 预分析模型（可用便宜的） |
| | `vision_model` | 多模态（图片识别）模型，留空自动回退 |
| | `temperature` / `max_tokens` | 每请求采样参数 |
| `multimodal` | `enabled` | 是否启用图片翻译（等价 `--translate-images`） |
| | `vision_model` | 全局 vision 模型（覆盖 provider 节的 `vision_model`） |
| | `output_mode` | 译文写回方式：`append`（图片后追加段落）/ `alt`（写入 alt 属性） |
| | `max_image_mb` | 超过该大小的图片跳过 |
| | `delay_between_requests` / `temperature` / `max_tokens` | 图片请求参数 |
| `translation` | `max_chunk_chars` | 分块目标大小（英文字符，约 ÷4 ≈ token 数） |
| | `workers` | 并行翻译章节数 |
| | `glossary_max_terms_in_prompt` | 注入 prompt 的术语条数（按词频取 top N） |
| `prompt_customization` | `translation_style` | `literary` / `colloquial` / `academic` / `technical` / `simple` |
| | `formality` | `formal` / `moderate` / `casual` |
| | `preserve_names` | 人名策略 |
| | `cultural_adaptation` | 文化适配 |
| `pre_analysis` | `enabled` | 是否启用预分析 |
| | `chapter_sample_chars` | 每章分析采样字符数 |
| | `chapters_per_summary_batch` | 每次摘要请求的章节数 |
| | `glossary_max_terms_total` | 术语表总量上限 |
| | `incremental_glossary` | 翻译中增量更新术语表 |

## Cost Estimation

预分析阶段使用便宜的 `analysis_model`（如 `gpt-4o-mini` / `claude-haiku`），成本通常不到翻译阶段的 10%。

Approximate costs per 80,000-word novel:

| Provider | Model | Total |
|----------|-------|-------|
| **OpenAI-compatible**（如 Kimi: `base_url=https://api.moonshot.cn/v1`） | 视网关定价 | **~¥0.20 (~$0.03) 起** |
| **Claude** | Sonnet 4 | **~$3.10** |
| **Claude** | Opus 4 | **~$15.30** |

## Troubleshooting

### "API key required" error
Set the API key in `config.json` for your chosen provider, or use the `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` env vars. `OPENAI_BASE_URL` / `ANTHROPIC_BASE_URL` env vars can override endpoints.

### "No HTML chapter files found" error
The EPUB structure may be non-standard. Try `--keep-work-dir` to inspect the extracted files.

### Translation seems truncated
Increase `max_tokens` for the provider in config.json.

### 术语不一致
检查 `glossary.json`（`--keep-work-dir` 保留工作目录），可用 `--reanalyze` 重新生成；提高 `glossary_max_terms_total` 收录更多术语。

### Rate limit errors
Increase `delay_between_requests` in `translation` section, or lower `workers`.

### Resume after interruption
Re-run the same command — the pipeline resumes from the per-chunk checkpoint without re-paying for finished chunks.

### Dependencies won't install
```bash
python --version
pip install --upgrade pip
pip install -r requirements.txt
```


## Important Legal Notes

⚠️ **Copyright Considerations:**
- Only translate books you own or have rights to translate
- Respect copyright laws in your jurisdiction
- This tool is for personal use or licensed content only
- Commercial translation may require publisher permission

## Output Format

The script generates two files:

1. **[output].epub** - The translated EPUB book
2. **[output].stats.json** - Translation statistics including:
   - Provider and model used
   - Files translated
   - Token counts
   - Estimated cost
   - Duration

## Advanced Customization

### Modify Translation Style

Edit the prompts in `translate_epub.py` to customize:
- Translation tone (formal vs casual)
- Target dialect (Simplified vs Traditional Chinese)
- Handling of names and technical terms
- Cultural adaptation level

### Change Target Language

The script can be adapted for other languages by:
1. Modifying the translation prompts in `translate_epub.py`
2. Updating the language code in `update_metadata()` function

### Add Custom Settings

You can customize behavior in `config.json`:
```json
{
  "translation": {
    "target_language": "zh",
    "temperature": 0.3,
    "max_tokens": 16384,
    "delay_between_chapters": 1.0
  },
  "processing": {
    "skip_files_shorter_than": 50,
    "work_directory": "./epub_work",
    "keep_work_directory": false
  }
}
```

**Happy translating!**
