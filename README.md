# EPUB Translator: English to Chinese

自用vide coding的epub电子书翻译器。  
Automatically translate English EPUB books to Chinese using AI (Kimi or Claude) while preserving formatting, structure, and metadata.

## Prerequisites

1. **Python 3.8+**
2. **API Key** from one of these providers:
   - **Kimi** (recommended, cost-effective): Sign up at [platform.moonshot.cn](https://platform.moonshot.cn)
   - **Claude**: Sign up at [console.anthropic.com](https://console.anthropic.com)
3. **Basic command line knowledge**

## Installation

### 1. Clone or download this repository

```bash
# If you have git
git clone <repository-url>
cd epub-translator

# Or download and extract the files manually
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API settings

Create a `config.json` file from the template:

```bash
cp config.template.json config.json
```

Then edit `config.json` and add your API key:

**For Kimi (recommended):**
```json
{
  "api_provider": "kimi",
  "kimi": {
    "api_key": "your-kimi-api-key-here",
    "base_url": "https://api.moonshot.cn/v1",
    "model": "moonshot-v1-128k"
  }
}
```

**For Anthropic Claude:**
```json
{
  "api_provider": "anthropic",
  "anthropic": {
    "api_key": "your-anthropic-api-key-here",
    "model": "claude-sonnet-4-6"
  }
}
```

## Usage

### Basic Usage

Translate an EPUB file to Chinese (uses settings from config.json):

```bash
python translate_epub.py input.epub output_chinese.epub
```

### Advanced Usage

**Switch between providers:**
```bash
# Use Kimi (if configured in config.json)
python translate_epub.py input.epub output.epub --provider kimi

# Use Claude (if configured in config.json)
python translate_epub.py input.epub output.epub --provider anthropic
```

**Override model:**
```bash
python translate_epub.py input.epub output.epub --model moonshot-v1-128k
```

**Keep working directory for inspection:**
```bash
python translate_epub.py input.epub output.epub --keep-work-dir
```

**Specify custom config file:**
```bash
python translate_epub.py input.epub output.epub --config my-config.json
```

### Full Command Reference

```bash
python translate_epub.py [-h] [--config CONFIG] [--provider {kimi,anthropic}]
                         [--model MODEL] [--api-key API_KEY]
                         [--work-dir WORK_DIR] [--keep-work-dir]
                         input_epub output_epub

Arguments:
  input_epub            Path to input English EPUB file
  output_epub           Path for output Chinese EPUB file

Options:
  --config              Path to config file (default: config.json)
  
  --provider            API provider: 'kimi' or 'anthropic' (overrides config)
  
  --model               Model to use (overrides config)
                        • Kimi: moonshot-v1-128k
                        • Claude: claude-sonnet-4-6, claude-opus-4-6
  
  --api-key             API key (overrides config)
  
  --work-dir            Working directory for extraction (default: ./epub_work)
  
  --keep-work-dir       Keep working directory after completion
  
  -h, --help            Show help message
```

## Example Workflow

```bash
# 1. Set up config.json with your API key (one time)
cp config.template.json config.json
# Edit config.json and add your Kimi or Claude API key

# 2. Translate your book
python translate_epub.py my_book.epub my_book_chinese.epub

# 3. Check the output
# - my_book_chinese.epub (translated book)
# - my_book_chinese.stats.json (translation statistics)
```

### Example Output

```
============================================================
EPUB Translation: my_book.epub
Provider: kimi
Model: moonshot-v1-128k
============================================================

Extracting EPUB: my_book.epub
✓ Extracted to: epub_work
✓ Found 25 HTML files to translate

Translating 25 files...

[1/25] Translating: chapter001.xhtml...
    ✓ Translated (3421 chars)
[2/25] Translating: chapter002.xhtml...
    ✓ Translated (4156 chars)
...

Updating metadata...
  ✓ Updated: content.opf

Rebuilding EPUB: my_book_chinese.epub
✓ Created: my_book_chinese.epub
✓ Cleaned up working directory

============================================================
TRANSLATION COMPLETE
============================================================
Output file: my_book_chinese.epub
Files translated: 25/25
Input tokens: 125,432
Output tokens: 178,654
Estimated cost: $0.25 USD
Duration: 8.3 minutes
============================================================
```

## Cost Estimation

Approximate costs per 80,000-word novel:

| Provider | Model | Input Cost | Output Cost | Total |
|----------|-------|-----------|-------------|-------|
| **Kimi** | moonshot-v1-128k | ~¥0.05 | ~¥0.15 | **~¥0.20 (~$0.03)** |
| **Claude** | Sonnet 4 | ~$0.40 | ~$2.70 | **~$3.10** |
| **Claude** | Opus 4 | ~$1.90 | ~$13.40 | **~$15.30** |

**Factors affecting cost:**
- Book length (primary factor)
- Model choice
- Complexity of text

**Tip:** Kimi offers excellent quality at very low cost - ideal for most books. Use Claude if you need premium quality or have specific requirements.

## How It Works

1. **Extraction**: Unzips EPUB file and identifies HTML chapter files
2. **Translation**: Sends each chapter to AI API with specialized translation prompt
3. **Preservation**: Maintains all HTML tags, CSS, images, and formatting
4. **Metadata Update**: Changes language metadata from English to Chinese
5. **Reconstruction**: Rebuilds valid EPUB file with translated content
6. **Statistics**: Tracks tokens used and estimates costs

## Translation Quality

The script uses carefully crafted prompts to ensure:

- ✅ **Literary quality** - Natural, fluent Chinese
- ✅ **Cultural adaptation** - Appropriate for Chinese readers
- ✅ **Tone preservation** - Maintains original style and voice
- ✅ **Consistency** - Uniform terminology and character names
- ✅ **Format integrity** - All formatting preserved exactly

## Troubleshooting

### "API key required" error
Make sure you've set the API key in `config.json` for your chosen provider.

### "Config file not found" error
Create `config.json` from the template:
```bash
cp config.template.json config.json
```
Then edit it to add your API key.

### "No HTML chapter files found" error
The EPUB structure may be non-standard. Try using `--keep-work-dir` to inspect the extracted files.

### Translation seems truncated
Increase the model's context window by editing the `max_tokens` parameter in config.json or the translate_epub.py code.

### Rate limit errors
The script includes 1-second delays between chapters. If you still hit rate limits, you can increase the delay in config.json.

### Dependencies won't install
Ensure you have Python 3.8+ and pip updated:
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
