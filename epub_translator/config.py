"""Configuration loading with deep defaults."""

import copy
import json
import sys
from pathlib import Path
from typing import Dict

DEFAULT_CONFIG: Dict = {
    "api_provider": "openai",
    "max_retries": 5,
    "retry_base_delay": 2,

    # Any OpenAI-compatible endpoint (OpenAI, Kimi, GLM, Qwen, DeepSeek, ...).
    # Kimi migration: set base_url to https://api.moonshot.cn/v1 and pick moonshot models.
    "openai": {
        "api_key": "",
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o",
        "analysis_model": "gpt-4o-mini",
        "vision_model": "",           # separate multimodal model; empty = auto fallback
        "temperature": 0.3,
        "max_tokens": 16384,
    },

    # Anthropic Messages API, including Claude-compatible gateways via base_url.
    "anthropic": {
        "api_key": "",
        "base_url": "",               # empty = official Anthropic endpoint
        "model": "claude-sonnet-4-6",
        "analysis_model": "claude-haiku-4-5",
        "vision_model": "",
        "temperature": 0.3,
        "max_tokens": 16384,
    },

    # Optional multimodal pass: OCR + translate text inside images.
    "multimodal": {
        "enabled": False,              # True, or use --translate-images
        "vision_model": "",           # global override (beats provider.vision_model)
        "output_mode": "append",      # append | alt
        "max_image_mb": 2,            # images larger than this are skipped
        "delay_between_requests": 1.0,
        "temperature": 0.2,
        "max_tokens": 4096,
    },

    "translation": {
        "target_language": "zh",
        "target_language_name": "Simplified Chinese (简体中文)",
        "max_chunk_chars": 15000,      # target chunk size (English chars) per request
        "min_chunk_chars": 200,        # chunks smaller than this get merged with neighbours
        "skip_files_shorter_than": 50, # files with less visible text are left untranslated
        "delay_between_requests": 1.0, # seconds between chunk requests
        "workers": 3,                  # parallel chapter workers
        "glossary_max_terms_in_prompt": 150,  # top-N terms injected into prompts
    },

    "prompt_customization": {
        "translation_style": "literary",
        "formality": "moderate",
        "preserve_names": True,
        "cultural_adaptation": True,
    },

    "pre_analysis": {
        "enabled": True,
        "chapter_sample_chars": 4000,       # sampled chars per chapter for analysis
        "chapters_per_summary_batch": 4,    # chapters summarized per analysis request
        "glossary_max_terms_total": 500,    # cap for the final consolidated glossary
        "incremental_glossary": True,       # extract new terms after each chapter
        "temperature": 0.2,
        "max_tokens": 4096,
    },

    "processing": {
        "work_directory": "./epub_work",
        "keep_work_directory": False,
        "include_outside_spine": False,     # translate files not listed in the spine
    },
}


def deep_merge(base: Dict, override: Dict) -> Dict:
    """Recursively merge override into base (base wins nowhere)."""
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(config_path: str = "config.json") -> Dict:
    """Load JSON config merged over built-in defaults."""
    config_file = Path(config_path)

    if not config_file.exists():
        print(f"Warning: Config file not found at {config_path}")
        print("Using built-in defaults. Create config.json from config.template.json to customize.")
        return copy.deepcopy(DEFAULT_CONFIG)

    try:
        with open(config_file, "r", encoding="utf-8") as f:
            user_config = json.load(f)
        return deep_merge(DEFAULT_CONFIG, user_config)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in config file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading config file: {e}")
        sys.exit(1)
