#!/usr/bin/env python3
"""
EPUB Translator - Translate English EPUB books to Chinese using Claude API
"""

import os
import sys
import json
import time
import shutil
import zipfile
import threading
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Optional, Set
import argparse
from datetime import datetime

try:
    from anthropic import Anthropic
    from openai import OpenAI
    from bs4 import BeautifulSoup
    from tqdm import tqdm
except ImportError:
    print("Error: Required dependencies not installed.")
    print("Please run: pip install anthropic openai beautifulsoup4 lxml tqdm")
    sys.exit(1)


def load_config(config_path: str = "config.json") -> Dict:
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to config file (default: config.json)
        
    Returns:
        Configuration dictionary
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        print(f"Warning: Config file not found at {config_path}")
        print("Using default settings. Create config.json from config.template.json to customize.")
        return {}
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in config file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading config file: {e}")
        sys.exit(1)


class EPUBTranslator:
    """Handles EPUB extraction, translation, and reconstruction."""
    
    def __init__(self, config: Optional[Dict] = None, api_key: Optional[str] = None, 
                 model: Optional[str] = None, provider: Optional[str] = None):
        """
        Initialize the EPUB translator.
        
        Args:
            config: Configuration dictionary from config.json
            api_key: API key (overrides config file)
            model: Model to use (overrides config file)
            provider: API provider ('kimi' or 'anthropic', overrides config file)
        """
        # Load config if not provided
        if config is None:
            config = load_config()
        
        # Determine provider
        self.provider = provider or config.get("api_provider", "kimi")
        
        # Get provider-specific config
        if self.provider == "kimi":
            provider_config = config.get("kimi", {})
            self.api_key = api_key or provider_config.get("api_key", "")
            self.model = model or provider_config.get("model", "moonshot-v1-128k")
            self.base_url = provider_config.get("base_url", "https://api.moonshot.cn/v1")
            
            if not self.api_key:
                # Fallback to env var for backward compatibility
                self.api_key = os.environ.get("KIMI_API_KEY", "")
            
            if not self.api_key:
                raise ValueError(
                    "Kimi API key required. Set it in config.json or pass api_key parameter."
                )
            
            # Initialize OpenAI client for Kimi
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
            
        elif self.provider == "anthropic":
            provider_config = config.get("anthropic", {})
            self.api_key = api_key or provider_config.get("api_key", "")
            self.model = model or provider_config.get("model", "claude-sonnet-4-6")
            
            if not self.api_key:
                # Fallback to env var for backward compatibility
                self.api_key = os.environ.get("ANTHROPIC_API_KEY", "")
            
            if not self.api_key:
                raise ValueError(
                    "Anthropic API key required. Set it in config.json or pass api_key parameter."
                )
            
            # Initialize Anthropic client
            self.client = Anthropic(api_key=self.api_key)
            
        else:
            raise ValueError(f"Unsupported provider: {self.provider}. Use 'kimi' or 'anthropic'.")
        
        self.translated_count = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self._lock = threading.Lock()
        self.failed_files: List[str] = []
        self.max_retries = config.get("max_retries", 5)
        self.retry_base_delay = config.get("retry_base_delay", 2)

    def extract_epub(self, epub_path: str, output_dir: str) -> Path:
        """
        Extract EPUB file to a directory.
        
        Args:
            epub_path: Path to input EPUB file
            output_dir: Directory to extract to
            
        Returns:
            Path to extracted directory
        """
        print(f"Extracting EPUB: {epub_path}")
        
        extract_path = Path(output_dir)
        extract_path.mkdir(parents=True, exist_ok=True)
        
        # EPUB is essentially a ZIP file
        with zipfile.ZipFile(epub_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        
        print(f"✓ Extracted to: {extract_path}")
        return extract_path
    
    def find_html_files(self, extract_path: Path) -> List[Path]:
        """
        Find all HTML/XHTML chapter files in the extracted EPUB.
        
        Args:
            extract_path: Path to extracted EPUB directory
            
        Returns:
            List of HTML file paths
        """
        html_files = []
        
        # Common patterns for chapter files
        for pattern in ["**/*.html", "**/*.xhtml", "**/*.htm"]:
            html_files.extend(extract_path.glob(pattern))
        
        # Filter out common non-chapter files
        html_files = [
            f for f in html_files 
            if not any(x in f.name.lower() for x in ['toc', 'nav', 'cover'])
        ]
        
        # Sort by path for consistent order
        html_files.sort()
        
        print(f"✓ Found {len(html_files)} HTML files to translate")
        return html_files
    
    def translate_text(self, text: str, context: str = "") -> str:
        """
        Translate English text to Chinese using configured API provider.
        
        Args:
            text: English text to translate
            context: Additional context about the content
            
        Returns:
            Translated Chinese text
        """
        if self.provider == "kimi":
            return self._translate_with_kimi(text, context)
        elif self.provider == "anthropic":
            return self._translate_with_anthropic(text, context)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def _translate_with_kimi(self, text: str, context: str = "") -> str:
        """
        Translate using Kimi API (OpenAI-compatible).
        
        Args:
            text: English text to translate
            context: Additional context about the content
            
        Returns:
            Translated Chinese text
        """
        prompt = f"""Translate the following English text to Simplified Chinese (简体中文).

Requirements:
- Provide high-quality literary translation with natural Chinese expression
- Maintain the tone, style, and nuance of the original text
- Preserve all HTML tags, attributes, and structure EXACTLY as they appear
- Adapt cultural references appropriately for Chinese readers
- Keep proper nouns and names consistent
- DO NOT add any preamble, explanation, or commentary
- Output ONLY the translated HTML content

Context: {context if context else "This is content from an English book being translated to Chinese."}

Input:
{text}

Output the translated Chinese HTML content:"""
        
        last_error: Exception = Exception("Unknown error")
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=1.0,  # Kimi requires temperature=1
                    max_tokens=16384
                )

                # Track token usage (thread-safe)
                with self._lock:
                    self.total_input_tokens += response.usage.prompt_tokens
                    self.total_output_tokens += response.usage.completion_tokens

                # Extract text from response
                translated = response.choices[0].message.content.strip()
                return translated

            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    delay = min(self.retry_base_delay * (2 ** attempt) + random.uniform(0, 1), 60)
                    print(f"\n  ⚠ Kimi API error (attempt {attempt + 1}/{self.max_retries}), "
                          f"retrying in {delay:.1f}s: {type(e).__name__}: {e}")
                    time.sleep(delay)

        print(f"✗ Translation failed after {self.max_retries} attempts: {last_error}")
        raise last_error
    
    def _translate_with_anthropic(self, text: str, context: str = "") -> str:
        """
        Translate using Anthropic Claude API.
        
        Args:
            text: English text to translate
            context: Additional context about the content
            
        Returns:
            Translated Chinese text
        """
        prompt = f"""<task>
Translate the following English text to Simplified Chinese (简体中文).

Requirements:
- Provide high-quality literary translation with natural Chinese expression
- Maintain the tone, style, and nuance of the original text
- Preserve all HTML tags, attributes, and structure EXACTLY as they appear
- Adapt cultural references appropriately for Chinese readers
- Keep proper nouns and names consistent
- DO NOT add any preamble, explanation, or commentary
- Output ONLY the translated HTML content
</task>

<context>
{context if context else "This is content from an English book being translated to Chinese."}
</context>

<input>
{text}
</input>

Output the translated Chinese HTML content:"""

        last_error: Exception = Exception("Unknown error")
        for attempt in range(self.max_retries):
            try:
                message = self.client.messages.create(
                    model=self.model,
                    max_tokens=16384,
                    temperature=0.3,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )

                # Track token usage (thread-safe)
                with self._lock:
                    self.total_input_tokens += message.usage.input_tokens
                    self.total_output_tokens += message.usage.output_tokens

                # Extract text from response
                translated = message.content[0].text.strip()
                return translated

            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    delay = min(self.retry_base_delay * (2 ** attempt) + random.uniform(0, 1), 60)
                    print(f"\n  ⚠ Anthropic API error (attempt {attempt + 1}/{self.max_retries}), "
                          f"retrying in {delay:.1f}s: {type(e).__name__}: {e}")
                    time.sleep(delay)

        print(f"✗ Translation failed after {self.max_retries} attempts: {last_error}")
        raise last_error
    
    def translate_html_file(self, html_path: Path, book_title: str = "") -> str:
        """
        Translate an HTML file while preserving structure.
        
        Args:
            html_path: Path to HTML file
            book_title: Title of the book for context
            
        Returns:
            Translated HTML content
        """
        # Read original HTML
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Parse to check if there's actual content to translate
        soup = BeautifulSoup(html_content, 'lxml')
        text_content = soup.get_text(strip=True)
        
        # Skip if minimal content (likely just a cover or blank page)
        if len(text_content) < 50:
            return html_content
        
        # Prepare context
        context = f"This is a chapter from the book '{book_title}'." if book_title else "This is a chapter from a book."
        
        # Translate the entire HTML
        translated_html = self.translate_text(html_content, context)
        
        with self._lock:
            self.translated_count += 1

        # Small delay to respect rate limits
        time.sleep(1)
        
        return translated_html

    def _load_checkpoint(self, checkpoint_file: Path) -> Set[str]:
        """Load set of already-translated file names from checkpoint."""
        if checkpoint_file.exists():
            try:
                with open(checkpoint_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                completed = set(data.get("completed", []))
                if completed:
                    print(f"✓ Resuming: {len(completed)} file(s) already translated")
                return completed
            except Exception:
                pass
        return set()

    def _save_checkpoint(self, checkpoint_file: Path, filename: str):
        """Thread-safe checkpoint save after each successfully translated file."""
        with self._lock:
            completed: Set[str] = set()
            if checkpoint_file.exists():
                try:
                    with open(checkpoint_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    completed = set(data.get("completed", []))
                except Exception:
                    pass
            completed.add(filename)
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump({"completed": list(completed)}, f, indent=2)

    def _translate_file_worker(
        self, html_file: Path, book_title: str, checkpoint_file: Path
    ) -> str:
        """Worker function for parallel translation of a single HTML file."""
        translated_html = self.translate_html_file(html_file, book_title)
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(translated_html)
        self._save_checkpoint(checkpoint_file, html_file.name)
        return html_file.name

    def update_metadata(self, extract_path: Path, target_lang: str = "zh"):
        """
        Update EPUB metadata to reflect Chinese translation.
        
        Args:
            extract_path: Path to extracted EPUB directory
            target_lang: Target language code (default: zh for Chinese)
        """
        print("Updating metadata...")
        
        # Find and update content.opf or metadata file
        opf_files = list(extract_path.glob("**/*.opf"))
        
        for opf_file in opf_files:
            with open(opf_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Update language
            content = content.replace('lang="en"', f'lang="{target_lang}"')
            content = content.replace('<dc:language>en</dc:language>', f'<dc:language>{target_lang}</dc:language>')
            content = content.replace('<dc:language>en-US</dc:language>', f'<dc:language>{target_lang}</dc:language>')
            content = content.replace('<dc:language>en-GB</dc:language>', f'<dc:language>{target_lang}</dc:language>')
            
            with open(opf_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"  ✓ Updated: {opf_file.name}")
    
    def rebuild_epub(self, extract_path: Path, output_path: str) -> Path:
        """
        Rebuild EPUB from extracted directory.
        
        Args:
            extract_path: Path to extracted EPUB directory
            output_path: Path for output EPUB file
            
        Returns:
            Path to created EPUB file
        """
        print(f"Rebuilding EPUB: {output_path}")
        
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        
        # Create ZIP file (EPUB is a ZIP)
        with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as epub_zip:
            # First, add mimetype (must be uncompressed and first)
            mimetype_path = extract_path / "mimetype"
            if mimetype_path.exists():
                epub_zip.write(mimetype_path, "mimetype", compress_type=zipfile.ZIP_STORED)
            
            # Add all other files
            for file_path in extract_path.rglob("*"):
                if file_path.is_file() and file_path.name != "mimetype":
                    arcname = file_path.relative_to(extract_path)
                    epub_zip.write(file_path, arcname)
        
        print(f"✓ Created: {output}")
        return output
    
    def translate_epub(
        self,
        input_epub: str,
        output_epub: str,
        work_dir: str = "./epub_work",
        keep_work_dir: bool = False,
        workers: int = 3
    ) -> Dict:
        """
        Complete EPUB translation workflow.

        Args:
            input_epub: Path to input EPUB file
            output_epub: Path for output translated EPUB
            work_dir: Working directory for extraction (default: ./epub_work)
            keep_work_dir: Keep working directory after completion
            workers: Number of parallel translation workers (default: 3)

        Returns:
            Dictionary with translation statistics
        """
        start_time = time.time()
        work_path = Path(work_dir)
        
        try:
            print("\n" + "="*60)
            print(f"EPUB Translation: {Path(input_epub).name}")
            print(f"Provider: {self.provider}")
            print(f"Model: {self.model}")
            print(f"Workers: {workers}")
            print("="*60 + "\n")
            
            # Step 1: Extract EPUB (skip if resuming from existing checkpoint)
            checkpoint_file = work_path / ".translation_checkpoint.json"
            if work_path.exists() and checkpoint_file.exists():
                print(f"✓ Resuming from existing extraction: {work_path}")
                extract_path = work_path
            else:
                extract_path = self.extract_epub(input_epub, work_dir)
            
            # Step 2: Find HTML files
            html_files = self.find_html_files(extract_path)
            
            if not html_files:
                raise ValueError("No HTML chapter files found in EPUB")
            
            # Step 3: Translate each HTML file (parallel with checkpoint/resume)
            book_title = Path(input_epub).stem

            completed = self._load_checkpoint(checkpoint_file)
            remaining_files = [f for f in html_files if f.name not in completed]
            skipped = len(html_files) - len(remaining_files)
            if skipped:
                print(f"  ↩ Skipping {skipped} already-translated file(s)")

            print(f"\nTranslating {len(remaining_files)} file(s) with {workers} worker(s)...\n")

            # Parallel translation with progress bar
            with tqdm(
                total=len(html_files), initial=skipped,
                desc="Translation Progress", unit="file"
            ) as pbar:
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    futures = {
                        executor.submit(
                            self._translate_file_worker, f, book_title, checkpoint_file
                        ): f
                        for f in remaining_files
                    }
                    for future in as_completed(futures):
                        html_file = futures[future]
                        try:
                            filename = future.result()
                            pbar.set_description(f"✓ {filename}")
                        except Exception as e:
                            pbar.set_description(f"✗ {html_file.name}")
                            tqdm.write(f"  ✗ Failed: {html_file.name}: {e}")
                            with self._lock:
                                self.failed_files.append(html_file.name)
                        pbar.update(1)
            
            # Step 4: Update metadata
            print()
            self.update_metadata(extract_path)
            
            # Step 5: Rebuild EPUB
            print()
            output_path = self.rebuild_epub(extract_path, output_epub)
            
            # Calculate statistics
            duration = time.time() - start_time
            
            # Estimate cost based on provider and model
            if self.provider == "kimi":
                # Kimi pricing (approximate - check official pricing)
                # moonshot-v1-128k: ¥12/M input tokens, ¥12/M output tokens
                # Convert to USD (approximate: 1 USD = 7 CNY)
                input_cost = self.total_input_tokens * 0.000012 / 7  # ~$0.0017/M tokens
                output_cost = self.total_output_tokens * 0.000012 / 7  # ~$0.0017/M tokens
            elif self.provider == "anthropic":
                # Anthropic pricing
                if "opus" in self.model:
                    input_cost = self.total_input_tokens * 0.000015  # $15/M tokens
                    output_cost = self.total_output_tokens * 0.000075  # $75/M tokens
                else:  # sonnet
                    input_cost = self.total_input_tokens * 0.000003  # $3/M tokens
                    output_cost = self.total_output_tokens * 0.000015  # $15/M tokens
            else:
                input_cost = 0
                output_cost = 0
            
            total_cost = input_cost + output_cost
            
            stats = {
                "input_file": input_epub,
                "output_file": str(output_path),
                "provider": self.provider,
                "model": self.model,
                "workers": workers,
                "files_translated": self.translated_count,
                "total_files": len(html_files),
                "failed_files": self.failed_files,
                "input_tokens": self.total_input_tokens,
                "output_tokens": self.total_output_tokens,
                "estimated_cost_usd": round(total_cost, 2),
                "duration_seconds": round(duration, 1)
            }
            
            # Cleanup
            if work_path.exists():
                if self.failed_files:
                    print(f"⚠ Keeping working directory: {len(self.failed_files)} file(s) failed")
                    print(f"  Re-run the same command to resume translation")
                elif not keep_work_dir:
                    shutil.rmtree(work_path)
                    print(f"✓ Cleaned up working directory")
            
            # Print summary
            print("\n" + "="*60)
            print("TRANSLATION COMPLETE")
            print("="*60)
            print(f"Output file: {output_path}")
            print(f"Files translated: {self.translated_count}/{len(html_files)}")
            if self.failed_files:
                print(f"Failed files ({len(self.failed_files)}):")
                for fn in self.failed_files:
                    print(f"  - {fn}")
            print(f"Input tokens: {self.total_input_tokens:,}")
            print(f"Output tokens: {self.total_output_tokens:,}")
            print(f"Estimated cost: ${total_cost:.2f} USD")
            print(f"Duration: {duration/60:.1f} minutes")
            print("="*60 + "\n")
            
            return stats
            
        except Exception as e:
            print(f"\n✗ Error during translation: {e}")
            raise
        

def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Translate English EPUB books to Chinese using AI (Kimi or Claude)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Translate using config.json settings (default: Kimi)
  python translate_epub.py input.epub output_chinese.epub
  
  # Use Anthropic Claude instead
  python translate_epub.py input.epub output.epub --provider anthropic
  
  # Override model from config
  python translate_epub.py input.epub output.epub --model moonshot-v1-128k
  
  # Keep working directory for inspection
  python translate_epub.py input.epub output.epub --keep-work-dir
        """
    )
    
    parser.add_argument(
        "input_epub",
        help="Path to input English EPUB file"
    )
    
    parser.add_argument(
        "output_epub",
        help="Path for output Chinese EPUB file"
    )
    
    parser.add_argument(
        "--config",
        default="config.json",
        help="Path to config file (default: config.json)"
    )
    
    parser.add_argument(
        "--provider",
        choices=["kimi", "anthropic"],
        help="API provider to use (overrides config file)"
    )
    
    parser.add_argument(
        "--model",
        help="Model to use (overrides config file)"
    )
    
    parser.add_argument(
        "--api-key",
        help="API key (overrides config file)"
    )
    
    parser.add_argument(
        "--work-dir",
        default="./epub_work",
        help="Working directory for extraction (default: ./epub_work)"
    )
    
    parser.add_argument(
        "--keep-work-dir",
        action="store_true",
        help="Keep working directory after completion"
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=3,
        help="Number of parallel translation workers (default: 3)"
    )

    args = parser.parse_args()
    
    # Validate input file
    if not Path(args.input_epub).exists():
        print(f"Error: Input file not found: {args.input_epub}")
        sys.exit(1)
    
    # Load config
    config = load_config(args.config)
    
    # Create translator and run
    try:
        translator = EPUBTranslator(
            config=config,
            api_key=args.api_key,
            model=args.model,
            provider=args.provider
        )
        
        stats = translator.translate_epub(
            input_epub=args.input_epub,
            output_epub=args.output_epub,
            work_dir=args.work_dir,
            keep_work_dir=args.keep_work_dir,
            workers=args.workers
        )
        
        # Save statistics
        stats_file = Path(args.output_epub).with_suffix(".stats.json")
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        print(f"Statistics saved to: {stats_file}")
        
    except KeyboardInterrupt:
        print("\n\nTranslation interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
