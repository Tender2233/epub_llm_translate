#!/usr/bin/env python3
"""
EPUB Translator - Translate English EPUB books to Chinese using LLM APIs.

Two-phase pipeline:
  1) Pre-analysis: chapter summaries + book summary + unified glossary
     (runs on a cheaper analysis model)
  2) Translation: parallel chapters / sequential chunks, with book summary,
     chapter summary and glossary injected into the system prompt.
"""

import argparse
import json
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Dict

from epub_translator.config import load_config
from epub_translator.extractor import EPubExtractor, rebuild_epub, update_metadata
from epub_translator.glossary import Glossary
from epub_translator.llm_client import LLMClient
from epub_translator.summarizer import BookAnalyzer
from epub_translator.translator import TranslationPipeline, estimate_cost
from epub_translator.vision import ImageTranslator


def parse_args():
    parser = argparse.ArgumentParser(
        description="Translate English EPUB books to Chinese using LLM APIs (OpenAI-compatible / Anthropic)."
    )
    parser.add_argument("input_epub", help="Path to input English EPUB file")
    parser.add_argument("output_epub", help="Path for output translated EPUB file")
    parser.add_argument("--config", default="config.json", help="Path to config file (default: config.json)")
    parser.add_argument("--provider", choices=["openai", "anthropic"], help="API provider (overrides config)")
    parser.add_argument("--model", help="Translation model (overrides config)")
    parser.add_argument("--analysis-model", help="Model used for pre-analysis/glossary (overrides config)")
    parser.add_argument("--vision-model", help="Vision model used for the image pass (overrides config)")
    parser.add_argument("--api-key", help="API key (overrides config)")
    parser.add_argument("--work-dir", help="Working directory for extraction (default from config)")
    parser.add_argument("--workers", type=int, help="Number of parallel translation workers")
    parser.add_argument("--keep-work-dir", action="store_true", help="Keep working directory after completion")
    parser.add_argument(
        "--translate-images",
        action="store_true",
        help="Enable the multimodal pass: OCR and translate text inside images",
    )
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Skip pre-analysis; reuse existing summary.json/glossary.json in the work directory",
    )
    parser.add_argument(
        "--reanalyze",
        action="store_true",
        help="Force re-running pre-analysis even if summary.json/glossary.json exist",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    provider = args.provider or config.get("api_provider", "openai")
    tcfg = config.get("translation", {})
    pcfg = config.get("processing", {})
    precfg = config.get("pre_analysis", {})

    workers = args.workers or tcfg.get("workers", 3)
    work_dir = args.work_dir or pcfg.get("work_directory", "./epub_work")
    keep_work = args.keep_work_dir or bool(pcfg.get("keep_work_directory", False))

    llm = LLMClient(
        provider, config, api_key=args.api_key, model=args.model, analysis_model=args.analysis_model
    )

    start_time = time.time()
    work_path = Path(work_dir)
    checkpoint_file = work_path / ".translation_checkpoint.json"

    print("\n" + "=" * 60)
    print(f"EPUB Translation: {Path(args.input_epub).name}")
    print(f"Provider: {provider} | Model: {llm.model} | Analysis model: {llm.analysis_model}")
    print(f"Workers: {workers}")
    print("=" * 60 + "\n")

    # Step 1: Extract EPUB (or resume from an existing work directory)
    if work_path.exists() and checkpoint_file.exists():
        print(f"✓ Resuming from existing extraction: {work_path}")
    else:
        if work_path.exists():
            shutil.rmtree(work_path)
        EPubExtractor.extract_epub(args.input_epub, work_dir)

    extractor = EPubExtractor(work_path)
    chapters = extractor.chapters(include_outside_spine=bool(pcfg.get("include_outside_spine", False)))
    if not chapters:
        raise ValueError("No HTML chapter files found in EPUB")
    print(f"✓ Found {len(chapters)} chapter file(s) in spine order")

    # Step 2: Pre-analysis (book summary + chapter summaries + glossary)
    summary_path = work_path / "summary.json"
    glossary_path = work_path / "glossary.json"
    glossary = Glossary(glossary_path)
    summary = {}

    if not precfg.get("enabled", True) or args.skip_analysis:
        if not summary_path.exists():
            raise ValueError(
                "--skip-analysis requires existing summary.json/glossary.json. "
                "Run once without this flag to generate them."
            )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        print(f"✓ Pre-analysis skipped; reusing saved summary/glossary ({len(glossary)} terms)")
    elif args.reanalyze or not summary_path.exists() or not glossary_path.exists():
        print("Running pre-analysis: chapter summaries, book summary, unified glossary ...")
        analyzer = BookAnalyzer(llm, config)
        result = analyzer.analyze(chapters, extractor)
        summary = {
            "book": result["book_summary"],
            "chapters": result["chapter_summaries"],
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        }
        glossary.replace_terms(result["glossary_terms"])
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        glossary.save()
        print(
            f"✓ Pre-analysis done: {len(result['chapter_summaries'])} chapter summaries, "
            f"{len(glossary)} glossary terms"
        )
    else:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        print(
            f"✓ Reusing saved pre-analysis: {len(summary.get('chapters', {}))} chapter summaries, "
            f"{len(glossary)} glossary terms"
        )

    # Step 3: Translation
    pipeline = TranslationPipeline(llm, config, extractor, glossary, summary, work_path, checkpoint_file)
    result = pipeline.translate(chapters, workers)

    # Step 4 (optional): Multimodal pass — OCR + translate text inside images
    vision_result: Dict = {}
    if args.translate_images or config.get("multimodal", {}).get("enabled"):
        if args.vision_model:
            config.setdefault("multimodal", {})["vision_model"] = args.vision_model
            llm.vision_model = args.vision_model  # CLI wins over every config source
        print()
        print(f"Running image translation with vision model: {llm.vision_model} ...")
        vision_result = ImageTranslator(llm, config, work_path, extractor).translate(chapters)

    # Step 5: Update metadata and rebuild the EPUB
    print()
    update_metadata(work_path)
    print()
    output_path = rebuild_epub(work_path, args.output_epub)

    duration = time.time() - start_time
    translation_cost = estimate_cost(provider, llm.model, llm.total_input_tokens, llm.total_output_tokens)
    analysis_cost = estimate_cost(
        provider, llm.analysis_model, llm.analysis_input_tokens, llm.analysis_output_tokens
    )
    vision_cost = estimate_cost(
        provider, llm.vision_model, llm.vision_input_tokens, llm.vision_output_tokens
    )
    total_cost = translation_cost + analysis_cost + vision_cost

    stats = {
        "input_file": args.input_epub,
        "output_file": str(output_path),
        "provider": provider,
        "model": llm.model,
        "analysis_model": llm.analysis_model,
        "workers": workers,
        "chapters_total": result["chapters_total"],
        "chapters_translated": result["translated"],
        "chapters_skipped": result["skipped"],
        "failed_files": result["failed"],
        "chunks_translated": result["chunks_translated"],
        "glossary_terms": len(glossary),
        "translation_input_tokens": llm.total_input_tokens,
        "translation_output_tokens": llm.total_output_tokens,
        "analysis_input_tokens": llm.analysis_input_tokens,
        "analysis_output_tokens": llm.analysis_output_tokens,
        "vision_model": llm.vision_model if vision_result else None,
        "vision_input_tokens": llm.vision_input_tokens,
        "vision_output_tokens": llm.vision_output_tokens,
        "images_translated": vision_result.get("images_translated", 0) if vision_result else 0,
        "images_skipped": vision_result.get("images_skipped", 0) if vision_result else 0,
        "chapters_updated_by_vision": vision_result.get("chapters_updated", 0) if vision_result else 0,
        "estimated_translation_cost_usd": round(translation_cost, 2),
        "estimated_analysis_cost_usd": round(analysis_cost, 2),
        "estimated_vision_cost_usd": round(vision_cost, 2),
        "estimated_total_cost_usd": round(total_cost, 2),
        "duration_seconds": round(duration, 1),
    }
    stats_path = Path(str(output_path) + ".stats.json")
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    # Cleanup
    if work_path.exists():
        if result["failed"]:
            print(f"⚠ Keeping working directory: {len(result['failed'])} chapter(s) failed")
            print("  Re-run the same command to resume translation")
        elif not keep_work:
            shutil.rmtree(work_path)
            print("✓ Cleaned up working directory")

    # Summary
    print("\n" + "=" * 60)
    print("TRANSLATION COMPLETE")
    print("=" * 60)
    print(f"Output file: {output_path}")
    print(f"Stats file: {stats_path}")
    print(
        f"Chapters: {result['translated']} translated / {result['skipped']} skipped "
        f"/ {len(result['failed'])} failed (total {result['chapters_total']})"
    )
    print(f"Chunks translated: {result['chunks_translated']}")
    if result["failed"]:
        print("Failed chapters:")
        for name in result["failed"]:
            print(f"  - {name}")
    print(f"Glossary terms: {len(glossary)}")
    print(f"Translation tokens: {llm.total_input_tokens:,} in / {llm.total_output_tokens:,} out")
    print(f"Analysis tokens: {llm.analysis_input_tokens:,} in / {llm.analysis_output_tokens:,} out")
    print(
        f"Estimated cost: ${translation_cost:.2f} (translation) + "
        f"${analysis_cost:.2f} (analysis) = ${total_cost:.2f} USD"
    )
    print(f"Duration: {duration / 60:.1f} minutes")
    print("=" * 60 + "\n")

    return stats


if __name__ == "__main__":
    main()

