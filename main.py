#!/usr/bin/env python3
"""
ML Research Agent — Entry Point

Usage:
  python main.py --goal "Compare SGD vs Adam on MNIST" --model llama3
  python main.py --goal "Find optimal learning rate for ResNet" --gpu
  python main.py --goal "Benchmark decision trees vs neural nets on tabular data"
"""

import argparse
import logging
import sys
import yaml
from pathlib import Path


def load_config(path: str = "config/settings.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("logs/agent.log"),
        ],
    )


def main():
    parser = argparse.ArgumentParser(description="ML Research Agent")
    parser.add_argument("--goal", required=True, help="Research question or hypothesis")
    parser.add_argument("--model", default=None, help="Override LLM model name")
    parser.add_argument("--backend", default=None, help="Override LLM backend (ollama|lmstudio|vllm)")
    parser.add_argument("--config", default="config/settings.yaml", help="Config file path")
    parser.add_argument("--max-iter", type=int, default=None, help="Max experiment iterations")
    parser.add_argument("--gpu", action="store_true", help="Enable GPU in Docker")
    parser.add_argument("--template", default=None, help="Paper template (ieee|neurips|icml|plain)")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    # Load & override config
    Path("logs").mkdir(exist_ok=True)
    setup_logging(args.log_level)

    config = load_config(args.config)

    if args.model:
        config["llm"]["model"] = args.model
    if args.backend:
        config["llm"]["backend"] = args.backend
    if args.max_iter:
        config["experiment"]["max_iterations"] = args.max_iter
    if args.gpu:
        config["docker"]["gpu"] = True
    if args.template:
        config["paper"]["template"] = args.template

    # Lazy import after config is ready
    from llm.client import LLMClient
    from core.agent import ResearchAgent

    logger = logging.getLogger("main")

    # Health check
    llm = LLMClient(config)
    if not llm.health_check():
        logger.error(
            f"Cannot reach LLM at {config['llm']['base_url']} — "
            f"is {config['llm']['backend']} running?"
        )
        sys.exit(1)

    logger.info(f"✅ LLM connected: {config['llm']['backend']} / {config['llm']['model']}")

    tg = config.get("telegram", {})
    if tg.get("bot_token") and tg.get("chat_id"):
        logger.info("📱 Telegram aktiviert — Kommunikation via Bot")
    else:
        logger.info("💻 Terminal-Modus — Kommunikation via Tastatur")

    agent = ResearchAgent(config)
    pdf_path = agent.run(goal=args.goal)

    if pdf_path is None:
        print(f"\n{'='*60}")
        print(f"  💾 Session durch Timeout gespeichert.")
        print(f"  Checkpoint laden: starte neu und antworte mit 'ja'")
        print(f"{'='*60}")
    else:
        print(f"\n{'='*60}")
        print(f"  ✅ Research complete!")
        print(f"  📄 Paper: {pdf_path}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
