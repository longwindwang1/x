"""Command-line entry point: ``parley serve`` / ``parley characters``."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="parley", description="Parley voice NPC runtime")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="Run the runtime server")
    serve.add_argument("--config", "-c", default=None, help="Path to server.yaml")
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=int, default=None)

    chars = sub.add_parser("characters", help="Character card utilities")
    chars_sub = chars.add_subparsers(dest="chars_command", required=True)
    validate = chars_sub.add_parser("validate", help="Validate all cards in a directory")
    validate.add_argument("directory", nargs="?", default="characters")

    args = parser.parse_args(argv)

    if args.command == "serve":
        return _serve(args)
    if args.command == "characters" and args.chars_command == "validate":
        return _validate(args.directory)
    return 2


def _serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .server import create_app

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    config_path = args.config
    if config_path is None and Path("server.yaml").is_file():
        config_path = "server.yaml"
    cfg = load_config(config_path)
    if args.host:
        cfg.host = args.host
    if args.port:
        cfg.port = args.port
    uvicorn.run(create_app(cfg), host=cfg.host, port=cfg.port)
    return 0


def _validate(directory: str) -> int:
    from .characters import CharacterRegistry, build_system_prompt

    try:
        registry = CharacterRegistry(directory)
    except Exception as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1
    cards = registry.all()
    if not cards:
        print(f"No character cards found in {directory}/", file=sys.stderr)
        return 1
    for card in cards:
        prompt_chars = len(build_system_prompt(card))
        mode = card.model.mode
        print(f"  ok  {card.id:<20} {card.name:<28} mode={mode:<7} prompt={prompt_chars} chars")
    print(f"{len(cards)} card(s) valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
