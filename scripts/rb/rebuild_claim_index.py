from __future__ import annotations

import argparse

from src.config.load_config import load_app_config
from src.storage.reasoningbank_store import ReasoningBankStore


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild derived RB claim docs in Chroma.")
    parser.add_argument(
        "--active-only",
        action="store_true",
        help="Only rebuild claim docs for active items (skip archived).",
    )
    args = parser.parse_args()

    cfg = load_app_config()
    rb = ReasoningBankStore.from_config(cfg)

    processed = rb.rebuild_claim_index(include_archived=not bool(args.active_only))
    print(f"Rebuilt claim index for {processed} items.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

