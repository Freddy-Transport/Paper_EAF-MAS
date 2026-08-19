"""Bootstrap official baseline repositories without running experiments.

Default behavior is dry-run: it prints clone commands and verifies local checkout
status. Use ``--clone`` only when the user explicitly wants to download the
official repositories into ``baseline/external``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from official_adapters import git_head, iter_sources


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--registry", default="official_sources.json")
    p.add_argument("--external_root", default="external")
    p.add_argument("--depth", default="1")
    p.add_argument("--clone", action="store_true", help="Actually run git clone for missing repositories.")
    p.add_argument("--manifest", default="official_bootstrap_manifest.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    external_root = Path(args.external_root)
    rows = []
    if args.clone:
        external_root.mkdir(parents=True, exist_ok=True)
    for source in iter_sources(args.registry):
        local_path = Path(source.local_path)
        cmd = ["git", "clone", "--depth", str(args.depth), source.official_repo, str(local_path)]
        if local_path.exists():
            status = "present"
        elif args.clone:
            subprocess.check_call(cmd)
            status = "cloned"
        else:
            status = "dry_run_missing"
        rows.append(
            {
                "source_id": source.source_id,
                "models": source.models,
                "official_repo": source.official_repo,
                "local_path": str(local_path),
                "status": status,
                "clone_command": " ".join(cmd),
                "resolved_commit": git_head(local_path),
            }
        )
    with open(args.manifest, "w", encoding="utf-8") as f:
        json.dump({"clone_executed": bool(args.clone), "sources": rows}, f, indent=2)
    print(json.dumps({"clone_executed": bool(args.clone), "sources": rows}, indent=2))


if __name__ == "__main__":
    main()
