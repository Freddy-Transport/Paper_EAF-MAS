#!/usr/bin/env python3
"""Check the Git index before publication; inspect content, not just .gitignore."""
from pathlib import Path
import re
import subprocess
import sys

ALLOWED_ROOTS = {"agents", "baseline", "event_post_training", "event_screening", "experiments",
                 "momentfm", "tests", "scripts", "licenses"}
ALLOWED_FILES = {".gitignore", "baseline/.gitignore", "licenses/MOMENT_LICENSE", ".gitattributes", ".env.example", "README.md", "ARTIFACTS.md",
                 "FORMAL_INTERFACE.md", "THIRD_PARTY_NOTICES.md", "requirements.txt", "pyproject.toml", "setup.py"}
ALLOWED_SUFFIXES = {".py", ".md", ".txt", ".toml", ".yaml", ".yml", ".sh", ".cfg", ".ini"}
ALLOWED_JSON = {"baseline/config.json", "baseline/official_sources.json"}
PRIVATE_PARTS = {"legacy", "private_runs", "outputs", "results", "data", "autotemp", "reports", "__pycache__", ".git"}


def main():
    paths = subprocess.check_output(["git", "ls-files", "-z"]).decode().split("\0")
    violations = []
    for name in filter(None, paths):
        p = Path(name)
        private_parts = set(p.parts) & PRIVATE_PARTS
        if name.startswith("momentfm/data/") and p.suffix == ".py":
            private_parts.discard("data")
        if private_parts:
            violations.append((name,"private path")); continue
        if name not in ALLOWED_FILES and (p.parts[0] not in ALLOWED_ROOTS or
                (p.suffix not in ALLOWED_SUFFIXES and name not in ALLOWED_JSON)):
            violations.append((name,"outside release allowlist")); continue
        content = subprocess.check_output(["git", "show", ":"+name])
        if b"\0" in content:
            violations.append((name,"binary content")); continue
        text = content.decode("utf-8",errors="strict")
        if re.search(r"-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----",text):
            violations.append((name,"private key"))
        if re.search(r"(?:ghp_|github_pat_|sk-)[A-Za-z0-9_]{24,}",text):
            violations.append((name,"possible credential"))
    for name,reason in violations: print(name+": "+reason)
    print("Checked staged public tree; violations:",len(violations))
    return bool(violations)


if __name__=="__main__": sys.exit(main())
