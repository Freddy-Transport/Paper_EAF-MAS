import json
import shutil
import tarfile
import time
from pathlib import Path


RUN_ROOT = Path("autotemp/ccfa_formal_rolling_20260607_115014")
manifest_path = RUN_ROOT / "reports/cleanup_manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
archive_dir = RUN_ROOT / "archive"
archive_dir.mkdir(parents=True, exist_ok=True)
archive_path = archive_dir / f"legacy_code_{time.strftime('%Y%m%d_%H%M%S')}.tar.gz"

archived = []
with tarfile.open(archive_path, "w:gz") as tar:
    for item in manifest.get("legacy_code", []):
        p = Path(item)
        if p.exists():
            tar.add(p, arcname=item)
            archived.append(item)

deleted = []
skipped = []
errors = []
for item in manifest.get("delete", []):
    p = Path(item)
    if not p.exists():
        skipped.append(item)
        continue
    try:
        if p.resolve() == RUN_ROOT.resolve():
            skipped.append(item)
            continue
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()
        deleted.append(item)
    except Exception as exc:
        errors.append({"path": item, "error": f"{type(exc).__name__}: {exc}"})

payload = {
    "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "archive_path": str(archive_path),
    "archived_legacy_code": archived,
    "deleted": deleted,
    "skipped_missing": skipped,
    "errors": errors,
    "manifest_path": str(manifest_path),
}
(RUN_ROOT / "reports/cleanup_execution.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"archive": str(archive_path), "archived": len(archived), "deleted": len(deleted), "errors": len(errors)}, ensure_ascii=False, indent=2))
if errors:
    raise SystemExit(1)
