"""Integrity only: verify the curated RQ3 export without recomputing results."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def verify():
    failures = []
    manifest = json.loads((ROOT / 'rq3_publication_manifest.json').read_text())
    for relative, expected in manifest['files'].items():
        path = ROOT / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            failures.append('RQ3 artifact missing or changed: ' + relative)
            continue
        if path.suffix == '.csv':
            with path.open(newline='') as stream:
                header = next(csv.reader(stream))
            if {'description', 'prompt', 'body', 'text', 'diff', 'file_contents'} & set(header):
                failures.append('Unexpected raw-text column: ' + relative)
    for relative in manifest['omitted_derived_caches']:
        if (ROOT / relative).exists():
            failures.append('Excluded derived cache is present: ' + relative)
    for name, expected in manifest['source_completion_hashes'].items():
        base = ROOT / 'phase3/analysis/outputs' / name
        path = base / 'completion.json'
        if not path.is_file():
            continue  # Already reported through the file manifest.
        completion = json.loads(path.read_text())
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            failures.append('Source completion manifest changed: ' + name)
        if completion['status'] != 'complete_and_verified':
            failures.append('Incomplete source run: ' + name)
        for relative, digest in completion['output_hashes'].items():
            key = str(Path('phase3/analysis/outputs') / name / relative)
            recorded = manifest['files'].get(key, manifest['omitted_derived_caches'].get(key))
            if recorded != digest:
                failures.append('Unaccounted source artifact: ' + key)
    return failures


if __name__ == '__main__':
    errors = verify()
    print(json.dumps({'status': 'failed' if errors else 'verified', 'failures': errors}, indent=2))
    raise SystemExit(bool(errors))
