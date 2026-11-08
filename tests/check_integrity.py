"""Source integrity checks; does not run imaging or scientific models."""
from pathlib import Path
import ast
import csv
import hashlib
import json

root = Path(__file__).resolve().parents[1]
with (root / 'provenance/code_manifest.tsv').open() as f:
    rows = list(csv.DictReader(f, delimiter='\t'))
for row in rows:
    path = root / row['repository_path']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256'], str(path)
    if path.suffix == '.py':
        ast.parse(path.read_text(), filename=str(path))
record = json.loads((root / 'provenance/fa_followup_record.json').read_text())
for name in ['run_fa_severity.py', 'run_fa_followup.py']:
    path = root / 'analysis/diffusion' / name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == record['SHA256'][name]
severity = (root / 'analysis/diffusion/run_fa_severity.py').read_text()
assert '"yes"' in severity, 'Eligibility YES fix is missing'
print(f'Passed: {len(rows)} source hashes, Python syntax and two recorded FA hashes.')
