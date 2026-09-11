#!/usr/bin/env python
"""Print missing lines/branches for a module from a coverage JSON file."""
import json
import sys

rel = sys.argv[1]
json_path = sys.argv[2] if len(sys.argv) > 2 else "/tmp/full.json"
d = json.load(open(json_path))
f = f"src/helixlang/{rel}"
m = d["files"][f]
print(f"=== {rel}  ({m['summary']['percent_covered']}%  "
      f"L{len(m['missing_lines'])} / B{len(m['missing_branches'])}) ===")
with open(f) as fh:
    lines = fh.readlines()
for ln in m["missing_lines"]:
    print(f"  L {ln}: {lines[ln - 1].rstrip()}")
for src, tgt in m["missing_branches"]:
    print(f"  B {src}->{tgt}")