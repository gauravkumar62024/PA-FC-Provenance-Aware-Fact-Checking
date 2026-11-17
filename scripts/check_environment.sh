#!/usr/bin/env bash
# Verify the environment matches what PA-FC was validated on.
set -uo pipefail
PY="${PYTHON:-python}"
echo "== PA-FC environment check =="
$PY - <<'PYEOF'
import importlib, sys
EXPECT = {"torch":"2.12.0","transformers":"5.6.0","peft":"0.18.1",
          "bitsandbytes":"0.49.2","accelerate":"1.11.0","datasets":"4.0.0",
          "trl":"0.24.0","sklearn":"1.3.0","tldextract":"5.3.1"}
print("python:", sys.version.split()[0], "(validated: 3.11.4 train / 3.10.17 features)")
bad = 0
for m, want in EXPECT.items():
    try:
        got = getattr(importlib.import_module(m), "__version__", "?")
        flag = "ok " if got == want else "DIFF"
        if got != want: bad += 1
        print(f"  {flag} {m:<16} have={got:<12} validated={want}")
    except Exception:
        print(f"  MISS {m:<16} not installed"); bad += 1
try:
    import torch
    print("  cuda:", torch.cuda.is_available(), "| torch.version.cuda:", torch.version.cuda)
    if torch.cuda.is_available(): print("  device:", torch.cuda.get_device_name(0))
except Exception: pass
print(f"\n{bad} package(s) differ from the validated set."
      "\nDifferences are not necessarily fatal, but tldextract affects F2 domain"
      "\nextraction and transformers/peft affect training numerics.")
PYEOF
