#!/usr/bin/env bash
# Re-runs every equivalence check against the original implementations. Needs setup_reference.sh first.
set -uo pipefail
cd "$(dirname "$0")"
export PYTEXTAD_REF=${PYTEXTAD_REF:-$(pwd)/reference}
fail=0
"$PYTEXTAD_REF/py38/bin/python" date_ref_side.py | tail -1 && python date_check.py || fail=1
python cvdd_check.py || fail=1
for norm in L21 MSE; do
  python rsrae_ref_side.py 20 $norm 2>/dev/null | tail -1 && python rsrae_check.py 20 $norm || fail=1
done
python fate_check.py || fail=1
[ $fail -eq 0 ] && echo "ALL CHECKS PASSED" || { echo "SOME CHECKS FAILED"; exit 1; }
