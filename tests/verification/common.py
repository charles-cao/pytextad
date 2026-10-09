import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.environ.get("PYTEXTAD_REF", os.path.join(HERE, "reference"))
DATA = os.path.join(REF, "data")
PKG_PARENT = os.path.abspath(os.path.join(HERE, "..", ".."))          # repository root
sys.path.insert(0, PKG_PARENT)

def report(name, ok, detail):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    return ok
