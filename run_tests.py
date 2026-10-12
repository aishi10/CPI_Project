"""Run every part's unit tests:  python3 run_tests.py"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
failed = False
for part, test in (("part1", "test_memory_store.py"), ("part2", "test_evidence.py"), ("part3", "test_revision_engine.py")):
    print(f"=== {part} ===", flush=True)
    r = subprocess.run([sys.executable, "-m", "unittest", test], cwd=os.path.join(ROOT, part))
    failed |= r.returncode != 0
print("=== integration (parts 1+2+3) ===", flush=True)
r = subprocess.run([sys.executable, "-m", "unittest", "test_integration.py"], cwd=ROOT)
failed |= r.returncode != 0
print("\nALL TESTS PASSED" if not failed else "\nSOME TESTS FAILED")
sys.exit(1 if failed else 0)
