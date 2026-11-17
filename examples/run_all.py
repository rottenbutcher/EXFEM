"""Run every example deck and script (smoke test). Usage: python examples/run_all.py"""
import os
import subprocess
import sys
import time

here = os.path.dirname(os.path.abspath(__file__))
decks = ["spring_mass_modal.inp", "bracket.inp", "bracket.bdf", "plate_hole.inp", "panel.inp", "panel.bdf", "contact.inp"]
scripts = ["shells_and_beams.py", "spectrum_random_fatigue.py", "friction_bolt_remote.py", "postbuckling.py",
           "multi_load_cases.py"]
failed = []
for d in decks:
    t = time.time()
    r = subprocess.run([sys.executable, "-m", "gfem", "run", os.path.join(here, "decks", d), "-o", os.path.join(here, "out")],
                       capture_output=True, text=True)
    ok = r.returncode == 0
    print(f"{'OK  ' if ok else 'FAIL'} deck   {d:<28s} {time.time() - t:6.1f}s  " + (r.stdout.strip().splitlines() or [""])[-1])
    failed += [] if ok else [d]
for s in scripts:
    t = time.time()
    r = subprocess.run([sys.executable, os.path.join(here, s)], capture_output=True, text=True, cwd=here)
    ok = r.returncode == 0
    print(f"{'OK  ' if ok else 'FAIL'} script {s:<28s} {time.time() - t:6.1f}s")
    for line in r.stdout.strip().splitlines()[-3:]:
        print("       " + line)
    if not ok:
        print(r.stderr[-1500:])
        failed.append(s)
print("all examples ran" if not failed else f"FAILED: {failed}")
sys.exit(1 if failed else 0)
