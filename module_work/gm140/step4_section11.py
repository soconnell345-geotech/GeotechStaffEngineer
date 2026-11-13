"""GM-140 step 4: run the published validation set of the account's
section 11 (every ``validation_examples`` file that uses slope_stability,
and the benchmark files of ``slope_stability/tests``) and print, for each
test, whether its pinned value still holds and, where it does not, the
value obtained beside the value pinned.

    python module_work/gm140/step4_section11.py [out.txt]

Run on master it is the "before" column (every pin holds); run on the
branch, the lines marked MOVED are the values step 4 moves.
"""
import glob
import re
import subprocess
import sys

FILES = sorted(
    f for f in glob.glob("validation_examples/test_published_*.py")
    if "slope_stability" in open(f, encoding="utf-8").read())
FILES += [
    "slope_stability/tests/test_slope_stability.py",
    "slope_stability/tests/test_duncan_verification.py",
    "slope_stability/tests/test_search_supplements.py",
    "slope_stability/tests/test_search_reinforcement.py",
    "slope_stability/tests/test_m_alpha.py",
    "slope_stability/tests/test_circle_slicing.py",
    "slope_stability/tests/test_rapid_drawdown.py",
    "slope_stability/tests/test_usability_features.py",
]


def main(out=None):
    import os
    files = [f for f in FILES if os.path.exists(f)]
    p = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:warnings",
         "-p", "no:cacheprovider", "-n", "auto", "-rA", "--tb=short",
         *files], capture_output=True, text=True)
    text = p.stdout
    lines = []
    for m in re.finditer(r"^(PASSED|FAILED|XFAIL|XPASS|SKIPPED) (\S+)(.*)$",
                         text, re.M):
        lines.append(f"{m.group(1):8s}{m.group(2)}{m.group(3)[:150]}")
    for m in re.finditer(r"^_+ (\S+) _+$(.*?)(?=^_+ \S+ _+$|^=+ )", text,
                         re.M | re.S):
        got = re.findall(r"^E\s+(?:assert |Obtained: |Expected: ).*$",
                         m.group(2), re.M)
        if got:
            lines.append(f"MOVED   {m.group(1)}: "
                         + " | ".join(g[1:].strip()[:110] for g in got[:3]))
    lines.append(text.strip().splitlines()[-1])
    report = "\n".join(lines)
    print(report)
    if out:
        open(out, "w", encoding="utf-8").write(report + "\n")
    return p.returncode


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else None))
