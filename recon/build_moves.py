#!/usr/bin/env python3
"""Build a dataset of accepted optimization MOVES from git history.

Deterministic. Stdlib only; shells out to `git` via subprocess.

For every commit reachable from HEAD whose subject starts with
'Accept submission', emit one JSON object describing the move (the diff of
that commit versus its first parent). Output is JSONL, oldest commit first,
written to ecdsa-model/data/moves_raw.jsonl.
"""

import json
import os
import re
import subprocess
import sys

REPO = "/Users/dennison/develop/quantum-project/ecdsafail-challenge"
OUT = "/Users/dennison/develop/quantum-project/ecdsa-model/data/moves_raw.jsonl"

# Files that count as a "small high-signal move" when they are the only
# non-memory / non-md file touched.
SMALL_MOVE_FILES = ("src/point_add/mod.rs",)
SMALL_MOVE_SUFFIXES = ("dialog/config.rs",)  # e.g. src/point_add/rounds/dialog/config.rs

# Big arithmetic primitive files of interest.
BIG_ARITH_TOKENS = ("adder", "modular", "multiply", "const_arith")

SMALL_MOVE_LINE_LIMIT = 120


def git(*args):
    """Run a git command in the repo and return stdout (text, utf-8, errors replaced)."""
    res = subprocess.run(
        ["git", *args],
        cwd=REPO,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if res.returncode != 0:
        raise RuntimeError(
            "git %s failed (%d): %s"
            % (" ".join(args), res.returncode, res.stderr.decode("utf-8", "replace"))
        )
    return res.stdout.decode("utf-8", "replace")


def is_excluded(path):
    """True for memory-path or markdown files (not counted toward small-move test)."""
    if path.endswith(".md"):
        return True
    # any path component named 'memory'
    parts = path.split("/")
    if "memory" in parts:
        return True
    return False


def parse_numstat(sha):
    """Return list of [adds, dels, file]. Binary files report '-' which we keep as None."""
    out = git("show", "--numstat", "--format=", sha)
    rows = []
    for line in out.splitlines():
        if not line.strip():
            continue
        # numstat is tab-separated: adds<TAB>dels<TAB>path
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        a, d, path = parts[0], parts[1], "\t".join(parts[2:])
        adds = None if a == "-" else int(a)
        dels = None if d == "-" else int(d)
        rows.append([adds, dels, path])
    return rows


def changed_lines_of_file(sha, path):
    """Sum of |adds|+|dels| for one file from its numstat row, or None if binary."""
    for adds, dels, p in parse_numstat(sha):
        if p == path:
            if adds is None or dels is None:
                return None
            return adds + dels
    return None


def matches_small_file(path):
    if path in SMALL_MOVE_FILES:
        return True
    for suf in SMALL_MOVE_SUFFIXES:
        if path.endswith(suf):
            return True
    return False


def unified_diff_for_file(sha, path):
    """Unified diff text for a single file in commit `sha` vs its parent."""
    return git("show", "--format=", sha, "--", path)


SCORE_PATTERNS = {
    "score": re.compile(r"\bscore\b[^0-9]{0,12}([0-9][0-9,]*)", re.IGNORECASE),
    "toffoli": re.compile(r"\btoffoli\b[^0-9]{0,12}([0-9][0-9,]*)", re.IGNORECASE),
    "qubits": re.compile(r"\bqubits?\b[^0-9]{0,12}([0-9][0-9,]*)", re.IGNORECASE),
}


def reconstruct_score(body):
    """Parse the commit body for score/toffoli/qubits integers. Return dict or None."""
    found = {}
    for key, pat in SCORE_PATTERNS.items():
        m = pat.search(body)
        if m:
            found[key] = int(m.group(1).replace(",", ""))
    return found or None


def collect_accept_commits():
    """Return list of (sha, subject) for accept commits reachable from HEAD, oldest first.

    git log defaults to newest-first; we reverse to oldest-first.
    """
    # NUL-delimited records, field-separated by 0x1f, to survive odd subjects.
    out = git("log", "HEAD", "--no-merges", "--format=%H%x1f%s%x1e")
    commits = []
    for rec in out.split("\x1e"):
        rec = rec.strip("\n")
        if not rec:
            continue
        sha, _, subject = rec.partition("\x1f")
        if subject.startswith("Accept submission"):
            commits.append((sha, subject))
    commits.reverse()  # oldest first
    return commits


def parent_of(sha):
    """First parent sha, or empty string for a root commit."""
    out = git("rev-list", "--parents", "-n", "1", sha).split()
    # format: <sha> [<parent1> <parent2> ...]
    if len(out) >= 2:
        return out[1]
    return ""


def build_record(sha, subject):
    parent = parent_of(sha)
    body = git("log", "-1", "--format=%b", sha)
    # %b can carry a trailing newline; keep verbatim but strip a single trailing newline
    body = body.rstrip("\n")
    numstat = parse_numstat(sha)
    n_files = len(numstat)

    # Determine the non-excluded (non-memory/non-md) changed files.
    non_excluded = [row[2] for row in numstat if not is_excluded(row[2])]
    unique_non_excluded = sorted(set(non_excluded))

    is_small_move = False
    small_diff = None
    if len(unique_non_excluded) == 1:
        only = unique_non_excluded[0]
        if matches_small_file(only):
            cl = changed_lines_of_file(sha, only)
            if cl is not None and cl < SMALL_MOVE_LINE_LIMIT:
                is_small_move = True
                small_diff = unified_diff_for_file(sha, only)

    reconstructed_score = reconstruct_score(body)

    return {
        "sha": sha,
        "parent": parent,
        "subject": subject,
        "body": body,
        "numstat": numstat,
        "n_files": n_files,
        "is_small_move": is_small_move,
        "small_diff": small_diff,
        "reconstructed_score": reconstructed_score,
    }


def main():
    commits = collect_accept_commits()
    records = []
    with open(OUT, "w", encoding="utf-8") as fh:
        for sha, subject in commits:
            rec = build_record(sha, subject)
            records.append(rec)
            fh.write(json.dumps(rec, ensure_ascii=False))
            fh.write("\n")

    # ---- stats ----
    total = len(records)
    small = sum(1 for r in records if r["is_small_move"])
    big_arith = 0
    for r in records:
        touched = False
        for _, _, path in r["numstat"]:
            base = os.path.basename(path)
            stem = base[:-3] if base.endswith(".rs") else base
            # match files like adder.rs, modular.rs, multiply.rs, const_arith.rs
            # plus directory-form arith/adder/*, arith/modular/* etc.
            if stem in BIG_ARITH_TOKENS or any(
                ("/arith/%s/" % t) in path or ("/arith/%s." % t) in path
                for t in BIG_ARITH_TOKENS
            ):
                touched = True
                break
        if touched:
            big_arith += 1
    with_score = sum(1 for r in records if r["reconstructed_score"] is not None)

    # distinct authors from Co-authored-by trailers in bodies
    authors = {}
    coauth = re.compile(r"Co-authored-by:\s*([^<\n]+?)\s*(?:<|$)", re.IGNORECASE)
    for r in records:
        for m in coauth.finditer(r["body"]):
            name = m.group(1).strip()
            if name:
                authors[name] = authors.get(name, 0) + 1

    print("=== MOVES dataset stats ===")
    print("output: %s" % OUT)
    print("total accept commits: %d" % total)
    print("small high-signal moves: %d" % small)
    print("touching big arith files (adder/modular/multiply/const_arith): %d" % big_arith)
    print("with reconstructable score (from body): %d" % with_score)
    print("distinct authors (Co-authored-by trailers): %d" % len(authors))
    for name, c in sorted(authors.items(), key=lambda kv: (-kv[1], kv[0])):
        print("    %3d  %s" % (c, name))


if __name__ == "__main__":
    main()
