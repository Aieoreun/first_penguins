"""Submission Pre-flight Validator & Runbox Score Logger.

Subcommands:
1. preflight:
   - Static AST audit across server.py and app/*.py to guarantee SPEC.md §8 compliance
     (no logging or printing of request messages, candidate_answer, or response answer).
   - Synchronize `submission.json` commit field with `git rev-parse --short HEAD`.

2. log-result:
   - Parse an official Runbox score message (e.g. "Score: 74.190000 ... 총점 74.19 (튜터 70.0, 가드 78.37) ...")
   - Append a structured entry to `experiments/ledger.jsonl` and print the markdown row for `docs/submissions.md`.
"""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]

_FORBIDDEN_LOG_VARS = frozenset({"messages", "candidate_answer", "answer", "raw_content", "user_prompt"})

_RESULT_RE = re.compile(
    r"총점\s*(?P<total>[0-9.]+)\s*\(\s*튜터\s*(?P<tutor>[0-9.]+)\s*,\s*가드\s*(?P<guard>[0-9.]+)\s*\)"
    r".*?시간\s*초과\s*(?P<timeouts>\d+)개"
    r".*?형식\s*오류[·/]?응답\s*없음\s*(?P<invalid>\d+)개"
    r".*?판정\s*오류\s*(?P<judge_err>\d+)개"
    r"(?:.*?\(\s*seed\s+(?P<seed>[a-fA-F0-9]+)\s*\))?",
    re.DOTALL,
)
_DURATION_RE = re.compile(r"Duration:\s*(?P<duration>[0-9.]+)\s*seconds", re.IGNORECASE)


def get_git_short_head() -> str:
    res = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return res.stdout.strip()


def audit_no_content_logging() -> List[str]:
    """Static AST check ensuring server.py and app/*.py never log or print forbidden content variables."""
    violations: List[str] = []
    targets = [ROOT / "server.py", *sorted((ROOT / "app").glob("*.py"))]
    for path in targets:
        if not path.exists():
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            is_print = isinstance(node.func, ast.Name) and node.func.id == "print"
            is_logger = (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in ("logger", "logging")
                and node.func.attr in ("debug", "info", "warning", "error", "critical", "exception")
            )
            if not (is_print or is_logger):
                continue
            for arg in node.args:
                # Allow len(answer), len(req.messages), type(exc).__name__, etc.
                if (
                    isinstance(arg, ast.Call)
                    and isinstance(arg.func, ast.Name)
                    and arg.func.id in ("len", "type", "bool", "int", "float")
                ):
                    continue
                if isinstance(arg, ast.Name) and arg.id in _FORBIDDEN_LOG_VARS:
                    violations.append(f"{path.name}:{node.lineno} logs forbidden variable '{arg.id}'")
    return violations


def cmd_preflight(args: argparse.Namespace) -> int:
    violations = audit_no_content_logging()
    if violations:
        print("[FAIL] SPEC.md §8 logging audit violations detected:", file=sys.stderr)
        for v in violations:
            print(f"  - {v}", file=sys.stderr)
        return 1

    commit_sha = get_git_short_head()
    sub_path = ROOT / "submission.json"
    if sub_path.exists() and args.sync_submission:
        data = json.loads(sub_path.read_text(encoding="utf-8"))
        data["commit"] = commit_sha
        if args.base_url:
            base = args.base_url.rstrip("/")
            data["tutor"] = f"{base}/tutor"
            data["guard"] = f"{base}/guard"
        if args.token:
            data["token"] = args.token
        sub_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[OK] Updated submission.json with commit={commit_sha}")
    else:
        print(f"[OK] Pre-flight passed. Current HEAD commit={commit_sha}")
    return 0


def parse_runbox_message(raw_text: str) -> Dict[str, Any]:
    m = _RESULT_RE.search(raw_text)
    if not m:
        raise ValueError(f"Could not parse Runbox result string: {raw_text!r}")
    dur_m = _DURATION_RE.search(raw_text)
    return {
        "total": float(m.group("total")),
        "tutor": float(m.group("tutor")),
        "guard": float(m.group("guard")),
        "timeouts": int(m.group("timeouts")),
        "invalid": int(m.group("invalid")),
        "judge_errors": int(m.group("judge_err")),
        "seed": m.group("seed") or "",
        "duration_sec": float(dur_m.group("duration")) if dur_m else None,
    }


def cmd_log_result(args: argparse.Namespace) -> int:
    parsed = parse_runbox_message(args.message)
    commit_sha = args.commit or get_git_short_head()
    now_kst = dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).strftime("%Y-%m-%d %H:%M")

    entry = {
        "timestamp_kst": now_kst,
        "author": args.author,
        "commit": commit_sha,
        "local_score": args.local_score,
        "note": args.note,
        **parsed,
    }

    ledger_path = ROOT / "experiments" / "ledger.jsonl"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    dur_note = f" (소요 {parsed['duration_sec']:.2f}s)" if parsed["duration_sec"] is not None else ""
    md_row = (
        f"| {now_kst} | {args.author} | `{commit_sha}` | {parsed['total']:.2f} | "
        f"{parsed['tutor']:.1f} | {parsed['guard']:.2f} | {parsed['timeouts']} | "
        f"{parsed['invalid']} | {parsed['judge_errors']} | `{parsed['seed']}` | "
        f"{args.local_score} | {args.note}{dur_note} |"
    )
    print("Recorded to experiments/ledger.jsonl. Markdown row for docs/submissions.md:")
    print(md_row)
    return 0


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Submission Pre-flight & Runbox Result Logger")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_pre = sub.add_parser("preflight", help="Run SPEC.md §8 audit and optionally sync submission.json")
    p_pre.add_argument("--sync-submission", action="store_true", help="Write current git short HEAD into submission.json")
    p_pre.add_argument("--base-url", default=None, help="Optional tunnel base URL for submission.json")
    p_pre.add_argument("--token", default=None, help="Optional TEAM_TOKEN for submission.json")
    p_pre.set_defaults(func=cmd_preflight)

    p_log = sub.add_parser("log-result", help="Parse Runbox score message and append to experiments/ledger.jsonl")
    p_log.add_argument("--message", required=True, help="Raw Runbox result text")
    p_log.add_argument("--commit", default=None, help="Commit hash (defaults to current HEAD)")
    p_log.add_argument("--author", default="dj (`bsuk1`)")
    p_log.add_argument("--local-score", default="-")
    p_log.add_argument("--note", default="")
    p_log.set_defaults(func=cmd_log_result)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
