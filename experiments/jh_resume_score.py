"""Resume interrupted PUBLIC scoring from a completed response collection."""
import argparse
import json
import os
from pathlib import Path

from jh_compare import ROOT, evaluator_hashes, load_public, score_rows


def main():
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    if not run.is_relative_to(ROOT / "experiments/.cache"):
        parser.error("Only local public experiment recordings can be resumed")
    if (run / "result.json").exists():
        parser.error("This run is already scored; refusing to overwrite it")
    manifest = json.loads((run / "manifest.json").read_text())
    if manifest["evaluator_hashes"] != evaluator_hashes():
        parser.error("Evaluator hashes differ from the recorded run")
    if manifest["model"] != os.environ.get("OPENAI_MODEL"):
        parser.error("Judge model differs from the recorded run")
    rows = [json.loads(s) for s in (run / "responses.jsonl").read_text().splitlines()]
    ids = {r["id"] for r in rows}
    items = [p for p in load_public() if p["id"] in ids]
    if len(rows) != manifest["count"] or len(items) != len(rows):
        parser.error("Response collection is incomplete or contains unknown/duplicate IDs")
    result = score_rows(items, rows, manifest["judge_workers"],
                        not manifest["judge_enabled"], run / "judge.jsonl")
    result["runtime"] = json.loads((run / "runtime.json").read_text())
    result["score_scope"] = "+".join(manifest.get("endpoints", ["tutor", "guard"]))
    (run / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({k: result[k] for k in ("total", "tutor", "guard", "judge_errors", "invalid")}))


if __name__ == "__main__":
    main()
