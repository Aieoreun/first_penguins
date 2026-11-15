"""Run an isolated experiment server using the repository's shared .env."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from dotenv import load_dotenv
    import uvicorn
    from pipelines.kimjaehak.config import VARIANTS
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=(*VARIANTS, "v4"), required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env", override=True)
    os.environ["PIPELINE"] = args.variant
    for key in ("TEAM_TOKEN", "OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL"):
        if not os.environ.get(key):
            raise SystemExit(f"Missing setting: {key}")
    from pipelines import load
    pipeline = load(args.variant)
    fields = ("answer_effort", "answer_max_tokens", "votes", "seek_first", "safe_by_rule",
              "deadline", "prepare_limit", "peace_input", "peace_output",
              "guided_input", "guided_output", "focus_input", "request_slots", "short_votes", "other_vote_count",
              "short_effort", "short_max_tokens", "short_timeout", "answer_sections", "material_review_timeout")
    effective = {key: getattr(pipeline, key, None) for key in fields}
    effective["class"] = pipeline.__module__ + "." + pipeline.__name__
    effective["answer_prompts_sha256"] = hashlib.sha256(json.dumps(
        {k.value: v for k, v in pipeline.answer_systems.items()}, sort_keys=True).encode()).hexdigest()
    effective["unsupported_prompt_sha256"] = hashlib.sha256(pipeline.unsupported_system.encode()).hexdigest()
    print("EXPERIMENT_CONFIG " + json.dumps(effective), flush=True)
    uvicorn.run("server:app", host=args.host, port=args.port, access_log=False)


if __name__ == "__main__":
    main()
