"""Run an isolated experiment server using the repository's shared .env."""
import argparse
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
    uvicorn.run("server:app", host=args.host, port=args.port, access_log=False)


if __name__ == "__main__":
    main()
