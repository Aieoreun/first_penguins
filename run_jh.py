"""Start the selected reference harness after loading .env in the shell."""
import argparse
import os


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    for name in ("TEAM_TOKEN", "OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL",
                 "EMBEDDING_API_KEY", "EMBEDDING_BASE_URL", "EMBEDDING_MODEL"):
        if not os.environ.get(name):
            parser.error(f"Missing environment variable: {name}; load .env first")
    os.environ["PIPELINE"] = "jh_r4_material_budget"
    from pipelines import load
    pipeline = load(os.environ["PIPELINE"])
    pipeline.request_slots = 64
    if pipeline.learned_models_enabled:
        raise RuntimeError("This launcher requires the non-trained reference variant")
    import uvicorn
    uvicorn.run("server:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
