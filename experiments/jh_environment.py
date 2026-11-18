"""Organizer process settings take precedence; .env supplies missing values."""
import json
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]


def load_environment(root=ROOT):
    load_dotenv(Path(root) / '.env', override=False)


def public_status():
    """Configuration only: no keys, tokens, URLs or request contents."""
    return {
        'OPENAI_MODEL': os.environ.get('OPENAI_MODEL'),
        'EMBEDDING_MODEL': os.environ.get('EMBEDDING_MODEL'),
        'chat_base_configured': bool(os.environ.get('OPENAI_BASE_URL')),
        'embedding_base_configured': bool(os.environ.get('EMBEDDING_BASE_URL')),
        'chat_key_configured': bool(os.environ.get('OPENAI_API_KEY')),
        'embedding_key_configured': bool(os.environ.get('EMBEDDING_API_KEY')),
        'embedding_effective_key_configured': bool(
            os.environ.get('EMBEDDING_API_KEY') or os.environ.get('OPENAI_API_KEY')),
        'team_token_configured': bool(os.environ.get('TEAM_TOKEN')),
    }


if __name__ == '__main__':
    load_environment()
    print(json.dumps(public_status(), ensure_ascii=False, indent=2))
