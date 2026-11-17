"""Content-free server telemetry in one append-only JSONL file."""
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path


class MetadataFormatter(logging.Formatter):
    def format(self, record):
        message = record.getMessage()
        # Only the explicitly structured runtime events belong in this file.
        payload = json.loads(message.split(' ', 1)[1])
        return json.dumps({**payload, 'timestamp': datetime.fromtimestamp(
            record.created, timezone.utc).isoformat(), 'pid': record.process}, ensure_ascii=True)


def configure_server_log():
    path = Path(__file__).resolve().parent / 'work/logs/server.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger('guard_runtime')
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not any(isinstance(h, logging.FileHandler) and h.baseFilename == str(path)
               for h in logger.handlers):
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        os.close(fd)
        handler = logging.FileHandler(path, encoding='utf-8')
        handler.setFormatter(MetadataFormatter())
        logger.addHandler(handler)
    if not any(type(h) is logging.StreamHandler for h in logger.handlers):
        logger.addHandler(logging.StreamHandler())
    return path


def server_event(event, **fields):
    allowed = {'endpoint', 'status', 'seconds', 'request_id', 'error_type',
               'review_mode', 'privacy_enabled', 'privacy_wait_cap', 'safety_review'}
    payload = {'event': event, **{k: v for k, v in fields.items() if k in allowed}}
    logging.getLogger('guard_runtime').info('server_event %s', json.dumps(payload))
