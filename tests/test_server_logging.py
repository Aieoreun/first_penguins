import json
import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import server_logging as s


class ServerLoggingTests(unittest.TestCase):
    def test_append_metadata_and_idempotent_setup(self):
        logger = logging.getLogger('guard_runtime')
        original = logger.handlers[:]
        level, propagate = logger.level, logger.propagate
        logger.handlers = []
        try:
            with tempfile.TemporaryDirectory() as d, patch.object(s, '__file__', str(Path(d)/'server_logging.py')):
                path=s.configure_server_log()
                s.configure_server_log()
                s.server_event('http_complete', request_id='test', status=503, body='SECRET', token='SECRET')
                logger.info('guard_event %s', json.dumps({'request_id':'test','stage':'input','event':'failure'}))
                lines=path.read_text().splitlines()
                self.assertEqual(len(lines),2)
                self.assertNotIn('SECRET',path.read_text())
                for line in lines:
                    self.assertIn('timestamp',json.loads(line))
                s.server_event('server_stop')
                self.assertEqual(len(path.read_text().splitlines()),3)
        finally:
            for h in logger.handlers:h.close()
            logger.handlers=original
            logger.setLevel(level)
            logger.propagate=propagate
