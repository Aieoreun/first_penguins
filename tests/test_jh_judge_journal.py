"""Concurrent development scoring must share identical in-flight calls."""
import json
import tempfile
import threading
import unittest
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from experiments.jh_compare import journal_judge


class JournalTests(unittest.TestCase):
    def run_concurrent(self, fail):
        entered = threading.Event()
        waiting = threading.Event()
        release = threading.Event()
        calls = []

        class ObservedFuture(Future):
            def result(self, timeout=None):
                waiting.set()
                return super().result(timeout)

        def judge(kind, payload):
            calls.append((kind, payload))
            entered.set()
            if not release.wait(3):
                raise TimeoutError('test synchronization')
            if fail and len(calls) == 1:
                raise ValueError('judge failed')
            return {'ok': True}

        with tempfile.TemporaryDirectory() as tmp, patch('experiments.jh_compare.Future', ObservedFuture):
            path = Path(tmp) / 'judge.jsonl'
            call = journal_judge(judge, path)
            with ThreadPoolExecutor(max_workers=2) as pool:
                first = pool.submit(call, 'faithful', {'answer': 'same'})
                self.assertTrue(entered.wait(3))
                second = pool.submit(call, 'faithful', {'answer': 'same'})
                try:
                    self.assertTrue(waiting.wait(3))
                finally:
                    release.set()
                for result in (first, second):
                    if fail:
                        with self.assertRaisesRegex(ValueError, 'judge failed'):
                            result.result(timeout=3)
                    else:
                        self.assertEqual(result.result(timeout=3), {'ok': True})
            self.assertEqual(len(calls), 1)
            if fail:
                self.assertFalse(path.exists())
                self.assertEqual(call('faithful', {'answer': 'same'}), {'ok': True})
                self.assertEqual(len(calls), 2)
            self.assertEqual(len(path.read_text().splitlines()), 1)
            self.assertEqual(json.loads(path.read_text())['result'], {'ok': True})

    def test_identical_calls_share_one_result(self):
        self.run_concurrent(False)

    def test_failure_reaches_waiters_and_is_not_cached(self):
        self.run_concurrent(True)

    def test_journal_write_failure_can_be_retried(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'missing' / 'judge.jsonl'
            calls = []
            call = journal_judge(lambda *args: calls.append(args) or {'ok': True}, path)
            with self.assertRaises(FileNotFoundError):
                call('faithful', {})
            path.parent.mkdir()
            self.assertEqual(call('faithful', {}), {'ok': True})
            self.assertEqual(len(calls), 2)
