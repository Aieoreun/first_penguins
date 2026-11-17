import asyncio
import json
import unittest

from experiments.peace_full_http import request


class HTTPDeadlineTests(unittest.IsolatedAsyncioTestCase):
    async def serve(self, response, delay=0):
        self.received = []
        self.handlers = set()

        async def handle(reader, writer):
            task = asyncio.current_task()
            self.handlers.add(task)
            try:
                head = await reader.readuntil(b'\r\n\r\n')
                size = int(next(line.split(b':', 1)[1] for line in head.split(b'\r\n')
                                if line.lower().startswith(b'content-length:')))
                self.received.append(json.loads(await reader.readexactly(size)))
                for chunk in response:
                    writer.write(chunk)
                    await writer.drain()
                    await asyncio.sleep(delay)
            except (ConnectionError, asyncio.IncompleteReadError):
                pass
            finally:
                writer.close()
                self.handlers.discard(task)
        self.server = await asyncio.start_server(handle, '127.0.0.1', 0)
        return self.server.sockets[0].getsockname()[1]

    async def asyncTearDown(self):
        if hasattr(self, 'server'):
            self.server.close()
            await self.server.wait_closed()
            tasks = list(self.handlers)
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    def item(self):
        return {'id': 'test', 'endpoint': 'guard', 'messages': [{'role': 'user', 'content': 'hello'}],
                'candidate_answer': 'answer', 'labels': {'secret_gold': True}}

    async def test_json_response_and_no_labels_sent(self):
        port = await self.serve([b'HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n{}'])
        row = await request(self.item(), port, 'test-token', deadline=1)
        self.assertEqual(row['response'], {})
        self.assertIsNone(row['error'])
        self.assertNotIn('labels', self.received[0])
        self.assertEqual(self.received[0]['candidate_answer'], 'answer')

    async def test_chunked_response(self):
        port = await self.serve([b'HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n',
                                 b'2\r\n{}\r\n0\r\n\r\n'])
        row = await request(self.item(), port, 'test-token', deadline=1)
        self.assertEqual(row['response'], {})

    async def test_total_deadline_applies_to_slow_body(self):
        port = await self.serve([b'HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n', b'{', b'}'], delay=.08)
        row = await request(self.item(), port, 'test-token', deadline=.12)
        self.assertEqual(row['error'], 'TimeoutError')
        self.assertIsNone(row['response'])
        self.assertLess(row['seconds'], .3)

    async def test_redirect_not_followed(self):
        port = await self.serve([b'HTTP/1.1 302 Found\r\nContent-Length: 0\r\nLocation: /other\r\n\r\n'])
        row = await request(self.item(), port, 'test-token', deadline=1)
        self.assertEqual(row['error'], 'HTTPError:302')
        self.assertIsNone(row['response'])


if __name__ == '__main__':
    unittest.main()
