import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.jh_environment import load_environment, public_status


class EnvironmentTests(unittest.TestCase):
    def test_organizer_process_values_win_and_file_fills_missing_credentials(self):
        with tempfile.TemporaryDirectory() as folder:
            Path(folder,'.env').write_text(
                'OPENAI_BASE_URL=https://old.invalid/v1\nOPENAI_MODEL=old-chat\n'
                'EMBEDDING_BASE_URL=https://old-embed.invalid/v1\n'
                'EMBEDDING_MODEL=text-embedding-3-small\n'
                'OPENAI_API_KEY=file-secret\nEMBEDDING_API_KEY=file-secret\nTEAM_TOKEN=file-token\n')
            provided={'OPENAI_BASE_URL':'https://organizer.invalid/v1',
                'OPENAI_MODEL':'organizer-chat','EMBEDDING_BASE_URL':'https://embedding.invalid/v1',
                'EMBEDDING_MODEL':'openai/text-embedding-3-small'}
            with patch.dict(os.environ,provided,clear=True):
                load_environment(folder)
                for key,value in provided.items(): self.assertEqual(os.environ[key],value)
                self.assertEqual(os.environ['EMBEDDING_API_KEY'],'file-secret')
                self.assertEqual(os.environ['TEAM_TOKEN'],'file-token')
                visible=str(public_status())
                for secret in ('file-secret','file-token','https://'):
                    self.assertNotIn(secret,visible)

    def test_local_file_defaults_and_existing_process_key_remain_usable(self):
        with tempfile.TemporaryDirectory() as folder:
            Path(folder,'.env').write_text('EMBEDDING_MODEL=text-embedding-3-small\n'
                'OPENAI_API_KEY=file-key\nEMBEDDING_API_KEY=\n')
            with patch.dict(os.environ,{'OPENAI_API_KEY':'process-key'},clear=True):
                load_environment(folder)
                self.assertEqual(os.environ['EMBEDDING_MODEL'],'text-embedding-3-small')
                self.assertEqual(os.environ['OPENAI_API_KEY'],'process-key')
                self.assertTrue(public_status()['embedding_effective_key_configured'])
