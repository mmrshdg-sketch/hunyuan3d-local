import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import model_setup


class DownloaderTests(unittest.TestCase):
    def test_anonymous_download_progress_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            specs = {'shape': ('test/repo', 'revision', 'shape-small', {'config.yaml': 3})}
            received = []
            def fake_download(**kwargs):
                received.append(kwargs['token'])
                path = Path(kwargs['local_dir']) / kwargs['filename']
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'abc')
                return str(path)
            def fake_background(url, known_hash, fname, path, progressbar):
                path.mkdir(parents=True, exist_ok=True)
                (path / fname).write_bytes(b'xy')
            with patch.object(model_setup, 'MODELS', root), patch.object(model_setup, 'SPECS', specs), \
                 patch.object(model_setup, 'BACKGROUND_SIZE', 2), \
                 patch('huggingface_hub.hf_hub_download', fake_download), patch('pooch.retrieve', fake_background):
                statuses = list(model_setup.download_models(''))
                self.assertEqual(received, [False])
                self.assertEqual(statuses[-1], 'Ready: all model files are installed.')
                list(model_setup.download_models(''))
                self.assertEqual(len(received), 1)
                self.assertEqual(model_setup.missing_models(), [])

    def test_authenticated_error_does_not_expose_token(self):
        with tempfile.TemporaryDirectory() as tmp:
            specs = {'shape': ('test/repo', 'revision', 'shape-small', {'config.yaml': 3})}
            def fail(**kwargs):
                raise RuntimeError('secret token: ' + kwargs['token'])
            with patch.object(model_setup, 'MODELS', Path(tmp)), patch.object(model_setup, 'SPECS', specs), \
                 patch('huggingface_hub.hf_hub_download', fail):
                statuses = list(model_setup.download_models('test-token-secret'))
                self.assertIn('Download failed', statuses[-1])
                self.assertNotIn('test-token-secret', ' '.join(statuses))

if __name__ == '__main__':
    unittest.main()
