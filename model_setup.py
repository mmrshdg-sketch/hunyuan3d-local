"""First-run model downloader. Credentials are never persisted by this module."""
from pathlib import Path
import hashlib
import queue
import threading
import time

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / 'models'
SPECS = {
    'shape': ('zimengxiong/hunyuan3d-mlx-shape-small', 'b7536809d38ad13fe6a9b7769a41fd5d42e520df', 'shape-small', {
        'config.yaml': 1628, 'model.fp16.safetensors': 3819958234}),
    'paint': ('zimengxiong/hunyuan3d-mlx-paint-small', '29bab9dbf2a4a4f9c6988a41b0e891156a517a23', 'paint-small', {
        'unet/config.json': 911, 'unet/diffusion_pytorch_model.safetensors': 3662636472,
        'vae/config.json': 553, 'vae/diffusion_pytorch_model.safetensors': 167335310,
        'realesrgan/rrdbnet_mlx.safetensors': 66857885}),
}
BACKGROUND_URL = 'https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx'
BACKGROUND_MD5 = '60024c5c889badc19c04ad937298a77b'
BACKGROUND_SIZE = 175997641  # displayed estimate; pooch verifies the actual checksum


def missing_models():
    missing = []
    for name, (_, _, folder, files) in SPECS.items():
        if any(not (MODELS / folder / rel).is_file() or (MODELS / folder / rel).stat().st_size != size for rel, size in files.items()):
            missing.append(name)
    background = MODELS / 'background' / 'u2net.onnx'
    if not background.is_file() or background.stat().st_size != BACKGROUND_SIZE:
        missing.append('background')
    return missing


def ensure_models(required):
    missing = [part for part in required if part in missing_models()]
    if missing:
        import gradio as gr
        raise gr.Error('Missing model files: ' + ', '.join(missing) + '. Open Setup & downloads and download the models first.')


def model_status_text():
    missing = missing_models()
    return ('Ready: all model files are installed.' if not missing else
            'Missing: ' + ', '.join(missing) + '. Download requires about 8 GB of disk space.')


def _disk_bytes():
    total = 0
    if MODELS.exists():
        for f in MODELS.rglob('*'):
            try:
                if f.is_file() and (f.suffix in ('.safetensors', '.onnx', '.incomplete')):
                    total += f.stat().st_size
            except OSError:
                pass
    return total


def download_models(token):
    """Yield progress while a background worker downloads into resumable local files."""
    token = (token or '').strip()
    events = queue.Queue()
    def worker():
        try:
            from huggingface_hub import hf_hub_download
            import pooch
            for name, (repo, revision, folder, files) in SPECS.items():
                dest = MODELS / folder
                for rel, size in files.items():
                    path = dest / rel
                    if path.is_file() and path.stat().st_size == size:
                        continue
                    events.put(('progress', f'Downloading {name}: {rel}'))
                    hf_hub_download(repo_id=repo, filename=rel, revision=revision,
                                    local_dir=str(dest), token=token if token else False)
                    if not path.is_file() or path.stat().st_size != size:
                        raise RuntimeError('Downloaded file has an unexpected size.')
            if 'background' in missing_models():
                events.put(('progress', 'Downloading background-removal model from GitHub'))
                pooch.retrieve(BACKGROUND_URL, known_hash='md5:' + BACKGROUND_MD5,
                               fname='u2net.onnx', path=MODELS / 'background', progressbar=False)
            events.put(('done', model_status_text()))
        except Exception as exc:
            # Never show exception text: authentication errors can contain request details.
            events.put(('error', f'Download failed ({type(exc).__name__}). Check your connection, token permissions, and available disk space, then retry.'))
    threading.Thread(target=worker, daemon=True).start()
    current = 'Starting downloads'
    expected = sum(sum(item[3].values()) for item in SPECS.values()) + BACKGROUND_SIZE
    while True:
        try:
            kind, message = events.get(timeout=1)
            if kind in ('done', 'error'):
                yield message
                return
            current = message
        except queue.Empty:
            pass
        done = min(_disk_bytes(), expected)
        yield f'{current}\nDownloaded approximately {done / 1e9:.2f} / {expected / 1e9:.2f} GB. Downloads can be resumed by clicking the button again.'
