import os
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
os.environ.setdefault('U2NET_HOME', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'background'))
import gc
import ctypes
import json
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from pathlib import Path

import gradio as gr
from PIL import Image
from model_setup import ensure_models, model_status_text, download_models

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / 'models' / 'shape-small'
OUTPUT = ROOT / 'generated'
OUTPUT.mkdir(exist_ok=True)


def prepare_image(image, remove_bg):
    if image is None:
        raise gr.Error('Choose a reference image first.')
    image = image.convert('RGBA')
    if remove_bg:
        import rembg
        session = rembg.new_session()
        image = rembg.remove(image, session=session)
        del session
        gc.collect()
    return image


def name_for_job():
    return time.strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:6]


def generate(image, remove_bg, steps, guidance, resolution, seed, precision, compile_dit, ram_limit, progress=gr.Progress()):
    ensure_models(('shape',) + (('background',) if remove_bg else ()))
    started = time.monotonic()
    progress(0, desc='Preparing image')
    image = prepare_image(image, remove_bg)
    name = name_for_job()
    png, target = OUTPUT / f'{name}-input.png', OUTPUT / f'{name}.glb'
    image.save(png)
    if float(ram_limit) not in (0, 4, 5, 6, 7):
        raise gr.Error('Choose a valid RAM target.')
    steps, resolution, seed = int(steps), int(resolution), int(seed)
    guidance = float(guidance)
    if not 5 <= steps <= 60 or not 1 <= guidance <= 10 or resolution not in (128, 192, 256, 320, 384) or not 0 <= seed <= 2147483647 or precision not in ('float16', '8-bit', '4-bit'):
        raise gr.Error('Invalid shape settings.')
    settings = {'steps': steps, 'guidance': guidance, 'resolution': resolution, 'seed': seed, 'precision': precision, 'compile_dit': bool(compile_dit), 'ram_limit_gb': float(ram_limit), 'remove_background': bool(remove_bg)}
    (OUTPUT / f'{name}-settings.json').write_text(json.dumps(settings, indent=2))
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join([str(ROOT / 'local-deps'), str(ROOT / 'python/shape'), env.get('PYTHONPATH', '')])
    env['HY3D_RAM_GB'] = str(ram_limit)
    env['HY3D_IMAGE_CACHE'] = str(ROOT / 'work' / 'shape-image-cache')
    cmd = [sys.executable, '-m', 'hy3dmlx.pipeline', str(png), '--out', str(target), '--weights', str(WEIGHTS), '--steps', str(steps), '--guidance', str(guidance), '--octree', str(resolution), '--seed', str(seed), '--dtype', 'float16', '--octree-decode']
    if precision != 'float16':
        cmd += ['--quantize', precision.split('-')[0]]
    if compile_dit:
        cmd += ['--compile-dit']
    progress(0.2, desc='Generating shape — this may take several minutes')
    _, base_swap = system_health()
    try:
        run_command(cmd, env, progress, base_swap, 0.2, 'shape')
    except RuntimeError as e:
        raise gr.Error('Shape generation failed. ' + str(e)) from e
    if not target.is_file():
        raise gr.Error('Shape generation finished without producing a file.')
    return str(target), str(target), image, f'Completed in {time.monotonic()-started:.1f} seconds. Generated parts may not be separate.'


def system_health():
    """Best-effort guard for dangerous machine-wide pressure; not an 8GB cap."""
    try:
        p = subprocess.run(['memory_pressure', '-Q'], text=True, capture_output=True, timeout=5)
        free = re.search(r'free percentage:\s*(\d+)%', p.stdout)
        s = subprocess.run(['sysctl', '-n', 'vm.swapusage'], text=True, capture_output=True, timeout=5)
        swap = re.search(r'used\s*=\s*([\d.]+)([MG])', s.stdout)
        used_mb = float(swap.group(1)) * (1024 if swap.group(2) == 'G' else 1) if swap else None
        return (int(free.group(1)) if free else None), used_mb
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return None, None



def process_footprint(pid):
    """macOS physical footprint of one subprocess, including its charged GPU allocations."""
    class RUsageV0(ctypes.Structure):
        _fields_ = [('uuid', ctypes.c_uint8 * 16)] + [(name, ctypes.c_uint64) for name in (
            'user_time', 'system_time', 'pkg_idle_wkups', 'interrupt_wkups', 'pageins',
            'wired_size', 'resident_size', 'phys_footprint', 'start', 'exit')]
    try:
        libproc = ctypes.CDLL('/usr/lib/libproc.dylib', use_errno=True)
        libproc.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        data = RUsageV0()
        if libproc.proc_pid_rusage(pid, 0, ctypes.byref(data)) != 0:
            return None
        return data.phys_footprint
    except (OSError, AttributeError):
        return None

def run_stage(stage, env, progress, base_swap, fraction):
    cmd = [sys.executable, '-u', str(ROOT / 'paint_staged.py'), stage]
    return run_command(cmd, env, progress, base_swap, fraction, stage)


def run_command(cmd, env, progress, base_swap, fraction, stage):
    p = subprocess.Popen(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=ROOT, env=env, bufsize=1)
    lines = queue.Queue()
    def read_output():
        for line in p.stdout:
            lines.put(line)
    threading.Thread(target=read_output, daemon=True).start()
    recent = deque(maxlen=20)
    last_health = 0
    last_footprint_check = 0
    over_limit_samples = 0
    ram_gb = float(env.get('HY3D_RAM_GB', '0'))
    try:
        while p.poll() is None or not lines.empty():
            try:
                line = lines.get(timeout=1)
                recent.append(line)
                if 'Denoise ' in line:
                    progress(fraction, desc='Texturing: step ' + line.split('Denoise ', 1)[1].strip())
                elif '[denoise]' in line and '/' in line:
                    progress(fraction, desc='Generating shape: ' + line.strip())
                elif 'Enhancing view ' in line:
                    progress(fraction, desc=line.split('] ', 1)[-1].strip() + ' — tiled detail enhancement')
                elif 'Encoding view ' in line:
                    progress(fraction, desc=line.split('] ', 1)[-1].strip())
            except queue.Empty:
                pass
            if p.poll() is None and ram_gb and time.monotonic() - last_footprint_check >= 2:
                last_footprint_check = time.monotonic()
                footprint = process_footprint(p.pid)
                over_limit_samples = over_limit_samples + 1 if footprint is not None and footprint > ram_gb * 1024**3 else 0
                if over_limit_samples >= 2:
                    raise RuntimeError(f'Stopped: this generation stage exceeded the {ram_gb:g} GiB RAM target. Try a higher target, smaller resolution, or a lower-memory model setting.')
            if p.poll() is None and time.monotonic() - last_health >= 15:
                last_health = time.monotonic()
                free, swap = system_health()
                if (free is not None and free < 12) or (base_swap is not None and swap is not None and swap - base_swap > 3072):
                    p.terminate()
                    try: p.wait(timeout=10)
                    except subprocess.TimeoutExpired: p.kill(); p.wait()
                    raise RuntimeError('Stopped because the Mac was under severe memory pressure. Close other apps or choose a smaller setting.')
        if p.wait():
            raise RuntimeError(''.join(recent)[-1200:] or f'{stage} exited with code {p.returncode}')
    finally:
        if p.poll() is None:
            p.terminate()
            try: p.wait(timeout=10)
            except subprocess.TimeoutExpired: p.kill(); p.wait()


def texture(mesh_path, image, remove_bg, resolution, atlas, chunk, steps, guidance, seed, ref_scale, shadow_bias, detail_scale, detail_tile, ram_limit, progress=gr.Progress()):
    if not mesh_path:
        raise gr.Error('Choose a GLB mesh first.')
    ensure_models(('paint',) + (('background',) if remove_bg else ()))
    if int(resolution) not in (256, 384, 512) or int(atlas) not in (512, 1024, 2048) or int(chunk) not in (256, 512, 1024, 2048) or not 5 <= int(steps) <= 50:
        raise gr.Error('Choose the available texture settings.')
    if not 0 <= float(guidance) <= 10 or not 0 <= int(seed) <= 2147483647 or not 0 <= float(ref_scale) <= 2 or not 0 <= float(shadow_bias) <= 4:
        raise gr.Error('Invalid texture settings.')
    if float(ram_limit) not in (0, 4, 5, 6, 7):
        raise gr.Error('Choose a valid RAM target.')
    if int(detail_scale) not in (1, 2, 4) or int(detail_tile) not in (64, 128, 256):
        raise gr.Error('Invalid texture detail settings.')
    started = time.monotonic()
    progress(0, desc='Preparing reference image')
    image = prepare_image(image, remove_bg)
    name = name_for_job()
    job = OUTPUT / f'{name}-paint-work'
    job.mkdir()
    png = job / 'input.png'
    image.save(png)
    env = os.environ.copy()
    ram_gb = float(ram_limit)
    effective_chunk = min(int(chunk), 256 if ram_gb and ram_gb <= 6 else 512 if ram_gb else int(chunk))
    effective_tile = min(int(detail_tile), 64 if ram_gb and ram_gb <= 5 else 128 if ram_gb else int(detail_tile))
    env.update({'HY3D_RAM_GB': str(ram_limit), 'PAINT_WORK': str(job), 'PAINT_MESH': str(Path(mesh_path).resolve()), 'PAINT_IMAGE': str(png),
                'PAINT_RES': str(resolution), 'PAINT_STEPS': str(steps), 'PAINT_TEX': str(atlas),
                'PAINT_TOKEN_CHUNK': str(effective_chunk), 'PAINT_VIEW_CHUNK': '1', 'PAINT_GUIDANCE': str(guidance), 'PAINT_SEED': str(int(seed)), 'PAINT_REF_SCALE': str(ref_scale), 'PAINT_SHADOW_BIAS': str(shadow_bias), 'PAINT_SR_SCALE': str(detail_scale), 'PAINT_SR_TILE': str(effective_tile)})
    (job / 'settings.json').write_text(json.dumps({'resolution': resolution, 'atlas': atlas, 'chunk': chunk, 'steps': steps, 'guidance': guidance, 'seed': int(seed), 'reference_strength': ref_scale, 'shadow_bias': shadow_bias, 'detail_scale': int(detail_scale), 'detail_tile': int(detail_tile), 'effective_chunk': effective_chunk, 'effective_tile': effective_tile, 'ram_limit_gb': ram_gb, 'remove_background': bool(remove_bg)}, indent=2))
    env['PYTHONPATH'] = os.pathsep.join([str(ROOT / 'local-deps'), str(ROOT / 'python/paint'), env.get('PYTHONPATH', '')])
    _, base_swap = system_health()
    try:
        stages = ['prepare', 'reference', 'denoise', 'decode'] + (['superres'] if int(detail_scale) > 1 else []) + ['finish']
        for i, stage in enumerate(stages):
            fraction = 0.1 + i * 0.8 / len(stages)
            progress(fraction, desc=f'Texturing: {stage}')
            run_stage(stage, env, progress, base_swap, fraction)
    except RuntimeError as e:
        raise gr.Error(str(e)) from e
    target = job / 'painted.glb'
    if not target.is_file():
        raise gr.Error('Texturing finished without producing a file.')
    return str(target), str(target), f'Completed in {(time.monotonic()-started)/60:.1f} minutes. Inspect the texture and seams; parts may not be separate.'


def saved_paints():
    return [p.name for p in sorted(OUTPUT.glob('*-paint-work'), reverse=True)
            if (p / 'mesh-data.npz').is_file() and (p / 'paint-views.png').is_file()]


def rebake_paint(job_name, atlas, shadow_bias, detail_scale, detail_tile, ram_limit, progress=gr.Progress()):
    if job_name not in saved_paints() or int(atlas) not in (512, 1024, 2048) or not 0 <= float(shadow_bias) <= 4 or int(detail_scale) not in (1, 2, 4) or int(detail_tile) not in (64, 128, 256) or float(ram_limit) not in (0, 4, 5, 6, 7):
        raise gr.Error('Choose a saved texture job and valid settings.')
    started = time.monotonic()
    source = OUTPUT / job_name
    job = OUTPUT / f'{name_for_job()}-paint-work'
    job.mkdir()
    for filename in ('mesh-data.npz', 'paint-views.png'):
        shutil.copy2(source / filename, job / filename)
    (job / 'settings.json').write_text(json.dumps({'source_job': job_name, 'atlas': int(atlas), 'shadow_bias': float(shadow_bias), 'detail_scale': int(detail_scale), 'detail_tile': int(detail_tile), 'ram_limit_gb': float(ram_limit)}, indent=2))
    env = os.environ.copy()
    effective_tile = min(int(detail_tile), 64 if float(ram_limit) and float(ram_limit) <= 5 else 128 if float(ram_limit) else int(detail_tile))
    env.update({'HY3D_RAM_GB': str(ram_limit), 'PAINT_WORK': str(job), 'PAINT_TEX': str(atlas), 'PAINT_SHADOW_BIAS': str(shadow_bias), 'PAINT_SR_SCALE': str(detail_scale), 'PAINT_SR_TILE': str(effective_tile)})
    env['PYTHONPATH'] = os.pathsep.join([str(ROOT / 'local-deps'), str(ROOT / 'python/paint'), env.get('PYTHONPATH', '')])
    _, base_swap = system_health()
    try:
        if int(detail_scale) > 1:
            run_stage('superres', env, progress, base_swap, 0.2)
        run_stage('rebake', env, progress, base_swap, 0.8)
    except RuntimeError as e:
        raise gr.Error('Rebake failed: ' + str(e)) from e
    target = job / 'painted.glb'
    if not target.is_file():
        raise gr.Error('Rebake finished without producing a file.')
    return str(target), str(target), f'Rebaked in {time.monotonic()-started:.1f} seconds. Dark areas with no brighter view may remain.'


with gr.Blocks(title='Hunyuan3D' , css_paths=[ROOT / 'studio.css'], theme=gr.themes.Soft(primary_hue='blue', neutral_hue='slate'), analytics_enabled=False) as app:
    gr.HTML('''<header class="site-header"><h1>Hunyuan3D</h1><p>Generate and texture 3D models locally.</p></header>''')
    with gr.Tabs(elem_id='workflow-tabs'):
        with gr.Tab('Setup & downloads'):
            gr.Markdown('## Download models\nDownload the model files before using the app. A Hugging Face token is optional for these public models; it can help with authentication or rate limits, but does not guarantee faster downloads. The token is used for this download only and is not saved by the app. The background-removal model is downloaded from GitHub.')
            model_status = gr.Textbox(label='Download status', value=model_status_text(), lines=4, interactive=False)
            hf_token = gr.Textbox(label='Hugging Face token (optional)', type='password', placeholder='Leave empty to download anonymously')
            download_button = gr.Button('Download missing models', variant='primary')
            download_button.click(download_models, [hf_token], [model_status], concurrency_id='generation', concurrency_limit=1).then(lambda: '', outputs=hf_token)
            gr.Button('Refresh model status').click(model_status_text, outputs=model_status)
        with gr.Tab('Generate shape'):
            with gr.Row(elem_classes='workspace-row'):
                with gr.Column(scale=5, elem_classes='panel'):
                    gr.HTML('''<div class="panel-heading"><h2>Input</h2></div>''')
                    image = gr.Image(label='Reference image', type='pil', image_mode='RGBA', sources=['upload', 'clipboard'], height=235)
                    remove = gr.Checkbox(label='Remove background', value=True)
                    shape_ram = gr.Dropdown([("No target", 0), ("4 GiB", 4), ("5 GiB", 5), ("6 GiB", 6), ("7 GiB", 7)], value=6, label='RAM target per generation stage', info='Best effort, not a system-wide cap. Lower targets can stop a job that cannot fit.')
                    shape_guidance = gr.Slider(1, 10, value=5, step=0.5, label='Image guidance', info='Try 3–4 for more freedom in the shape.')
                    with gr.Accordion('Advanced settings', open=False, elem_classes='advanced'):
                        shape_steps = gr.Slider(5, 60, value=30, step=1, label='Shape refinement steps', info='More steps take longer; improvements are not guaranteed.')
                        shape_resolution = gr.Dropdown([128, 192, 256, 320, 384], value=256, label='Shape mesh resolution', info='Higher can capture finer geometry, but takes more memory.')
                        shape_seed = gr.Number(value=42, precision=0, label='Shape seed', info='Keep fixed to compare settings.')
                        shape_precision = gr.Dropdown(['float16', '8-bit', '4-bit'], value='float16', label='Shape model precision', info='Lower precision uses less memory and can change results.')
                        shape_compile = gr.Checkbox(label='Experimental compiled shape model', value=False, info='May speed inference; compilation adds overhead and can change results.')
                    submit = gr.Button('Generate shape', variant='primary')
                with gr.Column(scale=7, elem_classes=['panel','output-panel']):
                    gr.HTML('''<div class="panel-heading"><h2>Result</h2></div>''')
                    mesh = gr.Model3D(label='3D preview', height=390)
                    file = gr.File(label='Download shape GLB')
                    status = gr.Textbox(label='Status', placeholder='Your generation status will appear here.')
                    with gr.Accordion('Processed reference', open=False, elem_classes='advanced'):
                        processed = gr.Image(label='Processed input')
            submit.click(generate, [image, remove, shape_steps, shape_guidance, shape_resolution, shape_seed, shape_precision, shape_compile, shape_ram], [mesh, file, processed, status], concurrency_id='generation', concurrency_limit=1)
        with gr.Tab('Add texture'):
            with gr.Row(elem_classes='workspace-row'):
                with gr.Column(scale=5, elem_classes='panel'):
                    gr.HTML('''<div class="panel-heading"><h2>Input</h2></div>''')
                    source_mesh = gr.File(label='Mesh to texture (.glb)', file_types=['.glb'], type='filepath')
                    source_image = gr.Image(label='Reference image', type='pil', image_mode='RGBA', sources=['upload', 'clipboard'], height=210)
                    paint_remove = gr.Checkbox(label='Remove background', value=True)
                    res = gr.Dropdown([256, 384, 512], value=256, label='View resolution (pixels)', info='Higher resolutions take more time and memory.')
                    paint_ram = gr.Dropdown([("No target", 0), ("4 GiB", 4), ("5 GiB", 5), ("6 GiB", 6), ("7 GiB", 7)], value=6, label='RAM target per generation stage', info='Uses smaller working chunks where possible. This is not a system-wide cap; jobs may stop if they cannot fit.')
                    detail_scale = gr.Dropdown([("Off", 1), ("2× detail (slower)", 2), ("4× detail (slowest)", 4)], value=1, label='AI texture detail', info='Enhances the generated views in tiles. Adds inferred detail; does not guarantee accuracy.')
                    paint_shadow = gr.Slider(0, 4, value=2, step=0.25, label='Reduce baked shadows', info='Favors brighter overlapping views; it cannot recover missing detail.')
                    with gr.Accordion('Advanced settings', open=False, elem_classes='advanced'):
                        atlas = gr.Dropdown([512, 1024, 2048], value=1024, label='Output texture size (pixels)')
                        chunk = gr.Dropdown([256, 512, 1024, 2048], value=256, label='Processing chunk size', info='Larger chunks may use more working memory.')
                        steps = gr.Slider(5, 50, value=15, step=1, label='Texture refinement steps', info='More steps take longer; results can vary.')
                        paint_guidance = gr.Slider(0, 10, value=2, step=0.25, label='Texture guidance', info='Changes the influence of the reference.')
                        paint_ref = gr.Slider(0, 2, value=1, step=0.1, label='Texture reference strength', info='Experimental reference influence inside the network.')
                        detail_tile = gr.Dropdown([64, 128, 256], value=128, label='Detail enhancement tile size', info='Smaller tiles use less temporary memory, but take longer.')
                        paint_seed = gr.Number(value=0, precision=0, label='Texture seed', info='Keep fixed when comparing settings.')
                    paint_button = gr.Button('Generate texture', variant='primary')
                with gr.Column(scale=7, elem_classes=['panel','output-panel']):
                    gr.HTML('''<div class="panel-heading"><h2>Result</h2></div>''')
                    painted_mesh = gr.Model3D(label='3D preview', height=390)
                    painted_file = gr.File(label='Download textured GLB')
                    paint_status = gr.Textbox(label='Status', placeholder='Your texture status will appear here.')
                    with gr.Accordion('Quick rebake previous texture', open=False, elem_classes='advanced'):
                        gr.Markdown('Reuses generated views. Shadows can only be reduced where another view contains brighter information.')
                        previous = gr.Dropdown(choices=saved_paints(), value=(saved_paints()[0] if saved_paints() else None), label='Saved texture job')
                        refresh = gr.Button('Refresh saved jobs')
                        rebake_atlas = gr.Dropdown([512, 1024, 2048], value=2048, label='Rebake texture size')
                        rebake_shadow = gr.Slider(0, 4, value=2, step=0.25, label='Reduce baked shadows')
                        rebake_detail = gr.Dropdown([("Off", 1), ("2× detail (slower)", 2), ("4× detail (slowest)", 4)], value=1, label='AI texture detail')
                        rebake_tile = gr.Dropdown([64, 128, 256], value=128, label='Detail enhancement tile size')
                        rebake_ram = gr.Dropdown([("No target", 0), ("4 GiB", 4), ("5 GiB", 5), ("6 GiB", 6), ("7 GiB", 7)], value=6, label='RAM target per generation stage')
                        rebake_button = gr.Button('Rebake saved views')
            paint_button.click(texture, [source_mesh, source_image, paint_remove, res, atlas, chunk, steps, paint_guidance, paint_seed, paint_ref, paint_shadow, detail_scale, detail_tile, paint_ram], [painted_mesh, painted_file, paint_status], concurrency_id='generation', concurrency_limit=1)
            refresh.click(lambda: gr.update(choices=saved_paints()), outputs=previous)
            rebake_button.click(rebake_paint, [previous, rebake_atlas, rebake_shadow, rebake_detail, rebake_tile, rebake_ram], [painted_mesh, painted_file, paint_status], concurrency_id='generation', concurrency_limit=1)
    app.load(None, js="() => { document.body.classList.remove('dark'); }")

if __name__ == '__main__':
    app.queue(default_concurrency_limit=1).launch(server_name='127.0.0.1', server_port=7864, inbrowser=True)
