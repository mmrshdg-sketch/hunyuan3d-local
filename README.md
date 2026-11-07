# Hunyuan3D Local (experimental)

An experimental local image-to-3D app for Apple Silicon Macs. Create a shape from an image, texture a GLB, preview the result, and download it. Generation runs on your Mac after setup. Model weights are **not** included in this download.

This project is experimental. Results may contain holes, distorted geometry, seams, baked-in shadows, or incorrect details. Generated objects generally do not have independently movable parts. Review assets before using them in a game or other project.

## How this was made

The local interface, staged low-memory workflow, downloader, packaging, and documentation were built by an AI assistant (OpenAI Codex) in collaboration with the project owner. This project builds on existing work: the underlying Hunyuan3D models were created by Tencent, the MLX implementation and original source are from [Hunyuan3D-Swift](https://github.com/ZimengXiong/Hunyuan3D-Swift), and texture enhancement uses Real-ESRGAN. Neither the AI assistant nor this project created those underlying models or libraries.

## Requirements

- An **Apple Silicon Mac** (M-series). Intel Macs and Windows are not supported.
- **macOS 26 or later** for the pinned MLX packages in this tested configuration.
- **Python 3.12**, installed from [python.org](https://www.python.org/downloads/macos/) or via `brew install python@3.12`.
- An internet connection for initial installation and downloads. At least 12 GB free disk space is recommended for models, dependencies, and temporary files; generated files need extra space.
- 8 GB unified memory was used for development. Close other memory-heavy apps; experience varies with macOS, model settings, and available swap.

## Install and launch

1. Download and unzip this repository or release. Keep the extracted folder in a writable location such as Documents.
2. Install **Python 3.12** if necessary.
3. Open **Start Hunyuan3D.command**. The first run creates a virtual environment and installs dependencies. If macOS blocks the unsigned script, use the normal macOS control-click **Open** flow if offered.
4. Keep the Terminal window open. The app opens at [http://127.0.0.1:7864/](http://127.0.0.1:7864/).
5. Open **Setup & downloads**, optionally enter a Hugging Face token, and click **Download missing models**. Expect approximately 7.9 GB of model downloads. The background-removal model comes from GitHub; the other models come from Hugging Face. Restarting a failed download can resume completed files.

The two MLX model repositories are public and can be downloaded without a token. A token can help with access or rate limits, but does not guarantee faster downloads. The app uses the token for the download request and clears the field afterward; it does not intentionally save it. Do not paste a token into an issue, screenshot, or public repository.

After setup, use **Generate shape** or **Add texture**. The shape and texture steps are separate: download the shape GLB and select it in the texture tab. Files and models are saved in `generated/` and `models/` inside the app folder. Close the Terminal window to stop the server. The service binds to localhost only and is not a public hosted service.

## Memory and performance

The app runs generation stages in separate processes, chunks selected attention operations, decodes and enhances views individually, and has a best-effort RAM target and a system-pressure stop. The target is **per generation subprocess**, not a hard limit on the whole Mac: the app, macOS, other programs, temporary spikes, and swap are additional. A job may stop rather than finish if it cannot fit.

On one 8 GB Apple Silicon development Mac, a full texture test at **256px views, 5 steps, 512px output texture, and enhancement off** completed with a highest sampled individual stage footprint of **2.51 GiB**. Sampling was approximately once per second; this does **not** measure total system RAM, background removal, all short spikes, or guarantee the same result elsewhere. The minimum settings are low quality and this measurement is not proof that the whole application requires less than 4 GB.

A separate 512px-view, 15-step, 2048px-texture run with 4× enhancement completed with a **6 GiB per-stage target** enabled on the same machine. This is evidence for that particular run, not a guarantee of less than 6 GB of total system memory or performance on all Macs. Higher settings can be slow, use swap, or fail.

## Settings

### Generate shape

- **Reference image:** source image for geometry. Remove background attempts to isolate the object; disable it if it cuts away important features or your input is already transparent.
- **RAM target:** best-effort per-stage target. Lower targets can force smaller temporary work, but can also stop a job that does not fit.
- **Image guidance:** strength of the model's adherence to the reference. Lower values give more freedom; higher values may change fidelity and distortion.
- **Shape refinement steps:** more model iterations generally take longer; improvement is not guaranteed.
- **Shape mesh resolution:** resolution used to extract geometry. Higher values can preserve more geometry but cost memory and time.
- **Shape seed:** controls random variation. Hold it fixed while comparing other settings.
- **Shape model precision:** `float16`, 8-bit, or 4-bit. Lower precision may save memory but may alter quality.
- **Experimental compiled shape model:** may speed inference after compilation; output can change slightly.

### Add texture

- **Mesh to texture / reference image / remove background:** choose an existing GLB and the image used for its appearance.
- **View resolution:** generated view size. Larger values can improve detail and are slower and more memory-intensive.
- **RAM target:** per-stage best-effort target; may reduce effective chunk and tile sizes.
- **AI texture detail:** off, 2×, or 4×. A tiled Real-ESRGAN pass enhances generated views before baking. It may invent details or artifacts; larger images alone are not a quality guarantee.
- **Reduce baked shadows:** favors brighter overlapping views. It cannot recover detail where every generated view is dark, and may alter intended dark materials.
- **Output texture size:** dimensions of the final texture atlas. A larger atlas preserves available information but cannot create missing information.
- **Processing chunk size:** temporary attention/feed-forward work size. Smaller values generally use less memory and can be slower. With a target of 6 GiB or less, the effective maximum is 256; with 7 GiB it is 512.
- **Texture refinement steps:** number of model iterations. More takes longer and may change quality.
- **Texture guidance / reference strength:** change the influence of the reference image. Results vary; higher is not always better.
- **Detail enhancement tile size:** tile size used for the separate AI upscaler. Smaller tiles use less temporary memory and generally take longer. The RAM target can reduce the chosen size.
- **Texture seed:** controls random variation in texture generation.

### Quick rebake

Choose a saved texture job to reuse its generated views, adjust texture size, shadow reduction, enhancement, tile size, and RAM target, then rebake without rerunning the full texture model. Enhancement still takes time. Rebaking cannot invent missing geometry or reliably restore hidden texture details.

## Known problems

- Generated lighting can become part of the texture. In particular, views invented by the model may contain very dark areas that hide detail; increasing shadow reduction cannot always fix them.
- Shape and texture generation can be slow on an 8 GB Mac, especially at 512px and higher settings. Memory targets are approximate and some jobs can stop under pressure.
- Models can have holes, geometry distortion, texture seams, inconsistent views, or invented details. Automatic separation and rigging of movable parts are not implemented.
- The upscaler may make artifacts sharper or introduce details that differ from the reference.
- This is an experimental local tool; downloading, setup, and system compatibility may need fixes. The installer is not a signed or notarized Mac app.
- Text-to-image/text-to-3D and Windows are not currently supported. If this project goes viral, the project owner hopes to add text generation and Windows support, improve quality, and fix bugs. These are aspirations, not promised features or timelines.

## Models, privacy, and licenses

The code includes or adapts [Hunyuan3D-Swift](https://github.com/ZimengXiong/Hunyuan3D-Swift) under its MIT license; see [LICENSE](LICENSE) and [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md). The downloaded [shape](https://huggingface.co/zimengxiong/hunyuan3d-mlx-shape-small) and [paint](https://huggingface.co/zimengxiong/hunyuan3d-mlx-paint-small) weights are subject to Tencent's [Hunyuan3D community license](https://github.com/Tencent-Hunyuan/Hunyuan3D-2/blob/main/LICENSE), not the MIT code license. Real-ESRGAN and rembg have separate upstream licenses. Review all applicable terms before sharing, publishing, or commercial use.

Generation is local. Initial setup downloads packages from PyPI, model files from Hugging Face, and the background-removal model from GitHub. A supplied token is sent to Hugging Face for the download. The app does not need an image-generation cloud service.
