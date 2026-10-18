"""Build a small, explicit Hugging Face Docker Space package from this checkout."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build(destination):
    destination = Path(destination).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise ValueError('Destination must be empty.')
    destination.mkdir(parents=True, exist_ok=True)

    def copy(relative):
        source = ROOT / relative
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    for relative in ['demo/server.py', 'demo/requirements.txt', 'src/mosaic_gym/__init__.py', 'src/mosaic_gym/contracts.py', 'LICENSE']:
        copy(relative)
    for source in sorted((ROOT / 'src/mosaic_gym/toolbox').rglob('*.py')):
        copy(source.relative_to(ROOT))
    for relative in ['style.css', 'alpha-canvas.css', 'alpha-canvas.js', 'gym-controls.mjs', 'assets/gym-examples.json', 'assets/fonts/OFL.txt', 'assets/mosaic-logo.png']:
        copy(Path('docs') / relative)
    for source in (ROOT / 'docs/assets/fonts').glob('*.ttf'):
        copy(source.relative_to(ROOT))
    examples = json.loads((ROOT / 'docs/assets/gym-examples.json').read_text())
    for example in examples:
        for image in example['images']:
            relative = Path('docs') / image['file']
            if not (ROOT / relative).resolve().is_relative_to(ROOT / 'docs/assets'):
                raise ValueError('Example path must be within docs/assets.')
            copy(relative)
    page = (ROOT / 'docs/index.html').read_text()
    start = page.index('  <section id="alpha-canvas"')
    end = page.index('  </section>', start) + len('  </section>')
    section = page[start:end].replace(' data-toolbox-api="https://gengyuanmax-mosaicgym.hf.space/api/gym"', '')
    (destination / 'docs/index.html').write_text('''<!doctype html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MosaicGym Toolbox</title><link rel="icon" href="assets/mosaic-logo.png">
<link rel="stylesheet" href="style.css"><link rel="stylesheet" href="alpha-canvas.css">
<style>main{max-width:1100px;margin:0 auto 28px}.toolbox-credit{margin:3px 0 0;font-size:11px;color:#778399}.toolbox-credit a{text-decoration:underline;text-underline-offset:3px}</style>
</head><body><main>''' + section + '</main><script type="module" src="alpha-canvas.js"></script></body></html>')
    shutil.copy2(ROOT / 'demo/Dockerfile', destination / 'Dockerfile')
    (destination / 'README.md').write_text('''---
title: MosaicGym
emoji: 🧩
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
license: mit
---

# MosaicGym Toolbox

Interactive image operations from [Mosaic](https://github.com/gengyuanmax/Mosaic).
The server directly calls the same Python Toolbox used by the evaluation runner.
No model inference is performed. The code is MIT-licensed; example images retain
their own source licenses and attribution, available in each example's Image
credits and `docs/assets/gym-examples.json`.

[Project page](https://gengyuanmax.github.io/Mosaic/) ·
[Source](https://github.com/gengyuanmax/Mosaic/tree/main/demo)
''')
    manifest = {str(p.relative_to(destination)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in destination.rglob('*') if p.is_file()}
    (destination / 'SHA256SUMS.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Packaged {len(manifest)} files; Toolbox source copied without changes.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    build(parser.parse_args().destination)
