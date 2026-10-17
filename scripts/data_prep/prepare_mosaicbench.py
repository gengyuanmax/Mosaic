"""Convert a local MosaicBench snapshot to the environment's input schema."""
import argparse
import json
from pathlib import Path


def prepare(root, output):
    root = Path(root).resolve()
    output = Path(output)
    samples, skipped = [], []
    seen = set()
    for line in (root / 'data/test.jsonl').read_text().splitlines():
        row = json.loads(line)
        sid = row['sample_id']
        if sid in seen:
            raise ValueError(f'Duplicate sample ID: {sid}')
        seen.add(sid)
        if not row['images_available']:
            skipped.append({'sample_id': sid, 'reason': 'Images not distributed; consult source mapping.'})
            continue
        paths, ids = row['image_paths'], row['image_ids']
        if not paths or len(paths) != len(ids):
            raise ValueError(f'Invalid image alignment: {sid}')
        images = []
        for image_id, relative in zip(ids, paths):
            path = (root / relative).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError(f'Missing or invalid image: {sid}/{image_id}')
            images.append({'image_id': image_id, 'uri_or_path': str(path)})
        samples.append({'sample_id': sid, 'dataset_name': 'MosaicBench',
                        'prompt': row['prompt'], 'images': images,
                        'task_type': row['task'], 'target': row['answer'],
                        'metadata': {'choices': row['choices']}})
    (output / 'datasets').mkdir(parents=True, exist_ok=True)
    (output / 'datasets/mosaicbench.json').write_text(json.dumps(samples))
    report = {'total': len(seen), 'included': len(samples), 'excluded': skipped,
              'scope': 'Only samples with distributed images; not the full benchmark.'}
    (output / 'coverage.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k != 'excluded'}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    prepare(args.snapshot, args.output)
