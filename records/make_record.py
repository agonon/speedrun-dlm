#!/usr/bin/env python3
"""Build a leaderboard record from per-training HellaSwag measurements."""
import argparse
import csv
import json
import statistics
from pathlib import Path


def select(rows, entry, position):
    rows = [r for r in rows if r['recipe'] == entry and r['time_key'] == position]
    draws = sorted({int(r['mc_samples']) for r in rows if int(r['mc_samples']) >= 32}, reverse=True)
    for k in draws:
        selected = [r for r in rows if int(r['mc_samples']) == k]
        if len(selected) == 10 and len({int(r['seed']) for r in selected}) == 10:
            if any(int(r['examples']) != 10042 for r in selected):
                raise ValueError('Every evaluation must cover all 10,042 examples.')
            for r in selected:
                if not 0 <= float(r['accuracy']) <= 1:
                    raise ValueError('Accuracy must be a fraction between zero and one.')
            return sorted(selected, key=lambda r: int(r['seed']))
    raise ValueError('Expected a complete set of ten independent trainings at one draw count.')


def build(rows, entry, position, output):
    selected = select(rows, entry, position)
    output.mkdir(parents=True, exist_ok=False)
    runs = [{
        'training_seed': int(r['seed']), 'checkpoint_step': int(r['checkpoint_step']),
        'checkpoint_sha256': r['checkpoint_sha256'],
        'training_tokens': int(r['pretraining_tokens']),
        'elapsed_training_seconds': float(r['timed_training_seconds']),
        'hellaswag_accuracy': float(r['accuracy']),
        'examples': int(r['examples']), 'corruptions_per_answer': int(r['mc_samples']),
    } for r in selected]
    summary = {
        'recipe': entry, 'position': position, 'training_runs': 10,
        'target_accuracy': .275,
        'average_accuracy': statistics.mean(r['hellaswag_accuracy'] for r in runs),
        'accuracy_standard_deviation': statistics.stdev(r['hellaswag_accuracy'] for r in runs),
        'average_training_seconds': statistics.mean(r['elapsed_training_seconds'] for r in runs),
        'training_seconds_standard_deviation': statistics.stdev(r['elapsed_training_seconds'] for r in runs),
        'average_training_tokens': statistics.mean(r['training_tokens'] for r in runs),
        'corruptions_per_answer': runs[0]['corruptions_per_answer'],
        'examples_per_training': 10042,
    }
    summary['reaches_target'] = summary['average_accuracy'] >= summary['target_accuracy']
    (output / 'training_runs.jsonl').write_text(''.join(json.dumps(r, sort_keys=True)+'\n' for r in runs))
    (output / 'hellaswag.json').write_text(json.dumps(summary, indent=2, sort_keys=True)+'\n')
    related = [r for r in rows if r['recipe'] == entry]
    with (output / 'measurements.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(related)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--entry', required=True)
    parser.add_argument('--position', required=True, help='Recorded position, e.g. minute:012.')
    parser.add_argument('--measurements', type=Path, required=True)
    parser.add_argument('--record-dir', type=Path, required=True)
    args = parser.parse_args()
    with args.measurements.open() as f:
        rows = list(csv.DictReader(f))
    print(json.dumps(build(rows, args.entry, args.position, args.record_dir), indent=2))


if __name__ == '__main__':
    main()
