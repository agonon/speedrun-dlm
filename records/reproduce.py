#!/usr/bin/env python3
"""Print or run the recorded training configuration on one or two GPU nodes."""
import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path


def command(record, seed, output, train_files, val_files):
    config = json.loads((record / 'recipe.json').read_text())
    nodes = int(os.environ.get('NNODES', '1'))
    per_node = int(os.environ.get('NPROC_PER_NODE', '8' if nodes == 1 else '4'))
    if nodes * per_node != 8:
        raise ValueError('The recorded configuration requires eight GPUs.')
    launch = ['torchrun', f'--nnodes={nodes}', f'--nproc_per_node={per_node}']
    if nodes == 1:
        launch += ['--standalone']
    else:
        launch += [f'--node_rank={os.environ["NODE_RANK"]}',
                   f'--master_addr={os.environ["MASTER_ADDR"]}',
                   f'--master_port={os.environ.get("MASTER_PORT", "29500")}']
    launch += ['-m', 'speedrun_dlm.train_dlm']
    args = dict(config['training_arguments'], seed=seed, output_dir=str(output),
                train_files=train_files, val_files=val_files)
    for key, value in args.items():
        if isinstance(value, bool):
            if key == 'time_conditioning':
                launch.append('--time_conditioning' if value else '--no_time_conditioning')
            elif value:
                launch.append('--'+key)
        else:
            launch += ['--'+key, str(value)]
    return launch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('record', type=Path)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--train-files', default='data/training-prefix/fineweb_train_*.bin')
    parser.add_argument('--val-files', default='data/fineweb10B/fineweb_val_*.bin')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    cmd = command(args.record, args.seed, args.output, args.train_files, args.val_files)
    print(shlex.join(cmd), flush=True)
    if args.execute:
        subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()
