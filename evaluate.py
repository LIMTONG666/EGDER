"""Evaluate EGDER and the official comparison method with one command."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys

import yaml

PACKAGE = Path(__file__).resolve().parent
DEFAULT_CONFIG = PACKAGE/'configs/evaluation.yaml'


def run_comparison(config_path):
    with Path(config_path).open() as stream:
        settings = yaml.safe_load(stream)
    paths = {}
    for name in ('runtime', 'backbone', 'data_root',
                 'egder_checkpoint', 'baseline_checkpoint', 'output'):
        path = Path(settings[name]).expanduser()
        paths[name] = path.resolve() if path.is_absolute() else (PACKAGE/path).resolve()
    for name in ('runtime', 'backbone', 'data_root'):
        if not paths[name].is_dir():
            raise FileNotFoundError('Configure the ' + name + ' directory: ' + str(paths[name]))
    for name in ('egder_checkpoint', 'baseline_checkpoint'):
        if not paths[name].is_file():
            raise FileNotFoundError('Configure the ' + name + ' file: ' + str(paths[name]))

    run_dir = paths['output']/datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')
    run_dir.mkdir(parents=True, exist_ok=False)
    environment = dict(os.environ)
    environment['CUDA_VISIBLE_DEVICES'] = str(settings.get('gpu', 2))
    environment['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    for method, label, checkpoint in (
        ('egder', 'egder', 'egder_checkpoint'),
        ('liveedit', 'baseline', 'baseline_checkpoint'),
    ):
        print('Evaluating ' + ('EGDER' if method == 'egder' else 'official LiveEdit'), flush=True)
        subprocess.run([
            sys.executable, str(PACKAGE/'evaluation_worker.py'),
            '--upstream', str(paths['runtime']),
            '--method', method,
            '--checkpoint', str(paths[checkpoint]),
            '--backbone', str(paths['backbone']),
            '--data-root', str(paths['data_root']),
            '--output', str(run_dir/label),
        ], check=True, env=environment)
    print('Evaluation completed: ' + str(run_dir), flush=True)
    return run_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG,
                        help='Evaluation settings (default: configs/evaluation.yaml).')
    args = parser.parse_args()
    run_comparison(args.config)


if __name__ == '__main__':
    main()
