#!/usr/bin/env python3
"""Run isolated, reproducible supervised learning comparisons sequentially.

Example:
  python3 tools/benchmark_learning.py --binary build/gomoku_train \
    --data data/gomoku_v2 --epochs 3 --batch-size 128 --seed 1 \
    --prefix verify_learning_20260926 --precisions fp32 bf16

Run from the desired experiment working directory. Checkpoints, metrics and
complete logs are retained. Existing experiment names are never overwritten.
Both modes enable TF32 training/inference and fused Adam; "fp32" describes
parameter storage and disabled BF16 autocast, not strict FP32 arithmetic.
Only Python's standard library is required by this driver.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import time


FORCED_ENVIRONMENT = {
    'GOMOK_TF32': '1',
    'GOMOK_INFERENCE_TF32': '1',
    'GOMOK_FUSED_ADAM': '1',
}
PRECISION_DEFINITIONS = {
    'fp32': 'FP32 parameters and Adam moments; BF16 autocast disabled; TF32 convolution/matmul permitted. This is not strict FP32 arithmetic.',
    'bf16': 'BF16 training autocast with FP32 parameters and Adam moments; FP32 loss/value output operations; TF32 convolution/matmul permitted.',
    'inference': 'Both modes evaluate with BF16 autocast disabled and TF32 permitted; validation settings are identical.',
}
DATA_SUFFIXES = ('_states.bin', '_probs.bin', '_winners.bin', '_game_ids.bin', '_meta.json')


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda: file.read(4*1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path):
    stat = path.stat()
    return {'path': str(path), 'bytes': stat.st_size, 'sha256': sha256_file(path)}


def command_metadata(command, cwd=None):
    """Collect optional host tools without requiring them for training."""
    try:
        result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=15)
        return {'command': command, 'returncode': result.returncode,
                'stdout': result.stdout.strip(), 'stderr': result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {'command': command, 'unavailable': str(error)}


def source_identity():
    root = Path(__file__).resolve().parents[1]
    paths = sorted(set([root/'CMakeLists.txt'] +
                       [path for folder in ('src', 'tools') for path in (root/folder).rglob('*')
                        if path.is_file() and path.suffix in ('.cpp', '.h', '.py')]))
    digest = hashlib.sha256()
    files = {}
    for path in paths:
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        fingerprint = sha256_file(path)
        files[relative] = fingerprint
        digest.update(relative.encode()+b'\0'+fingerprint.encode()+b'\n')
    return {'root': str(root), 'sha256': digest.hexdigest(), 'files': files,
            'git_head': command_metadata(['git', 'rev-parse', 'HEAD'], root),
            'git_status': command_metadata(['git', 'status', '--porcelain'], root)}


def host_metadata():
    packages = {}
    for package in ('torch', 'numpy', 'onnx', 'onnxruntime'):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {
        'platform': platform.platform(), 'machine': platform.machine(),
        'python': sys.version, 'python_executable': sys.executable,
        'logical_cpus': os.cpu_count(), 'packages': packages,
        'nvidia_smi': command_metadata([
            'nvidia-smi', '--query-gpu=index,name,uuid,driver_version,memory.total',
            '--format=csv,noheader,nounits']),
        'inherited_environment': {key: os.environ[key] for key in (
            'CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
            'CUBLAS_WORKSPACE_CONFIG', 'NVIDIA_TF32_OVERRIDE') if key in os.environ},
    }


def write_summary(path, report):
    """Replace only the summary reserved by this invocation, atomically."""
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', prefix='.'+path.stem,
                                     suffix='.tmp', dir=path.parent, delete=False) as file:
        temporary = Path(file.name)
        json.dump(report, file, indent=2, ensure_ascii=False, allow_nan=False)
        file.write('\n')
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def read_metrics(path, expected_epoch):
    with path.open(newline='') as file:
        rows = list(csv.DictReader(file))
    converted = []
    required = ('phase', 'index', 'train_policy', 'val_policy', 'train_top1', 'val_top1')
    for row in rows:
        if not all(key in row for key in required):
            raise ValueError(f'metrics file lacks required columns: {path}')
        values = {'phase': row['phase'], 'index': int(row['index'])}
        for key, value in row.items():
            if key in ('phase', 'index'):
                continue
            values[key] = int(value) if key.endswith('_samples') else float(value)
            if not math.isfinite(values[key]):
                raise ValueError(f'non-finite metric {key}: {path}')
        converted.append(values)
    initial = next((row for row in converted if row['phase'] == 'initial'), None)
    epochs = [row for row in converted if row['phase'] == 'pretrain']
    if initial is None or [row['index'] for row in epochs] != list(range(1, expected_epoch+1)):
        raise ValueError(f'expected initial metrics and complete supervised epochs 1..{expected_epoch}: {path}')
    return initial, epochs[-1], converted


def log_measurements(path):
    content = path.read_text(errors='replace')
    split = re.search(r'\[data\] (\d+) games, (\d+) train, (\d+) independent val \((\d+) D4 duplicates purged\)', content)
    records = {}
    if split:
        records['dataset_split'] = dict(zip(('games', 'train_samples', 'validation_samples', 'purged_validation'),
                                           map(int, split.groups())))
    epochs = re.findall(r'\[pretrain\] epoch (\d+) samples:(\d+) loss:([\d.eE+-]+) samples/s:([\d.eE+-]+)', content)
    records['epoch_training'] = [dict(epoch=int(epoch), samples=int(samples), loss=float(loss),
                                      samples_per_second=float(rate)) for epoch, samples, loss, rate in epochs]
    parameters = re.search(r'\[net\].*?parameters=(\d+)', content)
    if parameters:
        records['parameters'] = int(parameters[1])
    return records


def run_process(command, cwd, environment, log):
    # A separate process group lets an interrupted driver stop its trainer
    # instead of leaving an unobserved GPU process behind.
    process = subprocess.Popen(command, cwd=cwd, env=environment, stdout=log,
                               stderr=subprocess.STDOUT, start_new_session=True)
    try:
        return process.wait()
    except BaseException:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=10)
        except ProcessLookupError:
            pass
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        raise


def experiment(args, cwd, precision):
    tag = f'{args.prefix}_{precision}'
    log_path = cwd/'analysis'/f'learning_experiment_{tag}.log'
    metrics_path = cwd/'runs'/tag/'metrics.csv'
    checkpoint = cwd/f'best_policy_{tag}_bin'
    # Full held-out validation avoids measuring different sampled subsets.
    # The matching training probe has the same size as the validation set.
    command = [str(args.binary), 'train', '--games', '0', '--pretrain', str(args.epochs),
               '--mix', str(args.data), '--batch-size', str(args.batch_size),
               '--precision', precision, '--seed', str(args.seed), '--tag', tag,
               '--val-samples', '0']
    return {'precision': precision, 'precision_definition': PRECISION_DEFINITIONS[precision],
            'tag': tag, 'command': command, 'command_shell': shlex.join(command),
            'cwd': str(cwd), 'log': str(log_path), 'metrics_csv': str(metrics_path),
            'checkpoint': str(checkpoint), 'status': 'pending'}


def preflight(args, cwd, runs, summary):
    if not args.binary.is_file() or not os.access(args.binary, os.X_OK):
        raise ValueError(f'binary is missing or not executable: {args.binary}')
    if not 0 < args.epochs < (1 << 31) or not 0 < args.batch_size < (1 << 31):
        raise ValueError('epochs and batch-size must be positive 32-bit integers')
    if not 0 <= args.seed < (1 << 64):
        raise ValueError('seed must be an unsigned 64-bit integer')
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,160}', args.prefix):
        raise ValueError('prefix must contain 1..160 letters, digits, underscores or hyphens')
    if len(set(args.precisions)) != len(args.precisions):
        raise ValueError('precisions must not contain duplicates')
    for suffix in DATA_SUFFIXES[:-1]:
        path = Path(str(args.data)+suffix)
        if not path.is_file():
            raise ValueError(f'missing grouped dataset file: {path}')
    # This driver explicitly requests TF32; a global override would invalidate
    # that precision description even if both child processes exit normally.
    if os.environ.get('NVIDIA_TF32_OVERRIDE') == '0':
        raise ValueError('NVIDIA_TF32_OVERRIDE=0 conflicts with this TF32-enabled comparison; unset it first')
    reserved = [summary]
    for run in runs:
        reserved.extend([Path(run['log']), cwd/'runs'/run['tag']])
        for directory in (Path(run['checkpoint']), cwd/f"best_policy_{run['tag']}_val_bin"):
            reserved.extend([directory, Path(str(directory)+'.staging'), Path(str(directory)+'.previous')])
    conflicts = [str(path) for path in reserved if path.exists() or path.is_symlink()]
    if conflicts:
        raise ValueError('refusing to overwrite existing experiment artifacts:\n  '+'\n  '.join(conflicts))


def run_experiments(args):
    cwd = Path.cwd().resolve()
    args.binary, args.data = args.binary.resolve(), args.data.resolve()
    runs = [experiment(args, cwd, mode) for mode in args.precisions]
    summary = cwd/'analysis'/f'learning_experiment_{args.prefix}.json'
    preflight(args, cwd, runs, summary)
    dataset_files = {suffix: file_identity(Path(str(args.data)+suffix))
                     for suffix in DATA_SUFFIXES if Path(str(args.data)+suffix).is_file()}
    report = {
        'format_version': 1, 'created_utc': utc_now(), 'status': 'running',
        'prefix': args.prefix, 'epochs': args.epochs, 'batch_size': args.batch_size, 'seed': args.seed,
        'precision_definitions': PRECISION_DEFINITIONS,
        'environment_overrides': FORCED_ENVIRONMENT,
        'measurement_definitions': {
            'wall_seconds': 'Complete child-process duration, including loading, training, validation and checkpoint writes.',
            'final_train_metrics': 'Fixed training probe matched to the full held-out validation sample count.',
            'final_validation_metrics': 'Full D4-purged held-out game partition; hard policy targets, inference mode.',
            'epoch_training': 'Per-epoch sample counts and training throughput reported by the trainer.',
            'reproducibility': 'Fixed data, binary/source hashes, seed and options; CUDA kernels need not be bitwise deterministic.',
        },
        'binary': file_identity(args.binary), 'dataset_files': dataset_files,
        'dataset_metadata': None, 'source': source_identity(), 'host': host_metadata(), 'runs': runs,
    }
    metadata_path = Path(str(args.data)+'_meta.json')
    if metadata_path.is_file():
        report['dataset_metadata'] = json.loads(metadata_path.read_text())
    summary.parent.mkdir(parents=True, exist_ok=True)
    # Atomic exclusive creation prevents concurrent invocations from sharing
    # this experiment prefix after the preflight existence check.
    with summary.open('x', encoding='utf-8') as file:
        json.dump(report, file, indent=2, ensure_ascii=False, allow_nan=False)
        file.write('\n')
    environment = dict(os.environ, **FORCED_ENVIRONMENT)
    for run in runs:
        started = time.perf_counter()
        run.update(status='running', started_utc=utc_now())
        write_summary(summary, report)
        print(f"[{run['precision']}] starting; full output: {run['log']}", flush=True)
        try:
            with Path(run['log']).open('x', encoding='utf-8') as log:
                log.write('# '+run['command_shell']+'\n')
                log.write('# environment '+json.dumps(FORCED_ENVIRONMENT, sort_keys=True)+'\n')
                log.flush()
                run['returncode'] = run_process(run['command'], cwd, environment, log)
            run['wall_seconds'] = time.perf_counter()-started
            run['finished_utc'] = utc_now()
            if run['returncode'] != 0:
                raise RuntimeError(f"trainer exited with code {run['returncode']}; see {run['log']}")
            initial, final, metrics = read_metrics(Path(run['metrics_csv']), args.epochs)
            if not (Path(run['checkpoint'])/'training.meta').is_file():
                raise RuntimeError('successful training did not produce a training checkpoint')
            run.update(initial_metrics=initial, final_metrics=final, metrics=metrics,
                       metrics_sha256=sha256_file(Path(run['metrics_csv'])),
                       validation_policy_change=final['val_policy']-initial['val_policy'],
                       validation_top1_change=final['val_top1']-initial['val_top1'],
                       **log_measurements(Path(run['log'])))
            run['network_checkpoint_metadata'] = dict(
                line.split(maxsplit=1) for line in (Path(run['checkpoint'])/'training.meta').read_text().splitlines() if line.strip())
            if 'dataset_split' in run:
                run['end_to_end_training_samples_per_second'] = (
                    args.epochs*run['dataset_split']['train_samples']/run['wall_seconds'])
            run['status'] = 'complete'
            print(f"[{run['precision']}] {run['wall_seconds']:.3f}s; "
                  f"train_policy={final['train_policy']:.6f} val_policy={final['val_policy']:.6f} "
                  f"train_top1={final['train_top1']:.6f} val_top1={final['val_top1']:.6f}", flush=True)
        except BaseException as error:
            run.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                       error=str(error) or type(error).__name__,
                       wall_seconds=time.perf_counter()-started, finished_utc=utc_now())
            report.update(status=run['status'], finished_utc=utc_now())
            write_summary(summary, report)
            raise
        write_summary(summary, report)
    complete = {run['precision']: run for run in runs}
    if 'fp32' in complete and 'bf16' in complete:
        fp32, bf16 = complete['fp32'], complete['bf16']
        report['comparison'] = {
            'fp32_wall_seconds_over_bf16': fp32['wall_seconds']/bf16['wall_seconds'],
            'bf16_minus_fp32_final_validation_policy': bf16['final_metrics']['val_policy']-fp32['final_metrics']['val_policy'],
            'bf16_minus_fp32_final_validation_top1': bf16['final_metrics']['val_top1']-fp32['final_metrics']['val_top1'],
        }
    report.update(status='complete', finished_utc=utc_now())
    write_summary(summary, report)
    print(f'Summary: {summary}', flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True, help='grouped C++ dataset prefix')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--prefix', required=True, help='fresh experiment name; tags append _fp32 / _bf16')
    parser.add_argument('--precisions', nargs='+', choices=('fp32', 'bf16'), default=['fp32', 'bf16'])
    args = parser.parse_args()
    try:
        run_experiments(args)
    except KeyboardInterrupt:
        print('Learning experiment interrupted; logs and summary retained.', file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError) as error:
        print(f'Learning experiment failed: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
