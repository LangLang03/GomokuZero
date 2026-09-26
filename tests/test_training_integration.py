"""Opt-in GPU training lifecycle regression on a small compiled network.

Run with GOMOK_RUN_TRAINING_TESTS=1 and GOMOK_TRAIN_BINARY pointing to a
CUDA-enabled small-network gomoku_train. The ordinary test suite skips this
resource-intensive test. It uses temporary data/checkpoints and no real model.
"""
import csv
import hashlib
import math
import os
from pathlib import Path
import shutil
import shlex
import subprocess
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from audit_dataset import audit
from npz_to_bin import validate_arrays
from parse_renjunet import Game, write_dataset


def read_network_metadata(directory):
    return dict(line.split(maxsplit=1) for line in (directory/'training.meta').read_text().splitlines())


def read_trainer_metadata(directory):
    lines = (directory/'trainer.meta').read_text().splitlines()
    if lines[0] != 'GOMOKU_TRAINER 2':
        raise ValueError('unsupported trainer checkpoint format')
    batch, completed, horizon, supervised_horizon, seed = map(int, lines[1].split())
    best, stale, count, head = lines[2].split()
    return dict(batch=batch, completed=completed, horizon=horizon,
                supervised_horizon=supervised_horizon, seed=seed,
                best=float(best), stale=int(stale), replay_count=int(count), replay_head=int(head))


def tensor_digest(directory):
    digest = hashlib.sha256()
    for path in sorted(directory.glob('*.bin')):
        digest.update(path.name.encode()+b'\0')
        digest.update(path.read_bytes())
    return digest.hexdigest()


@unittest.skipUnless(os.environ.get('GOMOK_RUN_TRAINING_TESTS') == '1' and os.environ.get('GOMOK_TRAIN_BINARY'),
                     'GPU lifecycle test requires GOMOK_RUN_TRAINING_TESTS=1 and a small GOMOK_TRAIN_BINARY')
class TrainingIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.binary = Path(os.environ['GOMOK_TRAIN_BINARY']).expanduser().resolve()
        if not cls.binary.is_file():
            raise RuntimeError(f'GOMOK_TRAIN_BINARY does not exist: {cls.binary}')

    def run_training(self, working_directory, options, success=True):
        arguments = [str(self.binary), 'train']
        for option, value in options.items():
            arguments.extend([option, str(value)])
        result = subprocess.run(arguments, cwd=working_directory, capture_output=True,
                                text=True, input='', timeout=180,
                                env=dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'))
        output = result.stdout+result.stderr
        print('\n$ '+shlex.join(arguments), flush=True)
        print(output, end='' if output.endswith('\n') else '\n', flush=True)
        if success:
            self.assertEqual(result.returncode, 0, output)
            self.assertIn('[net] CUDA', output)
        else:
            self.assertGreater(result.returncode, 0, output)
        return output

    def test_pretraining_resume_rl_schedule_replay_and_rejected_checkpoints(self):
        with tempfile.TemporaryDirectory(prefix='gomoku-integration-') as temporary:
            directory = Path(temporary)
            prefix = directory/'fixture'
            sequences = [
                (112, 113, 127, 100, 145, 90, 42, 199),
                (112, 97, 129, 110, 43, 165, 32, 203),
                (112, 95, 170, 34, 46, 87, 26, 198),
                (112, 111, 80, 130, 61, 166, 14, 189),
            ]
            games = [Game(moves, -1, index+17, '7') for index, moves in enumerate(sequences)]
            write_dataset(games, {'samples': sum(map(len, sequences))}, 'integration_fixture',
                          bin_prefix=prefix, skip_opening=0)
            report = audit(prefix, fraction=.25, state_overlap=True, seed=1)
            train_count = report['position_overlap']['train_samples']
            self.assertEqual(train_count, 24)
            self.assertGreater(report['position_overlap']['validation_samples_after_purge'], 0)
            batch_size, pretrain_epochs, updates_per_batch = 7, 2, 2
            self.assertNotEqual(train_count % batch_size, 0)
            expected_steps = math.ceil(train_count/batch_size)*pretrain_epochs
            options = {
                '--mix': prefix, '--tag': 'integration', '--games': 0,
                '--pretrain': pretrain_epochs, '--batch-size': batch_size,
                '--epochs': updates_per_batch, '--buffer': 512,
                '--batch-games': 1, '--playout': 4, '--threads': 1,
                '--max-candidates': 8, '--inference-batch': 4,
                '--val-fraction': .25, '--val-samples': 0, '--val-freq': 0,
                '--kl-freq': 0, '--check-freq': 1, '--lr': .001,
                '--lr-final-fraction': .1, '--lr-warmup': 0,
                '--pretrain-warmup': 0, '--seed': 1,
            }
            checkpoint = directory/'best_policy_integration_bin'
            self.run_training(directory, options)
            network = read_network_metadata(checkpoint)
            trainer = read_trainer_metadata(checkpoint)
            self.assertEqual(int(network['optimizer_step']), expected_steps)
            self.assertEqual(int(network['value_optimizer_step']), 0)
            self.assertEqual(int(network['batch_index']), 0)
            self.assertEqual(trainer['completed'], pretrain_epochs)
            self.assertEqual(trainer['horizon'], 0)
            self.assertEqual(trainer['supervised_horizon'], pretrain_epochs)
            self.assertEqual(trainer['replay_count'], 0)
            self.assertEqual((checkpoint/'trainer.replay').stat().st_size, 0)
            before = tensor_digest(checkpoint)

            # Repeating the same absolute pretraining target performs zero
            # updates, including the final partial minibatch.
            options['--init'] = checkpoint
            self.run_training(directory, options)
            self.assertEqual(int(read_network_metadata(checkpoint)['optimizer_step']), expected_steps)
            self.assertEqual(tensor_digest(checkpoint), before)
            with (directory/'runs/integration/metrics.csv').open() as file:
                metrics = list(csv.DictReader(file))
            self.assertEqual(sum(row['phase'] == 'pretrain' for row in metrics), pretrain_epochs)

            # A supervised-only checkpoint initializes a fresh RL horizon.
            options['--games'] = 1
            options['--playout'] = 5  # first RL phase may choose its own search budget
            self.run_training(directory, options)
            first_rl = read_trainer_metadata(checkpoint)
            self.assertEqual(first_rl['batch'], 1)
            self.assertEqual(first_rl['horizon'], 1)
            self.assertEqual(first_rl['completed'], pretrain_epochs)
            self.assertEqual(int(read_network_metadata(checkpoint)['optimizer_step']), expected_steps+updates_per_batch)
            self.assertEqual(int(read_network_metadata(checkpoint)['value_optimizer_step']), updates_per_batch)
            self.assertGreater(first_rl['replay_count'], 0)
            replay = (checkpoint/'trainer.replay').read_bytes()
            self.assertEqual(len(replay), first_rl['replay_count']*1126*4)
            before = tensor_digest(checkpoint)

            # Loading and resaving without new batches must preserve both the
            # optimizer position and the actual replay records.
            output = self.run_training(directory, options)
            self.assertIn(f"restored {first_rl['replay_count']} replay samples", output)
            self.assertEqual((checkpoint/'trainer.replay').read_bytes(), replay)
            self.assertEqual(tensor_digest(checkpoint), before)
            self.assertEqual(int(read_network_metadata(checkpoint)['optimizer_step']), expected_steps+updates_per_batch)

            # Extending the absolute target preserves the established LR
            # horizon; existing replay remains a prefix until capacity is hit.
            options['--games'] = 2
            self.run_training(directory, options)
            second_rl = read_trainer_metadata(checkpoint)
            self.assertEqual(second_rl['batch'], 2)
            self.assertEqual(second_rl['horizon'], 1)
            self.assertGreater(second_rl['replay_count'], first_rl['replay_count'])
            self.assertLessEqual(second_rl['replay_count'], 450)  # two complete 15x15 games
            self.assertTrue((checkpoint/'trainer.replay').read_bytes().startswith(replay))
            self.assertEqual(int(read_network_metadata(checkpoint)['optimizer_step']), expected_steps+2*updates_per_batch)
            self.assertEqual(int(read_network_metadata(checkpoint)['value_optimizer_step']), 2*updates_per_batch)
            records = np.fromfile(checkpoint/'trainer.replay', dtype='<f4').reshape(-1, 1126)
            validate_arrays(records[:, :900], records[:, 900:1125], records[:, 1125])

            # A changed training contract must be diagnosed, rather than
            # silently resuming with a different numerical experiment.
            changed_search = dict(options, **{'--playout': 6})
            output = self.run_training(directory, changed_search, success=False)
            self.assertIn('configuration or dataset changed', output)
            changed = dict(options, **{'--batch-size': batch_size+1})
            output = self.run_training(directory, changed, success=False)
            self.assertIn('configuration or dataset changed', output)

            # A missing arena opponent must fail before any training/metrics,
            # and neither of this run's two checkpoint outputs can be baseline.
            missing_baseline = dict(options, **{'--eval-games': 2, '--baseline': directory/'missing_opponent'})
            output = self.run_training(directory, missing_baseline, success=False)
            self.assertIn('before training', output)
            self.assertNotIn('[val] initial', output)
            for opponent in (checkpoint, directory/'best_policy_integration_val_bin'):
                output = self.run_training(directory, dict(options, **{'--eval-games': 2, '--baseline': opponent}), success=False)
                self.assertIn('must not be a checkpoint output', output)

            if Path('/dev/full').exists():
                metrics_dir = directory/'runs/failing_metrics'
                metrics_dir.mkdir(parents=True)
                (metrics_dir/'metrics.csv').symlink_to('/dev/full')
                output = self.run_training(directory, dict(options, **{'--tag': 'failing_metrics'}), success=False)
                self.assertIn('cannot write training metrics', output)

            corrupt_replay = directory/'corrupt_replay'
            shutil.copytree(checkpoint, corrupt_replay)
            path = corrupt_replay/'trainer.replay'
            path.write_bytes(path.read_bytes()[:-4])
            output = self.run_training(directory, dict(options, **{'--init': corrupt_replay}), success=False)
            self.assertIn('replay', output.lower())

            # A correct byte count does not establish replay integrity. Each
            # mutation keeps every record and the total payload length intact.
            occupied_row = int(np.flatnonzero(records[:, :450].sum(axis=1) > 0)[0])
            occupied_cell = int(np.flatnonzero(records[occupied_row, :225]+records[occupied_row, 225:450])[0])
            for corruption, diagnostic in [
                ('nonbinary_state', 'state planes must be binary'),
                ('occupied_policy', 'policy targets an occupied cell'),
                ('nan_outcome', 'outcome outside'),
            ]:
                with self.subTest(replay_corruption=corruption):
                    corrupted = records.copy()
                    if corruption == 'nonbinary_state':
                        corrupted[0, 0] = .5
                    elif corruption == 'occupied_policy':
                        corrupted[occupied_row, 900:1125] = 0
                        corrupted[occupied_row, 900+occupied_cell] = 1
                    else:
                        corrupted[0, 1125] = np.nan
                    broken = directory/f'corrupt_{corruption}'
                    shutil.copytree(checkpoint, broken)
                    path = broken/'trainer.replay'
                    original_size = path.stat().st_size
                    corrupted.astype('<f4', copy=False).tofile(path)
                    self.assertEqual(path.stat().st_size, original_size)
                    output = self.run_training(directory, dict(options, **{'--init': broken}), success=False)
                    self.assertIn(diagnostic, output)

            corrupt_optimizer = directory/'corrupt_optimizer'
            shutil.copytree(checkpoint, corrupt_optimizer)
            (corrupt_optimizer/'training_optimizer.pt').unlink()
            output = self.run_training(directory, dict(options, **{'--init': corrupt_optimizer}), success=False)
            self.assertIn('model load failed', output)


if __name__ == '__main__':
    unittest.main()
