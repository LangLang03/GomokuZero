"""CLI regression tests, enabled by GOMOK_TRAIN_BINARY=/path/to/gomoku_train.

All cases must fail before starting training or interactive play. A temporary
working directory isolates accidental output if an argument regression returns.
"""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


@unittest.skipUnless(os.environ.get('GOMOK_TRAIN_BINARY'), 'set GOMOK_TRAIN_BINARY to test the compiled CLI')
class CommandLineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.binary = Path(os.environ['GOMOK_TRAIN_BINARY']).expanduser().resolve()
        if not cls.binary.is_file():
            raise RuntimeError(f'GOMOK_TRAIN_BINARY does not exist: {cls.binary}')

    def assert_rejected(self, *arguments):
        with tempfile.TemporaryDirectory(prefix='gomoku-cli-') as directory:
            result = subprocess.run([str(self.binary), *arguments], cwd=directory,
                                    input='', capture_output=True, text=True, timeout=15,
                                    env=dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'))
        self.assertGreater(result.returncode, 0, f'arguments were accepted or process crashed: {arguments}\n{result.stdout}\n{result.stderr}')
        self.assertTrue(result.stdout.strip() or result.stderr.strip(), 'failure needs an explanatory diagnostic')

    def test_unknown_commands_and_options(self):
        self.assert_rejected('not-a-command')
        for command in ('train', 'selfplay', 'human'):
            with self.subTest(command=command):
                self.assert_rejected(command, '--not-a-real-option')

    def test_missing_argument_values(self):
        for arguments in [('train', '--lr'), ('train', '--tag'), ('selfplay', '--model'), ('human', '--playout')]:
            with self.subTest(arguments=arguments):
                self.assert_rejected(*arguments)

    def test_negative_zero_and_out_of_range_arguments(self):
        for arguments in [
            ('train', '--games', '-1'), ('train', '--batch-size', '0'),
            ('train', '--augment', '2'), ('selfplay', '--games', '-1'),
            ('human', '--playout', '-1'),
        ]:
            with self.subTest(arguments=arguments):
                self.assert_rejected(*arguments)

    def test_nonfinite_and_partially_parsed_numbers(self):
        for option, value in [('--lr', 'nan'), ('--mix-ratio', 'inf'), ('--epochs', '1x'), ('--lr', '0.001oops')]:
            with self.subTest(option=option, value=value):
                self.assert_rejected('train', option, value)

    def test_requested_checkpoint_failure_is_fatal(self):
        self.assert_rejected('train', '--cpu', '--init', 'missing-checkpoint', '--games', '1', '--playout', '1')
        for command in ('selfplay', 'human'):
            with self.subTest(command=command):
                self.assert_rejected(command, '--cpu', '--model', 'missing-checkpoint', '--playout', '1')


if __name__ == '__main__':
    unittest.main()
