"""Regression coverage for data integrity, deployment search and model export.

Run: python3 -m unittest discover -s tests -p 'test_*.py' -v
"""
import importlib.util
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from audit_dataset import audit, holdout_game_ids, infer_game_ids, splitmix64
from gomoku import Board, MCTS, MCTSNode
from npz_to_bin import convert, terminal_positions, validate_arrays, validate_metadata
from parse_renjunet import canonical_moves, game_samples, read_games, validate_game, write_dataset
from server_onnx import GomokuServer


MOVES = 'h8 h9 i8 i9 j8 j9 k8 k9 a1 a2'


def game(body=MOVES, **changes):
    attrs = {'rule': '7', 'swap': '-', 'bresult': '1'}
    attrs.update(changes)
    return validate_game(attrs, f'<move>{body}</move>')


class BoardSearchTests(unittest.TestCase):
    def test_encoding_and_illegal_move_are_transactional(self):
        board = Board()
        board.play(112)
        expected = board.current_state().copy()
        self.assertEqual(expected[1, 7, 7], 1)
        self.assertEqual(expected[2, 7, 7], 1)
        self.assertEqual(expected[3].sum(), 0)
        for move in [112, -1, 225, True, 1.5]:
            with self.assertRaises(ValueError):
                board.play(move)
            np.testing.assert_array_equal(board.current_state(), expected)

    def test_puct_uses_current_node_visits(self):
        root = MCTSNode()
        root.expand([(1, 0.01), (2, 0.99)])
        root.children[1].value = 0.5
        root.visits = 1
        self.assertEqual(root.select(0.1)[0], 1)
        root.visits = 10000
        self.assertEqual(root.select(0.1)[0], 2)

    def test_terminal_search_and_forced_win_avoid_network(self):
        board = Board()
        for move in [0, 30, 1, 32, 2, 34, 3, 36]:
            board.play(move)
        def forbidden(_):
            self.fail('terminal / immediate win must not call network')
        search = MCTS(forbidden, n_playout=4)
        self.assertEqual(search.get_move(board), 4)
        self.assertEqual(len(board.states), 8)
        self.assertEqual(search.root.children[4].value, 1)
        board.play(4)
        with self.assertRaises(ValueError):
            search.get_move(board)
        expected = board.current_state().copy()
        with self.assertRaises(ValueError):
            board.play(100)
        np.testing.assert_array_equal(expected, board.current_state())
        self.assertEqual(board.copy().winner(), 1)

    def test_pruning_renormalizes_priors(self):
        board = Board()
        board.play(112)
        search = MCTS(lambda b: ([(m, 0.001) for m in b.availables], 0), n_playout=1)
        search.get_move(board)
        self.assertAlmostEqual(sum(node.prior for node in search.root.children.values()), 1)

    def test_only_mandatory_block_survives_pruning(self):
        board = Board()
        for move in [0, 30, 1, 32, 2, 34, 3]:
            board.play(move)
        search = MCTS(lambda b: ([(m, 1/len(b.availables)) for m in b.availables], 0), n_playout=2)
        self.assertEqual(search.get_move(board), 4)

    def test_server_rejects_coordinate_alias(self):
        server = GomokuServer(None)
        for coordinates in [(0, 15), (-1, 20), (True, 7), ('7', 7)]:
            with self.assertRaises(ValueError):
                server.human_move(*coordinates)
        self.assertFalse(server.board.states)

    def test_server_rolls_back_if_ai_inference_fails(self):
        class BrokenEngine:
            def get_move(self, _):
                raise RuntimeError('inference failed')
        server = GomokuServer(BrokenEngine())
        with self.assertRaises(RuntimeError):
            server.human_move(7, 7)
        self.assertFalse(server.board.states)
        self.assertEqual(server.board.current_player, 1)
        self.assertIsNone(server.winner)


class DatasetTests(unittest.TestCase):
    def test_cross_language_split_golden_values(self):
        self.assertEqual(splitmix64(0), 0xe220a8397b1dcdaf)
        self.assertEqual(splitmix64(1), 0x910a2dec89025cc1)
        # The same fixture is pinned in tests/dataset_test.cpp.
        self.assertEqual(holdout_game_ids([17, 18, 19], 0.34, 17), {18})
        self.assertEqual(len(holdout_game_ids(range(5), 0.5, 1)), 3)
        self.assertEqual(holdout_game_ids([7], 0, 1), set())
        for args in [([7], .1, 1), ([7, 8], .1, -1), ([7, 8], .1, 1 << 64), ([7, 8], float('nan'), 1)]:
            with self.assertRaises(ValueError):
                holdout_game_ids(*args)

    def test_terminal_detection_covers_both_sides_directions_and_overlines(self):
        states = np.zeros((8, 4, 15, 15), np.float32)
        for direction, (dr, dc) in enumerate(((0, 1), (1, 0), (1, 1), (1, -1))):
            for side in range(2):
                for distance in range(6):
                    states[direction*2+side, side, 4+distance*dr, 8+distance*dc] = 1
        np.testing.assert_array_equal(terminal_positions(states), np.ones(8, dtype=bool))
        states[:, :2, 4:10, 8] = 0
        # Construct a physical terminal board to exercise validation, not just
        # the vectorized detector's synthetic lines.
        board = Board()
        for move in [0, 30, 1, 32, 2, 34, 3, 36, 4]:
            board.play(move)
        state = board.current_state()[None]
        policy = np.zeros((1, 225), np.float32); policy[0, 100] = 1
        with self.assertRaisesRegex(ValueError, 'terminal'):
            validate_arrays(state, policy, np.zeros(1))

    def test_metadata_cannot_enable_wrong_rule_value_targets(self):
        self.assertEqual(validate_metadata({})['value_weight'], 0)
        self.assertEqual(validate_metadata({'target_rules': 'freestyle', 'value_weight': 1})['value_weight'], 1)
        for metadata in [
            {'target_rules': 'renju'}, {'value_weight': 1}, {'value_weight': float('nan')},
            {'value_weight': True}, {'value_weight': -1},
            {'target_rules': 'freestyle', 'value_weight': 1, 'value_supervision': 'disabled'},
        ]:
            with self.assertRaises(ValueError):
                validate_metadata(metadata)

    def test_rule_one_is_renju_not_default_gomoku(self):
        self.assertEqual(game(rule='1')[1], 'other_rule')

    def test_bad_record_rejected_before_any_sample(self):
        for text, attrs, reason in [
            (MOVES+' h8', {}, 'repeated_move'),
            (MOVES+' p1', {}, 'invalid_coordinate'),
            (MOVES, {'bresult': '?'}, 'unknown_result'),
            ('a1 a2 b1 b2 c1 c2 d1 d2 e1 e2', {}, 'moves_after_terminal'),
            ('a1 a2 b1 b2 c1 c2 d1 d2 e1 e2', {'bresult': '0'}, 'terminal_result_conflict'),
        ]:
            self.assertEqual(game(text, **attrs)[1], reason)

    def test_group_invariant_to_all_eight_symmetries(self):
        parsed, rejection = game()
        self.assertIsNone(rejection)
        key = canonical_moves(parsed.moves)
        rows, cols = np.divmod(parsed.moves, 15)
        for _ in range(4):
            self.assertEqual(canonical_moves(rows*15+cols), key)
            self.assertEqual(canonical_moves(rows*15+14-cols), key)
            rows, cols = cols, 14-rows

    def test_value_targets_follow_current_player_and_skip_opening(self):
        parsed, _ = game()
        states, probs, winners, ids = game_samples(parsed)
        self.assertEqual(len(states), 5)
        np.testing.assert_array_equal(winners, [-1, 1, -1, 1, -1])
        self.assertEqual(states[0, :2].sum(), 5)
        validate_arrays(states, probs, winners, ids)

    def test_reject_nonfinite_value_illegal_policy_wrong_last_move(self):
        parsed, _ = game()
        states, probs, winners, ids = game_samples(parsed)
        broken = winners.copy(); broken[0] = np.nan
        with self.assertRaises(ValueError):
            validate_arrays(states, probs, broken, ids)
        broken = probs.copy(); broken[0].fill(0); broken[0, 112] = 1
        with self.assertRaises(ValueError):
            validate_arrays(states, broken, winners, ids)
        broken = states.copy(); broken[0, 2].fill(0); broken[0, 2, 0, 0] = 1
        with self.assertRaises(ValueError):
            validate_arrays(broken, probs, winners, ids)
        broken = winners.copy(); broken[0] *= -1
        for chunk_size in (1, 4096):
            with self.assertRaisesRegex(ValueError, 'outcomes disagree'):
                validate_arrays(states, probs, broken, ids, chunk_size=chunk_size)

    def test_duplicate_games_and_conflicting_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'games.xml'
            def record(result):
                return f'<game rule="7" bresult="{result}"><move>{MOVES}</move></game>'
            path.write_text(record('1')*2)
            games, counts = read_games(path)
            self.assertEqual(len(games), 1)
            self.assertEqual(counts['duplicate_game'], 1)
            path.write_text(record('1')+record('0'))
            games, counts = read_games(path)
            self.assertFalse(games)
            self.assertEqual(counts['conflicting_duplicate'], 2)

    def test_streaming_npz_binary_roundtrip_keeps_game_groups(self):
        with tempfile.TemporaryDirectory() as tmp:
            parsed, _ = game()
            n = len(parsed.moves)-5
            npz, direct, converted = [Path(tmp)/name for name in ('data.npz', 'direct', 'converted')]
            write_dataset([parsed], {'samples': n}, 'fixture.xml', npz, direct)
            convert(npz, converted)
            for suffix in ('states', 'probs', 'winners', 'game_ids'):
                self.assertEqual(Path(f'{direct}_{suffix}.bin').read_bytes(), Path(f'{converted}_{suffix}.bin').read_bytes())
            report = audit(converted)
            self.assertEqual(report['games'], 1)
            self.assertEqual(report['samples'], n)
            self.assertEqual(report['metadata']['value_weight'], 0)

    def test_fortran_order_and_narrow_integer_npz_preserve_sample_layout(self):
        parsed, _ = game()
        states, probs, winners, ids = game_samples(parsed)
        with tempfile.TemporaryDirectory() as tmp:
            path, prefix = Path(tmp)/'data.npz', Path(tmp)/'converted'
            np.savez_compressed(path, states=np.asfortranarray(states.astype('u1')),
                                probs=np.asfortranarray(probs), winners=winners, game_ids=ids)
            convert(path, prefix)
            values = np.fromfile(f'{prefix}_states.bin', dtype='<f4', offset=8).reshape(states.shape)
            np.testing.assert_array_equal(values, states)

    def test_legacy_groups_require_consecutive_legal_states(self):
        board = Board(); states = []
        for move in [112, 127, 113, 128]:
            board.play(move); states.append(board.current_state().ravel())
        states = np.asarray(states)
        np.testing.assert_array_equal(infer_game_ids(states), [0, 0, 0, 0])
        wrong = states.copy(); wrong[2, 0] = 1; wrong[2, 113] = 0
        with self.assertRaises(ValueError):
            infer_game_ids(wrong)
        empty = Board().current_state().ravel()[None]
        np.testing.assert_array_equal(infer_game_ids(np.concatenate([empty, states, empty, states])),
                                      [0, 0, 0, 0, 0, 1, 1, 1, 1, 1])


@unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch is optional for data/deployment tests')
class ExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)

    def test_folded_and_training_tensor_exports_match(self):
        import torch
        from export_onnx import Architecture, GomokuNet, infer_architecture, layer_shapes, load_state_from_bin
        config = Architecture(channels=4, res_blocks=2, value_hidden=8)
        torch.manual_seed(1)
        model = GomokuNet(config).eval()
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                with torch.no_grad():
                    module.weight.uniform_(0.5, 1.5)
                    module.bias.uniform_(-0.2, 0.2)
                    module.running_mean.uniform_(-0.3, 0.3)
                    module.running_var.uniform_(0.3, 1.5)
        state = model.state_dict()
        with tempfile.TemporaryDirectory() as tmp:
            raw, folded = Path(tmp)/'raw', Path(tmp)/'folded'
            raw.mkdir(); folded.mkdir()
            def write(directory, name, tensor):
                values = tensor.detach().numpy().astype('<f4')
                (directory/name).write_bytes(struct.pack('<I', values.size)+values.tobytes())
            ci, fi = 0, 0
            for layer in layer_shapes(config):
                if layer.startswith('res'):
                    index, suffix = layer[3:].split('_')
                    target, bn = f'trunk.{index}.conv_{suffix}', f'trunk.{index}.bn_{suffix}'
                else:
                    target, bn = layer, {'stem': 'stem_bn', 'policy_conv1': 'policy_bn', 'val_conv1': 'val_bn'}.get(layer)
                prefix = f'training_conv{ci}' if bn else f'training_fc{fi}'
                ci += bn is not None; fi += bn is None
                weight, bias = state[target+'.weight'], state[target+'.bias']
                write(raw, prefix+'_weight.bin', weight); write(raw, prefix+'_bias.bin', bias)
                if bn:
                    for part, suffix in [('weight', 'gamma'), ('bias', 'beta'), ('running_mean', 'mean'), ('running_var', 'var')]:
                        write(raw, prefix+'_bn_'+suffix+'.bin', state[bn+'.'+part])
                    scale = state[bn+'.weight']/torch.sqrt(state[bn+'.running_var']+1e-5)
                    weight = weight*scale[:, None, None, None]
                    bias = state[bn+'.bias']+(bias-state[bn+'.running_mean'])*scale
                write(folded, layer+'_weight.bin', weight); write(folded, layer+'_bias.bin', bias)
            (raw/'training.meta').write_text(f'version 2\nconv_count {ci}\nfc_count {fi}\n')
            # Raw checkpoints are independently sufficient; folded files may be absent.
            data = torch.randn(3, 4, 15, 15)
            with torch.no_grad():
                expected = model(data)
                for directory in (raw, folded):
                    inferred = infer_architecture(directory)
                    self.assertEqual(inferred, config)
                    loaded = GomokuNet(inferred).eval()
                    loaded.load_state_dict(load_state_from_bin(directory), strict=True)
                    actual = loaded(data)
                    for want, got in zip(expected, actual):
                        torch.testing.assert_close(got, want, atol=2e-6, rtol=2e-6)

    def test_corrupt_tensor_header_is_rejected(self):
        from export_onnx import read_tensor
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'test.bin'
            path.write_bytes(struct.pack('<I', 2)+np.ones(1, '<f4').tobytes())
            with self.assertRaises(ValueError):
                read_tensor(tmp, path.name)

    @unittest.skipUnless(importlib.util.find_spec('onnx') and importlib.util.find_spec('onnxruntime'),
                         'ONNX dependencies are optional')
    def test_export_cli_and_runtime_legal_policy_match(self):
        import onnx
        import torch
        from export_onnx import Architecture, GomokuNet
        from gomoku import ONNXEngine
        torch.manual_seed(19)
        model = GomokuNet(Architecture(channels=4, res_blocks=1, value_hidden=8)).eval()
        board = Board()
        for move in [112, 128, 95, 170]:
            board.play(move)
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint, output = Path(tmp)/'model.pt', Path(tmp)/'model.onnx'
            torch.save(model.state_dict(), checkpoint)
            exporter = Path(__file__).resolve().parents[1]/'tools/export_onnx.py'
            result = subprocess.run([sys.executable, str(exporter), str(checkpoint), str(output)],
                                    capture_output=True, text=True, timeout=60,
                                    env=dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'))
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            engine = ONNXEngine(output, n_playout=2)
            policy, value = engine.policy_value(board)
            self.assertEqual(set(dict(policy)), set(board.availables))
            self.assertAlmostEqual(sum(prob for _, prob in policy), 1)
            with torch.no_grad():
                log_policy, expected_value = model(torch.from_numpy(board.current_state()[None]))
                expected = torch.softmax(log_policy[0, board.availables], dim=0).numpy()
            np.testing.assert_allclose([prob for _, prob in policy], expected, atol=1e-7, rtol=1e-5)
            self.assertAlmostEqual(value, expected_value.item(), places=5)
            exported = onnx.load(output)
            metadata = {field.key: field.value for field in exported.metadata_props}
            self.assertEqual(metadata['rules'], 'freestyle')
            for field in exported.metadata_props:
                if field.key == 'rules':
                    field.value = 'renju'
            onnx.save(exported, output)
            with self.assertRaisesRegex(ValueError, 'rules'):
                ONNXEngine(output)


if __name__ == '__main__':
    unittest.main()
