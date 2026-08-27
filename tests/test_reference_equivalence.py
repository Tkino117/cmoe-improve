"""移送した方式が、元実装と同じ入力に同じ答えを返すか。

GPU のアンカー（experiments/00_anchor.py）は end-to-end の PPL を突き合わせるが、
アンカーが通らない方式（方式2・3・6）もある。ここでは方式の関数を直接、同じ
乱数入力で呼んで**ビット一致**を確かめる。CPU で数秒。

元実装は参照専用の ``CMoE-ref/`` にあり、リポジトリでは追跡していない。無い環境
では丸ごと skip する — 移送済みのコードは元実装が無くても動く、というのが
`third_party を import しない` 方針の意味である。

元実装のパッケージ __init__ は matplotlib まで引き込むので、方式のファイルだけを
パス指定で読み込む（依存は torch と標準ライブラリのみ）。
"""

import importlib.util
from pathlib import Path
import sys
import types

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest

REFERENCE = Path(__file__).resolve().parents[1] / 'CMoE-ref' / 'routerlab' / 'methods'

pytestmark = pytest.mark.skipif(
    not REFERENCE.is_dir(), reason='参照実装 CMoE-ref/ が無い')

N, HIDDEN, GROUPS, TOKENS = 64, 32, 4, 128
PER_EXPERT = N // (GROUPS + 1)


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, REFERENCE / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def fixtures():
    torch.manual_seed(0)
    permutation = torch.randperm(N).tolist()
    groups = [tuple(permutation[i * PER_EXPERT:(i + 1) * PER_EXPERT])
              for i in range(GROUPS + 1)]
    return {
        'gate': torch.randn(N, HIDDEN),
        'up': torch.randn(N, HIDDEN),
        'z': torch.randn(TOKENS, HIDDEN),
        'groups': groups,
        'routed': groups[1:],
        'rates': torch.rand(N),
        'markers': (torch.rand(TOKENS, N) < 0.2).float(),
    }


def test_frequency_centroid_matches_reference(fixtures):
    from cmoe.router.methods.frequency_centroid import frequency_weighted_representatives

    reference = load('ref_fc', 'frequency_centroid.py')
    assert (reference.frequency_weighted_representatives(
        fixtures['routed'], fixtures['rates'], fixtures['markers'])
        == frequency_weighted_representatives(
            fixtures['routed'], fixtures['rates'], fixtures['markers']))


def test_oracle_correlation_matches_reference(fixtures):
    from cmoe.router.methods.oracle_correlation import select_correlated_representatives

    reference = load('ref_oc', 'oracle_correlation.py')
    expected = reference.select_oracle_correlated_representatives(
        fixtures['routed'], fixtures['z'], fixtures['gate'], fixtures['up'])
    got = select_correlated_representatives(
        fixtures['routed'], fixtures['z'], fixtures['gate'], fixtures['up'])
    assert expected.representatives == got.representatives
    assert expected.correlations == got.correlations


def test_oracle_recovery_matches_reference(fixtures):
    from cmoe.router.methods.oracle_recovery import select_recovery_representatives

    reference = load('ref_or', 'oracle_recovery.py')
    starts = (tuple(group[0] for group in fixtures['routed']),
              tuple(group[1] for group in fixtures['routed']))
    arguments = (fixtures['groups'], 2, fixtures['z'], fixtures['gate'],
                 fixtures['up'], starts)
    expected = reference.select_recovery_representatives(*arguments)
    got = select_recovery_representatives(*arguments)
    assert expected.representatives == got.representatives
    assert expected.recovery == got.recovery
    # 探索の経路まで一致する（受理した手の数と順序）
    assert ([[move['to_neuron'] for move in start['accepted_moves']]
             for start in expected.starts]
            == [[move['to_neuron'] for move in start['accepted_moves']]
                for start in got.starts])


@pytest.mark.parametrize('centered', [False, True])
def test_expert_mean_matches_reference(fixtures, centered):
    from cmoe.router.methods.expert_mean import expert_mean_rows

    reference = load('ref_em', 'expert_mean.py')
    expected = reference.expert_mean_rows(
        fixtures['routed'], fixtures['gate'], fixtures['up'], centered=centered)
    got = expert_mean_rows(
        fixtures['routed'], fixtures['gate'], fixtures['up'], centered=centered)
    assert torch.equal(expected[0], got[0])
    assert torch.equal(expected[1], got[1])
    assert expected[2] == got[2]


def test_oracle_abs_matches_reference(fixtures):
    from cmoe.router.methods.oracle_abs import OracleAbsRouter

    reference = load('ref_oa', 'oracle_abs.py')
    gate = torch.stack([fixtures['gate'][list(g)] for g in fixtures['routed']])
    up = torch.stack([fixtures['up'][list(g)] for g in fixtures['routed']])
    expected = reference.OracleAbsRouter(gate, up, 2)(fixtures['z'])[1]
    got = OracleAbsRouter(gate, up, 2)(fixtures['z'])[1]
    assert torch.equal(expected, got)


def test_diagnostics_match_reference(fixtures):
    from cmoe.adapters.base import DenseFFN
    from cmoe.carve.base import Partition
    from cmoe.router.base import build_baseline_router
    from cmoe.router.diagnostics import evaluate_routers_against_abs_oracle

    reference = load('ref_oc', 'oracle_correlation.py')
    dense = DenseFFN(HIDDEN, N, nn.Linear(HIDDEN, N, bias=False),
                     nn.Linear(HIDDEN, N, bias=False),
                     nn.Linear(N, HIDDEN, bias=False), F.silu)
    dense.gate_proj.weight.data = fixtures['gate'].clone()
    dense.up_proj.weight.data = fixtures['up'].clone()
    partition = Partition(GROUPS + 1, 1, tuple(fixtures['groups']),
                          tuple(group[0] for group in fixtures['routed']))
    router = build_baseline_router(dense, partition, 2)

    arguments = ({'r': router}, fixtures['z'], fixtures['groups'],
                 fixtures['gate'], fixtures['up'])
    expected = reference.evaluate_routers_against_abs_oracle(*arguments)['r']
    got = evaluate_routers_against_abs_oracle(*arguments)['r']
    for key in ('router_r', 'oracle_r', 'oracle_mean_recall',
                'oracle_exact_set_rate', 'mean_representative_mass_correlation'):
        assert expected[key] == got[key], key


def test_score_calibration_matches_reference(fixtures):
    from cmoe.router.methods.score_calibration import calibrate_router_scores

    # 元実装の方式5 は方式4 を ``routerlab.methods`` 経由で読む。パス指定で読む
    # ここでは、その1本だけを本物の名前で先に登録して繋ぐ。
    package = types.ModuleType('routerlab')
    package.__path__ = []
    methods = types.ModuleType('routerlab.methods')
    methods.__path__ = []
    sys.modules.setdefault('routerlab', package)
    sys.modules.setdefault('routerlab.methods', methods)
    load('routerlab.methods.oracle_recovery', 'oracle_recovery.py')
    reference = load('ref_sc', 'score_calibration.py')
    representatives = tuple(group[0] for group in fixtures['routed'])
    arguments = (fixtures['groups'], 2, fixtures['z'], fixtures['gate'],
                 fixtures['up'], representatives)
    expected = reference.calibrate_router_scores(*arguments)
    got = calibrate_router_scores(*arguments)
    assert expected.gains == got.gains
    assert expected.biases == got.biases
    assert expected.recovery == got.recovery
    assert expected.gain_recovery == got.gain_recovery
    assert expected.faithfulness == got.faithfulness
    # 探索の経路まで一致する（受理した手の段・expert・行き先）
    assert ([(move['stage'], move['expert'], move['to']) for move in expected.moves]
            == [(move['stage'], move['expert'], move['to']) for move in got.moves])
