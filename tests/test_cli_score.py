"""``cmoe score`` の引数まわり。モデルは読まない。

``score`` は「探索と同じ条件で、与えた配分を採点する」ためだけにある。だから
見るべきなのは採点の中身（それは ``score_allocation`` の試験にある）ではなく、
**``search`` と条件がずれていないこと**である。オラクルを組む引数がどちらかに
しか無ければ、2つのコマンドが出した score は比べられなくなる。
"""

import pytest

from cmoe.cli import build_parser, check_oracle_arguments, parse_alloc_specs

# オラクルを組むのに効く引数。ここに並んだものが search と score の両方に、
# 同じ既定値で無いといけない
ORACLE_OPTIONS = [
    '--model', '--adapter', '--oracle', '--carver', '--calib', '--seed',
    '--nsamples', '--nexperts', '--nactive', '--k-act', '--bias-speed',
    '--seqlen', '--batch-chunk', '--token-chunk', '--no-profiling-norm',
    '--no-router-norm', '--layers', '--out',
]


def options(command):
    for action in build_parser()._subparsers._group_actions[0].choices[command]._actions:
        for flag in action.option_strings:
            yield flag, action.default


def test_search_and_score_take_the_same_oracle_arguments():
    """条件がずれたら比べられない。名前と既定値の両方を見る。"""
    search, score = dict(options('search')), dict(options('score'))
    for flag in ORACLE_OPTIONS:
        assert flag in search, f'search に {flag} が無い'
        assert flag in score, f'score に {flag} が無い'
        assert search[flag] == score[flag], f'{flag} の既定値が食い違っている'


def test_search_keeps_its_own_arguments():
    """探索だけの引数は score に漏れない（採点に探索の幅は無い）。"""
    search, score = dict(options('search')), dict(options('score'))
    for flag in ('--search', '--width', '--budget', '--no-recheck'):
        assert flag in search
        assert flag not in score
    assert '--alloc' in score and '--alloc' not in search


def test_score_takes_several_allocations():
    args = build_parser().parse_args(
        ['score', '--alloc', 'uniform3', '--alloc', '4,5,6'])
    assert parse_alloc_specs(args.alloc) == ['uniform3', '4,5,6']


def test_score_without_alloc_scores_the_default():
    args = build_parser().parse_args(['score'])
    assert parse_alloc_specs(args.alloc) == ['uniform3']


def test_score_refuses_zero_layers():
    args = build_parser().parse_args(['score', '--layers', '0'])
    with pytest.raises(SystemExit):
        check_oracle_arguments(args)


def test_score_refuses_a_layer_local_oracle_that_measures_nothing():
    """A = N ではどの候補も routed を全部走らせる。mass は全候補で 0 を返す。"""
    args = build_parser().parse_args(
        ['score', '--oracle', 'mass', '--nactive', '8', '--nexperts', '8'])
    with pytest.raises(SystemExit):
        check_oracle_arguments(args)
    # suffix_kl は出力分布を見るので、A = N でも候補を区別する
    args = build_parser().parse_args(
        ['score', '--oracle', 'suffix_kl', '--nactive', '8', '--nexperts', '8'])
    check_oracle_arguments(args)
