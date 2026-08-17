"""配分の採点オラクル。極小モデルの CPU スモークと、元実装との一致。

見るのは3つ。

* 契約 — A を動かさない、割合として成り立つ、接頭辞のスコアの積み方が言う通り
* 一貫性 — オラクルが測った軌道と、組み立て役が実際に載せる軌道が同じ数を出す
* 一致 — 移送元 ``CMoE-ref/xsearch/`` と同じ入力に同じ答えを返す

3つ目は元実装が無い環境では丸ごと skip する。移送済みのコードは元実装が無くても
動く、というのが「参照専用ディレクトリを import しない」方針の意味である。
"""

import importlib
import json
from pathlib import Path
import sys
import types
from types import SimpleNamespace

import pytest
import torch

from cmoe.adapters.llama import LlamaAdapter
from cmoe.alloc.base import Allocation
from cmoe.alloc.oracles.base import LayerWalk, PrefixState, score_allocation
from cmoe.alloc.oracles.local_error import LocalErrorOracle, missed_energy
from cmoe.alloc.oracles.mass import (MassOracle, neuron_to_expert,
                                     recovered_mass, router_selection)
from cmoe.alloc.oracles.registry import create_oracle
from cmoe.alloc.oracles.suffix_kl import SuffixKLOracle, kl_divergence
from cmoe.alloc.search.registry import create_search
from cmoe.assemble import Converter, layer_factory
from cmoe.carve.registry import create_carver
from cmoe.data.base import TokenSet
from cmoe.router.registry import create_method

# A < N。Top-K = A - x が routed 数 N - x より必ず小さくなるので、どの候補でも
# ルーターが実際に選び、取りこぼしが出る。A = N だとどの候補も routed を全部
# 走らせてしまい、層ローカル指標は恒等的に 0 になって何も測らない。
N_EXPERTS, N_ACTIVE, SEQLEN = 4, 3, 16
# 「全部走らせたときはちょうど 0」という境界だけは A = N でしか作れない
N_ACTIVE_EXACT = N_EXPERTS


@pytest.fixture
def adapter():
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        vocab_size=128, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=SEQLEN * 8)
    torch.manual_seed(0)
    model = LlamaForCausalLM(config).to(torch.bfloat16)
    model.eval()
    model.config.use_cache = False
    return LlamaAdapter(model, seqlen=SEQLEN, device='cpu')


def token_set(name='calib', shape=(2, SEQLEN), seed=0):
    generator = torch.Generator().manual_seed(seed)
    return TokenSet(name, torch.randint(0, 128, shape, generator=generator))


def make_walk(adapter, tokens=None, batch_chunk=None, n_active_total=N_ACTIVE):
    tokens = tokens or token_set()
    inputs = adapter.capture_layer_inputs(tokens.input_ids)
    factory = layer_factory(create_carver('cmoe', N_EXPERTS), N_EXPERTS,
                            device=adapter.device)
    return LayerWalk(adapter, inputs, factory, N_EXPERTS,
                     n_active_total=n_active_total, batch_chunk=batch_chunk)


def measure_one(oracle, x, layer=0):
    """1層ぶん測る。オラクルの中身を直接見たいときの近道。"""
    walk = oracle.walk
    state = walk.root()
    profile = walk.profile(state, layer)
    carved = walk.carve(profile, x)
    child = walk.propagate(profile, carved)
    return oracle.measure(profile, carved, state, child), profile, carved, child


# -- 契約 -------------------------------------------------------------------


def test_every_candidate_runs_the_same_number_of_experts(adapter):
    walk = make_walk(adapter)
    assert walk.candidates(0) == (0, 1, 2, 3)      # x <= A、かつ x < N
    state = walk.root()
    profile = walk.profile(state, 0)
    for x in walk.candidates(0):
        carved = walk.carve(profile, x)
        assert carved.n_shared + carved.topk == N_ACTIVE
        assert carved.moe.n_shared_experts + carved.moe.n_activated_experts == N_ACTIVE
        assert len(carved.partition.routed_groups) == N_EXPERTS - x
        # A < N なので、どの候補も routed のうち少なくとも1つを走らせない
        assert carved.topk < len(carved.partition.routed_groups)


def test_a_candidate_outside_the_budget_is_refused(adapter):
    walk = make_walk(adapter)
    profile = walk.profile(walk.root(), 0)
    with pytest.raises(ValueError, match='候補'):
        walk.carve(profile, N_EXPERTS)


@pytest.mark.parametrize('squared', [False, True])
def test_nothing_is_missed_when_every_routed_expert_runs(adapter, squared):
    """routed を全部選べば、回収率はちょうど 1・出力誤差はちょうど 0。

    層ローカル指標の定義そのものの境界で、ここが合わない実装は「走らせたぶん」と
    「取りこぼしたぶん」を取り違えている。A < N である限りルーターがこの選択を
    することはない（Top-K = A-x は routed 数 N-x より必ず小さい）ので、選択を
    手で作って下位の関数に直接渡す。

    ちょうど 1 になるのは偶然ではない。回収の和が全体の和と同じ列を同じ順序で
    足すよう、gather ではなくマスクで組んであるからである。
    """
    walk = make_walk(adapter)
    profile = walk.profile(walk.root(), 0)
    carved = walk.carve(profile, 1)
    h = walk.true_activations(profile)
    n_routed = len(carved.partition.routed_groups)
    everything = torch.arange(n_routed).expand(h.shape[0], n_routed)

    r, _ = recovered_mass(h, carved.partition, everything, squared)
    assert r == 1.0

    group_of = neuron_to_expert(carved.partition, h.shape[1], h.device)
    w32 = profile.dense.down_proj.weight.to(torch.float32)
    missed = missed_energy(h, w32, group_of, everything, n_routed)
    assert float(missed.sum()) == 0.0


@pytest.mark.parametrize('name', ['mass', 'mass_squared', 'local_error'])
def test_a_layer_local_oracle_refuses_a_setting_it_cannot_measure(adapter, name):
    """A >= N ではどの候補も routed を全部走らせるので、層ローカル指標は無力。

    黙って 0 を返すと、ビームは同点崩しで並べただけの配分を「成功」として返す。
    測れない設定は、測る前に断る。
    """
    walk = make_walk(adapter, n_active_total=N_ACTIVE_EXACT)
    with pytest.raises(ValueError, match='何も測らない'):
        create_oracle(name, walk)
    # suffix_kl は A = N でも意味のある数を返すので、断らない
    assert create_oracle('suffix_kl', walk) is not None


@pytest.mark.parametrize('name', ['mass', 'mass_squared', 'local_error'])
def test_the_score_of_one_layer_is_a_fraction(adapter, name):
    """どの候補も何かを取りこぼし、その割合は 0 と 1 の間に入る。"""
    oracle = create_oracle(name, make_walk(adapter))
    scores = [measure_one(oracle, x)[0].score for x in (0, 1, 2, 3)]
    assert all(0.0 < score < 1.0 for score in scores)
    # 候補どうしが実際に区別されている（区別しないなら順位付けの意味が無い）
    assert len(set(scores)) == len(scores)


def test_the_prefix_score_is_the_sum_of_the_layers(adapter):
    """接頭辞のスコアは、層ごとの L をそのまま足したものである。

    「足した」ことまで見る。単に増えることを見るだけだと、``max`` に変わっても
    ``2 * L`` に変わっても通ってしまう。
    """
    oracle = create_oracle('local_error', make_walk(adapter))
    allocation = Allocation((2, 3), n_active_total=N_ACTIVE)
    result = score_allocation(oracle, allocation)
    rows = result.details['per_layer']
    assert [row['x'] for row in rows] == [2, 3]
    assert rows[0]['score'] == pytest.approx(rows[0]['details']['l'])
    assert rows[1]['score'] == pytest.approx(
        rows[0]['details']['l'] + rows[1]['details']['l'])
    # 最後の層のスコアが、その配分の答えになる
    assert result.score == rows[-1]['score']
    assert result.cost == oracle.spent
    assert result.details['calls'] == 2


def test_the_denominator_is_shared_by_the_candidates_of_a_layer(adapter):
    """L の分母は dense の FFN だけで決まるので、同じ層の候補で使い回す。

    層をまたいで使い回すと L が全部おかしくなる。使い回しの成立と、層が変われば
    作り直すことの両方を見る。
    """
    walk = make_walk(adapter)
    oracle = create_oracle('local_error', walk)
    root = walk.root()
    profile = walk.profile(root, 0)

    totals = []
    for x in (1, 2):
        carved = walk.carve(profile, x)
        child = walk.propagate(profile, carved)
        oracle.measure(profile, carved, root, child)
        totals.append(oracle._total)
    # 同じ層の2候補は、同じ分母テンソルそのものを使う
    assert totals[0][1] is totals[1][1]
    assert totals[0][0]() is profile

    # 次の層へ移れば作り直す
    carved = walk.carve(profile, 1)
    child = walk.propagate(profile, carved)
    next_profile = walk.profile(child, 1)
    next_carved = walk.carve(next_profile, 1)
    oracle.measure(next_profile, next_carved,
                   child, walk.propagate(next_profile, next_carved))
    assert oracle._total[1] is not totals[0][1]


def test_the_denominator_cache_does_not_keep_the_capture_alive(adapter):
    """分母のキャッシュは弱参照で持つ。

    強参照だと、``LayerWalk`` が捕捉を手放したあともここが掴んでいて、その層の
    z・residual・真の H が次の層の途中まで生き残る（1層ぶんで数 GiB）。
    """
    walk = make_walk(adapter)
    oracle = create_oracle('local_error', walk)
    root = walk.root()
    profile = walk.profile(root, 0)
    carved = walk.carve(profile, 1)
    oracle.measure(profile, carved, root, walk.propagate(profile, carved))

    reference = oracle._total[0]
    assert reference() is profile
    walk.release(root)          # 捕捉を手放す
    del profile, carved
    assert reference() is None  # オラクル側が掴んだままではない


def test_measuring_the_same_prefix_twice_gives_the_same_number(adapter):
    oracle = create_oracle('local_error', make_walk(adapter))
    first, _, _, _ = measure_one(oracle, 2)
    second, _, _, _ = measure_one(oracle, 2)
    assert first.score == second.score


def test_the_capture_and_the_statistics_are_shared_by_the_candidates(adapter):
    """同じ接頭辞・同じ層なら、捕捉と活性統計は1回しか作らない。

    x ごとに測り直すと、探索の費用が候補の数だけ増える。
    """
    walk = make_walk(adapter)
    state = walk.root()
    first = walk.profile(state, 0)
    assert walk.profile(state, 0) is first
    # 次の層へ移れば作り直す
    child = walk.propagate(first, walk.carve(first, 1))
    assert walk.profile(child, 1) is not first


def test_a_released_prefix_cannot_be_expanded(adapter):
    walk = make_walk(adapter)
    root = walk.root()
    profile = walk.profile(root, 0)
    state = walk.propagate(profile, walk.carve(profile, 1))
    walk.release(state)
    with pytest.raises(ValueError, match='解放済み'):
        walk.profile(state, 1)


def test_releasing_drops_the_capture_even_for_the_calibration_input(adapter):
    """根は pinned だが、その上で作った捕捉は手放す。

    キャリブレーション入力そのものは実行の最後まで要る（最後に測り直すのも
    そこから始まる）。捕まえた z・residual・真の H は層1本ぶんの大きさがあり、
    握ったままだと層0 の捕捉が最後まで残る。
    """
    walk = make_walk(adapter)
    root = walk.root()
    profile = walk.profile(root, 0)
    walk.true_activations(profile)
    walk.release(root)
    assert root.hidden is not None                 # 入力そのものは残る
    assert walk.profile(root, 0) is not profile    # 捕捉は作り直しになる


def test_the_oracle_never_converts_the_model(adapter):
    """探索はモデルを書き換えない。層を差し替えて戻す、をしない。

    戻し忘れという失敗の形が無いことが、同じ層の候補を好きな順で測れる理由
    でもある。
    """
    oracle = create_oracle('mass', make_walk(adapter))
    create_search('beam', width=2, n_active_total=N_ACTIVE).search(oracle, 2)
    assert not any(adapter.is_converted(index) for index in range(adapter.n_layers))


def test_the_winner_measures_the_same_when_walked_again(adapter):
    """探索が勝者に付けたスコアは、系譜をたどって組み立てた数である。

    同じ配分を頭から測り直して同じ数が出ることが、その帳簿が正しいことの確認に
    なる（元実装の final_recheck にあたる）。
    """
    search = create_search('beam', width=2, n_active_total=N_ACTIVE)
    allocation = search.search(create_oracle('mass', make_walk(adapter)), 2)
    again = score_allocation(create_oracle('mass', make_walk(adapter)), allocation)
    assert again.score == pytest.approx(search.records[-1]['beam'][0]['score'],
                                        rel=1e-12)


def test_splitting_the_batch_does_not_change_the_score(adapter):
    """--batch-chunk はメモリの都合であって、測る量ではない。"""
    tokens = token_set()
    whole = create_oracle('mass', make_walk(adapter, tokens))
    split = create_oracle('mass', make_walk(adapter, tokens, batch_chunk=1))
    assert (measure_one(whole, 2)[0].score
            == pytest.approx(measure_one(split, 2)[0].score, rel=1e-6))


def test_splitting_the_tokens_does_not_change_the_selection(adapter):
    """ルーターの選択はトークンごとに独立なので、分けても同じ。

    ここが崩れるなら、塊をまたいで何かが混ざっている（softmax も Top-K も
    行ごとに閉じているはず）。分けて進めるのは、z がホストに残っているときに
    重みの側へ塊だけを渡すためである。
    """
    walk = make_walk(adapter)
    profile = walk.profile(walk.root(), 0)
    carved = walk.carve(profile, 1)
    whole = router_selection(carved.moe, profile.z)
    split = router_selection(carved.moe, profile.z, token_chunk=3)
    assert torch.equal(whole, split)
    # 選択は z と同じ側に戻る。このあと同じ側にある H と突き合わせるため
    assert split.device == profile.z.device


@pytest.mark.parametrize('name', ['mass', 'local_error'])
def test_splitting_the_tokens_does_not_change_the_score(adapter, name):
    walk = make_walk(adapter)
    whole = create_oracle(name, walk)
    whole.token_chunk = None
    split = create_oracle(name, make_walk(adapter))
    split.token_chunk = 3
    assert (measure_one(whole, 2)[0].score
            == pytest.approx(measure_one(split, 2)[0].score, rel=1e-6))


# -- suffix KL --------------------------------------------------------------


def test_the_dense_readout_is_reproducible(adapter):
    """同じ dense モデルを2回読んだ差 = この機械の非決定性の床。

    CPU では 0 で、それは「読み出しの経路に、実行ごとに揺れるものが無い」と
    いう主張そのものである。
    """
    oracle = SuffixKLOracle(make_walk(adapter))
    assert oracle.nondeterminism_floor() == 0.0


def test_the_suffix_kl_counts_the_layers_it_actually_runs(adapter):
    oracle = SuffixKLOracle(make_walk(adapter))
    first, _, _, _ = measure_one(oracle, 2, layer=0)
    assert first.score > 0.0
    # 自分の層と、そのあとに走らせた層
    assert first.cost == float(adapter.n_layers)
    assert first.details['dense_suffix_layers'] == adapter.n_layers - 1


def test_the_score_describes_the_model_the_conversion_builds(adapter):
    """オラクルが測った軌道と、組み立て役が実際に載せる軌道が一致する。

    探索は層を差し替えずに ``moe(z) + residual`` で先へ進み、変換は層そのものを
    差し替えて進む。この2つが違う数を出すなら、探索は配備されないモデルを
    採点していたことになる。
    """
    tokens = token_set()
    walk = make_walk(adapter, tokens)
    oracle = SuffixKLOracle(walk)
    allocation = Allocation((1, 2), n_active_total=N_ACTIVE)
    result = score_allocation(oracle, allocation)
    dense_logits = oracle.dense_logits

    converter = Converter(adapter, create_carver('cmoe', N_EXPERTS),
                          [create_method('cmoe')], n_experts=N_EXPERTS)
    converter.convert(tokens, allocation)
    converted = adapter.forward_suffix(0, walk.inputs.hidden, walk.inputs)
    kl, _ = kl_divergence(dense_logits, converted)
    assert kl == pytest.approx(result.score, rel=1e-6, abs=1e-9)


# -- 元実装との一致 ---------------------------------------------------------

REFERENCE = Path(__file__).resolve().parents[1] / 'CMoE-ref' / 'xsearch'


@pytest.fixture(scope='module')
def xsearch():
    """元実装の xsearch を、パッケージの ``__init__`` を通さずに読み込む。

    ``xsearch/__init__.py`` は matplotlib まで引き込むので、必要な2枚だけを
    ディレクトリ指定で読む（依存は torch と標準ライブラリのみ）。
    """
    if not REFERENCE.is_dir():
        pytest.skip('参照実装 CMoE-ref/ が無い')
    package = types.ModuleType('xsearch')
    package.__path__ = [str(REFERENCE)]
    sys.modules.setdefault('xsearch', package)
    return SimpleNamespace(
        simulate=importlib.import_module('xsearch.simulate'),
        output_error=importlib.import_module('xsearch.output_error'))


def reference_pair(profile, carved, h):
    """元実装の evaluate に渡す candidate / capture（同じ層・同じ選択）。"""
    partition = carved.partition
    candidate = SimpleNamespace(
        n_experts=N_EXPERTS,
        n_shared=carved.n_shared,
        n_activated=carved.topk,
        shared_group=list(partition.shared_group),
        routed_groups=[list(group) for group in partition.routed_groups],
        router=carved.moe.gate,
        source_mlp=profile.dense,
    )
    return candidate, SimpleNamespace(h_true=h, z=profile.z)


@pytest.mark.parametrize('x', [1, 2, 3])
def test_the_router_selection_is_the_reference_selection(adapter, xsearch, x):
    walk = make_walk(adapter)
    profile = walk.profile(walk.root(), 0)
    carved = walk.carve(profile, x)
    mine = router_selection(carved.moe, profile.z)
    _, theirs = xsearch.simulate.select(carved.moe.gate, profile.z)
    assert torch.equal(mine, theirs)


@pytest.mark.parametrize('x', [1, 2, 3])
@pytest.mark.parametrize('squared', [False, True])
def test_recovered_mass_matches_the_reference(adapter, xsearch, x, squared):
    walk = make_walk(adapter)
    oracle = MassOracle(walk, squared=squared)
    result, profile, carved, child = measure_one(oracle, x)

    h = walk.true_activations(profile)
    candidate, capture = reference_pair(profile, carved, h)
    reference = xsearch.simulate.MassEvaluator(squared_mass=squared).evaluate(
        candidate, capture)
    # ビット一致を求める。回収率は候補に順序を付けるためだけの数で、順序が
    # 変わらないことが移送の条件である
    assert result.details['r'] == reference.r
    assert result.details['shared_mass'] == reference.shared_mass
    assert result.details['total_mass'] == reference.total_mass


@pytest.mark.parametrize('x', [1, 2, 3])
def test_output_error_matches_the_reference(adapter, xsearch, x):
    walk = make_walk(adapter)
    oracle = LocalErrorOracle(walk)
    result, profile, carved, child = measure_one(oracle, x)

    h = walk.true_activations(profile)
    candidate, capture = reference_pair(profile, carved, h)
    reference = xsearch.output_error.OutputErrorEvaluator().evaluate(
        candidate, capture)
    assert result.details['l'] == reference.l
    assert result.details['missed_energy'] == reference.missed_energy
    assert result.details['total_energy'] == reference.total_energy


# -- CLI ---------------------------------------------------------------------


def test_the_search_command_runs_end_to_end(tmp_path, monkeypatch):
    """``cmoe search`` が端から端まで通り、結果を書き出す。

    極小モデルとトークンを registry に差し込んで走らせる。見るのは配線 —
    引数がオラクルと探索に届き、層ごとに書き出され、勝った配分が再測定される
    ことで、数値そのものは他のテストが見ている。
    """
    import cmoe.adapters.registry as adapters
    import cmoe.data.registry as data
    from cmoe.cli import main
    from cmoe import runlog

    def load_tiny(name, seqlen=SEQLEN, device=None):
        from transformers import LlamaConfig, LlamaForCausalLM

        config = LlamaConfig(
            vocab_size=128, hidden_size=64, intermediate_size=128,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
            max_position_embeddings=SEQLEN * 8)
        torch.manual_seed(0)
        model = LlamaForCausalLM(config).to(torch.bfloat16)
        model.eval()
        model.config.use_cache = False
        return LlamaAdapter(model, seqlen=seqlen, device='cpu')

    monkeypatch.setitem(adapters.ADAPTERS, 'tiny',
                        type('Tiny', (), {'load': staticmethod(load_tiny)}))
    monkeypatch.setitem(
        data.CALIBRATION_SETS, 'tiny',
        lambda model, seqlen, n, seed: token_set('tiny-calib', (n, seqlen), seed))

    out = tmp_path / 'search'
    code = main(['search', '--adapter', 'tiny', '--model', 'tiny',
                 '--calib', 'tiny', '--oracle', 'local_error',
                 '--search', 'beam', '--width', '2',
                 '--nexperts', str(N_EXPERTS), '--nactive', str(N_ACTIVE),
                 '--nsamples', '2', '--seqlen', str(SEQLEN), '--out', str(out)])
    runlog.close_mirror()
    assert code == 0

    payload = json.loads((out / 'search.json').read_text())
    assert len(payload['allocation']['values']) == 2
    assert payload['allocation']['n_active_total'] == N_ACTIVE
    assert len(payload['layers']) == 2          # 層ごとに書き出されている
    assert payload['oracle'] == {'name': 'local_error',
                                 'cost_unit': 'layer_ffn_evals'}
    # 探索が付けたスコアと、同じ配分を頭から測り直した値が一致する。探索の
    # 帳簿（系譜をたどって組み立てた数）が正しいことの確認
    assert payload['recheck']['gap'] == 0.0
    assert payload['recheck']['score'] == payload['score']
    assert (out / 'run.txt').exists()


def test_the_kl_matches_the_reference(adapter, xsearch):
    """KL は元実装の downstream.kl_divergence と同じ数を返す。"""
    downstream = importlib.import_module('xsearch.downstream')
    torch.manual_seed(3)
    reference_logits = torch.randn(2, SEQLEN, 32, dtype=torch.bfloat16)
    candidate_logits = reference_logits + 0.1 * torch.randn(2, SEQLEN, 32).to(
        torch.bfloat16)
    mine, n_mine = kl_divergence(reference_logits, candidate_logits)
    theirs, n_theirs = downstream.kl_divergence(reference_logits, candidate_logits)
    assert (mine, n_mine) == (theirs, n_theirs)
