"""唯一のドライバ。コマンドは2つ。

``run`` は決まった配分でモデルを変換し、perplexity を測る。配分とルーターを
それぞれ複数指定でき、両方に複数を渡せばその直積が走るので、report/13 の 2×2 表
（一様 x=3 / ビーム配分 × 現行ルーター / 方式4）は1コマンドで出る。

  cmoe run --alloc uniform3,beam --router cmoe,oracle_recovery --seeds 0,1,2

同じ配分のルーター違いは**1回の変換**でまかなう。carve も expert 重みも共有し、
``MoE.gate`` だけを差し替えて評価するので、比較が「ルーターだけの差」になる。

``search`` は配分そのものを探す。採点オラクルと探索アルゴリズムを別々に選べる
ので、「同じ目的関数で探索を替える」「同じ探索で安いオラクルに替える」がどちらも
1行の違いになる。

  cmoe search --oracle suffix_kl --search beam --width 4

出てくるのは層ごとの x のベクトルで、そのまま ``run --alloc`` に貼れる。探索と
評価を分けてあるのは、探索が数時間かかる一方で、出た配分を使う実験はその後
何度も走るからである。

``score`` は探索を走らせず、**与えた配分を同じオラクルで採点するだけ**である。

  cmoe score --oracle suffix_kl --alloc uniform3 --alloc uniform4

対照の選び方を探索と揃えるためにある。探索は校正データだけを見て1本を選ぶので、
比べる相手の一様配分も同じ目的関数で選ばないと、評価指標を見てから7本の最良を
取ることになり、対照の側にだけ「たまたまの上振れ」が乗る。

方式の分岐はここに無い。名前は各軸の registry が解決する。
"""

import argparse
import gc
import os
import time
import traceback

import torch

from cmoe import runlog
from cmoe.adapters.registry import create_adapter, guess_adapter
from cmoe.alloc.oracles.base import LayerWalk, score_allocation
from cmoe.alloc.oracles.registry import check_oracle, create_oracle
from cmoe.alloc.search.beam import BudgetExceeded
from cmoe.alloc.search.fixed import parse_allocation
from cmoe.alloc.search.registry import check_search, create_search
from cmoe.assemble import Converter, install_routers, layer_factory
from cmoe.carve.registry import create_carver
from cmoe.data.base import load_tokenizer
from cmoe.data.harness import DEFAULT_CACHE as DEFAULT_BENCH_CACHE
from cmoe.data.registry import load_calibration, load_evaluation, load_splits
from cmoe.eval import bench, bench_stats
from cmoe.eval.ppl import evaluate_ppl
from cmoe.eval.stats import paired_differences, stratified_paired_bootstrap
from cmoe.router.diagnostics import gap_recovered
from cmoe.router.methods.score_calibration import GAIN_LIMIT
from cmoe.router.registry import create_method, resolve_chain

TEXT_NAME, JSON_NAME = 'run.txt', 'summary.json'
SEARCH_JSON = 'search.json'
SCORE_JSON = 'score.json'
# 選択問題の生の尤度の置き場所。summary.json に混ぜないのは、1構成で 1MB 前後
# あり、summary.json は1構成終わるたびに丸ごと書き直されるからである
BENCH_DIR = 'bench'
DEFAULT_ALLOC = 'uniform3'
log = runlog.log


def parse_list(spec):
    return [field.strip() for field in spec.split(',') if field.strip()]


def parse_seeds(spec):
    return [int(field) for field in parse_list(spec)]


def parse_alloc_specs(specs):
    """``--alloc`` を、直積に回す配分の並びに割る。

    1つの ``--alloc`` の中でカンマが2つの意味を持つ — 配分の並び
    （``uniform3,beam``）と、層ごとの値を並べた1本のベクトル（``3,6,6,...``）で
    ある。区別は「全部が数字か」で付く。配分の**名前**が数字だけになることは
    無いので、これは曖昧にならない。

    この判別が無いと、``search`` が「そのまま貼れる」と印字したベクトルを
    ``run`` が32個の未知の配分名として読む。探索の結果を使う経路がそこで切れる。

    ``--alloc`` は繰り返せる。名前とベクトルを混ぜるにはこれしかない —
    ``--alloc uniform3,3,6,6,...`` は1つのカンマ区切りの中で両方を名乗ることに
    なり、上の判別が働かない。探した配分を既定と並べて測るのが、そのまま
    ``--alloc uniform3 --alloc 3,6,6,...`` になる。
    """
    # argparse の append は、既定値を渡すとそこへ**足す**（--alloc beam が
    # ['uniform3', 'beam'] になる）。既定は None にしておいて、ここで入れる
    if specs is None:
        specs = [DEFAULT_ALLOC]
    if isinstance(specs, str):
        specs = [specs]
    rows = []
    for spec in specs:
        fields = parse_list(spec)
        if len(fields) > 1 and all(field.lstrip('-').isdigit() for field in fields):
            rows.append(spec)
            continue
        rows.extend(fields)
    return rows


def build_parser():
    parser = argparse.ArgumentParser(prog='cmoe')
    sub = parser.add_subparsers(dest='command', required=True)

    run = sub.add_parser('run', help='変換して perplexity を測る')
    run.add_argument('--model', default='meta-llama/Llama-2-7b-hf')
    run.add_argument('--adapter', default=None, help='省略時はモデル名から推測する')
    run.add_argument('--alloc', action='append', default=None,
                     help='配分。プリセット名か層数分のカンマ区切り。'
                          '繰り返すと直積（名前とベクトルを混ぜるにはこれ）')
    run.add_argument('--router', default='cmoe',
                     help='ルーター方式（複数可）。方式4 は先行方式を自動で前に挿す')
    run.add_argument('--carver', default='cmoe', help='分割方式')
    run.add_argument('--calib', default='wikitext2', help='キャリブレーションセット')
    run.add_argument('--datasets', default='wikitext2,c4-new', help='評価セット')
    run.add_argument('--seeds', default='0', help='キャリブレーションの seed')
    run.add_argument('--nsamples', type=int, default=8,
                     help='carve に使う系列数。既存の測定はすべて 8')
    run.add_argument('--fit-samples', type=int, default=64,
                     help='ルーター方式が代表を選ぶのに使う系列数')
    run.add_argument('--validation-samples', type=int, default=64,
                     help='ルーター診断に使う系列数')
    run.add_argument('--diagnostics', action='store_true',
                     help='回収率と Oracle 一致率を層ごとに測る（validation が要る）')
    run.add_argument('--nexperts', type=int, default=8)
    run.add_argument('--nactive', type=int, default=6,
                     help='1トークンあたりに走る expert 数 A。全構成で同じ')
    run.add_argument('--k-act', type=int, default=10)
    run.add_argument('--bias-speed', type=float, default=0.001)
    run.add_argument('--seqlen', type=int, default=2048)
    run.add_argument('--batch-chunk', type=int, default=None,
                     help='carve 中にバッチを分ける幅。n が大きいときだけ要る')
    run.add_argument('--fit-batch-chunk', type=int, default=4,
                     help='fit / validation を進めるときのバッチ幅')
    run.add_argument('--token-chunk', type=int, default=4096,
                     help='ルーター方式がトークンを分けて進める幅')
    run.add_argument('--search-token-chunk', type=int, default=8192,
                     help='方式4 の座標上昇がトークンを分けて進める幅')
    run.add_argument('--keep-top', type=int, default=16,
                     help='方式3 が記録する上位候補の数')
    run.add_argument('--gain-limit', type=float, default=GAIN_LIMIT,
                     help='方式5 の gain の探索幅（1/x 〜 x）')
    run.add_argument('--no-bias', action='store_true',
                     help='方式5 の offset 段を走らせない（gain だけ合わせる）')
    run.add_argument('--max-sweeps', type=int, default=10,
                     help='方式4 の座標上昇の掃引上限')
    run.add_argument('--no-profiling-norm', action='store_true')
    run.add_argument('--no-router-norm', action='store_true')
    run.add_argument('--layers', type=int, default=None,
                     help='先頭 N 層だけ変換する（動作確認用）')
    run.add_argument('--no-ppl', action='store_true',
                     help='PPL を測らない。診断だけ見るとき')
    run.add_argument('--bench', action='store_true',
                     help='選択問題ベンチマークも測る')
    run.add_argument('--bench-tasks', default=','.join(bench.DEFAULT_TASKS),
                     help='測るタスク（既定は CMoE 最新版 Table 1 の5つ）')
    run.add_argument('--bench-limit', type=int, default=None,
                     help='タスクあたりの問題数を絞る（動作確認用）')
    run.add_argument('--bench-batch-size', type=int, default=8)
    run.add_argument('--bench-fewshot', type=int, default=0)
    run.add_argument('--bench-cache-dir', default=DEFAULT_BENCH_CACHE,
                     help='ベンチのデータセットのキャッシュ。共有キャッシュを'
                          '使うには空文字を渡す')
    run.add_argument('--bench-reference', default=None,
                     help='測り済みの dense を基準に使う（bench/dense.json への'
                          'パス）。dense は seed にも配分にも校正にも依らないので、'
                          '同じモデル・同じタスクなら測り直す必要はない')
    run.add_argument('--no-bench-dense', action='store_true',
                     help='dense の基準を測らない。ref_kl / ref_agreement が落ちる')
    run.add_argument('--bootstrap-reps', type=int, default=10000)
    run.add_argument('--bootstrap-seed', type=int, default=20260813)
    run.add_argument('--out', default=None)

    search = sub.add_parser('search', help='層ごとの x を探す')
    add_oracle_arguments(search, layers_help='先頭 N 層だけ探索する（配線確認用）')
    search.add_argument('--search', default='beam', help='探索アルゴリズム')
    search.add_argument('--width', type=int, default=None,
                        help='各層で生き残る接頭辞の本数。省略時は探索ごとの既定'
                             '（beam は4、greedy は1）')
    search.add_argument('--budget', type=float, default=None,
                        help='オラクルのコスト上限。単位はオラクルが決める')
    search.add_argument('--no-recheck', action='store_true',
                        help='勝った配分を頭から測り直さない')

    score = sub.add_parser('score', help='与えた配分をオラクルで採点する')
    add_oracle_arguments(score, layers_help='先頭 N 層だけ採点する（配線確認用）')
    score.add_argument('--alloc', action='append', default=None,
                       help='採点する配分。プリセット名か層数分のカンマ区切り。'
                            '繰り返すと全部を採点して並べる')
    return parser


def add_oracle_arguments(parser, layers_help):
    """オラクルを組むのに要る引数。``search`` と ``score`` で同一である。

    採点の条件が1文字でもずれると2つのコマンドの数は比べられないので、
    別々に並べるのではなく同じ関数から生やす。
    """
    parser.add_argument('--model', default='meta-llama/Llama-2-7b-hf')
    parser.add_argument('--adapter', default=None, help='省略時はモデル名から推測する')
    parser.add_argument('--oracle', default='suffix_kl',
                        help='採点オラクル。安い順に mass / local_error / suffix_kl')
    parser.add_argument('--carver', default='cmoe', help='分割方式')
    parser.add_argument('--calib', default='wikitext2', help='キャリブレーションセット')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--nsamples', type=int, default=8,
                        help='キャリブレーション系列数。全コストがこれに比例する')
    parser.add_argument('--nexperts', type=int, default=8)
    parser.add_argument('--nactive', type=int, default=6,
                        help='1トークンあたりに走る expert 数 A。全候補で同じ')
    parser.add_argument('--k-act', type=int, default=10)
    parser.add_argument('--bias-speed', type=float, default=0.001)
    parser.add_argument('--seqlen', type=int, default=2048)
    parser.add_argument('--batch-chunk', type=int, default=None,
                        help='系列を分ける幅。n が大きいときだけ要る')
    parser.add_argument('--token-chunk', type=int, default=None,
                        help='オラクルがトークンを分ける幅')
    parser.add_argument('--no-profiling-norm', action='store_true')
    parser.add_argument('--no-router-norm', action='store_true')
    parser.add_argument('--layers', type=int, default=None, help=layers_help)
    parser.add_argument('--out', default=None)


def configure_method(method, args):
    """CLI の分割幅を方式へ渡す。方式ごとの分岐ではなく、持っていれば設定する。"""
    for name, value in (('chunk_size', args.token_chunk),
                        ('keep_top', args.keep_top),
                        ('max_sweeps', args.max_sweeps),
                        ('gain_limit', args.gain_limit),
                        ('fit_bias', not args.no_bias)):
        if hasattr(method, name):
            setattr(method, name, value)
    if hasattr(method, 'token_chunk'):
        method.token_chunk = args.search_token_chunk
    return method


def configurations(args):
    """(配分名, ルーター名) の直積。先頭が対照になる。"""
    rows = []
    for alloc in parse_alloc_specs(args.alloc):
        for router in parse_list(args.router):
            rows.append((alloc, router))
    return rows


def load_carving_data(args, seed, methods):
    """carve と、必要なら fit / validation を読む。

    fit を読むのは、それを要求する方式があるときだけ。要らないときに 64 系列を
    トークナイズして流すのは、そのぶん丸ごと無駄になる。
    """
    needs_fit = any(getattr(method, 'requires_fit_z', False) for method in methods)
    if not needs_fit and not args.diagnostics:
        carve = load_calibration(
            args.calib, args.model, args.seqlen, args.nsamples, seed)
        return carve, None, None
    splits = load_splits(
        args.calib, args.model, args.seqlen, seed,
        carve_count=args.nsamples, fit_count=args.fit_samples,
        validation_count=args.validation_samples)
    validation = splits.validation if args.diagnostics else None
    return splits.carve, splits.fit, validation


def run_one(args, alloc_spec, router_names, seed, evaluation_sets):
    """1配分 × 1 seed。要求された全ルーター方式を1回の変換でまかなう。"""
    adapter_name = args.adapter or guess_adapter(args.model)
    adapter = create_adapter(adapter_name, args.model, seqlen=args.seqlen)

    build_order = resolve_chain(router_names)
    methods = [configure_method(create_method(name), args) for name in build_order]
    carve, fit, validation = load_carving_data(args, seed, methods)
    log(f'  carve: {carve.name} {tuple(carve.input_ids.shape)} '
        f'hash={carve.metadata()["token_hash"][:12]}')
    if fit is not None:
        log(f'  fit:   {fit.name} {tuple(fit.input_ids.shape)} '
            f'hash={fit.metadata()["token_hash"][:12]}')
    if validation is not None:
        log(f'  val:   {validation.name} {tuple(validation.input_ids.shape)} '
            f'hash={validation.metadata()["token_hash"][:12]}')
    if build_order != list(dict.fromkeys(router_names)):
        log(f'  構築順: {" -> ".join(build_order)}')

    search = parse_allocation(alloc_spec, n_active_total=args.nactive)
    n_layers = args.layers or adapter.n_layers
    allocation = search.search(None, n_layers)

    converter = Converter(
        adapter,
        create_carver(args.carver, args.nexperts),
        methods,
        n_experts=args.nexperts,
        k_act=args.k_act,
        bias_speed=args.bias_speed,
        profiling_norm=not args.no_profiling_norm,
        router_norm=not args.no_router_norm,
        batch_chunk=args.batch_chunk,
        fit_batch_chunk=args.fit_batch_chunk,
        token_chunk=args.token_chunk,
        n_layers=args.layers,
        log=lambda message: None,
    )
    started = time.time()
    report = converter.convert(carve, allocation, fit=fit, validation=validation)
    log(f'  変換 {report.seconds:.1f}s (平均 x={allocation.mean_x:.2f})')

    diagnostics = summarize_diagnostics(report)
    for name, values in diagnostics.items():
        log(f'  診断 {name:<20} R={values["router_r"]:.6f} '
            f'gap回収={values["gap_recovered"]:.2%} '
            f'recall={values["oracle_mean_recall"]:.4f} '
            f'一致率={values["oracle_exact_set_rate"]:.4f}')

    results = {}
    if not args.no_ppl:
        for name in router_names:
            install_routers(adapter, report.routers[name])
            results[name] = {}
            for dataset, token_set in evaluation_sets.items():
                result = evaluate_ppl(adapter, token_set)
                results[name][dataset] = result
                log(f'  {name:<20} {dataset:<10} {result.ppl:.6f}')

    bench_samples = {}
    if args.bench:
        tokenizer = load_tokenizer(args.model)
        for name in router_names:
            # PPL を測らなかった経路でもルーターは載せる必要がある
            install_routers(adapter, report.routers[name])
            adapter.to_device()
            started_bench = time.time()
            bench_samples[name] = bench.evaluate_bench(
                adapter.model, tokenizer,
                tasks=parse_list(args.bench_tasks),
                batch_size=args.bench_batch_size, limit=args.bench_limit,
                num_fewshot=args.bench_fewshot, max_length=args.seqlen,
                cache_dir=args.bench_cache_dir or None, model_name=args.model)
            rows = bench_stats.summarize(bench_samples[name])
            for task, values in rows['tasks'].items():
                log(f'  {name:<20} {task:<16} acc={values["acc"]:.4f} '
                    f'acc_norm={values["acc_norm"]:.4f} '
                    f'gold_nll={values["gold_nll"]:.6f}')
            log(f'  {name:<20} {"macro":<16} acc={rows["macro"]["acc"]:.4f} '
                f'acc_norm={rows["macro"]["acc_norm"]:.4f} '
                f'gold_nll={rows["macro"]["gold_nll"]:.6f} '
                f'({time.time() - started_bench:.1f}s)')

    payload = {
        'allocation': allocation.metadata(),
        'routers': list(router_names),
        'build_order': build_order,
        'carver': args.carver,
        'seed': seed,
        'data': {
            'carve': carve.metadata(),
            'fit': fit.metadata() if fit is not None else None,
            'validation': validation.metadata() if validation is not None else None,
        },
        'conversion': report.as_dict(),
        'diagnostics': diagnostics,
        'ppl': {name: {dataset: result.as_dict()
                       for dataset, result in rows.items()}
                for name, rows in results.items()},
        # 生の尤度は別ファイル。ここには平均だけ置く
        'bench': {name: bench_stats.summarize(rows)
                  for name, rows in bench_samples.items()},
        'seconds': time.time() - started,
    }

    del converter, adapter, report
    gc.collect()
    torch.cuda.empty_cache()
    return payload, results, bench_samples


def summarize_diagnostics(report):
    """層ごとの診断を、routing する層の平均にまとめる。

    Top-K=0 の層は入らない。何も選ばない層では回収率は全方式で shared 質量に
    等しく、平均に混ぜると方式の差が薄まるだけである（report/13 と同じ扱い）。
    """
    rows = [record.diagnostics for record in report.layers if record.diagnostics]
    if not rows:
        return {}
    baseline = report.baseline_method
    summary = {}
    for name in rows[0]:
        values = [row[name] for row in rows]
        router_r = sum(value['router_r'] for value in values) / len(values)
        oracle_r = sum(value['oracle_r'] for value in values) / len(values)
        baseline_r = sum(row[baseline]['router_r'] for row in rows) / len(rows)
        recovered = gap_recovered(baseline_r, router_r, oracle_r)
        summary[name] = {
            'router_r': router_r,
            'oracle_r': oracle_r,
            'baseline_r': baseline_r,
            'gap_recovered': recovered if recovered is not None else 0.0,
            'oracle_mean_recall': sum(
                value['oracle_mean_recall'] for value in values) / len(values),
            'oracle_exact_set_rate': sum(
                value['oracle_exact_set_rate'] for value in values) / len(values),
            'n_routing_layers': len(values),
        }
    return summary


def summarize(records, configs, datasets, reps, seed):
    """構成ごとの平均 PPL と、先頭構成に対する対応のある差。"""
    summary = {'configurations': [], 'baseline': f'{configs[0][0]}+{configs[0][1]}'}
    baseline_key = configs[0]
    for config in configs:
        row = {'allocation': config[0], 'router': config[1], 'datasets': {}}
        for dataset in datasets:
            values = [record['ppl'][dataset].ppl
                      for record in records if record['config'] == config]
            if not values:
                continue
            entry = {'ppl_per_seed': values, 'mean_ppl': sum(values) / len(values)}
            if config != baseline_key:
                differences = []
                for record in records:
                    if record['config'] != config:
                        continue
                    base = next(
                        (other for other in records
                         if other['config'] == baseline_key
                         and other['seed'] == record['seed']), None)
                    if base is None:
                        continue
                    differences.append(paired_differences(
                        base['ppl'][dataset], record['ppl'][dataset]))
                if differences:
                    entry['paired_vs_baseline'] = stratified_paired_bootstrap(
                        differences, reps=reps, seed=seed)
            row['datasets'][dataset] = entry
        summary['configurations'].append(row)
    return summary


def summarize_bench(records, configs, reference, reps, seed):
    """構成ごとの指標の平均と、先頭構成に対する対応のある差。

    再抽出の層は **(seed × タスク)** である。タスクごとに問題数が 1,200〜10,000
    と一桁違うので、層を等しく重み付けする既定の挙動がそのままマクロ平均になる
    — まとめて1つの層にすると、平均が HellaSwag の話になる。
    """
    baseline_key = configs[0]
    summary = {'baseline': f'{baseline_key[0]}+{baseline_key[1]}',
               'reference': 'dense' if reference else None,
               'configurations': []}
    for config in configs:
        mine = [record for record in records
                if record['config'] == config and record.get('bench')]
        if not mine:
            continue
        per_seed = [bench_stats.summarize(record['bench'], reference)
                    for record in mine]
        row = {
            'allocation': config[0], 'router': config[1], 'n_seeds': len(mine),
            'n_docs': per_seed[0]['n_docs'],
            'tasks': {
                task: {metric: sum(one['tasks'][task][metric] for one in per_seed)
                       / len(per_seed)
                       for metric in per_seed[0]['tasks'][task]}
                for task in per_seed[0]['tasks']},
            'macro': {metric: sum(one['macro'][metric] for one in per_seed)
                      / len(per_seed)
                      for metric in per_seed[0]['macro']},
        }
        if config != baseline_key:
            row['paired_vs_baseline'] = paired_bench(
                records, mine, baseline_key, reference,
                list(per_seed[0]['macro']), reps, seed)
        summary['configurations'].append(row)
    return summary


def paired_bench(records, mine, baseline_key, reference, metrics, reps, seed):
    """指標ごとに、(seed × タスク) を層とした対応のある差の信頼区間。"""
    paired = {}
    for metric in metrics:
        strata = []
        for record in mine:
            base = next((other for other in records
                         if other['config'] == baseline_key
                         and other['seed'] == record['seed']
                         and other.get('bench')), None)
            if base is None:
                continue
            for task, samples in record['bench'].items():
                strata.append(bench_stats.paired_differences(
                    base['bench'][task], samples, metric,
                    reference=reference.get(task) if reference else None))
        if not strata:
            continue
        result = stratified_paired_bootstrap(
            strata, reps=reps, seed=seed, key='mean_difference',
            unit='paired-question-within-seed-and-task',
            lower_is_better=not bench_stats.HIGHER_IS_BETTER[metric])
        # 層は seed ではなく (seed × タスク) なので、名前もそう呼ぶ
        result['improved_strata'] = result.pop('improved_seeds')
        result['n_strata'] = result.pop('n_seeds')
        result['higher_is_better'] = bench_stats.HIGHER_IS_BETTER[metric]
        paired[metric] = result
    return paired


def adopt_dense_bench(args, out):
    """測り済みの dense を基準に取り込む。

    dense は seed にも配分にも校正にも依らないので、同じモデル・同じタスクなら
    測り直す意味が無い（1回 8.8分。seed 3本 × 校正2種なら 53分がそのまま消える）。

    ただし「同じ問題を同じ順に測ったもの」でなければ対応のある比較にならない。
    タスクの顔ぶれ・問題数・shot 数・limit を先に突き合わせ、食い違えば取り込ま
    ずに止める。問題単位のずれはこの先 ``bench_stats.check_aligned`` が捕まえる。

    取り込んだものは出力先にも複製する。あとから読むとき、その実行の
    ディレクトリだけで完結していないと基準が辿れなくなる。
    """
    samples = bench.load_samples(args.bench_reference)
    wanted = parse_list(args.bench_tasks)
    if sorted(samples) != sorted(wanted):
        raise SystemExit(
            f'{args.bench_reference} のタスクは {sorted(samples)} で、'
            f'いま測る {sorted(wanted)} と違う')
    for name in wanted:
        row = samples[name]
        if row.model and row.model != args.model:
            raise SystemExit(
                f'{args.bench_reference} の {name} は {row.model} を測ったもので、'
                f'いま測る {args.model} と違う。doc_hash は問題側のハッシュなので'
                'そのままでは食い違いに気づけない')
        if row.num_fewshot != args.bench_fewshot or row.limit != args.bench_limit:
            raise SystemExit(
                f'{args.bench_reference} の {name} は '
                f'{row.num_fewshot}-shot / limit={row.limit} で、'
                f'いま測る {args.bench_fewshot}-shot / limit={args.bench_limit} と違う')
    log()
    log(f'== dense の基準（{args.bench_reference} から取り込み）==')
    rows = bench_stats.summarize(samples)
    for task, values in rows['tasks'].items():
        log(f'  {task:<16} {rows["n_docs"][task]:>6} 問  acc={values["acc"]:.4f} '
            f'acc_norm={values["acc_norm"]:.4f} gold_nll={values["gold_nll"]:.6f}')
    target = os.path.join(out, BENCH_DIR, 'dense.json')
    runlog.write_json(target, {name: row.as_dict() for name, row in samples.items()})
    with open(os.path.join(out, BENCH_DIR, 'dense_from.txt'), 'w') as handle:
        handle.write(f'{args.bench_reference}\n')
    return samples


def measure_dense_bench(args, out):
    """変換前の dense を1回だけ測る。``ref_kl`` / ``ref_agreement`` の基準。

    seed にも配分にもルーターにも依らないので、実行あたり1回で足りる。基準が
    無くても残り4指標は出るが、「どれだけ壊したか」を正誤と切り離して見る道は
    ここでしか作れない。
    """
    adapter_name = args.adapter or guess_adapter(args.model)
    adapter = create_adapter(adapter_name, args.model, seqlen=args.seqlen)
    adapter.to_device()
    log()
    log('== dense の基準 ==')
    started = time.time()
    samples = bench.evaluate_bench(
        adapter.model, load_tokenizer(args.model),
        tasks=parse_list(args.bench_tasks), batch_size=args.bench_batch_size,
        limit=args.bench_limit, num_fewshot=args.bench_fewshot,
        max_length=args.seqlen, cache_dir=args.bench_cache_dir or None,
        model_name=args.model)
    rows = bench_stats.summarize(samples)
    for task, values in rows['tasks'].items():
        log(f'  {task:<16} {rows["n_docs"][task]:>6} 問  acc={values["acc"]:.4f} '
            f'acc_norm={values["acc_norm"]:.4f} gold_nll={values["gold_nll"]:.6f}')
    log(f'  {"macro":<16} {"":>6}     acc={rows["macro"]["acc"]:.4f} '
        f'acc_norm={rows["macro"]["acc_norm"]:.4f} '
        f'gold_nll={rows["macro"]["gold_nll"]:.6f} ({time.time() - started:.1f}s)')
    runlog.write_json(
        os.path.join(out, BENCH_DIR, 'dense.json'),
        {name: row.as_dict() for name, row in samples.items()})
    del adapter
    gc.collect()
    torch.cuda.empty_cache()
    return samples


def command_run(args):
    configs = configurations(args)
    allocs = parse_alloc_specs(args.alloc)
    routers = parse_list(args.router)
    seeds = parse_seeds(args.seeds)
    datasets = parse_list(args.datasets)

    out = args.out or runlog.default_out_dir('run')
    runlog.prepare_out_dir(out, JSON_NAME)
    runlog.open_mirror(os.path.join(out, TEXT_NAME))

    log(f'model={args.model} carve={args.calib} n={args.nsamples} '
        f'N={args.nexperts} A={args.nactive}')
    log(f'配分 {len(allocs)} 種 × ルーター {len(routers)} 種 × seed {len(seeds)} 本 '
        f'= 変換 {len(allocs) * len(seeds)} 回 / 評価 {len(configs) * len(seeds)} 通り')
    log(f'出力 {out}')

    evaluation_sets = {}
    if not args.no_ppl:
        evaluation_sets = {
            name: load_evaluation(name, args.model, args.seqlen) for name in datasets}
        for name, token_set in evaluation_sets.items():
            meta = token_set.metadata()
            log(f'評価 {name}: {meta["n_tokens"]} トークン '
                f'hash={meta["token_hash"][:12]}')

    payload = {
        'arguments': vars(args),
        'configurations': [{'allocation': a, 'router': r} for a, r in configs],
        'seeds': seeds,
        'datasets': {name: token_set.metadata()
                     for name, token_set in evaluation_sets.items()},
        'runs': [],
    }
    reference = None
    if args.bench:
        os.makedirs(os.path.join(out, BENCH_DIR), exist_ok=True)
        log(f'ベンチ {args.bench_tasks}'
            + (f' limit={args.bench_limit}' if args.bench_limit else '')
            + f' {args.bench_fewshot}-shot')
        if args.bench_reference:
            reference = adopt_dense_bench(args, out)
        elif not args.no_bench_dense:
            reference = measure_dense_bench(args, out)
    records = []
    failures = []
    for seed in seeds:
        for alloc_spec in allocs:
            log()
            log(f'== seed {seed} / {alloc_spec} / {",".join(routers)} ==')
            try:
                run_payload, results, bench_samples = run_one(
                    args, alloc_spec, routers, seed, evaluation_sets)
            except Exception as error:  # 1構成の失敗で残りを捨てない
                log(f'  失敗: {type(error).__name__}: {error}')
                # 落ちた場所まで残す。ミラーが唯一の記録になることがある
                log(traceback.format_exc())
                failures.append(
                    {'seed': seed, 'allocation': alloc_spec, 'routers': routers,
                     'error': f'{type(error).__name__}: {error}'})
                payload['failures'] = failures
                runlog.write_json(os.path.join(out, JSON_NAME), payload)
                gc.collect()
                torch.cuda.empty_cache()
                continue
            index = len(payload['runs'])
            payload['runs'].append(run_payload)
            # 生の尤度は構成ごとに1ファイル。名前に配分をそのまま使うと、探索が
            # 出したベクトル（'3,6,6,...'）がファイル名になるので通し番号にする
            for router, samples in bench_samples.items():
                relative = os.path.join(BENCH_DIR, f'run{index:03d}_{router}.json')
                runlog.write_json(
                    os.path.join(out, relative),
                    {name: row.as_dict() for name, row in samples.items()})
                run_payload.setdefault('bench_samples', {})[router] = relative
            for router in routers:
                if router in results or router in bench_samples:
                    records.append({'config': (alloc_spec, router), 'seed': seed,
                                    'ppl': results.get(router),
                                    'bench': bench_samples.get(router)})
            if any(record['ppl'] for record in records):
                payload['summary'] = summarize(
                    [record for record in records if record['ppl']],
                    configs, datasets, args.bootstrap_reps, args.bootstrap_seed)
            if any(record['bench'] for record in records):
                payload['bench_summary'] = summarize_bench(
                    records, configs, reference, args.bootstrap_reps,
                    args.bootstrap_seed)
            runlog.write_json(os.path.join(out, JSON_NAME), payload)

    if 'summary' in payload:
        log()
        log('== まとめ (平均 PPL、小さいほど良い) ==')
    for row in payload.get('summary', {}).get('configurations', []):
        for dataset, entry in row['datasets'].items():
            line = (f'{row["allocation"]:>10} + {row["router"]:<20} '
                    f'{dataset:<10} {entry["mean_ppl"]:.6f}')
            paired = entry.get('paired_vs_baseline')
            if paired:
                line += (f'  対照比 NLL {paired["mean_nll_difference"]:+.6f} '
                         f'[{paired["lower"]:+.6f}, {paired["upper"]:+.6f}] '
                         f'{paired["improved_seeds"]}/{paired["n_seeds"]} seed 改善')
            log(line)
    if 'bench_summary' in payload:
        log()
        log('== ベンチマーク（マクロ平均。acc/acc_norm/margin/ref_agreement は'
            '大きいほど、gold_nll/ref_kl は小さいほど良い）==')
        for row in payload['bench_summary']['configurations']:
            head = f'{row["allocation"]:>10} + {row["router"]:<20}'
            log(head + '  ' + '  '.join(
                f'{metric}={value:.6f}' for metric, value in row['macro'].items()))
            for metric, paired in row.get('paired_vs_baseline', {}).items():
                log(f'{"":>10}   {metric:<16} 対照比 '
                    f'{paired["mean_difference"]:+.6f} '
                    f'[{paired["lower"]:+.6f}, {paired["upper"]:+.6f}] '
                    f'{paired["improved_strata"]}/{paired["n_strata"]} '
                    '(seed × タスク) 改善')
    if failures:
        log(f'失敗した構成: {len(failures)}')
    runlog.close_mirror()
    return 1 if failures else 0


def build_walk(args, adapter, calibration):
    """探索が使う層まわしを組む。

    層に載る MoE を作る役だけは組み立て役から借りる（``layer_factory``）。
    オラクルは分割規則もルーター方式も知らないまま、候補の層を測れる。
    """
    inputs = adapter.capture_layer_inputs(calibration.input_ids)
    factory = layer_factory(
        create_carver(args.carver, args.nexperts), args.nexperts,
        bias_speed=args.bias_speed, router_norm=not args.no_router_norm,
        device=adapter.device)
    return LayerWalk(
        adapter, inputs, factory, args.nexperts,
        n_active_total=args.nactive, k_act=args.k_act,
        profiling_norm=not args.no_profiling_norm, batch_chunk=args.batch_chunk)


def check_search_arguments(args):
    """モデルを読み込む前に、引数だけで分かる誤りを出し切る。

    7B を読み終えてから引数の綴り違いで落ちると、十数分がそのために消える。
    runlog が「終わった結果がある場所」を先に拒むのと同じ理由で、断れるものは
    測る前に断る。
    """
    check_search(args.search, args.width)
    check_oracle_arguments(args)


def check_oracle_arguments(args):
    """``search`` と ``score`` に共通の、オラクルまわりの引数検査。"""
    check_oracle(args.oracle)
    if args.layers is not None and args.layers < 1:
        # `args.layers or n_layers` は 0 を falsy として全層に化かす。配線確認の
        # つもりの --layers 0 で本番が始まる
        raise SystemExit(f'--layers は 1 以上（{args.layers}）')
    if args.nactive >= args.nexperts and args.oracle != 'suffix_kl':
        raise SystemExit(
            f'A={args.nactive} は N={args.nexperts} 以上。どの候補も routed を'
            f'全部走らせるので、{args.oracle} は全候補で 0 を返し、何も測らない')


def command_search(args):
    check_search_arguments(args)

    out = args.out or runlog.default_out_dir(f'search_{args.search}')
    runlog.prepare_out_dir(out, SEARCH_JSON)
    runlog.open_mirror(os.path.join(out, TEXT_NAME))
    json_path = os.path.join(out, SEARCH_JSON)

    adapter_name = args.adapter or guess_adapter(args.model)
    log(f'model={args.model} carve={args.calib} n={args.nsamples} seed={args.seed} '
        f'N={args.nexperts} A={args.nactive}')
    log(f'探索 {args.search}(幅 {args.width or "既定"}) × オラクル {args.oracle}')
    log(f'出力 {out}')

    adapter = create_adapter(adapter_name, args.model, seqlen=args.seqlen)
    calibration = load_calibration(
        args.calib, args.model, args.seqlen, args.nsamples, args.seed)
    log(f'  carve: {calibration.name} {tuple(calibration.input_ids.shape)} '
        f'hash={calibration.metadata()["token_hash"][:12]}')

    walk = build_walk(args, adapter, calibration)
    oracle = create_oracle(args.oracle, walk)
    if args.token_chunk is not None and hasattr(oracle, 'token_chunk'):
        oracle.token_chunk = args.token_chunk

    n_layers = min(args.layers or adapter.n_layers, adapter.n_layers)
    partial = n_layers != adapter.n_layers
    if partial:
        log(f'層 0..{n_layers - 1} だけを探索する。出てくるのは接頭辞で、'
            'そのままでは変換に使えない')

    payload = {
        'arguments': vars(args),
        'model': args.model,
        'n_layers': n_layers,
        'oracle': {'name': oracle.name, 'cost_unit': oracle.cost_unit},
        'search': {'name': args.search, 'width': args.width,
                   'budget': args.budget},
        'calibration': calibration.metadata(),
        'layers': [],
    }
    # 床は最後の測り直しの合否を決めるので、それを持つオラクルでは必ず測る。
    # 読み出し1回ぶんで、探索本体に比べれば無視できる
    floor = None
    if hasattr(oracle, 'nondeterminism_floor'):
        floor = oracle.nondeterminism_floor()
        payload['nondeterminism'] = floor
        log(f'dense 読み出しを2回: KL {floor:.3e}（この機械の非決定性の床。'
            'これより小さい差は区別できない）')
    runlog.write_json(json_path, payload)

    def on_layer(records):
        payload['layers'] = records
        runlog.write_json(json_path, payload)

    search = create_search(args.search, width=args.width,
                           n_active_total=args.nactive, budget=args.budget,
                           log=log, on_layer=on_layer)
    log()
    started = time.time()
    try:
        allocation = search.search(oracle, n_layers)
    except BudgetExceeded as error:
        # 予算切れは失敗ではなく結果である。決まったところまでを残す
        log(f'予算切れ: {error}')
        payload['budget_exceeded'] = {
            'prefix': list(error.prefix), 'spent': error.spent,
            'budget': error.budget}
        runlog.write_json(json_path, payload)
        runlog.close_mirror()
        return 1
    seconds = time.time() - started

    best_score = search.records[-1]['beam'][0]['score']
    payload.update({
        'allocation': allocation.metadata(),
        'score': best_score,
        'spent': oracle.spent,
        'calls': oracle.calls,
        'seconds': seconds,
    })
    log()
    log(f'{allocation.name}: score {best_score:.6e}  平均 x={allocation.mean_x:.2f}  '
        f'コスト {oracle.spent:g} {oracle.cost_unit}（{oracle.calls} 回）')
    if partial:
        log(f'接頭辞 {alloc_flag(allocation)}（{n_layers}/{adapter.n_layers} 層。'
            'run --alloc は全層ぶんを要求するので、これは貼るためのものではない）')
    else:
        log(f'配分: --alloc {alloc_flag(allocation)}')
    runlog.write_json(json_path, payload)

    if not args.no_recheck:
        # 探索が勝者に付けたスコアは、系譜をたどって組み立てた数である。同じ配分を
        # 頭から測り直すと、その帳簿が正しかったかが分かる（伝播も採点も同じ関数を
        # 通るので、一致するはずである）
        log()
        log('勝った配分を頭から測り直す ...')
        result = score_allocation(oracle, allocation)
        gap = abs(result.score - best_score)
        # 床があるならそれと比べて合否を言う。合否を言わない検査は、数時間の
        # ログの中で誰も読まない1行になる
        above = None if floor is None else gap > floor
        log(f'  {result.score:.6e} / 探索の {best_score:.6e}  差 {gap:.3e}'
            + ('' if floor is None else f'（床 {floor:.3e}）'))
        if above:
            # 例外にはしない。表はもうディスクにあり、traceback で終わる実行は
            # 警告ごと結果を捨てられる
            log('  WARNING: 探索と測り直しが、この機械の非決定性の床を超えて'
                '食い違っている。表のスコアはそれが名指すモデルの数ではない — '
                '配分を使う前に読むこと')
        payload['recheck'] = {'score': result.score, 'gap': gap,
                              'above_floor': above,
                              'per_layer': result.details['per_layer']}
        runlog.write_json(json_path, payload)

    log(f'{seconds / 60:.1f} 分')
    runlog.close_mirror()
    return 0


def alloc_flag(allocation):
    return ','.join(str(x) for x in allocation)


def command_score(args):
    """与えた配分を、探索と同じオラクルで頭から採点する。

    探索を1回も呼ばない。``score_allocation`` は探索が勝った配分を測り直すのに
    使っているのと同じ関数で、同じ層の進め方・同じ採点を通る。だから、ここで出る
    数と ``search`` の ``score`` は直接比べられる。
    """
    check_oracle_arguments(args)
    specs = parse_alloc_specs(args.alloc)
    # 配分の綴り違いはモデルを読む前に出す。層数はまだ分からないので、ここで
    # 落とせるのは名前と整数の並びだけである
    searches = [(spec, parse_allocation(spec, args.nactive)) for spec in specs]

    out = args.out or runlog.default_out_dir(f'score_{args.oracle}')
    runlog.prepare_out_dir(out, SCORE_JSON)
    runlog.open_mirror(os.path.join(out, TEXT_NAME))
    json_path = os.path.join(out, SCORE_JSON)

    adapter_name = args.adapter or guess_adapter(args.model)
    log(f'model={args.model} carve={args.calib} n={args.nsamples} seed={args.seed} '
        f'N={args.nexperts} A={args.nactive}')
    log(f'採点 オラクル {args.oracle} × 配分 {len(searches)} 本')
    log(f'出力 {out}')

    adapter = create_adapter(adapter_name, args.model, seqlen=args.seqlen)
    calibration = load_calibration(
        args.calib, args.model, args.seqlen, args.nsamples, args.seed)
    log(f'  carve: {calibration.name} {tuple(calibration.input_ids.shape)} '
        f'hash={calibration.metadata()["token_hash"][:12]}')

    walk = build_walk(args, adapter, calibration)
    oracle = create_oracle(args.oracle, walk)
    if args.token_chunk is not None and hasattr(oracle, 'token_chunk'):
        oracle.token_chunk = args.token_chunk

    n_layers = min(args.layers or adapter.n_layers, adapter.n_layers)
    payload = {
        'arguments': vars(args),
        'model': args.model,
        'n_layers': n_layers,
        'oracle': {'name': oracle.name, 'cost_unit': oracle.cost_unit},
        'calibration': calibration.metadata(),
        'scores': [],
    }
    floor = None
    if hasattr(oracle, 'nondeterminism_floor'):
        floor = oracle.nondeterminism_floor()
        payload['nondeterminism'] = floor
        log(f'dense 読み出しを2回: KL {floor:.3e}（この機械の非決定性の床。'
            'これより小さい差は区別できない）')
    runlog.write_json(json_path, payload)

    started = time.time()
    for index, (spec, search) in enumerate(searches):
        allocation = search.search(oracle, n_layers)
        log()
        log(f'== {index + 1}/{len(searches)} {allocation.name} '
            f'({alloc_flag(allocation)}) ==')
        spent_before = oracle.spent
        elapsed = time.time()
        result = score_allocation(oracle, allocation)
        elapsed = time.time() - elapsed
        log(f'  score {result.score:.6e}  平均 x={allocation.mean_x:.2f}  '
            f'コスト {oracle.spent - spent_before:g} {oracle.cost_unit}'
            f'（{elapsed / 60:.1f} 分）')
        payload['scores'].append({
            'spec': spec,
            'allocation': allocation.metadata(),
            'score': result.score,
            'cost': oracle.spent - spent_before,
            'seconds': elapsed,
            'per_layer': result.details['per_layer'],
        })
        runlog.write_json(json_path, payload)

    # 並べて出す。この表の一番良い1本が、探索と対等な選び方をした対照である
    log()
    log(f'{"配分":<24} {"平均 x":>7} {"score":>14}')
    best = min(payload['scores'], key=lambda row: row['score'])
    for row in payload['scores']:
        mark = ' <- 最良' if row is best else ''
        log(f'{row["allocation"]["name"]:<24} '
            f'{row["allocation"]["mean_x"]:>7.2f} {row["score"]:>14.6e}{mark}')
    payload['best'] = {'spec': best['spec'],
                       'name': best['allocation']['name'],
                       'score': best['score']}
    runlog.write_json(json_path, payload)
    log()
    log('score は同じ校正トークンの上でしか比べられない。'
        '校正が違う行を並べても意味は無い')
    log(f'{(time.time() - started) / 60:.1f} 分')
    runlog.close_mirror()
    return 0


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == 'run':
        return command_run(args)
    if args.command == 'search':
        return command_search(args)
    if args.command == 'score':
        return command_score(args)
    raise SystemExit(f'未知のコマンド {args.command!r}')


if __name__ == '__main__':
    raise SystemExit(main())
