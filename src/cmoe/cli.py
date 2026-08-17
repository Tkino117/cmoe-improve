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
from cmoe.data.registry import load_calibration, load_evaluation, load_splits
from cmoe.eval.ppl import evaluate_ppl
from cmoe.eval.stats import paired_differences, stratified_paired_bootstrap
from cmoe.router.diagnostics import gap_recovered
from cmoe.router.registry import create_method, resolve_chain

TEXT_NAME, JSON_NAME = 'run.txt', 'summary.json'
SEARCH_JSON = 'search.json'
log = runlog.log


def parse_list(spec):
    return [field.strip() for field in spec.split(',') if field.strip()]


def parse_seeds(spec):
    return [int(field) for field in parse_list(spec)]


def parse_alloc_specs(spec):
    """``--alloc`` を、直積に回す配分の並びに割る。

    カンマが2つの意味を持つ — 配分の並び（``uniform3,beam``）と、層ごとの値を
    並べた1本のベクトル（``3,6,6,...``）である。区別は「全部が数字か」で付く。
    配分の**名前**が数字だけになることは無いので、これは曖昧にならない。

    この判別が無いと、``search`` が「そのまま貼れる」と印字したベクトルを
    ``run`` が32個の未知の配分名として読む。探索の結果を使う経路がそこで切れる。
    """
    fields = parse_list(spec)
    if len(fields) > 1 and all(field.lstrip('-').isdigit() for field in fields):
        return [spec]
    return fields


def build_parser():
    parser = argparse.ArgumentParser(prog='cmoe')
    sub = parser.add_subparsers(dest='command', required=True)

    run = sub.add_parser('run', help='変換して perplexity を測る')
    run.add_argument('--model', default='meta-llama/Llama-2-7b-hf')
    run.add_argument('--adapter', default=None, help='省略時はモデル名から推測する')
    run.add_argument('--alloc', default='uniform3',
                     help='配分。プリセット名か層数分のカンマ区切り。複数渡すと直積')
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
    run.add_argument('--max-sweeps', type=int, default=10,
                     help='方式4 の座標上昇の掃引上限')
    run.add_argument('--no-profiling-norm', action='store_true')
    run.add_argument('--no-router-norm', action='store_true')
    run.add_argument('--layers', type=int, default=None,
                     help='先頭 N 層だけ変換する（動作確認用）')
    run.add_argument('--no-ppl', action='store_true',
                     help='PPL を測らない。診断だけ見るとき')
    run.add_argument('--bootstrap-reps', type=int, default=10000)
    run.add_argument('--bootstrap-seed', type=int, default=20260813)
    run.add_argument('--out', default=None)

    search = sub.add_parser('search', help='層ごとの x を探す')
    search.add_argument('--model', default='meta-llama/Llama-2-7b-hf')
    search.add_argument('--adapter', default=None, help='省略時はモデル名から推測する')
    search.add_argument('--oracle', default='suffix_kl',
                        help='採点オラクル。安い順に mass / local_error / suffix_kl')
    search.add_argument('--search', default='beam', help='探索アルゴリズム')
    search.add_argument('--width', type=int, default=None,
                        help='各層で生き残る接頭辞の本数。省略時は探索ごとの既定'
                             '（beam は4、greedy は1）')
    search.add_argument('--budget', type=float, default=None,
                        help='オラクルのコスト上限。単位はオラクルが決める')
    search.add_argument('--carver', default='cmoe', help='分割方式')
    search.add_argument('--calib', default='wikitext2', help='キャリブレーションセット')
    search.add_argument('--seed', type=int, default=0)
    search.add_argument('--nsamples', type=int, default=8,
                        help='キャリブレーション系列数。探索の全コストがこれに比例する')
    search.add_argument('--nexperts', type=int, default=8)
    search.add_argument('--nactive', type=int, default=6,
                        help='1トークンあたりに走る expert 数 A。全候補で同じ')
    search.add_argument('--k-act', type=int, default=10)
    search.add_argument('--bias-speed', type=float, default=0.001)
    search.add_argument('--seqlen', type=int, default=2048)
    search.add_argument('--batch-chunk', type=int, default=None,
                        help='系列を分ける幅。n が大きいときだけ要る')
    search.add_argument('--token-chunk', type=int, default=None,
                        help='オラクルがトークンを分ける幅')
    search.add_argument('--no-profiling-norm', action='store_true')
    search.add_argument('--no-router-norm', action='store_true')
    search.add_argument('--layers', type=int, default=None,
                        help='先頭 N 層だけ探索する（配線確認用）')
    search.add_argument('--no-recheck', action='store_true',
                        help='勝った配分を頭から測り直さない')
    search.add_argument('--out', default=None)
    return parser


def configure_method(method, args):
    """CLI の分割幅を方式へ渡す。方式ごとの分岐ではなく、持っていれば設定する。"""
    for name, value in (('chunk_size', args.token_chunk),
                        ('keep_top', args.keep_top),
                        ('max_sweeps', args.max_sweeps)):
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
        'seconds': time.time() - started,
    }

    del converter, adapter, report
    gc.collect()
    torch.cuda.empty_cache()
    return payload, results


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
    records = []
    failures = []
    for seed in seeds:
        for alloc_spec in allocs:
            log()
            log(f'== seed {seed} / {alloc_spec} / {",".join(routers)} ==')
            try:
                run_payload, results = run_one(
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
            payload['runs'].append(run_payload)
            for router in routers:
                if router in results:
                    records.append({'config': (alloc_spec, router), 'seed': seed,
                                    'ppl': results[router]})
            if records:
                payload['summary'] = summarize(
                    records, configs, datasets, args.bootstrap_reps,
                    args.bootstrap_seed)
            runlog.write_json(os.path.join(out, JSON_NAME), payload)

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
    check_oracle(args.oracle)
    check_search(args.search, args.width)
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


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == 'run':
        return command_run(args)
    if args.command == 'search':
        return command_search(args)
    raise SystemExit(f'未知のコマンド {args.command!r}')


if __name__ == '__main__':
    raise SystemExit(main())
