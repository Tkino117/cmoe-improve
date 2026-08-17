"""唯一のドライバ。

配分とルーターをそれぞれ複数指定できる。両方に複数を渡せばその直積が走るので、
report/13 の 2×2 表（一様 x=3 / ビーム配分 × 現行ルーター / 改良ルーター）は
1コマンドで出る。

  cmoe run --alloc uniform3,beam --router cmoe --datasets wikitext2,c4-new

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
from cmoe.alloc.search.fixed import parse_allocation
from cmoe.carve.registry import create_carver
from cmoe.data.registry import load_calibration, load_evaluation
from cmoe.eval.ppl import evaluate_ppl
from cmoe.eval.stats import paired_differences, stratified_paired_bootstrap
from cmoe.router.registry import create_method
from cmoe.assemble import Converter

TEXT_NAME, JSON_NAME = 'run.txt', 'summary.json'
log = runlog.log


def parse_list(spec):
    return [field.strip() for field in spec.split(',') if field.strip()]


def parse_seeds(spec):
    return [int(field) for field in parse_list(spec)]


def build_parser():
    parser = argparse.ArgumentParser(prog='cmoe')
    sub = parser.add_subparsers(dest='command', required=True)

    run = sub.add_parser('run', help='変換して perplexity を測る')
    run.add_argument('--model', default='meta-llama/Llama-2-7b-hf')
    run.add_argument('--adapter', default=None,
                     help='省略時はモデル名から推測する')
    run.add_argument('--alloc', default='uniform3',
                     help='配分。プリセット名か層数分のカンマ区切り。カンマ区切りの'
                          'プリセット名を複数渡すと直積で走る')
    run.add_argument('--router', default='cmoe', help='ルーター方式（複数可）')
    run.add_argument('--carver', default='cmoe', help='分割方式')
    run.add_argument('--calib', default='wikitext2', help='キャリブレーションセット')
    run.add_argument('--datasets', default='wikitext2,c4-new', help='評価セット')
    run.add_argument('--seeds', default='0', help='キャリブレーションの seed')
    run.add_argument('--nsamples', type=int, default=8,
                     help='carve に使う系列数。既存の測定はすべて 8')
    run.add_argument('--nexperts', type=int, default=8)
    run.add_argument('--nactive', type=int, default=6,
                     help='1トークンあたりに走る expert 数 A。全構成で同じ')
    run.add_argument('--k-act', type=int, default=10)
    run.add_argument('--bias-speed', type=float, default=0.001)
    run.add_argument('--seqlen', type=int, default=2048)
    run.add_argument('--batch-chunk', type=int, default=None,
                     help='carve 中にバッチを分ける幅。n が大きいときだけ要る')
    run.add_argument('--no-profiling-norm', action='store_true')
    run.add_argument('--no-router-norm', action='store_true')
    run.add_argument('--layers', type=int, default=None,
                     help='先頭 N 層だけ変換する（動作確認用）')
    run.add_argument('--bootstrap-reps', type=int, default=10000)
    run.add_argument('--out', default=None)
    return parser


def configurations(args):
    """(配分名, ルーター名) の直積。先頭が対照になる。"""
    rows = []
    for alloc in parse_list(args.alloc):
        for router in parse_list(args.router):
            rows.append((alloc, router))
    return rows


def run_one(args, alloc_spec, router_name, seed, evaluation_sets):
    """1構成 × 1 seed。モデルは構成ごとに読み直す（変換は破壊的）。"""
    adapter_name = args.adapter or guess_adapter(args.model)
    adapter = create_adapter(adapter_name, args.model, seqlen=args.seqlen)

    calibration = load_calibration(
        args.calib, args.model, args.seqlen, args.nsamples, seed)
    log(f'  carve: {calibration.name} {tuple(calibration.input_ids.shape)} '
        f'hash={calibration.metadata()["token_hash"][:12]}')

    search = parse_allocation(alloc_spec, n_active_total=args.nactive)
    n_layers = args.layers or adapter.n_layers
    allocation = search.search(None, n_layers)

    converter = Converter(
        adapter,
        create_carver(args.carver, args.nexperts),
        create_method(router_name),
        n_experts=args.nexperts,
        k_act=args.k_act,
        bias_speed=args.bias_speed,
        profiling_norm=not args.no_profiling_norm,
        router_norm=not args.no_router_norm,
        batch_chunk=args.batch_chunk,
        n_layers=args.layers,
        log=lambda message: None,
    )
    started = time.time()
    report = converter.convert(calibration, allocation)
    log(f'  変換 {report.seconds:.1f}s (平均 x={allocation.mean_x:.2f})')

    results = {}
    for name, token_set in evaluation_sets.items():
        result = evaluate_ppl(adapter, token_set)
        results[name] = result
        log(f'  {name}: {result.ppl:.6f}')

    payload = {
        'allocation': allocation.metadata(),
        'router': router_name,
        'carver': args.carver,
        'seed': seed,
        'calibration': calibration.metadata(),
        'conversion': report.as_dict(),
        'ppl': {name: result.as_dict() for name, result in results.items()},
        'seconds': time.time() - started,
    }

    del converter, adapter, report
    gc.collect()
    torch.cuda.empty_cache()
    return payload, results


def summarize(records, configs, datasets, reps):
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
                        differences, reps=reps)
            row['datasets'][dataset] = entry
        summary['configurations'].append(row)
    return summary


def command_run(args):
    configs = configurations(args)
    seeds = parse_seeds(args.seeds)
    datasets = parse_list(args.datasets)

    out = args.out or runlog.default_out_dir('run')
    runlog.prepare_out_dir(out, JSON_NAME)
    runlog.open_mirror(os.path.join(out, TEXT_NAME))

    log(f'model={args.model} carve={args.calib} n={args.nsamples} '
        f'N={args.nexperts} A={args.nactive}')
    log(f'構成 {len(configs)} 種 × seed {len(seeds)} 本 = '
        f'{len(configs) * len(seeds)} 実行')
    log(f'出力 {out}')

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
        for alloc_spec, router_name in configs:
            log()
            log(f'== seed {seed} / {alloc_spec} + {router_name} ==')
            try:
                run_payload, results = run_one(
                    args, alloc_spec, router_name, seed, evaluation_sets)
            except Exception as error:  # 1構成の失敗で残りを捨てない
                log(f'  失敗: {type(error).__name__}: {error}')
                # 落ちた場所まで残す。ミラーが唯一の記録になることがある
                log(traceback.format_exc())
                failures.append(
                    {'seed': seed, 'allocation': alloc_spec,
                     'router': router_name, 'error': f'{type(error).__name__}: {error}'})
                payload['failures'] = failures
                runlog.write_json(os.path.join(out, JSON_NAME), payload)
                gc.collect()
                torch.cuda.empty_cache()
                continue
            records.append({'config': (alloc_spec, router_name), 'seed': seed,
                            'ppl': results})
            payload['runs'].append(run_payload)
            payload['summary'] = summarize(
                records, configs, datasets, args.bootstrap_reps)
            runlog.write_json(os.path.join(out, JSON_NAME), payload)

    log()
    log('== まとめ (平均 PPL、小さいほど良い) ==')
    for row in payload.get('summary', {}).get('configurations', []):
        for dataset, entry in row['datasets'].items():
            line = (f'{row["allocation"]:>10} + {row["router"]:<16} '
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


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == 'run':
        return command_run(args)
    raise SystemExit(f'未知のコマンド {args.command!r}')


if __name__ == '__main__':
    raise SystemExit(main())
