"""22 配分の対照を3本足す（順序対照 / LExI 相当 / OWL 相当）。5段を通す。

[report/21](../../report/21_ew-rule-baseline.md) で比較相手は EW-rule 1本だけに
なった。**規則型が1本しか無いと、負けが「EW の既定 α が N=8 に合っていない」で
説明できてしまう**（report/21 の注意点）。加えて、EW-rule も一様配分も平均 x が
提案と違うので、「層別に配分を決めることに意味がある」を平均 x の差から分離
できていない。ここはその2つを埋める。

  1. probe   ``owl_probe.py``  外れ値比率 D_ℓ   → result_logs/owl_probe_*_seed<N>
  2. probe   ``lexi_probe.py`` 層ローカル誤差表 → result_logs/lexi_probe_*_seed<N>
  3. alloc   ``allocate.py``   候補ベクトル     → result_logs/alloc22*_seed<N>.json
  4. score   ``cmoe score``    校正上の suffix_kl → result_logs/score22*_seed<N>
  5. bench   ``cmoe run``      PPL と選択問題    → result_logs/bench22*_seed<N>

**校正・動作点・分割・ルーター・評価・dense の基準は experiments/06（25%）・
07（50%）・21 と同一で、動かしたのは配分だけである。** 段5 は対照の一様配分を
毎回同梱する — それが「report/19・20 で触った ``assemble.py`` / ``moe/modules.py``
が現行 CMoE の挙動を動かしていない」ことの検査になる（report/21 と同じ理由）。

段4 が要るのは、規則型の自由度を**校正だけで**決めるためである。OWL には
この軸での既定値が原典に無く、EW-rule の既定 α は N=8 向けではない。片方だけ
校正で選び直すのは不公平なので、両方に同じ格子を与えて同じ目的関数で選ぶ。
主表に載るのは原典の既定どうし（``owl_default`` と report/21 の EW-rule）で、
選び直した側は「選び直しても結論が変わらないか」の付記に使う。

済んだ段は飛ばす。GPU が落ちたらそのまま同じコマンドを叩き直せばよい。

  uv run python experiments/22_alloc_baselines/run.py --smoke        # 2層・8問
  uv run python experiments/22_alloc_baselines/run.py --seed 0 --nactive 6
  uv run python experiments/22_alloc_baselines/run.py --seed 0 --nactive 4
"""

import argparse
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path

from cmoe.cli import build_parser

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

CALIB = 'slimpajama'
NSAMPLES = 16
ORACLE = 'suffix_kl'
ROUTER = 'cmoe'
CARVER = 'cmoe'
NEXPERTS = 8
DATASETS = 'wikitext2,c4-new'
BENCH_BATCH = 32
# report/04 が測った dense。experiments/06・07・21 と同じものを取り込む
DENSE_REFERENCE = 'result_logs/bench_h4/wikitext2_seed0/bench/dense.json'
# 結果を変えない引数。同じ実験かの判定から外す（experiments/06 と同じ集合）
NOT_IDENTITY = {'out', 'batch_chunk', 'token_chunk', 'search_token_chunk',
                'no_recheck', 'bench_reference', 'bench_cache_dir'}
# 主表に載る対照。この順で ``cmoe run`` に渡す（先頭が対応比較の基準）
BENCH_ROWS = ('reverse', 'shuffle1', 'shuffle2', 'lexi', 'owl_default')
# smoke の層数。2層だと提案の先頭2つが多重集合として偏りすぎ、順序対照が
# 「元とも反転とも違う並び」を引けない
SMOKE_LAYERS = 4


def log(message=''):
    print(message, flush=True)


def load(path):
    with Path(path).open() as handle:
        return json.load(handle)


def retire(path):
    """書きかけを退避する。消さない（落ちた跡は原因調べに要る）。"""
    for index in itertools.count(1):
        target = path.with_name(f'{path.name}.partial{index}')
        if not target.exists():
            path.rename(target)
            log(f'  書きかけの {path.name} を {target.name} へ退避した')
            return target


def run_command(command):
    log(f'$ {" ".join(str(part) for part in command)}')
    subprocess.run(command, cwd=ROOT, check=True)


def cli_differences(record, argv):
    """``cmoe`` の段が、いま走らせようとしているものと同じか。"""
    wanted = vars(build_parser().parse_args(argv))
    given = record.get('arguments', {})
    return [key for key in sorted(wanted)
            if key not in NOT_IDENTITY and given.get(key) != wanted[key]]


def script_differences(record, wanted):
    """自作スクリプトの段。引数の辞書を直に突き合わせる。"""
    given = record.get('arguments', {})
    return [key for key in sorted(wanted)
            if key != 'out' and given.get(key) != wanted[key]]


def ensure(out_dir, result_name, command, is_finished, differences):
    """段を1つ通す。済んでいれば飛ばし、書きかけなら退避してから引き直す。

    完了印はファイルの存在ではない（開始直後から真になる）。段ごとに
    ``is_finished`` が中身を見て決める。
    """
    out_dir = Path(out_dir)
    payload = out_dir / result_name
    record = load(payload) if payload.exists() else None
    if record is not None and is_finished(record):
        log(f'  {out_dir.name}: 済み')
    else:
        if out_dir.exists():
            if record is not None and differences(record):
                raise SystemExit(
                    f'{out_dir} は違う実験の出力である'
                    f'（{", ".join(differences(record))}）。別の出力先にするか、'
                    'そのディレクトリを退けること')
            retire(out_dir)
        run_command(command)
        record = load(payload) if payload.exists() else None
        if record is None or not is_finished(record):
            raise SystemExit(f'{out_dir} に終わった段が残らなかった')
    gaps = differences(record)
    if gaps:
        raise SystemExit(f'{out_dir} は違う実験の出力である（{", ".join(gaps)}）')
    return record


# -- 段1・2: プローブ -----------------------------------------------------

def owl_stage(seed, plan):
    out = ROOT / f'result_logs/owl_probe_{CALIB}{plan["suffix"]}_seed{seed}'
    wanted = {'model': plan['model'], 'calib': CALIB, 'seed': seed,
              'nsamples': plan['nsamples'], 'seqlen': 2048,
              'batch_chunk': 4, 'layers': plan['layers']}
    command = ['uv', 'run', 'python', str(HERE / 'owl_probe.py'),
               '--seed', str(seed), '--calib', CALIB,
               '--nsamples', str(plan['nsamples']), '--out', str(out)]
    if plan['layers']:
        command += ['--layers', str(plan['layers'])]
    return ensure(out, 'owl.json', command,
                  lambda record: len(record.get('layers', [])) == record.get(
                      'n_layers', -1) and record['n_layers'] > 0,
                  lambda record: script_differences(record, wanted))


def lexi_stage(seed, n_active, plan):
    tag = '' if n_active == 6 else f'_a{n_active}'
    out = ROOT / f'result_logs/lexi_probe_{CALIB}{plan["suffix"]}{tag}_seed{seed}'
    wanted = {'model': plan['model'], 'calib': CALIB, 'seed': seed,
              'nsamples': plan['nsamples'], 'seqlen': 2048,
              'nexperts': NEXPERTS, 'nactive': n_active, 'carver': CARVER,
              'k_act': 10, 'bias_speed': 0.001, 'layers': plan['layers']}
    command = ['uv', 'run', 'python', str(HERE / 'lexi_probe.py'),
               '--seed', str(seed), '--calib', CALIB,
               '--nsamples', str(plan['nsamples']),
               '--nactive', str(n_active), '--out', str(out)]
    if plan['layers']:
        command += ['--layers', str(plan['layers'])]
    return ensure(out, 'lexi.json', command,
                  lambda record: 'allocation' in record,
                  lambda record: script_differences(record, wanted))


# -- 段3: 候補ベクトル（CPU） ---------------------------------------------

def allocate_stage(seed, n_active, plan, owl_dir, lexi_dir):
    tag = '' if n_active == 6 else f'_a{n_active}'
    out = ROOT / f'result_logs/alloc22{plan["suffix"]}{tag}_seed{seed}.json'
    command = ['uv', 'run', 'python', str(HERE / 'allocate.py'),
               '--seed', str(seed), '--nactive', str(n_active),
               '--nexperts', str(NEXPERTS), '--calib', CALIB,
               '--owl-probe', str(owl_dir), '--lexi-probe', str(lexi_dir),
               '--out', str(out.relative_to(ROOT))]
    if plan['layers']:
        command += ['--layers', str(plan['layers'])]
    # CPU で数秒なので、済み判定を持たずに毎回引き直す（プローブが更新されて
    # いれば追随してほしい側である）
    run_command(command)
    payload = load(out)
    missing = [name for name in BENCH_ROWS
               if not any(entry['name'] == name
                          or name in entry['aliases']
                          for entry in payload['candidates'])]
    if missing:
        raise SystemExit(f'主表の行 {missing} が候補に出ていない')
    return payload


def spec_label(payload, spec):
    """配分ベクトルに、表で使う短い名前を当てる。"""
    if spec == payload['proposal']['spec']:
        return '提案'
    for entry in payload['candidates']:
        if entry['spec'] == spec:
            return entry['name']
    return '一様'


def pick(candidates, name):
    """名前（または別名）で候補を1本引く。"""
    for entry in candidates:
        if entry['name'] == name or name in entry['aliases']:
            return entry
    raise SystemExit(f'候補に {name} が無い')


# -- 段4: 校正上の採点 ----------------------------------------------------

def score_specs(payload, plan, rows):
    """採点する配分の並び。``rows`` が None なら全候補（格子込み）。"""
    if rows is None:
        specs = [entry['spec'] for entry in payload['candidates']]
    else:
        specs = [pick(payload['candidates'], name)['spec'] for name in rows]
    n_layers = len(payload['proposal']['values'])
    uniform = payload['calibration_uniform']
    uniform_spec = ','.join([uniform.replace('uniform', '')] * n_layers)
    # 提案と、校正が選んだ一様。表を自己完結させるために同じ走査に入れる
    for extra in (payload['proposal']['spec'], uniform_spec):
        if extra not in specs:
            specs.append(extra)
    return list(dict.fromkeys(specs))


def score_stage(seed, n_active, plan, payload, rows, name='score22'):
    """配分を、探索と同じオラクル（``suffix_kl``）で採点する。

    主表の行だけを採点する既定の段と、規則型の格子を全部採点する段（``grid``）
    に分かれている。**格子は1本あたり数分かかるので、主表が先に出る順にして
    ある** — 途中で止まっても、主表の数字は揃っている側から積み上がる。
    """
    tag = '' if n_active == 6 else f'_a{n_active}'
    out = ROOT / f'result_logs/{name}{plan["suffix"]}{tag}_seed{seed}'
    specs = score_specs(payload, plan, rows)

    argv = ['score', '--calib', CALIB, '--nsamples', str(plan['nsamples']),
            '--oracle', ORACLE, '--seed', str(seed),
            '--nexperts', str(NEXPERTS), '--nactive', str(n_active),
            '--carver', CARVER]
    for spec in specs:
        argv += ['--alloc', spec]
    if plan['layers']:
        argv += ['--layers', str(plan['layers'])]
    argv += ['--out', str(out)]

    record = ensure(
        out, 'score.json', ['uv', 'run', 'cmoe', *argv],
        lambda record: len(record.get('scores', [])) == len(specs),
        lambda record: cli_differences(record, argv))
    return {'dir': str(out.relative_to(ROOT)),
            'scores': {','.join(str(x) for x in row['allocation']['values']):
                       row['score'] for row in record['scores']}}


def calibration_best(payload, scores, kind):
    """``kind`` の候補のうち、校正の score が最小の1本。"""
    rows = [entry for entry in payload['candidates']
            if any(detail.get('kind') == kind for detail in entry['details'])]
    rows = [entry for entry in rows if entry['spec'] in scores]
    if not rows:
        return None
    return min(rows, key=lambda entry: scores[entry['spec']])


# -- 段5: 測定 ------------------------------------------------------------

def bench_stage(seed, n_active, plan, payload, extra_specs, rows=None,
                name='bench22'):
    """配分を PPL と選択問題で測る。

    ``name`` を分けてあるのは、あとから足す段（格子で選び直した規則型）が
    **済んだ主表の run を書き換えないため**である。``ensure`` は同じ出力先に
    違う ``--alloc`` の並びが来たら止まるので、そこを分けないと主表を測り直す
    ことになる。
    """
    tag = '' if n_active == 6 else f'_a{n_active}'
    out = ROOT / f'result_logs/{name}{plan["suffix"]}{tag}_seed{seed}'
    # 先頭が cmoe run の対応比較の基準になる。experiments/06・07・21 と同じ
    # 一様を置くので、既存の bench_slimpajama* との突き合わせがそのまま検査になる
    baseline = f'uniform{n_active // 2}'
    specs = [baseline]
    for row in (plan['bench_rows'] if rows is None else rows):
        entry = pick(payload['candidates'], row)
        if entry['spec'] not in specs:
            specs.append(entry['spec'])
    for spec in extra_specs:
        if spec not in specs:
            specs.append(spec)

    argv = ['run']
    for spec in specs:
        argv += ['--alloc', spec]
    argv += ['--router', ROUTER, '--carver', CARVER, '--seeds', str(seed),
             '--calib', CALIB, '--nsamples', str(plan['nsamples']),
             '--nexperts', str(NEXPERTS), '--nactive', str(n_active),
             '--datasets', DATASETS, '--bench',
             '--bench-batch-size', str(plan['bench_batch'])]
    if plan['bench_limit']:
        argv += ['--bench-limit', str(plan['bench_limit'])]
    if plan['layers']:
        argv += ['--layers', str(plan['layers'])]
    if plan['reference'] is not None:
        reference = ROOT / plan['reference']
        if not reference.exists():
            raise SystemExit(f'{reference} が無い。dense の基準が要る')
        argv += ['--bench-reference', str(reference)]
    argv += ['--out', str(out)]

    ensure(out, 'summary.json', ['uv', 'run', 'cmoe', *argv],
           lambda record: (not record.get('failures') and 'summary' in record
                           and 'bench_summary' in record
                           and len(record.get('runs', [])) == len(specs)),
           lambda record: cli_differences(record, argv))
    return {'dir': str(out.relative_to(ROOT)), 'specs': specs,
            'baseline': baseline}


def summarize(path, payload):
    payload = dict(payload)
    payload.update({
        'commit': subprocess.run(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True,
            text=True).stdout.strip(),
        'dirty': bool(subprocess.run(
            ['git', 'status', '--porcelain'], cwd=ROOT, capture_output=True,
            text=True).stdout.strip()),
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as handle:
        json.dump(payload, handle, indent=1, ensure_ascii=False)
    log(f'\n段の一覧を {path} に書いた')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--nactive', type=int, default=6)
    parser.add_argument('--stages', default='probe,alloc,score,bench',
                        help='走らせる段。grid を足すと規則型の格子も採点して'
                             '測る（重い。主表には要らない）')
    parser.add_argument('--smoke', action='store_true',
                        help='2層・7本・8問。経路の確認')
    args = parser.parse_args(argv)

    stages = [field.strip() for field in args.stages.split(',') if field.strip()]
    plan = {
        'model': 'meta-llama/Llama-2-7b-hf',
        'layers': SMOKE_LAYERS if args.smoke else None,
        # slimpajama は7成分に最低1本ずつ配る。smoke でもそれ未満には落とせない
        'nsamples': 7 if args.smoke else NSAMPLES,
        'bench_limit': 8 if args.smoke else None,
        'bench_batch': 8 if args.smoke else BENCH_BATCH,
        # smoke は問題を8問に絞る。report/04 の dense は全件で測ったものなので
        # 取り込めない（``cmoe run`` が limit の違いを見て断る）
        'reference': None if args.smoke else DENSE_REFERENCE,
        'suffix': '_smoke' if args.smoke else '',
        'bench_rows': BENCH_ROWS,
    }

    log(f'seed {args.seed} / A={args.nactive}'
        f'（スパース率 {100 - 100 * args.nactive // NEXPERTS}%）/ '
        f'校正 {CALIB} n={plan["nsamples"]} / 段 {",".join(stages)}'
        + ('（smoke）' if args.smoke else ''))

    started = time.time()
    record = {'seed': args.seed, 'n_active': args.nactive, 'calib': CALIB,
              'nsamples': plan['nsamples'], 'smoke': args.smoke}

    log(f'\n=== 段1・2 プローブ / seed {args.seed} ===')
    owl_dir = ROOT / f'result_logs/owl_probe_{CALIB}{plan["suffix"]}_seed{args.seed}'
    tag = '' if args.nactive == 6 else f'_a{args.nactive}'
    lexi_dir = (ROOT / f'result_logs/lexi_probe_{CALIB}{plan["suffix"]}{tag}'
                f'_seed{args.seed}')
    if 'probe' in stages:
        owl_stage(args.seed, plan)
        lexi_stage(args.seed, args.nactive, plan)
    record['probes'] = {'owl': str(owl_dir.relative_to(ROOT)),
                        'lexi': str(lexi_dir.relative_to(ROOT))}

    log(f'\n=== 段3 候補ベクトル / seed {args.seed} ===')
    payload = allocate_stage(args.seed, args.nactive, plan, owl_dir, lexi_dir)
    record['n_candidates'] = len(payload['candidates'])

    if 'score' in stages:
        log(f'\n=== 段4 校正上の採点（主表の行）/ seed {args.seed} ===')
        record['score'] = score_stage(args.seed, args.nactive, plan, payload,
                                      plan['bench_rows'])
        scores = record['score']['scores']
        for spec, value in sorted(scores.items(), key=lambda kv: kv[1]):
            label = spec_label(payload, spec)
            log(f'  {label:<14} {value:.6e}  {spec[:40]}')

    if 'bench' in stages:
        log(f'\n=== 段5 測定 / seed {args.seed} ===')
        record['bench'] = bench_stage(args.seed, args.nactive, plan, payload, [])
        log(f'  {len(record["bench"]["specs"])} 構成')

    if 'grid' in stages:
        log(f'\n=== 段6 規則型の格子を校正で採点 / seed {args.seed} ===')
        record['grid'] = score_stage(args.seed, args.nactive, plan, payload,
                                     None, name='grid22')
        scores = record['grid']['scores']
        chosen = {}
        extra_specs = []
        for kind in ('owl', 'ew'):
            best = calibration_best(payload, scores, kind)
            if best is None:
                continue
            chosen[kind] = {'name': best['name'], 'spec': best['spec'],
                            'score': scores[best['spec']],
                            'aliases': best['aliases']}
            log(f'  {kind} の校正最良: {best["name"]}  '
                f'score {scores[best["spec"]]:.6e}  {best["spec"][:40]}')
            if best['spec'] not in extra_specs:
                extra_specs.append(best['spec'])
        record['calibration_choice'] = chosen
        log(f'  参考: 提案 score '
            f'{scores.get(payload["proposal"]["spec"], float("nan")):.6e}')
        log(f'\n=== 段7 選び直した規則型を測定 / seed {args.seed} ===')
        record['bench_grid'] = bench_stage(args.seed, args.nactive, plan,
                                           payload, extra_specs, rows=(),
                                           name='bench22grid')

    record['seconds'] = time.time() - started
    summarize(ROOT / f'result_logs/exp22{plan["suffix"]}{tag}'
              f'_stages_seed{args.seed}.json', record)
    log(f'所要 {record["seconds"] / 60:.1f}分')
    return 0


if __name__ == '__main__':
    sys.exit(main())
