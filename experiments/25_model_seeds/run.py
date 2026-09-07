"""25 06・07 と同じ3段を、モデルと動作点と seed を引数にして回す。

report/06（25%）と report/07（50%）は Llama-2-7b を seed 3本で回したもので、
動作点ごとに別のスクリプトに分かれ、モデルは CLI の既定に任せていた。ここは
その2本を1本に畳み、**モデルを引数にした**ものである。足したいのは2つ。

* seed を 0..4 の5本にする。3本では、提案が全 seed で勝っても符号検定の片側 p の
  下限が 1/8 = 0.125 で、seed 水準の主張が原理的に有意にならない。5本なら
  1/32 = 0.031 になる。探索は seed ごとに引き直す（配分そのものが seed ごとに
  独立に決まる）ので、対応のある検定がそのまま使える
* モデルを2本にする。Llama-2-7b と Mistral-7B-v0.1 で、どちらも32層・
  intermediate が8で割り切れる（11008 / 8 = 1376、14336 / 8 = 1792）。層数が
  揃っているので、report/09 の「どの層に routing が載るか」を層番号ごとに
  そのまま並べられる

**Llama-2-7b の出力先は report/06・07 の頃の名前のままである。** モデルの札が
名前に入るのは Llama 以外だけで、seed 3・4 を足しても既存の seed 0..2 と同じ
ディレクトリの並びに落ちる（``summarize_seeds.py`` がそのまま読める）。

  1. 探索  ``cmoe search`` 幅2/3/4  → result_logs/slimpajama[_<model>][_a<A>]_w<W>_n16_seed<N>
  2. 採点  ``cmoe score``  一様 x=0..A → result_logs/score_uniform_slimpajama[_<model>][_a<A>]_n16_seed<N>
  3. 測定  ``cmoe run --bench``      → result_logs/bench_slimpajama[_<model>][_a<A>]_seed<N>
  4. MMLU  ``cmoe run --bench``      → result_logs/bench_mmlu_slimpajama[_<model>][_a<A>]_seed<N>

段4 は段3 と同じ配分を MMLU で測り直すもので、experiments/17 が既存の実行に
対してやっていることを、最初から同じスクリプトの中でやる。**PPL は段3 で
測ってあるので段4 は ``--no-ppl``** である。

dense の基準はモデルごとに1つ。Llama は report/04（5タスク）と experiments/17
（MMLU）が測ったものを取り込む。他のモデルは A=6 / seed 0 の実行の中で1回だけ
測り、残りは全部それを取り込む（dense は A にも校正にも seed にも配分にも
依らない）。

  # 経路の確認（3層・8問。数分）
  uv run python experiments/25_model_seeds/run.py --model mistral-7b --smoke

  # 変換前後が妥当な acc を出すかの確認（dense と uniform3 だけ。1時間弱）
  uv run python experiments/25_model_seeds/run.py --model mistral-7b --check

  # 本実行
  uv run python experiments/25_model_seeds/run.py --model mistral-7b --nactive 6 --seed 0
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

CALIB = 'slimpajama'
NSAMPLES = 16
ORACLE = 'suffix_kl'
ROUTER = 'cmoe'
WIDTHS = (2, 3, 4)
DATASETS = 'wikitext2,c4-new'
BENCH_BATCH = 32
MMLU_TASKS = 'mmlu'
# N は動かさない — N を変えると expert 1個あたりのニューロン数（分割の粒度）まで
# 動いて、比べているものが「配分」でなくなる
NEXPERTS = 8

# 短い札 → HuggingFace の名前。札のほうが出力先の名前に入る。
# どちらも32層で、intermediate が NEXPERTS で割り切れることを確かめてある
MODELS = {
    'llama2-7b': 'meta-llama/Llama-2-7b-hf',
    'mistral-7b': 'mistralai/Mistral-7B-v0.1',
}
# 出力先の名前に札が入らないモデル。report/06・07 の頃の名前を保つためで、
# seed 3・4 を足したときに既存の seed 0..2 と同じ並びに落ちる
UNTAGGED_MODEL = 'llama2-7b'
# 既に測ってある dense。ここに無いモデルは A=6 / seed 0 の実行の中で1回測る
KNOWN_DENSE = {
    ('llama2-7b', 'bench'): 'result_logs/bench_h4/wikitext2_seed0/bench/dense.json',
    ('llama2-7b', 'mmlu'): 'result_logs/bench_mmlu_slimpajama_seed0/bench/dense.json',
}
# dense を測る動作点と seed。ここ以外は全部これを取り込む
DENSE_AT = (6, 0)

# 結果を変えない引数（出力先・チャンク幅・基準の取り込み元）。同じ実験かの判定から外す
NOT_IDENTITY = {'out', 'batch_chunk', 'token_chunk', 'search_token_chunk',
                'no_recheck', 'bench_reference', 'bench_cache_dir'}


def log(message=''):
    print(message, flush=True)


def run_cli(argv):
    command = ['uv', 'run', 'cmoe', *argv]
    log(f'$ {" ".join(command)}')
    subprocess.run(command, cwd=ROOT, check=True)


def load(path):
    with Path(path).open() as handle:
        return json.load(handle)


def retire(path):
    """書きかけのディレクトリを退避する。消さない（落ちた跡は原因調べに要る）。"""
    for index in itertools.count(1):
        target = path.with_name(f'{path.name}.partial{index}')
        if not target.exists():
            path.rename(target)
            log(f'  書きかけの {path.name} を {target.name} へ退避した')
            return target


def same_experiment(record, argv):
    """記録された引数が、いま走らせようとしている実験と同じものか。

    ``model`` はここに含まれる。**出力先の名前にモデルの札を入れないと**、
    別モデルの実行が既存のディレクトリに当たって毎回ここで止まる。
    """
    wanted = vars(build_parser().parse_args(argv))
    given = record.get('arguments', {})
    return [key for key in sorted(wanted)
            if key not in NOT_IDENTITY and given.get(key) != wanted[key]]


def ensure(out_dir, result_name, argv, is_finished):
    """段を1つ通す。済んでいれば飛ばし、書きかけなら退避してから引き直す。

    完了印はファイルの存在ではない（開始直後から真になる）。段ごとに
    ``is_finished`` が中身を見て決める。
    """
    payload = out_dir / result_name
    record = load(payload) if payload.exists() else None
    if record is not None and is_finished(record):
        log(f'  {out_dir.name}: 済み')
    else:
        if out_dir.exists():
            if record is not None:
                differences = same_experiment(record, argv)
                if differences:
                    raise SystemExit(
                        f'{out_dir} は違う実験の出力である'
                        f'（{", ".join(differences)}）。別の --out を指定するか、'
                        'そのディレクトリを退けること')
            retire(out_dir)
        run_cli(argv)
        record = load(payload) if payload.exists() else None
        if record is None or not is_finished(record):
            raise SystemExit(f'{out_dir} に終わった段が残らなかった')
    differences = same_experiment(record, argv)
    if differences:
        raise SystemExit(
            f'{out_dir} は違う実験の出力である（{", ".join(differences)}）')
    return record


# -- 名前 -----------------------------------------------------------------

def model_tag(model_key):
    """出力先に入るモデルの札。Llama-2-7b だけ空である。"""
    return '' if model_key == UNTAGGED_MODEL else f'_{model_key}'


def sparsity_tag(n_active):
    """動作点の札。A=6 だけ空である（report/06 の頃の名前を保つ）。"""
    return '' if n_active == 6 else f'_a{n_active}'


def stem(model_key, n_active):
    return f'{CALIB}{model_tag(model_key)}{sparsity_tag(n_active)}'


def uniform_specs(n_active):
    """一様の並び。**先頭が ``cmoe run`` の対応のある比較の基準になる。**

    A の半分（shared と routed が半々）を固定対照として先頭に置く。A=6 なら
    uniform3、A=4 なら uniform2 で、report/06・07 と同じ並びである。
    """
    baseline = f'uniform{n_active // 2}'
    return [baseline] + [f'uniform{x}' for x in range(n_active + 1)
                         if f'uniform{x}' != baseline]


def dense_reference(root, model_key, kind, n_active, seed):
    """取り込む dense。無ければ None（その実行が自分で測る）。

    kind は ``bench``（5タスク）か ``mmlu``。**タスクの顔ぶれが違うと
    ``cmoe run`` が取り込みを断る**ので、2つを別に持つ。

    探し先は3つある。既に測ってあるもの（``KNOWN_DENSE``）→ 本実行の
    A=6 / seed 0 → ``--check`` が測ったもの、の順である。**check は全件で
    測る**ので、その dense は本実行がそのまま取り込める（MMLU の dense は
    14,042問で1時間強かかるので、2度測る意味が無い）。取り込んでよいかの
    判定は ``cmoe run`` 自身がやる — モデル・タスク・limit が違えば断る。
    """
    known = KNOWN_DENSE.get((model_key, kind))
    if known is not None:
        return ROOT / known
    prefix = 'bench' if kind == 'bench' else 'bench_mmlu'
    name = f'{prefix}_{stem(model_key, DENSE_AT[0])}_seed{DENSE_AT[1]}'
    if (n_active, seed) != DENSE_AT:
        return root / name / 'bench' / 'dense.json'
    # ここが測る側だが、check が先に全件で測っていればそれを使う
    from_check = ROOT / 'result_logs' / 'exp25_check' / name / 'bench' / 'dense.json'
    if root != from_check.parents[2] and from_check.exists():
        return from_check
    return None


# -- 段 -------------------------------------------------------------------

def sparsity_flags(n_active):
    return ['--nexperts', str(NEXPERTS), '--nactive', str(n_active)]


def model_flags(model_key):
    """モデルの指定。アダプタは名前から推測させる（mistral → auto）。"""
    return ['--model', MODELS[model_key]]


def common_oracle_flags(plan, seed):
    argv = ['--calib', CALIB, '--nsamples', str(plan['nsamples']),
            '--oracle', ORACLE, '--seed', str(seed),
            *model_flags(plan['model']), *sparsity_flags(plan['nactive'])]
    if plan['layers']:
        argv += ['--layers', str(plan['layers'])]
    return argv


def search_stage(root, plan, seed, width):
    """段1。層ごとの x を探す。"""
    out_dir = (root / f'{stem(plan["model"], plan["nactive"])}'
                      f'_w{width}_n{plan["nsamples"]}_seed{seed}')
    argv = ['search', *common_oracle_flags(plan, seed),
            '--search', 'beam', '--width', str(width), '--out', str(out_dir)]

    def is_finished(record):
        # 予算切れの記録が残っていても 'allocation' は入らない
        return 'allocation' in record and 'recheck' in record

    record = ensure(out_dir, 'search.json', argv, is_finished)
    return {'width': width, 'dir': str(out_dir.relative_to(ROOT)),
            'values': record['allocation']['values'],
            'score': record['score'],
            'recheck_gap': record['recheck']['gap']}


def score_stage(root, plan, seed):
    """段2。一様配分を、探索と同じオラクルで採点する。

    対照を「評価指標を見て最良の1本」で選ぶと対照の側にだけ上振れが乗るので、
    ここで校正だけを見て1本に決める（report/06 の段2 と同じ）。
    """
    out_dir = (root / f'score_uniform_{stem(plan["model"], plan["nactive"])}'
                      f'_n{plan["nsamples"]}_seed{seed}')
    argv = ['score', *common_oracle_flags(plan, seed)]
    for spec in plan['scored_uniforms']:
        argv += ['--alloc', spec]
    argv += ['--out', str(out_dir)]

    def is_finished(record):
        return ('best' in record
                and len(record.get('scores', [])) == len(plan['scored_uniforms']))

    record = ensure(out_dir, 'score.json', argv, is_finished)
    return {'dir': str(out_dir.relative_to(ROOT)),
            'best': record['best'],
            'scores': {row['allocation']['name']: row['score']
                       for row in record['scores']}}


def bench_specs(plan, searched):
    """測る配分の並びと、表で使う名前。"""
    specs = list(plan['uniforms'])
    labels = {}
    for record in searched or []:
        spec = ','.join(str(value) for value in record['values'])
        # 幅が違っても同じ配分に着くことがある。行を消さずに束ねる
        labels[spec] = (f'{labels[spec]}/{record["width"]}' if spec in labels
                        else f'beam 幅{record["width"]}')
        specs.append(spec)
    return list(dict.fromkeys(specs)), labels


def bench_stage(root, plan, seed, searched, kind='bench'):
    """段3・段4。配分を PPL とベンチマークで測る。

    kind='mmlu' のときは MMLU だけを測る。**PPL と5タスクは段3 にあるので
    ``--no-ppl --bench-tasks mmlu``** で、同じ数字を引き直さない。
    """
    prefix = 'bench' if kind == 'bench' else 'bench_mmlu'
    out_dir = root / f'{prefix}_{stem(plan["model"], plan["nactive"])}_seed{seed}'
    specs, labels = bench_specs(plan, searched)

    argv = ['run']
    for spec in specs:
        argv += ['--alloc', spec]
    argv += ['--router', ROUTER, '--seeds', str(seed), '--calib', CALIB,
             '--nsamples', str(plan['nsamples']),
             *model_flags(plan['model']), *sparsity_flags(plan['nactive'])]
    if kind == 'bench':
        argv += ['--datasets', DATASETS]
    else:
        argv += ['--no-ppl', '--bench-tasks', MMLU_TASKS]
    argv += ['--bench', '--bench-batch-size', str(plan['bench_batch'])]
    if plan['bench_limit']:
        argv += ['--bench-limit', str(plan['bench_limit'])]
    if plan['layers']:
        argv += ['--layers', str(plan['layers'])]
    reference = plan['reference'][kind]
    if reference is not None:
        if not reference.exists():
            raise SystemExit(
                f'{reference} が無い。dense の基準が要る。'
                f'{plan["model"]} は A={DENSE_AT[0]} / seed {DENSE_AT[1]} を先に通すこと')
        argv += ['--bench-reference', str(reference)]
        log(f'  dense は {reference} から取り込む')
    else:
        log('  dense をここで測る（このモデルではこの1回だけ）')
    argv += ['--out', str(out_dir)]

    def is_finished(record):
        # --no-ppl の段には 'summary' が入らない。見るのはベンチ側だけ
        has_ppl = 'summary' in record if kind == 'bench' else True
        return (not record.get('failures') and has_ppl
                and 'bench_summary' in record
                and len(record.get('runs', [])) == len(specs))

    ensure(out_dir, 'summary.json', argv, is_finished)
    return {'dir': str(out_dir.relative_to(ROOT)), 'labels': labels,
            'specs': specs}


def summarize(path, payload):
    """段の一覧と、測ったコードのコミットを残す。数値そのものは集計しない。"""
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


def build_plan(args, root):
    """1回ぶんの設定。smoke と check はここで既定を差し替える。

    **check は探索を回さない。** 見たいのは「このモデルで dense と変換後が
    妥当な acc を出すか」だけで、配分の良し悪しではない。dense を全件で測り、
    固定対照（A の半分の一様）1本と並べる。
    """
    n_active = args.nactive
    plan = {
        'model': args.model,
        'nactive': n_active,
        'layers': 3 if args.smoke else None,
        # slimpajama は7成分に最低1本ずつ配る。smoke でもそれ未満には落とせない
        'nsamples': 7 if args.smoke else NSAMPLES,
        'bench_limit': 8 if args.smoke else None,
        'bench_batch': 8 if args.smoke else BENCH_BATCH,
        'uniforms': uniform_specs(n_active),
        'scored_uniforms': [f'uniform{x}' for x in range(n_active + 1)],
    }
    if args.smoke:
        # 見たいのは経路であって表ではない。PPL は評価セット全体を走るので
        # 構成数がそのまま時間になる
        plan['uniforms'] = [f'uniform{n_active // 2}', f'uniform{n_active}']
        plan['scored_uniforms'] = list(plan['uniforms'])
    if args.check:
        plan['uniforms'] = [f'uniform{n_active // 2}']
        plan['scored_uniforms'] = list(plan['uniforms'])
    # smoke は問題を絞るので、全件で測った dense は取り込めない
    # （``cmoe run`` が limit の違いを見て断る）。自分で測る
    plan['reference'] = {
        kind: None if args.smoke else
              dense_reference(root, args.model, kind, n_active, args.seed)
        for kind in ('bench', 'mmlu')
    }
    return plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='mistral-7b', choices=sorted(MODELS),
                        help='モデルの札。出力先の名前にも入る'
                             f'（{UNTAGGED_MODEL} だけ入らない）')
    parser.add_argument('--nactive', type=int, default=6,
                        help='動作点。6=スパース率25%% / 4=50%%')
    parser.add_argument('--seed', type=int, default=0, help='校正の seed')
    parser.add_argument('--widths', default=','.join(str(w) for w in WIDTHS))
    parser.add_argument('--stages', default='search,score,bench,mmlu',
                        help='走らせる段（既定は全部）')
    parser.add_argument('--out', default=None, help='出力の置き場所')
    parser.add_argument('--smoke', action='store_true',
                        help='3層・7本・8問。経路の確認')
    parser.add_argument('--check', action='store_true',
                        help='探索を回さず、dense と固定対照だけを全件で測る。'
                             '本実行の前に acc が妥当かを見るためのもの')
    args = parser.parse_args(argv)

    if args.smoke and args.check:
        raise SystemExit('--smoke と --check は同時に指定できない')
    if not 0 <= args.nactive <= NEXPERTS:
        raise SystemExit(f'--nactive は 0..{NEXPERTS} の中')

    widths = [int(field) for field in args.widths.split(',') if field.strip()]
    stages = [field.strip() for field in args.stages.split(',') if field.strip()]
    if args.smoke:
        widths = widths[:1]
    if args.check:
        # 探索も採点も回さない。見たいのは変換前後の acc だけである
        stages = [name for name in stages if name in ('bench', 'mmlu')]

    default_root = ROOT / 'result_logs'
    if args.smoke:
        default_root = default_root / 'exp25_smoke'
    elif args.check:
        default_root = default_root / 'exp25_check'
    root = Path(args.out) if args.out else default_root
    root.mkdir(parents=True, exist_ok=True)
    plan = build_plan(args, root)

    log(f'{args.model}（{MODELS[args.model]}）/ seed {args.seed} / '
        f'N={NEXPERTS} A={args.nactive}'
        f'（スパース率 {100 * (NEXPERTS - args.nactive) // NEXPERTS}%）/ '
        f'校正 {CALIB} n={plan["nsamples"]} / '
        f'幅 {",".join(str(w) for w in widths)} / 段 {",".join(stages)}'
        + ('（smoke）' if args.smoke else '')
        + ('（check）' if args.check else ''))
    log(f'出力 {root}')

    started = time.time()
    record = {'model': args.model, 'model_name': MODELS[args.model],
              'seed': args.seed, 'calib': CALIB, 'nsamples': plan['nsamples'],
              'nexperts': NEXPERTS, 'nactive': args.nactive,
              'widths': widths, 'smoke': args.smoke, 'check': args.check}

    if 'search' in stages:
        record['search'] = []
        for width in widths:
            log(f'\n=== 段1 探索 幅{width} / seed {args.seed} ===')
            record['search'].append(search_stage(root, plan, args.seed, width))
            got = record['search'][-1]
            mean_x = sum(got['values']) / len(got['values'])
            log(f'  score {got["score"]:.6e}  測り直しの差 {got["recheck_gap"]:.3e}'
                f'  平均 x={mean_x:.4f}')

    if 'score' in stages:
        log(f'\n=== 段2 一様の採点 / seed {args.seed} ===')
        record['score'] = score_stage(root, plan, args.seed)
        best = record['score']['best']
        log(f'  校正で選んだ対照: {best["name"]}（score {best["score"]:.6e}）')

    if 'bench' in stages or 'mmlu' in stages:
        if 'search' not in record and not (args.check or args.smoke):
            record['search'] = [search_stage(root, plan, args.seed, width)
                                for width in widths]
    if 'bench' in stages:
        log(f'\n=== 段3 測定（PPL と5タスク）/ seed {args.seed} ===')
        record['bench'] = bench_stage(root, plan, args.seed,
                                      record.get('search'), kind='bench')
        log(f'  {len(record["bench"]["specs"])} 配分')

    if 'mmlu' in stages:
        log(f'\n=== 段4 測定（MMLU）/ seed {args.seed} ===')
        record['mmlu'] = bench_stage(root, plan, args.seed,
                                     record.get('search'), kind='mmlu')
        log(f'  {len(record["mmlu"]["specs"])} 配分')

    record['seconds'] = time.time() - started
    summarize(root / f'exp25_stages_{args.model}'
                     f'_a{args.nactive}_seed{args.seed}.json', record)
    log(f'所要 {record["seconds"] / 60:.1f}分')
    return 0


if __name__ == '__main__':
    sys.exit(main())
