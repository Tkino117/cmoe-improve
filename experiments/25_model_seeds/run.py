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

**dense はモデルごとに1つで、``--check`` が測ったものに一本化する。** 旧マシンで
測った Llama の dense（report/04 と experiments/17）は取り込まない — 5 seed を
1台で測り直す方針なので、基準だけ別のハードウェアのものになるのを避ける。
check を先に通しておけば (モデル × A × seed) のジョブは互いに何も待たずに
走れる。通していないと A=6 / seed 0 だけが dense を持ち、残りはそれを待つ。

  # 経路の確認（3層・8問。数分）
  uv run python experiments/25_model_seeds/run.py --model mistral-7b --smoke

  # dense と、変換前後が妥当な acc を出すかの確認（モデルごとに1回。35分ほど）
  uv run python experiments/25_model_seeds/run.py --model mistral-7b --check

  # 本実行を1本
  uv run python experiments/25_model_seeds/run.py --model mistral-7b --nactive 6 --seed 0

  # 20本を4枚の GPU に配る（sweep.py を見ること）
  uv run python experiments/25_model_seeds/sweep.py --gpus 0,1,2,3
"""

import argparse
import itertools
import json
import os
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
# 既存の seed 0..2 と同じ並びに落ちる
UNTAGGED_MODEL = 'llama2-7b'
# dense を測る動作点と seed。``--check`` を先に通していないときの測る側である
DENSE_AT = (6, 0)

# **旧マシンで測った dense は取り込まない。** report/04 の
# ``bench_h4/wikitext2_seed0`` と report/17 の ``bench_mmlu_slimpajama_seed0``
# には Llama の dense があるが、どちらも RTX PRO 5000 Blackwell / cu128 で
# 測ったものである。5 seed を1台のマシンで測り直す方針にしたので、これらを
# 取り込むと基準だけが別のハードウェアのものになる。GPU をまたぐと浮動小数の
# 積み方が変わり、ここで見ている効果量（acc で 0.005〜0.012）と同じ桁の差が
# 出かねない。取り込み先は ``--check`` が測ったものに一本化する。

# 結果を変えない引数（出力先・チャンク幅・基準の取り込み元）。同じ実験かの判定から外す
NOT_IDENTITY = {'out', 'batch_chunk', 'token_chunk', 'search_token_chunk',
                'no_recheck', 'bench_reference', 'bench_cache_dir'}


def log(message=''):
    print(message, flush=True)


def require_single_gpu():
    """見えている GPU が1枚であることを確かめる。

    アダプタは ``device_map='auto'`` で読むので、**2枚以上見えていると
    accelerate が7Bを分割する**。ジョブが1枚に閉じないので並列にならず、
    それ以上に困るのは dense である — ``--check`` を4枚見える状態で回すと、
    20ジョブ全部が取り込む基準だけが別のデバイス構成で測られる。効果量が
    ``acc`` で 0.005〜0.012 の実験なので、ここは黙って通さない。

    ``sweep.py`` は ``CUDA_VISIBLE_DEVICES`` を1枚だけ渡すので、この検査は
    素通りする。手で1本走らせるときだけ引っかかる。
    """
    import torch

    count = torch.cuda.device_count()
    if count == 1:
        return
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if count == 0:
        raise SystemExit(
            'GPU が見えない。--gpus を渡しているか、ドライバが torch の'
            'ホイールに足りているかを見ること'
            '（experiments/25_model_seeds/preflight.py）')
    raise SystemExit(
        f'GPU が {count} 枚見えている（CUDA_VISIBLE_DEVICES='
        f'{visible if visible else "未設定"}）。1ジョブは1枚に閉じること — '
        "2枚以上あると device_map='auto' が7Bを分割し、この実行が測る値だけが"
        '別のデバイス構成のものになる。`CUDA_VISIBLE_DEVICES=0` を付けるか、'
        'sweep.py 越しに回すこと')


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

    **記録に無いキーは差分に数えない。** CLI に引数が1つ足されると、それ以前に
    測った記録にはそのキーが無く、``given.get(key)`` が ``None`` を返す。既定値が
    ``None`` でなければ（``0`` や ``'all'``）必ず差分になり、中身は同じ実験なのに
    「違う実験の出力である。そのディレクトリを退けること」と案内されて、
    数時間ぶんの正しい結果を捨てることになる。実際 report/06・07 の記録は
    ``lookahead`` / ``profile_positions`` を持たないので、この除外が無いと
    seed 0..2 の段1 が即死する。足された引数は既定値で走ったものとみなし、
    数えた代わりに ``old_arguments`` として呼ぶ側へ返す。
    """
    wanted = vars(build_parser().parse_args(argv))
    given = record.get('arguments', {})
    differences = [key for key in sorted(wanted)
                   if key not in NOT_IDENTITY and key in given
                   and given[key] != wanted[key]]
    missing = sorted(key for key in wanted
                     if key not in NOT_IDENTITY and key not in given)
    return differences, missing


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
                differences, _ = same_experiment(record, argv)
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
    differences, missing = same_experiment(record, argv)
    if differences:
        raise SystemExit(
            f'{out_dir} は違う実験の出力である（{", ".join(differences)}）')
    if missing:
        # 記録が古い引数集合で測られている。既定値で走ったものとみなして通すが、
        # 何を仮定したかは残す
        log(f'  {out_dir.name}: 記録に無い引数 {", ".join(missing)} '
            f'— 既定値で測られたものとみなす')
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


def check_dense(model_key, kind):
    """``--check`` が測った dense の置き場所。"""
    prefix = 'bench' if kind == 'bench' else 'bench_mmlu'
    name = f'{prefix}_{stem(model_key, DENSE_AT[0])}_seed{DENSE_AT[1]}'
    return ROOT / 'result_logs' / 'exp25_check' / name / 'bench' / 'dense.json'


def dense_reference(root, model_key, kind, n_active, seed):
    """取り込む dense。無ければ None（その実行が自分で測る）。

    kind は ``bench``（5タスク）か ``mmlu``。**タスクの顔ぶれが違うと
    ``cmoe run`` が取り込みを断る**ので、2つを別に持つ。

    **``--check`` が測ったものを最優先に取り込む。** これが並列実行の要である。
    check を先に1回通しておけば、(モデル × A × seed) のジョブは互いに何も
    待たずに走れる。check を通していないと、A=6 / seed 0 だけが dense を持ち、
    残りのジョブはそれを待つ — 4枚の GPU に投げると seed 0 以外が
    「dense の基準が要る」で即死する。

    dense は変換していないモデルなので A にも校正にも seed にも配分にも
    依らない。取り込んでよいかの判定は ``cmoe run`` 自身がやる（モデル・
    タスク・shot・limit が違えば断る）ので、緩めても安全側が保たれる。
    """
    # check 自身の実行がここで自分を指すことになるが、それは ``bench_stage`` が
    # 「出力先が参照元の親」で弾く。ここで root を見て弾こうとすると、本実行の
    # root（``result_logs``）が check の親でもあるので、拾えるはずの dense まで
    # 落としてしまう
    from_check = check_dense(model_key, kind)
    if from_check.exists():
        return from_check
    prefix = 'bench' if kind == 'bench' else 'bench_mmlu'
    name = f'{prefix}_{stem(model_key, DENSE_AT[0])}_seed{DENSE_AT[1]}'
    if (n_active, seed) != DENSE_AT:
        # check が無いときの直列の経路。A=6 / seed 0 が先に済んでいる前提
        return root / name / 'bench' / 'dense.json'
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
    if reference is not None and out_dir in reference.parents:
        # 自分が書き先にしているディレクトリを基準として読もうとしている
        # （llama2-7b / A=6 / seed 0 の MMLU 段がこれに当たる）。済みで抜ける
        # 限り無害だが、``retire()`` が走ると dense ごと退避されて、直後の
        # ``cmoe run`` が --bench-reference の欠損で落ちる
        log(f'  dense の置き場所が出力先と同じ（{out_dir.name}）。'
            '取り込みではなくこの実行が持っているものを使う')
        reference = None
    if reference is not None:
        if not reference.exists():
            raise SystemExit(
                f'{reference} が無い。dense の基準が要る。'
                f'並列で回すなら先に '
                f'`--model {plan["model"]} --check` を1回通すこと'
                f'（これで全ジョブが待ち無しで走れる）')
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
    """段の一覧と、測ったコードのコミットを残す。数値そのものは集計しない。

    **既にある一覧には上書きせず足す。** 段を分けて回すことがあるためで
    （``--stages search,score,bench`` を先に通し、あとから ``--stages mmlu``）、
    毎回まるごと書き直すと、後の回に含まれない段の記録が消える。消えると
    ``sweep.py`` の ``is_done`` が「その段はまだ」と誤って判定する。
    """
    if path.exists():
        try:
            previous = json.loads(path.read_text())
        except json.JSONDecodeError:
            previous = {}
        payload = {**previous, **payload}
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
    # 絶対にしておく。相対のままだと ``relative_to(ROOT)`` が段の**後**で落ち、
    # 探索やベンチを走らせきってから traceback で終わる
    root = Path(args.out).resolve() if args.out else default_root
    root.mkdir(parents=True, exist_ok=True)
    require_single_gpu()
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
