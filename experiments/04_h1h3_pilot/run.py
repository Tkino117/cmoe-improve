"""04 H1〜H3 の下見。設計は docs/03、固定する条件は同じディレクトリの README。

seed 1本（探索用①と変換用②を揃える）で H1〜H3 をひととおり出す。seed は
``--seed`` で振り、出力先も seed ごとに分かれる。複数 seed のまとめは
同じディレクトリの ``summarize_seeds.py``。

  uv run python experiments/04_h1h3_pilot/run.py --smoke   # 2層だけの動作確認
  uv run python experiments/04_h1h3_pilot/run.py           # seed 0（約4時間）
  uv run python experiments/04_h1h3_pilot/run.py --seed 1  # seed 1

段（探索1本 / 評価1回）ごとに別ディレクトリへ書き、終わった段は飛ばす。途中で
落ちても済んだ段は残り、同じコマンドで続きから走る。書きかけのディレクトリは
完了印を持たないので ``*.partial<N>`` へ退避してから引き直す — 「途中まで」を
「済み」と取り違えないことが、後から分析するための最低条件である。

集約は走らせた段だけでは作らない。済んでいる段はディスクから読み直すので、
``--stages h2`` を打っても pilot.md から H1・H3 が消えることはない。
"""

import argparse
import itertools
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from cmoe.cli import build_parser
from cmoe.eval.stats import paired_differences, stratified_paired_bootstrap

ROOT = Path(__file__).resolve().parents[2]

ORACLE = 'suffix_kl'
SEARCH = 'beam'
ROUTER = 'cmoe'
BASELINE = 'uniform3'
DATASETS = 'wikitext2,c4-new'
WIDTHS = (2, 3, 4)
BOOTSTRAP_REPS = 10000
BOOTSTRAP_SEED = 20260813
# 対照。先頭が run 内の対応のある比較の基準になるので uniform3 を先に置く
UNIFORMS = [BASELINE] + [f'uniform{x}' for x in (0, 1, 2, 4, 5, 6)]

# 引数が完全に一致するときだけ取り込む、既存の探索。report/01 は wikitext2・幅2 を
# seed 0〜4 で済ませてある。探索が見るのは①だけなので、①=② に揃えた本実験でも
# 探索の出力はそのまま使える（組み直しと評価はこの実験で改めて走る）
REUSE = {('wikitext2', 2, seed):
         ROOT / f'result_logs/calib_seed_variance_suffix_kl_w2/search_seed{seed}'
         for seed in range(5)}
# 結果を変えない引数（出力先・チャンク幅・測り直しの有無）。同じ実験かの判定から外す
NOT_IDENTITY = {'out', 'batch_chunk', 'token_chunk', 'search_token_chunk',
                'no_recheck'}
REUSE_MARK = 'reused_from.txt'


class Missing(Exception):
    """集約したい段がまだ済んでいない。"""


def log(message=''):
    print(message, flush=True)


def run_cli(argv):
    command = ['uv', 'run', 'cmoe', *argv]
    log(f'$ {" ".join(command)}')
    subprocess.run(command, cwd=ROOT, check=True)


def retire(path):
    """書きかけのディレクトリを退避する。消さない（落ちた跡は原因調べに要る）。"""
    for index in itertools.count(1):
        target = path.with_name(f'{path.name}.partial{index}')
        if not target.exists():
            path.rename(target)
            log(f'  書きかけの {path.name} を {target.name} へ退避した')
            return target


def load(path):
    with path.open() as handle:
        return json.load(handle)


def same_experiment(record, argv):
    """記録された引数が、いま走らせようとしている実験と同じものか。

    既定値も含めて突き合わせたいので、同じパーサに通してから比べる。
    """
    wanted = vars(build_parser().parse_args(argv))
    given = record.get('arguments', {})
    return [key for key in sorted(wanted)
            if key not in NOT_IDENTITY and given.get(key) != wanted[key]]


def finished_search(path):
    """終わった探索なら記録を返す。無い・書きかけなら None。

    ``allocation`` は探索が終わってから他の集計値と一度に書かれるので、
    完了印になる（ファイルの存在は開始直後から真になり、印にならない）。
    """
    payload = path / 'search.json'
    if not payload.exists():
        return None
    record = load(payload)
    if 'allocation' not in record:
        return None
    marker = path / REUSE_MARK
    if marker.exists():
        record['reused_from'] = marker.read_text().strip()
    return record


def finished_run(path, expected):
    """終わった評価なら記録を返す。無い・書きかけ・失敗込みなら None。"""
    payload = path / 'summary.json'
    if not payload.exists():
        return None
    record = load(payload)
    if record.get('failures') or 'summary' not in record:
        return None
    return record if len(record.get('runs', [])) == expected else None


def clear(out_dir, argv, name, plan):
    """書きかけを退避する。ただし別実験の出力なら、触らずに止める。"""
    if not out_dir.exists():
        return
    payload = out_dir / name
    if payload.exists():
        differences = same_experiment(load(payload), argv)
        if differences:
            raise SystemExit(
                f'{out_dir} は違う実験の出力である（{", ".join(differences)}）。'
                '別の --out を指定するか、そのディレクトリを退けること')
    if plan['dry']:
        raise Missing(out_dir)
    retire(out_dir)


def adopt(source, out_dir, argv):
    """既存の探索を取り込む。引数が食い違うなら取り込まない。"""
    record = finished_search(source)
    if record is None:
        log(f'  取り込み見送り: {source} に終わった探索が無い')
        return None
    differences = same_experiment(record, argv)
    if differences:
        log(f'  取り込み見送り: {source} は引数が違う（{", ".join(differences)}）')
        return None
    shutil.copytree(source, out_dir)
    (out_dir / REUSE_MARK).write_text(f'{source}\n', encoding='utf-8')
    log(f'  {out_dir.name}: {source} から取り込んだ')
    return finished_search(out_dir)


def search_one(calib, width, out_dir, plan):
    """1本の探索。済んでいれば飛ばす。"""
    argv = ['search', '--search', SEARCH, '--width', str(width),
            '--oracle', ORACLE, '--calib', calib,
            '--seed', str(plan['seed'])]
    if plan['layers']:
        argv += ['--layers', str(plan['layers'])]
    argv += ['--out', str(out_dir)]

    record = finished_search(out_dir)
    if record is not None:
        log(f'  {out_dir.name}: 済み'
            + (f'（{record["reused_from"]} から）' if 'reused_from' in record
               else ''))
    else:
        clear(out_dir, argv, 'search.json', plan)
        source = (None if plan['dry']
                  else plan['reuse'].get((calib, width, plan['seed'])))
        record = adopt(source, out_dir, argv) if source else None
    if record is None:
        if plan['dry']:
            raise Missing(out_dir)
        run_cli(argv)
        record = finished_search(out_dir)
        if record is None:
            raise SystemExit(f'{out_dir} に終わった探索が残らなかった')
    differences = same_experiment(record, argv)
    if differences:
        raise SystemExit(
            f'{out_dir} は違う実験の出力である（{", ".join(differences)}）。'
            '別の --out を指定するか、そのディレクトリを退けること')
    return record


def evaluate(out_dir, allocations, calib, plan):
    """1回の評価。配分は重ねない（同じベクトルは同じ構成として集計される）。"""
    unique = list(dict.fromkeys(allocations))
    argv = ['run']
    for spec in unique:
        argv += ['--alloc', spec]
    argv += ['--router', ROUTER, '--seeds', str(plan['seed']), '--calib', calib,
             '--datasets', plan['datasets']]
    if plan['layers']:
        argv += ['--layers', str(plan['layers'])]
    argv += ['--out', str(out_dir)]

    record = finished_run(out_dir, len(unique))
    if record is None:
        clear(out_dir, argv, 'summary.json', plan)
        if plan['dry']:
            raise Missing(out_dir)
        try:
            run_cli(argv)
        except subprocess.CalledProcessError:
            report_failures(out_dir)
            raise
        record = finished_run(out_dir, len(unique))
        if record is None:
            raise SystemExit(f'{out_dir} に終わった評価が残らなかった')
    else:
        log(f'  {out_dir.name}: 済み')
    differences = same_experiment(record, argv)
    if differences:
        raise SystemExit(
            f'{out_dir} は違う実験の出力である（{", ".join(differences)}）。'
            '別の --out を指定するか、そのディレクトリを退けること')
    return record


def report_failures(out_dir):
    """落ちた構成を名指しする。次の起動で全構成を引き直すことも伝える。"""
    payload = out_dir / 'summary.json'
    if not payload.exists():
        return
    record = load(payload)
    for failure in record.get('failures', []):
        log(f'  失敗した構成: seed {failure["seed"]} / {failure["allocation"]} '
            f'/ {failure["error"]}')
    log(f'  済んだ {len(record.get("runs", []))} 構成は {out_dir} に残るが、'
        '次の起動では退避のうえ全構成を引き直す（cmoe run に途中再開は無い）')


def vector(record):
    return ','.join(str(value) for value in record['allocation']['values'])


def spec_of(entry):
    """評価の記録1件を、``--alloc`` に渡したときの綴りへ戻す。"""
    name = entry['allocation']['name']
    return name if name.startswith('uniform') else vector(entry)


def stage_h1(out, plan):
    log('\n=== H1: 一様 x=0〜6（校正 wikitext2）===')
    return {'calib': 'wikitext2', 'labels': {},
            'eval': evaluate(out / 'h1_uniform', plan['uniforms'],
                             'wikitext2', plan)}


def stage_searched(name, calib, out, plan):
    """校正データを1つ選び、幅ごとに探索して、対照と並べて測る。"""
    searches, labels, allocations = {}, {}, [BASELINE]
    for width in plan['widths']:
        record = search_one(calib, width, out / f'{name}_search_w{width}', plan)
        spec = vector(record)
        searches[width] = record
        # 幅が違っても同じ配分に着くことがある。行を消さずに束ねる
        labels[spec] = (f'{labels[spec]}/{width}' if spec in labels
                        else f'beam 幅{width}')
        allocations.append(spec)
    if len(set(allocations)) < len(allocations):
        log(f'  探索 {len(plan["widths"])} 本中 '
            f'{len(set(allocations)) - 1} 本が相異なる')
    return {'calib': calib, 'searches': searches, 'labels': labels,
            'eval': evaluate(out / f'{name}_eval', allocations, calib, plan)}


def stage_h2(out, plan):
    log('\n=== H2: wikitext2 で校正して探索 ===')
    return stage_searched('h2', 'wikitext2', out, plan)


def stage_h3(out, plan):
    log('\n=== H3: c4 で校正して探索 ===')
    uniform = evaluate(out / 'h3_uniform', plan['uniforms'], 'c4', plan)
    stage = stage_searched('h3', 'c4', out, plan)
    stage['uniform'] = uniform
    return stage


STAGES = {'h1': stage_h1, 'h2': stage_h2, 'h3': stage_h3}


def records_of(stage):
    return [record for record in (stage.get('uniform'), stage['eval'])
            if record is not None]


def collect(stages):
    """全段の評価から、塊ごとの NLL を（校正, 配分, 評価セット）で引ける形に集める。

    段をまたいだ対応のある比較（H1 の一様 × H2 の探索配分）は、これが無いと
    できない。``cmoe run`` の対照は同じ run の先頭構成に限られる。
    """
    pool, conflicts = {}, []
    for stage in stages.values():
        for record in records_of(stage):
            calib = record['arguments']['calib']
            for entry in record['runs']:
                key = (calib, spec_of(entry))
                for dataset, result in entry['ppl'][ROUTER].items():
                    seen = pool.setdefault(key, {}).get(dataset)
                    if seen is None:
                        pool[key][dataset] = result
                    elif seen['chunk_mean_nlls'] != result['chunk_mean_nlls']:
                        # 同じ校正・同じ配分を別の run で組み直した結果である。
                        # 一致しないなら、どこかに決定的でない要素がある
                        conflicts.append(
                            {'calib': calib, 'allocation': key[1],
                             'dataset': dataset, 'ppl': [seen['ppl'],
                                                         result['ppl']]})
    return pool, conflicts


def compare(baseline, candidate):
    """塊ごとの対応のある差と、その95%区間。負なら candidate が良い。"""
    differences = paired_differences(
        SimpleNamespace(chunk_mean_nlls=baseline['chunk_mean_nlls']),
        SimpleNamespace(chunk_mean_nlls=candidate['chunk_mean_nlls']))
    return stratified_paired_bootstrap(
        [differences], reps=BOOTSTRAP_REPS, seed=BOOTSTRAP_SEED)


def interval(paired):
    return (f'{paired["mean_nll_difference"]:+.6f} '
            f'[{paired["lower"]:+.6f}, {paired["upper"]:+.6f}]')


def mean_x(spec, record):
    """配分の平均 x。一様は名前で、探索の結果はベクトルで引き当てる。"""
    for entry in record['runs']:
        if spec_of(entry) == spec:
            return entry['allocation']['mean_x']
    return float('nan')


def table(record, labels, datasets):
    """1回の評価を、構成 × 評価セットの表にする。"""
    rows = []
    for row in record['summary']['configurations']:
        spec = row['allocation']
        cells = {}
        for name in datasets:
            entry = row['datasets'].get(name)
            if entry is None:
                continue
            cells[name] = {'ppl': entry['mean_ppl'],
                           'paired': entry.get('paired_vs_baseline')}
        rows.append({'label': labels.get(spec, spec), 'allocation': spec,
                     'mean_x': mean_x(spec, record), 'datasets': cells})
    return rows


def render(stage, name, datasets, lines):
    """1段ぶんの表を markdown にする。"""
    for part, record in [('一様', stage.get('uniform')), ('', stage['eval'])]:
        if record is None:
            continue
        lines += [f'### {name}' + (f'（{part}）' if part else ''), '',
                  f'校正 {stage["calib"]} / '
                  f'変換 {sum(e["seconds"] for e in record["runs"]) / 60:.1f} 分', '']
        header = ['配分', '平均 x']
        for dataset in datasets:
            header += [f'{dataset} PPL', f'{dataset} 対照比 NLL [95%]']
        lines += ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
        for row in table(record, stage['labels'], datasets):
            cells = [row['label'], f'{row["mean_x"]:.2f}']
            for dataset in datasets:
                entry = row['datasets'].get(dataset)
                if entry is None:
                    cells += ['—', '—']
                    continue
                cells.append(f'{entry["ppl"]:.6f}')
                cells.append('—（対照）' if entry['paired'] is None
                             else interval(entry['paired']))
            lines.append('| ' + ' | '.join(cells) + ' |')
        lines.append('')


def crosswalk(stages, pool, plan, datasets, lines):
    """探索した配分を、すべての一様配分と突き合わせる（段をまたぐ）。

    docs/03 の主張は「**どの**一様配分よりも良い」なので、run 内の uniform3
    との比較だけでは足りない。塊ごとの NLL が残っているので GPU は要らない。
    """
    payload = {}
    order = sorted(plan['uniforms'], key=lambda name: int(name[len('uniform'):]))
    for name, stage in stages.items():
        if not stage['labels']:
            continue
        calib = stage['calib']
        for dataset in datasets:
            rows = []
            for uniform in order:
                base = pool.get((calib, uniform), {}).get(dataset)
                if base is None:
                    continue
                cells = {}
                for spec, label in stage['labels'].items():
                    candidate = pool.get((calib, spec), {}).get(dataset)
                    if candidate is not None:
                        cells[label] = compare(base, candidate)
                rows.append({'uniform': uniform, 'ppl': base['ppl'],
                             'cells': cells})
            if not rows:
                continue
            payload.setdefault(name, {})[dataset] = rows
            labels = list(stage['labels'].values())
            lines += [f'### {name.upper()} 探索配分 × 一様配分すべて'
                      f'（校正 {calib} / 評価 {dataset}）', '',
                      '各セルは、その行の一様配分を対照とした塊ごとの平均 NLL 差。'
                      '負なら探索配分が良い。', '',
                      '| 対照 | PPL | ' + ' | '.join(labels) + ' |',
                      '|' + '---|' * (len(labels) + 2)]
            for row in rows:
                cells = [interval(row['cells'][label]) if label in row['cells']
                         else '—' for label in labels]
                lines.append(f'| {row["uniform"]} | {row["ppl"]:.6f} | '
                             + ' | '.join(cells) + ' |')
            counts = []
            for label in labels:
                won = sum(1 for row in rows if label in row['cells']
                          and row['cells'][label]['upper'] < 0)
                counts.append(f'{label}: {won}/{len(rows)}')
            lines += ['', '区間が0を含まず負（探索配分が良い）だった対照の本数 — '
                      + ' / '.join(counts), '']
    return payload


def summarize(out, stages, plan, seconds):
    datasets = plan['datasets'].split(',')
    pool, conflicts = collect(stages)
    lines = ['# 04 H1〜H3 の下見（seed 1本）', '',
             f'探索 {SEARCH} × {ORACLE} / router {ROUTER} / '
             f'seed {plan["seed"]}（①=②）/ '
             f'評価 {plan["datasets"]}', '',
             '対照比 NLL は、対照との塊ごとの平均 NLL の差（負なら良い）。'
             '区間は評価塊の再標本化のみで、**seed の振り直しを含まない** — '
             'seed 間のばらつきは report/01 の標準偏差 0.0255（PPL, wikitext2。'
             'ただし①だけを振った値）を目安にすること。複数 seed を回したら '
             'summarize_seeds.py の実測に置き換える。', '']
    titles = {'h1': 'H1 一様 x=0〜6（校正 wikitext2）',
              'h2': 'H2 wikitext2 で校正して探索',
              'h3': 'H3 c4 で校正して探索'}
    payload = {'plan': {key: value for key, value in plan.items()
                        if key not in ('reuse', 'dry')},
               'repository': revision(), 'seconds': seconds, 'stages': {}}
    for name, stage in stages.items():
        render(stage, titles[name], datasets, lines)
        entry = {'calib': stage['calib'],
                 'rows': table(stage['eval'], stage['labels'], datasets)}
        if stage.get('uniform'):
            entry['uniform_rows'] = table(stage['uniform'], {}, datasets)
        entry['searches'] = {
            str(width): {'allocation': record['allocation']['values'],
                         'score': record['score'], 'seconds': record['seconds'],
                         'spent': record['spent'], 'calls': record['calls'],
                         'recheck': record.get('recheck', {}).get('gap'),
                         'above_floor': record.get('recheck', {}).get('above_floor'),
                         'nondeterminism': record.get('nondeterminism'),
                         'reused_from': record.get('reused_from')}
            for width, record in stage.get('searches', {}).items()}
        payload['stages'][name] = entry

    payload['crosswalk'] = crosswalk(stages, pool, plan, datasets, lines)

    searches = [(name, width, record)
                for name, stage in stages.items()
                for width, record in stage.get('searches', {}).items()]
    if searches:
        lines += ['### 探索そのもの', '',
                  '| 段 | 幅 | 最終スコア | 候補評価 | コスト | 測り直し | 所要 |',
                  '|---|---|---|---|---|---|---|']
        for name, width, record in searches:
            recheck = record.get('recheck')
            floor = record.get('nondeterminism')
            if recheck is None:
                check = '未実施'
            else:
                verdict = '不一致' if recheck.get('above_floor') else '一致'
                check = (f'{recheck["gap"]:.3e}'
                         + ('' if floor is None else f'（床 {floor:.3e}）')
                         + f' {verdict}')
            elapsed = f'{record["seconds"] / 60:.1f} 分'
            if record.get('reused_from'):
                elapsed += '（再利用）'
            lines.append(
                f'| {name} | {width} | {record["score"]:.6e} | '
                f'{record["calls"]} | {record["spent"]:g} | {check} | {elapsed} |')
        lines += ['', '出てきた配分', '', '```']
        for name, width, record in searches:
            lines.append(f'{name} 幅{width}: {vector(record)}')
        lines += ['```', '']

    payload['consistency'] = conflicts
    if conflicts:
        lines += ['### 食い違い', '',
                  '同じ校正・同じ配分を別の評価で組み直した結果が一致しなかった。'
                  'どこかに決定的でない要素がある。', '']
        for item in conflicts:
            lines.append(f'- {item["calib"]} / {item["allocation"]} / '
                         f'{item["dataset"]}: PPL {item["ppl"]}')
        lines.append('')
    else:
        lines += ['重複して組み直した構成（各校正の uniform3）は、別の評価の'
                  'あいだで塊ごとの NLL まで完全に一致した。', '']

    lines += [f'全体 {seconds / 60:.1f} 分', '']
    report = '\n'.join(lines)
    (out / 'pilot.md').write_text(report, encoding='utf-8')
    with (out / 'pilot.json').open('w') as handle:
        json.dump(payload, handle, indent=1, ensure_ascii=False)
    log('\n' + report)
    log(f'書き出し: {out / "pilot.md"} / {out / "pilot.json"}')


def revision():
    """どのコードで測ったか。後から結果を突き合わせるときに要る。"""
    def git(*arguments):
        result = subprocess.run(['git', *arguments], cwd=ROOT,
                                capture_output=True, text=True)
        return result.stdout.strip()
    return {'head': git('rev-parse', 'HEAD'),
            'dirty': bool(git('status', '--porcelain'))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stages', default=','.join(STAGES),
                        help='走らせる段。カンマ区切り。済んでいる他の段も集約には入る')
    parser.add_argument('--seed', type=int, default=0,
                        help='校正に引く8本を選ぶ seed（①=②）')
    parser.add_argument('--smoke', action='store_true',
                        help='2層・幅2・wikitext2 だけの動作確認')
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    names = [field.strip() for field in args.stages.split(',') if field.strip()]
    unknown = [name for name in names if name not in STAGES]
    if unknown:
        raise SystemExit(f'未知の段 {unknown}。{list(STAGES)} から選ぶ')

    plan = {'layers': 2 if args.smoke else None,
            'widths': (2,) if args.smoke else WIDTHS,
            'uniforms': UNIFORMS[:2] if args.smoke else UNIFORMS,
            'datasets': 'wikitext2' if args.smoke else DATASETS,
            'reuse': {} if args.smoke else REUSE,
            'seed': args.seed,
            'dry': False}
    # seed 0 の出力先は既存の実行に合わせて添字を付けない
    stem = 'h1h3_pilot' + ('_smoke' if args.smoke else '')
    if args.seed:
        stem += f'_seed{args.seed}'
    out = (Path(args.out).resolve() if args.out else
           ROOT / 'result_logs' / stem)
    out.mkdir(parents=True, exist_ok=True)
    log(f'出力 {out} / seed {args.seed} / 段 {names}' + ('（smoke: 2層だけ）' if args.smoke else ''))

    started = time.time()
    stages = {}
    for name, stage in STAGES.items():
        plan['dry'] = name not in names
        try:
            stages[name] = stage(out, plan)
        except Missing as error:
            log(f'  {name}: 済んでいないので集約に入れない（{error}）')
        if not plan['dry']:
            log(f'  ここまで {(time.time() - started) / 60:.1f} 分')
    summarize(out, stages, plan, time.time() - started)
    return 0


if __name__ == '__main__':
    sys.exit(main())
