"""22-c プローブと既存の探索結果から、対照の配分ベクトルを全部作る。GPU は要らない。

出すのは3群である。**どれも提案手法ではない。**

* **順序対照**（``reverse`` / ``shuffle1`` / ``shuffle2``）… 提案（beam 幅4）の
  ベクトルの**多重集合を保ったまま**層の並びだけを崩す。平均 x も x の度数分布も
  提案と同一になるので、差が出れば「層別に配分を決めること」自体に意味がある
* **LExI 相当**（``lexi``）… ``lexi_probe.py`` が作った層ローカル誤差の表の、
  層ごとの独立な argmin
* **OWL 相当**（``owl_*``）… ``owl_probe.py`` が作った外れ値比率を、EW-rule と
  同じ写像（``alloc.score_rule``）に載せたもの

OWL には「この軸での既定値」が原典に無い（原典が出すのは層別スパース率であって
shared の割合ではない）ので、自由度を2段に分けて扱う。

* ``owl_default`` … 原典の既定 M=5、向きは OWL の意味論（外れ値の多い層ほど
  重要 → 削らない → routing を広く取る = x を小さく）、α レンジは **EW-rule の
  既定と同じ [0.2, 0.7]**。N=8 という共通の制約のもとで、2本の規則に同じ値域を
  与えるのが対等な扱いである。**これが主表の行**
* ``owl_grid``    … M・向き・α を振った候補。**どれを使うかは校正データ上の
  目的関数（``score.py`` = ``cmoe score --oracle suffix_kl``）で選ぶ**。評価
  ベンチを見て選ぶと、EW の既定値が評価ベンチ上の掃引で決まっていることを
  批判している側が同じことをすることになる。EW-rule 側にも同じ格子を用意し、
  **両方の規則を対称に扱う**（片方だけ校正で選び直すのは不公平である）

  uv run python experiments/22_alloc_baselines/allocate.py --seed 0 --nactive 6
  uv run python experiments/22_alloc_baselines/allocate.py --seed 0 --nactive 4 --out ...
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from cmoe.alloc import lexi, order, score_rule
from cmoe.alloc.ew_rule import ew_allocation
from cmoe.alloc.owl import DEFAULT_M

ROOT = Path(__file__).resolve().parents[2]

# 提案手法の配分の出どころ。report/09（25%）・report/16（50%）と同じ表
PROPOSAL_SOURCE = {6: 'result_logs/exp06_seeds/seeds.json',
                   4: 'result_logs/exp07_seeds/seeds.json'}
PROPOSAL_LABEL = 'beam 幅4'

# OWL の主行。原典の既定 M と、EW-rule と同じ α レンジ
OWL_DEFAULT = {'m': DEFAULT_M, 'direction': 'desc',
               'alpha': (0.2, 0.7), 'normalize': 'minmax'}
# 校正で選び直す格子。M は原典の4点から3点（5 を必ず含む）、α は既定と全域
OWL_GRID_M = (3.0, 5.0, 10.0)
OWL_GRID_DIRECTION = ('desc', 'asc')
OWL_GRID_ALPHA = ((0.2, 0.7), (0.0, 1.0))
# EW-rule 側の格子。同じ本数・同じ α レンジで対称にする。τ は EW 既定 0.6 と、
# report/21 が「CV 分布から外れて縮退する」と書いた側を避けた2点
EW_GRID_TAU = (0.15, 0.30, 0.60)
EW_GRID_ALPHA = OWL_GRID_ALPHA


def spec_of(values):
    return ','.join(str(int(x)) for x in values)


def load_json(path):
    with (ROOT / path).open() as handle:
        return json.load(handle)


def proposal_values(seed, n_active, layers=None):
    payload = load_json(PROPOSAL_SOURCE[n_active])
    spec = payload['allocations'][str(seed)][PROPOSAL_LABEL]
    values = [int(field) for field in spec.split(',')]
    return values[:layers] if layers else values


def calibration_uniform(seed, n_active):
    """校正が ``suffix_kl`` で選んだ一様配分の名前。表の対照の1本。"""
    payload = load_json(PROPOSAL_SOURCE[n_active])
    return payload['calibration_choice'][str(seed)]['best']


def order_controls(values, seed, n_active):
    """反転と、独立な並べ替え2本。seed は決定的に作る。"""
    rows = [('reverse', list(order.reverse(values)), {'kind': 'reverse'})]
    for index in (1, 2):
        shuffle_seed = seed * 1000 + n_active * 10 + index
        permuted = list(order.shuffle(values, seed=shuffle_seed))
        rows.append((f'shuffle{index}', permuted,
                     {'kind': 'shuffle', 'shuffle_seed': shuffle_seed}))
    return [(name, permuted, {**detail, **order.displacement(values, permuted)})
            for name, permuted, detail in rows]


def lexi_row(probe_dir, n_active):
    payload = load_json(Path(probe_dir) / 'lexi.json')
    if payload['arguments']['nactive'] != n_active:
        raise SystemExit(
            f'{probe_dir} は A={payload["arguments"]["nactive"]} の表である'
            f'（要るのは A={n_active}）')
    table = [{int(x): float(v) for x, v in row.items()}
             for row in payload['allocation']['table']]
    values, detail = lexi.argmin_allocation(table, n_active)
    return ('lexi', values, {'kind': 'lexi', 'probe': str(probe_dir),
                             'mean_x': detail['mean_x'],
                             'n_routing_layers': detail['n_routing_layers'],
                             'n_no_routing_layers':
                                 detail['n_no_routing_layers'],
                             'sum_local_error': detail['sum_local_error'],
                             'degenerate': detail['degenerate']})


def owl_scores(probe_dir, m):
    payload = load_json(Path(probe_dir) / 'owl.json')
    rows = payload['layers']
    key = str(float(m)) if str(float(m)) in rows[0]['ratios'] else str(m)
    try:
        return [row['ratios'][key] for row in rows]
    except KeyError:
        raise SystemExit(
            f'{probe_dir} に M={m} が無い（あるのは {sorted(rows[0]["ratios"])}）')


def owl_rows(probe_dir, n_experts, n_active):
    """主行1本と、校正で選び直すための格子。"""
    rows = []
    default = OWL_DEFAULT
    values, detail = score_rule.rule_allocation(
        owl_scores(probe_dir, default['m']), n_experts, n_active,
        *default['alpha'], direction=default['direction'],
        normalize=default['normalize'])
    rows.append(('owl_default', values,
                 {'kind': 'owl', 'role': 'default', 'm': default['m'],
                  'direction': default['direction'],
                  'alpha': list(default['alpha']),
                  'mean_x': detail['mean_x'], 'n_clipped': detail['n_clipped'],
                  'degenerate': detail['degenerate']}))
    for m in OWL_GRID_M:
        scores = owl_scores(probe_dir, m)
        for direction in OWL_GRID_DIRECTION:
            for alpha in OWL_GRID_ALPHA:
                values, detail = score_rule.rule_allocation(
                    scores, n_experts, n_active, *alpha, direction=direction,
                    normalize='minmax')
                name = f'owl_m{m:g}_{direction}_a{alpha[0]:g}-{alpha[1]:g}'
                rows.append((name, values,
                             {'kind': 'owl', 'role': 'grid', 'm': m,
                              'direction': direction, 'alpha': list(alpha),
                              'mean_x': detail['mean_x'],
                              'n_clipped': detail['n_clipped'],
                              'degenerate': detail['degenerate']}))
    return rows


def ew_rows(probe_dir, n_experts, n_active, layers=None):
    """EW-rule の格子。OWL と対称に扱うためだけにある。

    既定（τ=0.6, α=[0.2, 0.7]）の行は report/21 で既に測ってあるので、ここは
    格子だけを出す（既定も格子に含まれる）。
    """
    array = np.load(ROOT / probe_dir / 'cv.npy')
    cvs = [torch.from_numpy(row) for row in array]
    if layers:
        cvs = cvs[:layers]
    rows = []
    for tau in EW_GRID_TAU:
        for alpha in EW_GRID_ALPHA:
            values, detail = ew_allocation(cvs, n_experts, n_active, tau=tau,
                                           alpha_min=alpha[0],
                                           alpha_max=alpha[1])
            rows.append((f'ew_t{tau:g}_a{alpha[0]:g}-{alpha[1]:g}', values,
                         {'kind': 'ew', 'role': 'grid', 'tau': tau,
                          'alpha': list(alpha), 'mean_x': detail['mean_x'],
                          'n_clipped': detail['n_clipped'],
                          'degenerate': len(set(values)) == 1}))
    return rows


def deduplicate(rows):
    """同じベクトルの候補を1本にまとめる。採点は1本あたり数分かかる。

    まとめた名前は ``aliases`` に残す — どの格子点がそこへ落ちたかは、規則の
    自由度が実際にどれだけ効いているかの証拠になる。
    """
    merged = {}
    for name, values, detail in rows:
        spec = spec_of(values)
        if spec in merged:
            merged[spec]['aliases'].append(name)
            merged[spec]['details'].append(detail)
            continue
        merged[spec] = {'name': name, 'values': [int(x) for x in values],
                        'spec': spec, 'aliases': [], 'details': [detail]}
    return list(merged.values())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--nactive', type=int, default=6)
    parser.add_argument('--nexperts', type=int, default=8)
    parser.add_argument('--calib', default='slimpajama')
    parser.add_argument('--lexi-probe', default=None)
    parser.add_argument('--owl-probe', default=None)
    parser.add_argument('--ew-probe', default=None)
    parser.add_argument('--layers', type=int, default=None,
                        help='先頭 n 層に切る。動作確認用。プローブ側も同じ層数で'
                             '走っていないと、規則の入力と出力の長さが合わない')
    parser.add_argument('--out', default=None, help='JSON の書き出し先')
    args = parser.parse_args(argv)

    tag = '' if args.nactive == 6 else f'_a{args.nactive}'
    lexi_probe = args.lexi_probe or \
        f'result_logs/lexi_probe_{args.calib}{tag}_seed{args.seed}'
    owl_probe = args.owl_probe or \
        f'result_logs/owl_probe_{args.calib}_seed{args.seed}'
    ew_probe = args.ew_probe or \
        f'result_logs/ew_probe_{args.calib}_signed_seed{args.seed}'

    proposal = proposal_values(args.seed, args.nactive, args.layers)
    uniform = calibration_uniform(args.seed, args.nactive)
    print(f'seed {args.seed}  N={args.nexperts} A={args.nactive}'
          f'（スパース率 {100 - 100 * args.nactive // args.nexperts}%）')
    print(f'提案（{PROPOSAL_LABEL}）  {spec_of(proposal)}')
    print(f'  平均x={sum(proposal) / len(proposal):.4f}  '
          f'度数={ {x: proposal.count(x) for x in sorted(set(proposal))} }')
    print(f'校正が選んだ一様: {uniform}')

    rows = []
    print('\n=== 順序対照（多重集合は提案と同一）===')
    for name, values, detail in order_controls(proposal, args.seed,
                                               args.nactive):
        rows.append((name, values, detail))
        print(f'  {name:<10} {spec_of(values)}')
        print(f'{"":<12} x が変わった層={detail["n_changed"]}/{len(values)} '
              f'平均|Δx|={detail["mean_abs_shift"]:.3f} '
              f'最大|Δx|={detail["max_abs_shift"]}')

    print('\n=== LExI 相当 ===')
    name, values, detail = lexi_row(lexi_probe, args.nactive)
    rows.append((name, values, detail))
    print(f'  {name:<10} {spec_of(values)}')
    print(f'{"":<12} 平均x={detail["mean_x"]:.4f} '
          f'routing のある層={detail["n_routing_layers"]}/{len(values)} '
          f'Top-K=0 の層={detail["n_no_routing_layers"]}/{len(values)}'
          + ('  ** 一様に潰れている **' if detail['degenerate'] else ''))

    print('\n=== OWL 相当 ===')
    owl = owl_rows(owl_probe, args.nexperts, args.nactive)
    rows.extend(owl)
    for name, values, detail in owl:
        mark = '  <- 主表の行' if detail['role'] == 'default' else ''
        flat = '  ** 縮退 **' if detail['degenerate'] else ''
        print(f'  {name:<26} 平均x={detail["mean_x"]:.3f} '
              f'clip={detail["n_clipped"]:2d}  {spec_of(values)}{mark}{flat}')

    print('\n=== EW-rule の格子（OWL と対称に扱うため）===')
    ew = ew_rows(ew_probe, args.nexperts, args.nactive, args.layers)
    rows.extend(ew)
    for name, values, detail in ew:
        flat = '  ** 縮退 **' if detail['degenerate'] else ''
        print(f'  {name:<26} 平均x={detail["mean_x"]:.3f} '
              f'clip={detail["n_clipped"]:2d}  {spec_of(values)}{flat}')

    lengths = {len(values) for _, values, _ in rows} | {len(proposal)}
    if len(lengths) != 1:
        raise SystemExit(
            f'候補の層数が揃っていない（{sorted(lengths)}）。'
            'プローブと提案の層数を合わせること')

    unique = deduplicate(rows)
    print(f'\n候補 {len(rows)} 本 → 重複を除いて {len(unique)} 本')
    for entry in unique:
        if entry['aliases']:
            print(f'  {entry["name"]} = ' + ' = '.join(entry['aliases']))

    payload = {
        'seed': args.seed, 'n_experts': args.nexperts,
        'n_active': args.nactive, 'calib': args.calib,
        'probes': {'lexi': lexi_probe, 'owl': owl_probe, 'ew': ew_probe},
        'proposal': {'label': PROPOSAL_LABEL, 'values': proposal,
                     'spec': spec_of(proposal)},
        'calibration_uniform': uniform,
        'candidates': unique,
    }
    out = args.out or f'result_logs/alloc22{tag}_seed{args.seed}.json'
    with (ROOT / out).open('w') as handle:
        json.dump(payload, handle, indent=1, ensure_ascii=False)
    print(f'\n書いた: {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
