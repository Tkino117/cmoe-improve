"""19 低ランクの活性質量ルーター（方式7）を、選択問題ベンチマークで測る。

report/18 は、真の |h| を読むオラクル ``oracle_abs`` が `acc` をマクロ +0.0221
上げること（dense との差の 33.5% を埋めること）を示した。ただしそのオラクルは
選ぶ前に H を作ってしまうので配備できない。report/10 で測った配備できる5方式は、
オラクルとの Top-K 一致率が 0.15、gap 回収が 16% で頭打ちだった。

方式7 ``spectral_mass`` は「expert を代表する実在ニューロン1本」という族を出て、
質量の和そのものを重み行列の低ランク近似で見積もる。オフラインの探査
（``probe.py``）では、rank 32 で一致率 0.27・gap 回収 46% まで来ている。

ここで測るのは1つだけ — **その一致率がベンチの `acc` に出るか**である。

対照も測り方も report/10・18 と同一にしてある（同じ配分・分割・Top-K・校正・
seed・dense の基準）ので、この実験の `cmoe` 行は report/10 の方式1 と同じ数字に
なるはずであり、そうならなければ設定が動いている。

  uv run python experiments/19_spectral_router/run.py --smoke   # 2層・タスク8問
  uv run python experiments/19_spectral_router/run.py           # 本測定（3 seed）
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# report/10・18 と同じ動作点（S3A3E8。スパース率25%、Top-K=3、routed 5）
ALLOC = 'uniform3'
CALIB = 'wikitext2'
NEXPERTS, NACTIVE = 8, 6
SEEDS = '0,1,2'
# 1回の変換の中に方式を並べる。ルーターだけを差し替えて比べるので、分割も
# expert 重みも共有され、差は ``MoE.gate`` だけになる
#
#   spectral_mass:r  … 固定 Top-K のまま、score を低ランクの質量見積もりにする
#   dynamic_cmoe     … score は現行のまま、固定 Top-K をやめる（追加コスト無し）
#   dynamic_spectral … 両方
#
# rank を振った表は PPL（`result_logs/spectral_r*_seed0`）にあり、ベンチは
# 軸ごとの寄与を切り分けるためだけに使う。
#
# **両方を同時に入れた `dynamic_spectral` はここに無い。** `result_logs/
# dynamic_ppl_seed0` の PPL が rank 32 で対照より悪く（wikitext2 7.332 対
# 7.072）、rank 128 でも単独の2方式に届かなかった（6.990 対 6.877 / 6.944）
# ため、ベンチを測る対象から外してある。層ローカルの誤差ではこの方式が最良
# だったので、そこで落ちたことも含めて report/19 に記録する
ROUTERS = 'cmoe,spectral_mass:128,dynamic_cmoe'
BENCH_BATCH = 8
# dense は変換していないモデルなので、seed にも配分にも校正にも依らない。
# report/04 で測ったものをそのまま取り込む（report/18 と同じ基準）
DENSE = 'result_logs/bench_h4/wikitext2_seed0/bench/dense.json'
OUT = 'result_logs/spectral_router_bench'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true',
                        help='2層・タスクあたり8問で配線だけ確かめる')
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    argv = [
        'run',
        '--alloc', ALLOC,
        '--router', ROUTERS,
        '--calib', CALIB,
        '--nexperts', str(NEXPERTS),
        '--nactive', str(NACTIVE),
        '--diagnostics',
        '--bench',
        '--bench-batch-size', str(BENCH_BATCH),
    ]
    if args.smoke:
        argv += ['--layers', '2', '--seeds', '0', '--bench-limit', '8',
                 '--no-ppl', '--no-bench-dense',
                 '--out', args.out or 'result_logs/spectral_router_smoke']
    else:
        argv += ['--seeds', SEEDS, '--bench-reference', DENSE,
                 '--out', args.out or OUT]

    command = ['uv', 'run', 'cmoe', *argv]
    print('$ ' + ' '.join(command), flush=True)
    return subprocess.run(command, cwd=ROOT).returncode


if __name__ == '__main__':
    sys.exit(main())
