"""25 の (モデル × 動作点 × seed) を、複数の GPU に配って回す。

``run.py`` は1ジョブ（1つのモデル・1つの A・1つの seed）を回す。ここはその
ジョブを並べて、空いた GPU に次を渡すだけの配り役である。**1ジョブは1枚の GPU
に閉じる** — ``CUDA_VISIBLE_DEVICES`` を1枚だけ見せるので、アダプタの
``device_map='auto'`` はそのGPUに全部載せる。4枚見せると7Bが4枚に分散して、
並列にならないどころか遅くなる。

    ジョブ = 2モデル × 2動作点(A=6, A=4) × 5 seed = 20本

**先に ``--check`` を通しておくこと。** dense（5タスクと MMLU）はモデルごとに
1回でよく、check がそれを測る。通してあれば20本は互いに何も待たない。通して
いないと A=6 / seed 0 以外が「dense の基準が要る」で即死する。段取りは:

    # 1. dense と、変換後が妥当かの確認（モデルごとに1回。各35分ほど）
    uv run python experiments/25_model_seeds/run.py --model llama2-7b  --check
    uv run python experiments/25_model_seeds/run.py --model mistral-7b --check

    # 2. 20本を4枚に配る
    uv run python experiments/25_model_seeds/sweep.py --gpus 0,1,2,3

    # 落ちたジョブだけ、あるいは特定のモデルだけ
    uv run python experiments/25_model_seeds/sweep.py --gpus 0,1,2,3 --models mistral-7b
    uv run python experiments/25_model_seeds/sweep.py --gpus 0,1,2,3 --dry-run

**同じジョブを2枚に渡さない。** ジョブは (モデル, A, seed) で一意で、出力先も
その3つで決まる。同じ ``result_logs`` を共有していても衝突しない。ただし
**同じジョブを2回同時に走らせると衝突する**ので、この配り役を2つ立てないこと。

済んだジョブは ``run.py`` 側の ``ensure()`` が飛ばすので、落ちたところから同じ
コマンドで再開できる。ログはジョブごとに ``result_logs/sweep_logs/`` に残す。
"""

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

MODELS = ('llama2-7b', 'mistral-7b')
NACTIVES = (6, 4)
SEEDS = (0, 1, 2, 3, 4)
LOG_DIR = ROOT / 'result_logs' / 'sweep_logs'


def log(message=''):
    print(message, flush=True)


def jobs(models, nactives, seeds):
    """走らせる順。**長いものから**投げる。

    A=6 は A=4 より探索が長い（幅2/3/4 で 2.6h 対 2.1h）ので、先に出しておく
    ほうが最後に1本だけ残って GPU が3枚遊ぶ形になりにくい。seed は
    その次の軸で、モデルは最後（片方のモデルだけ先に揃うより、両方が同じ
    ペースで進むほうが途中で表を書き始められる）。
    """
    ordered = [(model, nactive, seed)
               for nactive in sorted(nactives, reverse=True)
               for seed in sorted(seeds)
               for model in models]
    return ordered


def job_name(model, nactive, seed):
    return f'{model}_a{nactive}_seed{seed}'


def stages_path(model, nactive, seed):
    return ROOT / 'result_logs' / f'exp25_stages_{job_name(model, nactive, seed)}.json'


def is_done(model, nactive, seed):
    """段の一覧が残っていれば済み。``run.py`` が最後に書く。"""
    path = stages_path(model, nactive, seed)
    if not path.exists():
        return False
    try:
        record = json.loads(path.read_text())
    except json.JSONDecodeError:
        return False
    return 'bench' in record and 'mmlu' in record


def launch(model, nactive, seed, gpu, widths):
    """1ジョブを1枚の GPU で起こす。標準出力はジョブごとのログへ。"""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    name = job_name(model, nactive, seed)
    handle = (LOG_DIR / f'{name}.log').open('a')
    handle.write(f'\n===== {time.strftime("%Y-%m-%d %H:%M:%S")} GPU{gpu} =====\n')
    handle.flush()
    environment = dict(os.environ)
    # **1枚だけ見せる。** アダプタは device_map='auto' なので、見せた枚数に
    # 応じて分散する。1枚なら cuda:0 に全部載る
    environment['CUDA_VISIBLE_DEVICES'] = str(gpu)
    # 段は ``uv run cmoe`` を呼ぶ。4本が同時に venv を触ると解決がぶつかるので、
    # 揃っている前提で走らせる（イメージの build 時に uv sync 済み）
    environment['UV_FROZEN'] = '1'
    environment['UV_NO_SYNC'] = '1'
    # lm-eval / tokenizers が並列で走ると、1ジョブが CPU を占有して他を待たせる
    environment.setdefault('OMP_NUM_THREADS', '8')
    environment.setdefault('TOKENIZERS_PARALLELISM', 'false')
    command = [sys.executable, str(HERE / 'run.py'),
               '--model', model, '--nactive', str(nactive), '--seed', str(seed),
               '--widths', widths]
    process = subprocess.Popen(command, cwd=ROOT, env=environment,
                               stdout=handle, stderr=subprocess.STDOUT)
    return {'name': name, 'gpu': gpu, 'process': process, 'handle': handle,
            'started': time.time()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus', default='0',
                        help='使う GPU の番号（カンマ区切り）。1枚に1ジョブ')
    parser.add_argument('--models', default=','.join(MODELS))
    parser.add_argument('--nactives', default=','.join(str(a) for a in NACTIVES))
    parser.add_argument('--seeds', default=','.join(str(s) for s in SEEDS))
    parser.add_argument('--widths', default='2,3,4')
    parser.add_argument('--poll', type=float, default=20.0,
                        help='空きを見に行く間隔（秒）')
    parser.add_argument('--dry-run', action='store_true',
                        help='走らせるジョブの一覧だけ出す')
    args = parser.parse_args(argv)

    gpus = [int(field) for field in args.gpus.split(',') if field.strip()]
    models = [field.strip() for field in args.models.split(',') if field.strip()]
    nactives = [int(field) for field in args.nactives.split(',') if field.strip()]
    seeds = [int(field) for field in args.seeds.split(',') if field.strip()]

    pending = [job for job in jobs(models, nactives, seeds)
               if not is_done(*job)]
    skipped = len(jobs(models, nactives, seeds)) - len(pending)

    log(f'GPU {gpus} / ジョブ {len(pending)} 本'
        + (f'（済み {skipped} 本は飛ばす）' if skipped else ''))
    for model, nactive, seed in pending:
        log(f'  {job_name(model, nactive, seed)}')
    if args.dry_run:
        return 0
    if not pending:
        log('走らせるものが無い')
        return 0

    started = time.time()
    queue = list(pending)
    free = list(gpus)
    running = []
    failed = []

    while queue or running:
        while queue and free:
            gpu = free.pop(0)
            model, nactive, seed = queue.pop(0)
            entry = launch(model, nactive, seed, gpu, args.widths)
            running.append(entry)
            log(f'[{time.strftime("%H:%M:%S")}] 起動 {entry["name"]} → GPU{gpu} '
                f'（残り {len(queue)} 本）')

        time.sleep(args.poll)

        for entry in list(running):
            code = entry['process'].poll()
            if code is None:
                continue
            running.remove(entry)
            entry['handle'].close()
            free.append(entry['gpu'])
            minutes = (time.time() - entry['started']) / 60
            if code == 0:
                log(f'[{time.strftime("%H:%M:%S")}] 完了 {entry["name"]} '
                    f'（{minutes:.1f}分）')
            else:
                failed.append(entry['name'])
                log(f'[{time.strftime("%H:%M:%S")}] **失敗** {entry["name"]} '
                    f'(exit {code}, {minutes:.1f}分) — '
                    f'{LOG_DIR / (entry["name"] + ".log")}')

    log(f'\n全体 {(time.time() - started) / 3600:.1f}時間')
    if failed:
        log(f'失敗 {len(failed)} 本: {", ".join(failed)}')
        log('同じコマンドで再開できる（済んだ段は飛ばす）')
        return 1
    log('全ジョブ完了')
    return 0


if __name__ == '__main__':
    sys.exit(main())
