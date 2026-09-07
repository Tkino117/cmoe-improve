"""測定を始める前に、その環境で本当に走るかを確かめる。

**108時間ぶんを投げる前に、これを1回通すこと。** 見るのは4つで、どれも数分で
終わる。落ちるとしたら最初の2つで、どちらもドライバとホイールの組み合わせの
問題である。

1. torch がその GPU を使えるか。**cu128 のホイールを CUDA 12.2 のドライバで
   動かす**構成なので（minor version compatibility に頼っている）、ここが
   通らなければ torch を落とす判断が要る。sm_89 のカーネルが入っているかは
   実際に行列積を走らせて確かめる — ``is_available()`` だけでは
   「カーネルが無い」を見逃す。
2. GPU が何枚見えているか。**1ジョブには1枚だけ見せる**のが前提なので、
   ``CUDA_VISIBLE_DEVICES`` を設けたときに1枚になることを確認する。
3. モデルが読めるか。2本とも層構造の検査（``check_shape``）まで通す。
4. データセットが引けるか。校正・評価・ベンチのどれも、初回はダウンロードが
   走る。**測定の途中でこれが起きると数時間の実行が止まる**ので先に温める。

  uv run python experiments/25_model_seeds/preflight.py
  CUDA_VISIBLE_DEVICES=2 uv run python experiments/25_model_seeds/preflight.py
  uv run python experiments/25_model_seeds/preflight.py --skip-models   # 速く
"""

import argparse
import os
import sys
import time
import traceback

CHECKS = []


def check(name):
    def register(function):
        CHECKS.append((name, function))
        return function
    return register


def log(message=''):
    print(message, flush=True)


@check('torch と GPU')
def check_torch():
    import torch
    log(f'  torch {torch.__version__} / cu{torch.version.cuda}')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA が使えない。--gpus を渡しているか、'
                           'ドライバが torch のホイールに足りているかを見る')
    count = torch.cuda.device_count()
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '(未設定 = 全部)')
    log(f'  見えている GPU {count} 枚 / CUDA_VISIBLE_DEVICES={visible}')
    for index in range(count):
        major, minor = torch.cuda.get_device_capability(index)
        name = torch.cuda.get_device_name(index)
        total = torch.cuda.get_device_properties(index).total_memory / 1024 ** 3
        log(f'    [{index}] {name} sm_{major}{minor} {total:.1f}GiB')
    if count > 1:
        log('  注意: 1ジョブには1枚だけ見せること。複数見せると '
            "device_map='auto' が7Bを分散させて、並列にならない")
    # is_available() は通ってもカーネルが無いことがある。bf16 の行列積で見る。
    # **最初の数回は測らない** — cuBLAS のハンドル生成と自動調整が入り、
    # 実力の数分の1の数字になって「このGPUは遅い」と誤読させる
    size = 8192
    a = torch.randn(size, size, device='cuda', dtype=torch.bfloat16)
    b = torch.randn(size, size, device='cuda', dtype=torch.bfloat16)
    for _ in range(3):
        c = a @ b
    torch.cuda.synchronize()
    started = time.time()
    for _ in range(10):
        c = a @ b
    torch.cuda.synchronize()
    flops = 10 * 2 * size ** 3 / (time.time() - started) / 1e12
    if not torch.isfinite(c).all():
        raise RuntimeError('bf16 の行列積が有限でない値を返した')
    log(f'  bf16 matmul {flops:.0f} TFLOP/s（カーネルの有無を見るためのもので、'
        'ベンチマークではない）')


@check('モデル')
def check_models():
    import importlib.util
    from pathlib import Path
    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location('exp25', here / 'run.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from cmoe.adapters.registry import create_adapter, guess_adapter

    for key, name in module.MODELS.items():
        started = time.time()
        adapter = create_adapter(guess_adapter(name), name)
        intermediate = adapter.layers[0].mlp.gate_proj.out_features
        if intermediate % module.NEXPERTS:
            raise RuntimeError(
                f'{name} の intermediate {intermediate} が '
                f'N={module.NEXPERTS} で割り切れない')
        log(f'  {key:<12} {adapter.n_layers}層 hidden={adapter.hidden_size} '
            f'intermediate={intermediate} '
            f'(/{module.NEXPERTS}={intermediate // module.NEXPERTS}) '
            f'{time.time() - started:.0f}秒')
        del adapter
        import torch
        torch.cuda.empty_cache()


@check('データセット')
def check_datasets():
    from cmoe.data.registry import load_calibration, load_evaluation
    from cmoe.eval.bench import DEFAULT_TASKS
    from cmoe.data.harness import DEFAULT_CACHE, load_tasks, subtasks

    model = 'meta-llama/Llama-2-7b-hf'
    token_set = load_calibration('slimpajama', model, 2048, 16, 0)
    log(f'  校正 slimpajama n=16: {token_set.metadata()["n_tokens"]} トークン')
    for name in ('wikitext2', 'c4-new'):
        meta = load_evaluation(name, model, 2048).metadata()
        log(f'  評価 {name}: {meta["n_tokens"]} トークン '
            f'hash={meta["token_hash"][:12]}')
    # **本番とまったく同じ引き方をする。** (1) ベンチは隔離したキャッシュから
    # 読む — 既定の HuggingFace キャッシュに新しい datasets が書いた索引が
    # 混じっていると、固定してある 2.21.0 が知らない特徴量の型で落ちる
    # （``datasets_cache`` の経緯を見ること）。(2) ``mmlu`` はグループ名なので
    # ``subtasks`` で素のタスクへ開いてから渡す。ここを本番と揃えないと、
    # preflight だけが落ちる／通ることになる
    log(f'  ベンチのキャッシュ {DEFAULT_CACHE}')
    names = [sub for name in list(DEFAULT_TASKS) + ['mmlu']
             for sub in subtasks(name)]
    tasks = load_tasks(names, cache_dir=DEFAULT_CACHE)
    log(f'  ベンチ {len(tasks)} タスク（5タスク + mmlu の57科目）')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-models', action='store_true',
                        help='モデルの読み込みを飛ばす（14GB × 2 の読み出し）')
    parser.add_argument('--skip-datasets', action='store_true')
    args = parser.parse_args(argv)

    skip = set()
    if args.skip_models:
        skip.add('モデル')
    if args.skip_datasets:
        skip.add('データセット')

    failures = []
    for name, function in CHECKS:
        if name in skip:
            log(f'== {name}: 飛ばす')
            continue
        log(f'== {name}')
        try:
            function()
        except Exception as error:  # noqa: BLE001 — 全部拾って一覧にする
            failures.append((name, error))
            log(f'  **失敗** {type(error).__name__}: {error}')
            traceback.print_exc()
    log()
    if failures:
        log(f'{len(failures)} 件が通らない: '
            + ', '.join(name for name, _ in failures))
        log('この状態で測定を始めない')
        return 1
    log('すべて通った。sweep.py を回してよい')
    return 0


if __name__ == '__main__':
    sys.exit(main())
