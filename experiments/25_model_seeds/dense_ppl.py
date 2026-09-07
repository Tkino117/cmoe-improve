"""変換前（dense）の perplexity を測る。

``cmoe run`` が測る PPL は変換後の構成だけで、dense は測らない（dense の基準を
持つのはベンチ側だけである）。そのため res16 の PPL の表には dense の行が無く、
**PPL は配分どうしの比較にしか使えない**という注意書きが付いていた。ここが
その穴を埋める段で、変換を一切かけずに評価セットを走らせるだけである。

dense は配分にも校正にも seed にも依らないので、**モデルごとに1回**でよい。

  uv run python experiments/25_model_seeds/dense_ppl.py --model meta-llama/Llama-2-7b-hf
  uv run python experiments/25_model_seeds/dense_ppl.py --model mistralai/Mistral-7B-v0.1

出力は ``result_logs/dense_ppl_<札>/dense_ppl.json``。塊ごとの平均 NLL まで
残すので、変換後の実行と対応のあるブートストラップが後から組める（``cmoe run``
が残しているものと同じ形である）。
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from cmoe.adapters.registry import create_adapter, guess_adapter
from cmoe.data.registry import load_evaluation
from cmoe.eval.ppl import evaluate_ppl

ROOT = Path(__file__).resolve().parents[2]

DATASETS = ('wikitext2', 'c4-new')
SEQLEN = 2048


def log(message=''):
    print(message, flush=True)


def slug(model):
    """出力先に使う札。``org/Name`` の後ろだけを小文字にしたもの。"""
    return model.rsplit('/', 1)[-1].lower()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='meta-llama/Llama-2-7b-hf')
    parser.add_argument('--adapter', default=None,
                        help='省略時はモデル名から推測する')
    parser.add_argument('--datasets', default=','.join(DATASETS))
    parser.add_argument('--seqlen', type=int, default=SEQLEN)
    parser.add_argument('--out', default=None)
    args = parser.parse_args(argv)

    datasets = [name.strip() for name in args.datasets.split(',') if name.strip()]
    out_dir = Path(args.out) if args.out else (
        ROOT / 'result_logs' / f'dense_ppl_{slug(args.model)}')
    payload_path = out_dir / 'dense_ppl.json'
    if payload_path.exists():
        record = json.loads(payload_path.read_text())
        if sorted(record.get('results', {})) == sorted(datasets):
            log(f'{payload_path} は測り済み')
            for name, row in record['results'].items():
                log(f'  {name:<10} ppl={row["ppl"]:.6f} '
                    f'mean_nll={row["mean_nll"]:.6f}')
            return 0

    started = time.time()
    log(f'model={args.model} seqlen={args.seqlen} / 変換はしない')
    adapter = create_adapter(args.adapter or guess_adapter(args.model),
                             args.model, seqlen=args.seqlen)

    results = {}
    metadata = {}
    for name in datasets:
        token_set = load_evaluation(name, args.model, args.seqlen)
        meta = token_set.metadata()
        metadata[name] = meta
        log(f'評価 {name}: {meta["n_tokens"]} トークン '
            f'hash={meta["token_hash"][:12]}')
        result = evaluate_ppl(adapter, token_set)
        results[name] = result.as_dict()
        log(f'  {name:<10} ppl={result.ppl:.6f} mean_nll={result.mean_nll:.6f} '
            f'塊 {result.n_chunks}')

    out_dir.mkdir(parents=True, exist_ok=True)
    with payload_path.open('w') as handle:
        json.dump({
            'arguments': vars(args),
            'datasets': metadata,
            'results': results,
            'seconds': time.time() - started,
            'commit': subprocess.run(
                ['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True,
                text=True).stdout.strip(),
        }, handle, indent=1, ensure_ascii=False)
    log(f'\n{payload_path} に書いた（{(time.time() - started) / 60:.1f}分）')
    return 0


if __name__ == '__main__':
    sys.exit(main())
