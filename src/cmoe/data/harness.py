"""[軸2] lm-eval のタスクを読む1本。

タスク実体（``doc_to_text`` / ``doc_to_choice`` / ``doc_to_target``）が要る場所は
2つある — ベンチマークで正解番号を出すところ（``cmoe.eval.bench``）と、問題文
そのものを校正データにするところ（``cmoe.data.benchtrain``）である。2実装ある
と、校正が読んだ問題と評価が測った問題が黙って別物になりうるので、ここに1本
だけ置く。

依存の向きは ``data ← ... ← eval`` なので、``eval`` 側がここを読む。逆は無い。
"""

import contextlib
import os

# ベンチのデータセットは既定の共有キャッシュを使わない。固定した
# ``datasets==2.21.0`` では読めない索引が共有キャッシュに混じっているためで、
# 経緯は ``datasets_cache`` にある。校正側も同じ場所を見る（同じ生データから
# train split と eval split を引くので、キャッシュが分かれる理由が無い）
DEFAULT_CACHE = os.environ.get(
    'CMOE_BENCH_CACHE', os.path.join('.cache', 'hf-datasets'))


@contextlib.contextmanager
def datasets_cache(path):
    """タスクのデータセットを、指定したキャッシュから読む。

    このプロジェクトは ``datasets==2.21.0`` に固定してある（層の呼び出し規約の
    都合で transformers 4.47.1 に固定され、その組み合わせで検証されているのが
    そこまでのため）。一方 HuggingFace の既定のキャッシュは計算機で共有されて
    おり、そこに新しい ``datasets`` が書いた索引が混じっていると、2.21.0 は
    知らない特徴量の型（``List`` など）で読めずに落ちる。実際 ARC と HellaSwag
    がそうなっていた。

    ベンチマークのぶんだけ別のキャッシュを見ることで、PPL 側が使っている
    WikiText-2 / C4 の経路（既存の測定と6桁一致することが確かめてある）に
    手を触れずに済む。``path`` が None なら既定のキャッシュのまま。
    """
    if path is None:
        yield
        return
    import datasets

    before = datasets.config.HF_DATASETS_CACHE
    datasets.config.HF_DATASETS_CACHE = path
    try:
        yield
    finally:
        datasets.config.HF_DATASETS_CACHE = before


def load_tasks(names, cache_dir=None):
    """タスク名から lm-eval のタスク実体を作る。

    実体を自分で持つのは、正解番号を出すのに ``doc_to_target`` /
    ``doc_to_choice`` が要るからである。``simple_evaluate`` は実体をそのまま
    受け取れるので、データセットの読み込みは1回で済む。
    """
    from lm_eval.tasks import TaskManager

    names = list(names)
    with datasets_cache(cache_dir):
        loaded = TaskManager().load(names)['tasks']
    missing = [name for name in names if name not in loaded]
    if missing:
        raise ValueError(f'lm-eval に無いタスク: {", ".join(missing)}')
    for name in names:
        output_type = loaded[name].get_config('output_type')
        if output_type != 'multiple_choice':
            raise ValueError(
                f'{name} は output_type={output_type} で、選択肢ごとの尤度が'
                '出ない。ここが扱うのは multiple_choice だけ')
    return {name: loaded[name] for name in names}


def gold_index(task, doc, n_choices):
    """lm-eval が正誤の判定に使う正解番号。

    ``api/task.py`` の multiple_choice 版 ``process_results`` の導出を写した
    もの。``multiple_input`` のタスク（WinoGrande）では選択肢が「続き」ではなく
    「文脈」の側に立つので、正解番号は ``doc_to_text`` から来る。
    """
    if getattr(task, 'multiple_input', 0):
        gold = task.doc_to_text(doc)
    else:
        gold = task.doc_to_target(doc)
    if isinstance(gold, list):
        raise ValueError(f'{task.config.task}: 正解が複数ある問題は扱わない')
    if isinstance(gold, str):
        choices = task.doc_to_choice(doc)
        gold = choices.index(gold) if gold in choices else -100
    gold = int(gold)
    if not 0 <= gold < n_choices:
        raise ValueError(
            f'{task.config.task}: 正解番号 {gold} が選択肢 {n_choices} 個に収まらない')
    return gold


def render_document(task, doc):
    """1問を「文脈 + 正解の続き」の1本のテキストにする。

    不正解の選択肢は入れない。lm-eval は選択肢ごとに別々の系列を採点するので、
    K本すべてを連結したものはモデルが実際に読む形のどれでもない。正解肢だけを
    残せば、少なくとも「問題文のあとに正しい続きが来る」という、K本のうち1本と
    同じ形になる。

    繋ぎ目は lm-eval と同じ ``target_delimiter``（既定は空白1つ）にする。ここが
    ずれると、校正データだけが評価と違う繋ぎ方をすることになる。

    WinoGrande（``multiple_input``）は選択肢が文脈の側に立つので、正解の文脈に
    共通の続きを足す形になる。
    """
    choices = task.doc_to_choice(doc)
    index = gold_index(task, doc, len(choices))
    delimiter = task.config.target_delimiter
    if getattr(task, 'multiple_input', 0):
        return choices[index] + delimiter + task.doc_to_target(doc)
    return task.doc_to_text(doc) + delimiter + choices[index]
