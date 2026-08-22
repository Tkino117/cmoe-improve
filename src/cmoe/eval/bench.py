"""[軸6] 選択問題ベンチマーク。lm-eval-harness を通し、生の尤度を持ち帰る。

**なぜ正答率だけでは足りないか。** 選択問題の採点は「選択肢ごとの対数尤度を出して
argmax を取る」であり、正答率はマージン（正解の尤度 − 最良の不正解の尤度）の
**符号**しか見ない。このプロジェクトで問題になる差は PPL で 0.03、1トークンあたり
0.004 nat 程度で、そのずれでマージンの符号をまたぐ問題は全体のごく一部しかない。
正答率の標本誤差（1万問で ±1ポイント弱）に埋もれる。

したがってここが持ち帰るのは集計値ではなく、**問題ごと・選択肢ごとの対数尤度**
そのものである。K択NLL・マージン・dense との KL といった指標はすべてこの配列から
後で計算できる（``cmoe.eval.bench_stats``）ので、どれを主役にするかを測る前に
決めなくてよい。PPL 側で塊ごとの平均 NLL を全件残しているのと同じ方針である。

既定のタスクは CMoE 最新版（ACL 2026, arXiv:2502.04416）Table 1 と同じ5つで、
ExpertWeaver（arXiv:2602.15521）Table 2 との共通部分でもある。どちらの論文も
主表の動作点はスパース率 25% で、このリポジトリの N=8 / A=6 がちょうどそれに当たる。

正解番号の導出は lm-eval の ``api/task.py`` の写しである。写経はずれる余地がある
ので、``_check_against_harness`` が全問について lm-eval 自身の正誤と突き合わせる。
"""

import contextlib
import random
from dataclasses import dataclass, field

import numpy
import torch

# CMoE 最新版 Table 1 と同じ並び。ExpertWeaver Table 2 との共通部分でもある
DEFAULT_TASKS = ('piqa', 'winogrande', 'arc_easy', 'arc_challenge', 'hellaswag')


@dataclass
class TaskSamples:
    """1タスク分の、問題ごと・選択肢ごとの生の対数尤度。

    ``loglikelihoods[d][c]`` は問題 d の選択肢 c を続けたときの対数尤度で、
    lm-eval が argmax を取る前の値そのもの。``choice_lengths`` は acc_norm の
    分母（選択肢の文字数）である。

    ``doc_hashes`` は「構成をまたいで同じ問題を並べている」ことの確認用で、
    対応のある比較の前提が崩れていないかをここで見る。
    """

    task: str
    gold: list
    loglikelihoods: list
    choice_lengths: list
    doc_hashes: list
    harness_metrics: dict = field(default_factory=dict)
    num_fewshot: int = 0
    limit: object = None
    # どのモデルを測ったか。doc_hash は問題側のハッシュでモデルに依らないので、
    # これが無いと「別モデルで測った dense」を基準に取り込んでも何も気づけない
    model: str = ''

    @property
    def n_docs(self):
        return len(self.gold)

    def as_dict(self):
        return {
            'task': self.task,
            'n_docs': self.n_docs,
            'gold': list(self.gold),
            'loglikelihoods': [list(row) for row in self.loglikelihoods],
            'choice_lengths': [list(row) for row in self.choice_lengths],
            'doc_hashes': list(self.doc_hashes),
            'harness_metrics': dict(self.harness_metrics),
            'num_fewshot': self.num_fewshot,
            'limit': self.limit,
            'model': self.model,
        }

    @classmethod
    def from_dict(cls, payload):
        return cls(
            task=payload['task'],
            gold=payload['gold'],
            loglikelihoods=payload['loglikelihoods'],
            choice_lengths=payload['choice_lengths'],
            doc_hashes=payload['doc_hashes'],
            harness_metrics=payload.get('harness_metrics', {}),
            num_fewshot=payload.get('num_fewshot', 0),
            limit=payload.get('limit'),
            model=payload.get('model', ''),
        )


def load_samples(path):
    """``result_logs/<name>/bench/*.json`` を読み戻す。

    指標を後から足す・レポートを書く経路はここを通る。GPU も lm-eval も要らず、
    保存された生の尤度だけで ``bench_stats`` の全指標が出せる。
    """
    import json

    with open(path) as handle:
        payload = json.load(handle)
    return {name: TaskSamples.from_dict(row) for name, row in payload.items()}


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


@contextlib.contextmanager
def _isolated_rng():
    """大域の乱数状態を、入る前の値に戻して出る。

    lm-eval の ``simple_evaluate`` は ``random`` / ``numpy`` / ``torch`` の
    大域シードを自分の既定値で踏む。同じ実行の中でこの後に走るものが大域の
    乱数を引いていた場合、ベンチマークを足しただけで数字が変わることになる。
    測る道具が測られる対象を動かさない、という一点のためだけの囲い。
    """
    states = (random.getstate(), numpy.random.get_state(), torch.get_rng_state())
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(states[0])
        numpy.random.set_state(states[1])
        torch.set_rng_state(states[2])
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


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


def _gold_index(task, doc, n_choices):
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


def _check_against_harness(name, example, gold, lls, lengths):
    """写した正解番号が lm-eval 自身の正誤と一致するかを1問ずつ見る。

    ここが黙って通ることが、``_gold_index`` を自前で持ってよい理由である。
    一致しないなら正解番号が違うので、その先の指標はすべて意味を失う。
    """
    for metric, values in (('acc', lls),
                           ('acc_norm', [ll / length for ll, length in zip(lls, lengths)])):
        if metric not in example:
            continue
        predicted = max(range(len(values)), key=values.__getitem__)
        ours = 1.0 if predicted == gold else 0.0
        if ours != float(example[metric]):
            raise ValueError(
                f'{name} doc_id={example["doc_id"]}: {metric} が lm-eval と'
                f'食い違う（こちら {ours} / lm-eval {example[metric]}）。'
                '正解番号の導出がずれている')


def _extract(name, task, examples):
    """lm-eval が残した1タスク分の生ログを ``TaskSamples`` の中身に均す。"""
    gold, loglikelihoods, choice_lengths, doc_hashes = [], [], [], []
    for example in sorted(examples, key=lambda row: row['doc_id']):
        # filtered_resps は選択肢ごとの (対数尤度, greedy か)。前半だけが
        # 選択肢に対応する — mutual information を使うタスクは後半に無条件の
        # 尤度を足すが、ここが扱う5タスクはどれもそれを使わない
        choices = task.doc_to_choice(example['doc'])
        responses = example['filtered_resps'][:len(choices)]
        lls = [float(response[0]) for response in responses]
        lengths = [float(len(choice)) for choice in choices]
        index = _gold_index(task, example['doc'], len(choices))
        _check_against_harness(name, example, index, lls, lengths)
        gold.append(index)
        loglikelihoods.append(lls)
        choice_lengths.append(lengths)
        doc_hashes.append(example['doc_hash'])
    return gold, loglikelihoods, choice_lengths, doc_hashes


def evaluate_bench(model, tokenizer, tasks=DEFAULT_TASKS, batch_size=8,
                   limit=None, num_fewshot=0, max_length=None, cache_dir=None,
                   model_name='', log=None):
    """変換済み（あるいは dense の）モデルを選択問題にかける。

    model はすでに置かれているデバイスのまま使う（lm-eval は動かさない）。
    返すのは集計値ではなく問題ごとの生の尤度で、指標は ``bench_stats`` が作る。
    """
    from lm_eval.evaluator import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    task_objects = load_tasks(tasks, cache_dir=cache_dir)
    was_training = model.training
    model.eval()
    # softmax_dtype を明示するのは必須である。lm-eval の既定（None）は
    # log_softmax も対数尤度の総和もモデルの dtype のまま計算する。bf16 は
    # 仮数8ビットしかなく、−32 付近の刻みが 0.125、−46 付近では 0.25 nat に
    # なる。ここで見たい差は問題あたり 0.06 nat 程度なので、**測りたいものが
    # 丸めより小さい**ことになり、連続量を残す意味そのものが消える。
    # fp32 なら同じ範囲の刻みは 1e-5 以下で、丸めは差の1万分の1に落ちる。
    # 正答率しか見ない使い方では効かないので lm-eval の既定は bf16 のままだが、
    # このプロジェクトの使い方では効く。
    wrapper = HFLM(pretrained=model, tokenizer=tokenizer,
                   batch_size=batch_size, max_length=max_length,
                   softmax_dtype=torch.float32)
    with _isolated_rng(), datasets_cache(cache_dir), torch.no_grad():
        results = simple_evaluate(
            model=wrapper,
            tasks=list(task_objects.values()),
            num_fewshot=num_fewshot,
            limit=limit,
            log_samples=True,
            # 信頼区間はこちらで問題ごとの対応を取って出す。lm-eval 側の
            # ブートストラップ（既定10万回）は使わないので回さない
            bootstrap_iters=0,
            verbosity='ERROR',
        )
    if was_training:
        model.train()

    samples = {}
    for name, task in task_objects.items():
        gold, lls, lengths, hashes = _extract(name, task, results['samples'][name])
        samples[name] = TaskSamples(
            task=name, gold=gold, loglikelihoods=lls, choice_lengths=lengths,
            doc_hashes=hashes,
            harness_metrics={key: value for key, value in results['results'][name].items()
                             if isinstance(value, (int, float))},
            num_fewshot=num_fewshot, limit=limit, model=model_name)
        if log is not None:
            log(f'  {name:<16} {samples[name].n_docs} 問')
    return samples
