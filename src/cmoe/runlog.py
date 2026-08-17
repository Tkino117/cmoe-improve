"""測定結果の置き場所。実験が失われないための3つの規則。

CMoE-ref の ``scripts/runlog.py`` の移送。GPU を数分〜数時間使って出した表は
実行そのものより価値があるので、

* 表示したものはすべてファイルにも書く（リダイレクトは誰も打ち忘れる）
* 1単位終わるたびに書き直す（最後の層で落ちても最初の31層は残す）
* 終わった結果がある場所への上書きは拒否する（気づくのが何日も後になる）

規則をここに置くのは、「上書きを拒否する」実装が2箇所あれば間違える機会も
2箇所になるからである。
"""

import json
import os
import time

RESULTS_DIR = 'result_logs'

# 現在の実行のミラーファイル。log() の引数にしないのは、引数は呼び出し側が
# 渡し忘れられるからで、ミラーの意味は「書き漏らしが起きないこと」にある。
_MIRROR = None


def log(message=''):
    """1行表示し、同じものをミラーにも書く。"""
    print(message, flush=True)
    if _MIRROR is not None:
        _MIRROR.write(message + '\n')


def open_mirror(path):
    """log() の出力先をこの実行のテキストファイルに向ける。

    追記であって切り詰めではない。途中で死んだ実行のログは、しばしばなぜ死んだ
    かの唯一の記録なので、'w' で開くと再実行が消したい相手ごと消してしまう。
    """
    global _MIRROR
    if _MIRROR is not None:
        _MIRROR.close()
    _MIRROR = open(path, 'a', buffering=1)
    return _MIRROR


def close_mirror():
    global _MIRROR
    if _MIRROR is not None:
        _MIRROR.close()
        _MIRROR = None


def default_out_dir(prefix, now=None, root=RESULTS_DIR):
    stamp = time.strftime('%Y%m%d_%H%M%S', time.localtime(now))
    return os.path.join(root, f'{prefix}_{stamp}')


def prepare_out_dir(path, result_name):
    """出力先を作る。終わった実行がある場所は拒否する。

    result_name: これがあれば「ここで実行が終わっている」と見なすファイル名。
    空の、あるいは途中までのディレクトリへ書くのは許す。
    """
    os.makedirs(path, exist_ok=True)
    existing = os.path.join(path, result_name)
    if os.path.exists(existing):
        raise SystemExit(
            f'{existing} がすでにある。終わった結果は上書きしない — '
            '--out で別のディレクトリを指定する')
    return path


def write_json(path, payload):
    """結果ファイルを、前の版を壊さずに置き換える。

    一時ファイルに書いてから rename する。実ファイルを切り詰めてから途中で
    死ぬと、すでに無事だった結果まで失う。数値は丸めずに入れる。
    """
    tmp = path + '.tmp'
    with open(tmp, 'w') as handle:
        json.dump(payload, handle, indent=1, ensure_ascii=False)
    os.replace(tmp, path)
    return path
