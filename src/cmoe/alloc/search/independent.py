"""層を独立に決める。各層を単独で変換し、ほかの層は dense のまま測る。

ビーム（``beam.py``）の対照である。幅1のビームとは**評価回数が同じ**（層ごとに
候補数ぶん）で、違うのは次の層へ渡す状態だけ — 幅1 は決めた x で変換した状態を
渡し、ここは dense のまま渡す。したがって層 ℓ の候補は「層 ℓ だけを変換した
モデル」の採点で並び、先行層の選択に依存しない。2つの差が「先行層の変換を
反映して決めること」の効果になる。

層ごとのスコアは単独変換のモデルの値であって、決まった配分全体の値ではない。
配分全体の値は頭から測り直すまで分からないので、``scores_whole_model`` を
False にして呼び出し側に知らせる。
"""

from cmoe.alloc.base import Allocation
from cmoe.alloc.presets import N_ACTIVE


class IndependentSearch:
    name = 'independent'
    # 記録の score は単独変換のモデルの値。配分全体の値は測り直しで作る
    scores_whole_model = False

    def __init__(self, n_active_total=N_ACTIVE, budget=None, log=None,
                 on_layer=None):
        if budget is not None:
            # 評価回数は層数 × 候補数で決まっていて、途中で止める意味が無い
            raise ValueError('independent は予算を取らない')
        self.n_active_total = n_active_total
        self.log = log or (lambda message='': None)
        self.on_layer = on_layer
        self.records = []

    def allocation_name(self):
        return self.name

    def search(self, oracle, n_layers):
        if oracle is None:
            raise ValueError('この探索はオラクルを要求する')
        if not n_layers >= 1:
            raise ValueError(f'{n_layers} 層は探索できない')
        n_active_total = getattr(oracle, 'n_active_total', self.n_active_total)

        self.log(f'{"層":>5} '
                 + ' '.join(f'{f"x={x}":>10}' for x in oracle.candidates(0))
                 + '  最善（各層を単独で変換したモデルの値）')
        self.log('-' * 60)
        self.records = []
        chosen = []
        state = oracle.root()
        try:
            for layer in range(n_layers):
                row = []
                for x in oracle.candidates(layer):
                    result, child = oracle.extend(state, layer, x)
                    oracle.release(child)
                    row.append({'x': x, 'score': result.score,
                                'details': result.details})
                # 同点は小さい x を取る（ビームの同点崩しと同じ向き）
                ranked = sorted(row, key=lambda entry: (entry['score'], entry['x']))
                best = ranked[0]
                chosen.append(best['x'])
                margin = (ranked[1]['score'] - best['score']
                          if len(ranked) > 1 else None)
                following = oracle.pass_dense(state, layer)
                oracle.release(state)
                state = following
                record = {
                    'layer': layer,
                    'rows': [{'parent': 0, 'children': row}],
                    'beam': [{'rank': 0, 'parent': 0, 'x': best['x'],
                              'score': best['score'],
                              'allocation': list(chosen)}],
                    'margin': margin,
                    'spent': oracle.spent,
                    'cost_unit': oracle.cost_unit,
                }
                self.records.append(record)
                self.log(f'{layer:>5} '
                         + ' '.join(f'{entry["score"]:>10.3e}' for entry in row)
                         + f'  x={best["x"]}')
                if self.on_layer is not None:
                    self.on_layer(self.records)
        finally:
            oracle.release(state)

        return Allocation(tuple(chosen), name=self.allocation_name(),
                          n_active_total=n_active_total).check_layers(n_layers)
