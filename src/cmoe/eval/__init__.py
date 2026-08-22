"""[軸6] 評価の境界。

PPL（``ppl``）と選択問題ベンチマーク（``bench``）の2本立て。どちらも
「集計値ではなく、再抽出できる単位の生の値を全件残す」という同じ方針で、
信頼区間は後から GPU なしで付け直せる。

``bench`` は lm-eval-harness を import するので、ここでは持ち上げない
（PPL だけの実行に選択問題の依存を引かせない）。
"""

from cmoe.eval.ppl import PPLResult, evaluate_ppl
from cmoe.eval.stats import paired_differences, stratified_paired_bootstrap

__all__ = ['PPLResult', 'evaluate_ppl', 'paired_differences',
           'stratified_paired_bootstrap']
