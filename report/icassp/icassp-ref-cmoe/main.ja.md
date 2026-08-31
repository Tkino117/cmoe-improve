# 活性化パターン分析による解析的なFFNからMoEへの再構成

Zehua Pei$^1$, Hui-Ling Zhen$^2$, Lancheng Zou$^1$, Xianzhi Yu$^2$, Wulong Liu$^2$,
Sinno Jialin Pan$^1$, Mingxuan Yuan$^2$, Bei Yu$^1$

$^1$香港中文大学 $^2$Huawei Technologies Co., Ltd

## 概要

大規模言語モデル(LLM)はスケールさせるほど性能が向上するが、その分推論コストも大きく増加し、中でもフィードフォワードネットワーク(FFN)が計算資源の大部分を消費する。Mixture-of-Experts(MoE)アーキテクチャはスパースな活性化によってこのコストを削減できるが、既存の密なモデルをMoEへ再構成するには通常、数千億トークン規模の大規模な再学習が必要となる。本研究では、小規模なキャリブレーションデータセットのみを用いて、FFNをスパースなMoEアーキテクチャへ迅速に再構成する解析的な事後学習(post-training)フレームワークを提案する。この手法はニューロンの活性化パターンを分析し、常に活性化する共有エキスパートと、条件付きで活性化するルーティングされたエキスパートへとニューロンを分割したうえで、代表的なニューロン統計量からルーターを解析的に構築する。これにより、即座のデプロイ、あるいは任意で軽量なファインチューニングを行うことが可能になる。このアプローチは密なモデルだけでなく、既存のMoEモデルに再帰的に適用することで階層的なスパース性を実現することもできる。実験の結果、わずか数分の処理と2,000サンプルのファインチューニングだけで、計算律速なシナリオにおいて最大1.17倍の高速化を達成し、桁違いに多くの資源を必要とする既存手法を上回る性能を示した。[^1]

[^1]: コード: https://github.com/JarvisPei/CMoE

## 1 はじめに

大規模言語モデル(LLM)は幅広いタスクにおいて高い性能を達成してきた(Zhang et al., 2022; Touvron et al., 2023; Liu et al., 2024b,a)。しかし、モデルサイズが増大し続けることで、特にリソースが限られたハードウェア上や厳しいレイテンシ制約の下では、高い計算需要による導入上の課題が生じている。これが様々な推論高速化技術の開発を促してきた。その中でも、Mixture-of-Experts(MoE)アーキテクチャ(Lepikhin et al., 2020; Du et al., 2022; Fedus et al., 2022; Dai et al., 2024)は、ルーターを用いて各入力トークンに対して動的にパラメータのスパースな部分集合を選択することで、モデルの容量と計算コストを切り離す。しかし、従来MoEモデルの恩恵を得るには、ゼロからの高コストな事前学習が必要であり、モデル性能と学習コストの間に厳しいトレードオフが存在していた。

Transformerアーキテクチャにおける計算のボトルネックは、フィードフォワードネットワーク(FFN)ブロックに不釣り合いに集中している。複数の研究が、FFNニューロンにおける高い活性化スパース性を報告しており(Liu et al., 2023; Zhang et al., 2021; Pei et al., 2024)、これはどのような入力に対してもごく一部のニューロンしか活性化しないことを意味する。この自然に生じるスパース性は、事前学習のコストをかけずに推論を高速化できる機会を提供する。既存研究では密なモデルをMoEへ再構成する試みがなされてきたが、既存手法はニューロンの活性化挙動を考慮せず全てのニューロンを一様に扱う重み(weight)ベースのクラスタリング(Zhang et al., 2021; Qiu et al., 2023)に頼るか、あるいは品質を回復するために最大2,000億トークンにも及ぶ大規模な継続学習(Zhu et al., 2024; Qu et al., 2024)を必要とするかのいずれかである。これらの手法が見落としている重要な観察は、ニューロンの活性化頻度が2つの異なるグループに分かれるという点である。すなわち、大多数のニューロンは特定の入力に対してのみ活性化する一方、一部のニューロンはどの入力に対しても一貫して活性化する。全てのニューロンを一様に扱うことで、既存手法は一貫して活性化するニューロンを複数のエキスパートに分散させてしまい、効果的なルーティングを学習するために大規模な学習が必要になっていた。これが、この明確な構造を明示的に活用する解析的アプローチを動機づけている。すなわち、高頻度で活性化するニューロンを共有エキスパートにまとめ、低頻度で活性化するニューロンをルーティングされるエキスパートにまとめ、ルーティング自体も活性化統計量から直接導出するというアプローチである。

これらの限界を克服するため、我々はLLM高速化における性能とコストのトレードオフを改善する解析的な事後学習フレームワークを提案する。このフレームワークは、ごく小さなキャリブレーションデータセットのみを用いた高速な解析的プロセスによってFFNを再構成する。具体的には、ニューロンの活性化パターンを分析し、頻繁に活性化するニューロン(「共有」エキスパートにグループ化される)とまばらにしか活性化しないニューロンとを区別することで動作する。まばらに活性化するニューロンは、その後バランスの取れた割り当てアルゴリズム(Jonker and Volgenant, 1988)を用いて専門化された「ルーティング」エキスパートへとクラスタリングされる。この再構成は広く適用可能であり、密なモデルの単一の巨大なFFNをスパースなMoEアーキテクチャへ変換することも、既存のMoEモデルの個々のエキスパートに再帰的に適用してよりきめ細かい階層的なスパース性を誘導することもできる。このフレームワークは活性化統計量からルーターを解析的に構築し、高コストなルーター学習の必要性を回避することで、学習不要のベースラインによる迅速な導入、あるいは任意で軽量なファインチューニングを可能にする。

我々の貢献は以下の通りである。

- FFNニューロンの活性化頻度において2つの異なるグループを明らかにする。一部のニューロンはどの入力に対しても一貫して活性化する一方、大多数のニューロンは条件付きで活性化する。我々は、既存のFFN-to-MoE手法がこの構造を見落としていることを示す。
- この観察を活用し、ニューロンを共有エキスパートとルーティングエキスパートへと分割する解析的フレームワークを提案する。ルーターは、代表的なニューロン統計量から学習なしで直接構築される。
- 提案手法は、わずか2,000サンプルのファインチューニングのみで既存のMoE再構成手法を精度において上回り、最大1.17倍の推論高速化を達成する。この手法は密なモデルと既存のMoEアーキテクチャの両方に適用でき、階層的なスパース性を実現する。

## 2 関連研究

ゼロからMoEモデルを事前学習するのとは対照的に、近年の研究は既存の密なLLMを再利用してMoEアーキテクチャを構築することを検討してきた。現行の手法は概ね2つのパラダイムに従う。(1) 総パラメータ数を維持したままFFNのパラメータを分割する方法(Zuo et al., 2022; Zhang et al., 2021; Yang et al., 2024)、あるいは(2) 活性化次元を保持しながら容量を拡張する方法(Komatsuzaki et al., 2022; Wu et al., 2024)である。本研究は前者を優先する。MoE-BERT(Zuo et al., 2022)は重要度駆動の戦略を用いてスコアの高いニューロンをエキスパート間に再配分するが、重要なニューロンを共有する点では類似しているものの、構造的に分離された共有エキスパートを形成するのではなく、全てのエキスパート内部にそれらを複製しており、またタスクに依存しない活性化プロファイリングではなく、タスク固有の勾配ベースの重要度スコアに依存している。MoEfication(Zhang et al., 2021)はReLUベースのFFNにおけるスパースな活性化パターンを活用し、学習されたルーターとともに層をエキスパートグループへ分解する。G-MoEfication(Lee et al., 2024)は、選択されなかったエキスパートの代表値を保持することで、MoEficationを非ReLUモデルへ一般化する。LLaMA-MoE(Zhu et al., 2024)とその後継手法(Qu et al., 2024)はFFNをエキスパートへ分割するが、それぞれ2,000億トークンおよび70億トークンという大規模な継続学習を必要とする。EMoE(Qiu et al., 2023)はファインチューニング中にキーベクトルによってニューロンをクラスタリングし、Read-ME(Cai et al., 2024)はシステム協調設計を伴うドメイン認識型のエキスパート構築に焦点を当てる。これに対し、我々の手法は活性化パターンに基づいてニューロンを共有エキスパートとルーティングエキスパートへ解析的に分割してFFNを再構成し、その後代表的なニューロン統計量からルーターを構築するもので、わずか2,000サンプルのファインチューニングしか必要としない。

これと並行する研究の流れとして、完全に微分可能なルーティングを研究するものがある。ReMoE(Wang et al., 2024)はハードなTop-Kルーティングの代わりにReLUベースのルーティングを用い、Lory(Zhong et al., 2024)は大規模なトークン予算のもとで微分可能なエキスパート統合を行う。これらの手法は事前学習中にルーターを学習するのに対し、我々の軽量な学習で済む解析的な再構成手法はそれらと相補的な関係にある。

FFNからMoEへの変換とは直交する形で、他の効率化技術としては、モデルの構成要素を静的に削除する構造的プルーニング手法、例えばSliceGPT(Ashkboos et al., 2024)やSLEB(Song et al., 2024)が挙げられる。活性化スパース性を利用する手法はFFNの隠れ状態に内在する自然なスパース性を活用するもので、DejaVu(Liu et al., 2023)は文脈依存のスパース性を活用し、TEAL(Liu et al., 2024c)とWINA(Chen et al., 2025)は大きさ(magnitude)または重みに基づく閾値を用いる。Learn-To-be-Efficient(Zheng et al., 2024)は、より少ないニューロンを活性化するようにモデルを学習する。これらのアプローチは異なる粒度で動作しており、我々のMoE再構成と相補的に組み合わせられる可能性がある。

## 3 動機:活性化パターン

手法を提示する前に、FFN層における活性化パターンを分析する。これらの観察結果が、我々の解析的な再構成アプローチの設計の基礎となっている。

### 3.1 高い活性化スパース性

入力 $\mathbf{x} \in \mathbb{R}^d$ を受け取り、出力 $F(\mathbf{x}) \in \mathbb{R}^d$ を生成するFFN層を考える。この計算には隠れ状態 $\mathbf{h} \in \mathbb{R}^{d_h}$ が関わり、$d_h$ は隠れ次元(通常 $d_h \gg d$)である。各ニューロンの出力への寄与は次のように書ける。

$$F(\mathbf{x}) = \sum_{i=1}^{d_h} h_i \mathbf{w}_i, \tag{1}$$

ここで $h_i$ は $i$ 番目の隠れ活性化であり、$\mathbf{w}_i \in \mathbb{R}^d$ は出力射影行列の対応する列である。

図1は、代表的な層における隠れ活性化の分布を示している。この分布は0付近に鋭いピークを持っており、これはどのような入力に対しても大多数のニューロンが出力へほとんど寄与していないことを示している。このスパース性は、ニューロンを選択的に活性化させることで、出力の品質を保ちながら計算量を削減できる可能性を示唆している。しかし、これを利用するためには、*どの*入力に対して*どの*ニューロンが重要なのかを理解する必要がある。

図1: FFN隠れ状態 $\mathbf{h}$ の分布(Llama-2-7B)。活性化の大部分は0付近に集中している。

図2: ニューロン活性化率 $\boldsymbol{\mu}$ の分布(Llama-2-7B、可視化のため $K_a = 1000$ を使用。より小さな $K_a$ でも同様のパターンが見られる)。大多数のニューロンは活性化率が低い一方、一部のニューロンは一貫して活性化している。

### 3.2 多様な活性化パターン

入力全体にわたるニューロンの振る舞いを特徴づけるため、キャリブレーションセット上で活性化率を計算する。各ニューロン $i$ について、その活性化率 $\mu_i$ を、そのニューロンが大きさの上位$K_a$番目までの活性化に入るトークンの割合として定義する。ここで $K_a$ は、上位何個の活性化を考慮するかを指定するハイパーパラメータである。

図2は2つの明確なグループを明らかにしている。大多数のニューロンは低い活性化率を示し(0.07付近にピークがある)、これは特定の入力に対してのみ活性化することを意味する。しかし、一部のニューロンは1に近い一貫して高い活性化率を示しており、これはほぼ全ての入力に寄与していることを示している。この違いは、自然な分割の存在を示唆している。すなわち、常に重要なニューロンと、条件付きで重要なニューロンという分割である。

この観察はMoE設計に対して示唆を持つ。既存手法(Zhang et al., 2021; Qiu et al., 2023)は、エキスパートを構築する際に全てのニューロンを一様に扱い、この二峰性(bimodal)のパターンを無視している。これらの手法は一貫して活性化するニューロンと条件付きで活性化するニューロンとを区別しないため、高頻度に活性化するニューロンが異なるルーティングエキスパートへ分散してしまう可能性がある。これにより、ルーターは入力にかかわらずほとんどのエキスパートを活性化せざるを得なくなり、MoEが効率のために依拠しているスパース性が損なわれてしまう。対照的に、高頻度のニューロンを常に活性化する共有エキスパートへ明示的に分離し、低頻度のニューロンを条件付きで活性化するルーティングエキスパートへグループ化することで、ルーターが本当に入力依存的なエキスパートの中からのみ選択すればよい、よりよく構造化されたアーキテクチャが得られる。

### 3.3 問題の定式化

これらの観察に基づき、再構成の目的を定式化する。元のFFN出力 $F(\mathbf{x})$ が与えられたとき、我々は再構成誤差を最小化するMoEアーキテクチャを求める。

$$\min \mathbb{E}_{\mathbf{x}} \left[ \| F_{MoE}(\mathbf{x}) - F(\mathbf{x}) \|^2 \right]. \tag{2}$$

MoEの出力は $F_{MoE}(\mathbf{x}) = E^s(\mathbf{x}) + \sum_{i=1}^{N_r} g_i \cdot E_i^r(\mathbf{x})$ と定義される。ここで $E^s$ は常に活性化する共有エキスパートであり、$\{E_i^r\}_{i=1}^{N_r}$ は $N_r$ 個のルーティングエキスパート、$g_i \in \{0, 1\}$ はルーターによって生成されるゲート値である。

エキスパートはパラメータを追加することなく、元のニューロンを分割することによって構築されるため、この目的は「どのルーティングエキスパートを非活性化するか」を決定する問題に帰着する。一部のルーティングエキスパートが非活性化される場合(すなわち $g_i = 0$ を受け取る場合)、再構成誤差はそれらの出力の総和に等しくなる。この誤差を最小化するためには、非活性化されたエキスパートの期待される寄与が最小になるように、エキスパートの分割とルーターを構築する必要がある。

上記の観察は、共有エキスパートは高頻度のニューロンを含むべきであり、ルーティングエキスパートは共活性化する(co-activated)ニューロンをグループ化すべきであり、ルーターは活性化パターンから導出できることを示唆している。次節では、我々の解析的な解法を提示する。

## 4 手法

3.3節で定式化した問題を土台として、ここで我々の解析的な解法を提示する。このフレームワークは3つの段階で動作する。(A) ニューロンパターンのプロファイリング、(B) ニューロンを共有エキスパートとルーティングエキスパートへ分割する、(C) 解析的なルーターを構築する、の3段階である。図3は全体のパイプラインを示している。

図3: 提案する解析的FFN-to-MoE再構成フレームワークの概要。

### 4.1 エキスパートの構築

隠れ次元 $d_h$ を持つFFNに対し、我々はサイズ $m = d_h/N$ の $N$ 個のエキスパートを構築する。これは $N_s$ 個の共有エキスパートと $N_r$ 個のルーティングエキスパート($N_s + N_r = N$)からなる。SwiGLU活性化を用いるLLaMA系のモデルでは、FFNは次のように計算される。

$$\mathbf{h} = \text{Swish}(\mathbf{W}_{\text{gate}}^\top \mathbf{x}) \odot (\mathbf{W}_{\text{up}}^\top \mathbf{x}),$$
$$F(\mathbf{x}) = \mathbf{W}_{\text{down}}^\top \mathbf{h}, \tag{3}$$

ここで $\mathbf{x} \in \mathbb{R}^d$、$\mathbf{W}_{\text{up}}, \mathbf{W}_{\text{gate}} \in \mathbb{R}^{d \times d_h}$、$\mathbf{W}_{\text{down}} \in \mathbb{R}^{d_h \times d}$ である。

**活性化プロファイリング。** 小規模なキャリブレーションデータセットを用いて、$q$ 個のトークンに対する隠れ状態 $\mathbf{H} \in \mathbb{R}^{q \times d_h}$ を計算する。各トークンについて、活性化の大きさによる上位$K_a$個のニューロン、すなわち $|h_i|$ に対する絶対値による上位$K$選択(ATopK)を特定し、二値の活性化行列 $\mathbf{A} \in \{0,1\}^{q \times d_h} = [\mathbf{c}_1\ \mathbf{c}_2\ \cdots\ \mathbf{c}_{d_h}]$ を得る。各列 $\mathbf{c}_i$ はキャリブレーションセット全体にわたるニューロン $i$ の活性化パターンを表し、活性化率 $\mu_i = \text{mean}(\mathbf{c}_i)$ はニューロン $i$ がどれくらいの頻度で活性化しているかを表す(パイプライン全体についてはAppendixのA.2節を参照)。

**共有エキスパート。** 3.2節で観察した多様な活性化パターンに基づき、活性化率が最も高い $N_s \cdot m$ 個のニューロンを選んで共有エキスパート $E^s$ を構成する。これらのニューロンは入力全体を通して一貫して活性化しており、共通の知識を捉えている。共有エキスパートの重みは、選択されたニューロンのインデックスに従って元のFFN行列をスライスすることで構築される。

**ルーティングエキスパート。** 残りのニューロンは $N_r$ 個のルーティングエキスパートへ分割される。似た機能を持つニューロンは共活性化しやすい傾向があるため、我々はそれらの活性化特徴ベクトル $\mathbf{c}_i$ の類似度に基づいてクラスタリングを行う。クラスタ内の距離を最小化しながらニューロンを $N_r$ 個の等サイズのクラスタへグループ化する、バランスの取れた割り当てアルゴリズムを用いる。詳細はAppendixのA.3節に示す。

得られるMoEアーキテクチャは次のように計算される。

$$F_{MoE}(\mathbf{x}) = E^s(\mathbf{x}) + \sum_{i=1}^{N_r} g_i \cdot E_i^r(\mathbf{x}), \tag{4}$$

ここで $g_i \in \{0, 1\}$ はゲート値であり、入力ごとに選択されるルーティングエキスパートの数を表す $N_k$ を用いて、ルータースコアに基づく上位$N_k$個のルーティングエキスパートのみが活性化される。

### 4.2 解析的なルーター構築

3.3節の問題定式化から、再構成誤差を最小化するには、非活性化されたエキスパートの出力への寄与を最小にする必要がある。非活性化されたルーティングエキスパートの集合を $S_{de}$ とすると、再構成誤差は次のようになる。

$$F_{MoE}(\mathbf{x}) - F(\mathbf{x}) = -\sum_{i \in S_{de}} E_i^r(\mathbf{x}). \tag{5}$$

各エキスパートの出力は、その中のニューロンの寄与の総和として分解できる。3.1節で観察したスパース性のもとでは、隠れ活性化が小さいニューロンは無視できるほどしか寄与しない(形式的な仮定についてはAppendixのA.1節を参照)。これは、あるエキスパートの隠れ状態の $L_1$ ノルム $\|\mathbf{h}_i^r\|_1$ が、その出力の大きさの代理指標(proxy)として機能することを示唆している。したがって、再構成誤差の最小化は近似的に次の問題に帰着する(完全な導出はAppendixのA.4節に示す)。

$$\min_G \mathbb{E}_{\mathbf{x}} \left[ \sum_{i \in S_{de}} \|\mathbf{h}_i^r\|_1 \right], \tag{6}$$

ここで $G$ は、上位$N_k$選択によって $S_{de}$ を決定するルーターである。

この目的関数は、ルータースコア $\mathbf{s} = [s_1, \ldots, s_{N_r}]$ が各エキスパートの期待される隠れ状態の大きさに基づいてエキスパートを順位付けし、寄与の大きいエキスパートは活性化され、寄与の小さいエキスパートは非活性化されるようにすることで最小化される。

**代表ニューロンの選択。** このようなルーターを解析的に構築するため、各エキスパート $j$ について、活性化パターンがクラスタ重心 $\hat{\mathbf{c}}_j$ に最も近いニューロンを**代表ニューロン** $R_j$ として特定する。

$$R_j = \operatorname*{argmin}_{i \in \text{cluster } j} \|\mathbf{c}_i - \hat{\mathbf{c}}_j\|_2, \tag{7}$$

ここで $\hat{\mathbf{c}}_j$ は、バランスの取れたクラスタリングの段階で得られたクラスタ $j$ の重心である。$R_j$ はそのエキスパートの典型的な活性化挙動を最もよく代表しているため、その隠れ活性化 $h_{R_j}$ はそのエキスパート全体の寄与を近似する。

ルーターは、代表ニューロンのパラメータのみを用いて構築される。

$$G(\mathbf{x}) = \text{Swish}(\mathbf{W}_{\text{gate}}^{R\top}\mathbf{x}) \odot (\mathbf{W}_{\text{up}}^{R\top}\mathbf{x}), \tag{8}$$

ここで $\mathbf{W}_{\text{gate}}^R, \mathbf{W}_{\text{up}}^R \in \mathbb{R}^{d \times N_r}$ は、代表ニューロンに対応する列のみを含む。これにより、各エキスパートの期待されるルーティング上の寄与を近似するルータースコア $\mathbf{s} = [s_1, \ldots, s_{N_r}] = G(\mathbf{x})$ が得られる。

### 4.3 ファインチューニングによる改善

解析的なルーターは、学習不要のベースラインを提供する。さらなる改善のため、任意の軽量なファインチューニング向けに2つの改良を導入する。

**学習可能なスケーリング。** 初期のゲート値は二値($g_i \in \{0,1\}$)である。勾配ベースの最適化を可能にするため、ゼロで初期化された学習可能なスケーリングパラメータ $\mathbf{u} = [u_1, \ldots, u_{N_r}]$ を導入する。選択されたエキスパートについて、ゲートは $g_i = 1 + s_i' \cdot u_i$ となる。ここで $\mathbf{s}' = \text{Softmax}(\mathbf{s})$ である。

**負荷分散。** 補助的な損失関数を用いずにエキスパートの利用を均衡させるため、上位$N_k$選択の前にスコアへ加算する適応的なバイアス項 $\mathbf{b} = [b_1, \ldots, b_{N_r}]$ を導入する(Liu et al., 2024a)。最終的なゲーティングのロジックは次の通りである。

$$g_i = \begin{cases} 1 + s_i' \cdot u_i, & \text{if } s_i' + b_i \in \text{Top-}N_k, \\ 0, & \text{otherwise.} \end{cases} \tag{9}$$

バイアスは各ステップ後に更新される。もしエキスパート $i$ が過負荷であれば($p_i > p^*$)、$b_i$ を $\gamma$ だけ減少させ、過小負荷であれば($p_i < p^*$)、$b_i$ を $\gamma$ だけ増加させる。ここで $p_i$ はエキスパート $i$ の利用率、$p^* = 1/N_r$ は均等な目標値、$\gamma = 10^{-3}$ である。

### 4.4 既存のMoEモデルへの適用

このフレームワークは、密なFFNだけでなく既存のMoEモデルにも適用できる。エキスパート $\{E_i\}$ を持つMoE層に対して、我々は各エキスパートに個別に再構成を適用し、共有サブエキスパートとルーティングサブエキスパートからなる階層構造へ変換する。

$$E_i(\mathbf{x}) \to E_i^s(\mathbf{x}) + \sum_{j=1}^{N_r'} g_{i,j}' \cdot E_{i,j}^r(\mathbf{x}). \tag{10}$$

これにより2段階の階層が作られる。上位レベルのルーターが主要なエキスパートを選択し、活性化された各エキスパートの内部では、サブルーターが専門化されたサブエキスパートを選択する。これにより、さらなる高速化のためのよりきめ細かいスパース性が誘導される。

## 5 実験

我々は、大規模言語モデルの推論高速化のための事後学習(post-training)スパース化手法として、提案フレームワークを評価する。

### 5.1 実験設定

**モデルと実装。** 我々は、Hugging Face Transformers(Wolf, 2019)とPyTorch(Paszke et al., 2019)を用いて、Llama-2 7B、Llama-2 70B、Qwen-2.5-7B、Qwen-3-30B-A3Bで評価を行う。Qwen-2.5 72Bは、表9の産業応用向け高速化評価において追加で使用する。

**キャリブレーション。** 我々は、活性化プロファイリングのためにWikiText-2(Merity et al., 2016)から8個の例(各2,048トークン)を使用し、上位$K$活性化選択には $K_a = 10$ を用いる。

**ファインチューニング。** 我々は、2,048個のWikiText-2サンプルに対して1エポック、Adam(Kingma, 2014)($\beta_1 = 0.9$、$\beta_2 = 0.95$)を用いてLoRA(Hu et al., 2021)(ランク8、アルファ32)を適用する。学習率は、ルータースケーリングには0.001、LoRAには5.95e-5を用いる。負荷分散には $\gamma = 0.001$ を用いる。

**ベースライン(比較対象手法)。** 我々は以下と比較する。(1) *構造的プルーニング*:SliceGPT(Ashkboos et al., 2024)およびSLEB(Song et al., 2024)を20%削減の設定で使用。(2) *MoE再構成*:LLaMA-MoE(Zhu et al., 2024)、LLaMA-MoE-v2(Qu et al., 2024)、EMoE(Qiu et al., 2023)。全てのベースラインは我々自身が再実装し、公平な比較のために同一のデータ予算(2,000サンプル)のもとでLoRAによってファインチューニングした。構造的プルーニングのベースラインが20%削減を用いるのは、これらの手法がFFNとAttentionの両方のパラメータを削減するため、実質的なFFNのスパース性が、我々の(FFNのみを対象とした)25%スパース性と同程度になるようにするためである。

**構成。** 特に断りのない限り、25%のスパース性(すなわち、各トークンにつきFFNニューロンの75%が活性化される)を用い、S3A3E8(共有3個+活性化されるルーティング3個 / 合計8個のエキスパート)の構成を用いる。全てのMoE手法は、公平な比較のために8個のエキスパートを使用する。

### 5.2 主な結果

**ゼロショットの下流タスク。** 表1は、PIQA(Bisk et al., 2020)、WinoGrande(Sakaguchi et al., 2021)、ARC-Easy、ARC-Challenge(Clark et al., 2018)、HellaSwag(Zellers et al., 2019)の5つのベンチマークにおける結果を示している。25%のスパース性のもとで、我々の手法は7Bから30Bのパラメータ規模にまたがる4つのモデル全てにおいて、一貫して全てのベースラインを上回っている。Llama-2 7Bでは、PIQAで74.34%、HellaSwagで69.36%を達成し、構造的プルーニングおよびMoE再構成のアプローチを上回った。この改善はモデル規模が変わっても一貫して見られ、Llama-2 70Bでも同様の向上が観察され、Qwen-3-30B-A3Bでは我々の手法はPIQAで80.23%、HellaSwagで80.71%を達成した。

**知識・コーディング・数学に関するより広範な評価。** これら5つのゼロショットタスクに加えて、知識集約型タスクおよび推論ベンチマークを網羅するため、25%スパース性(S3A3E8)のLlama-2 7BをMMLU-5shot、HumanEval pass@1、GSM8K-8shotでも評価する。表2にまとめられているように、我々の解析的なMoE再構成は、MMLU-5shotで44.02%を達成し、コーディングおよび数学の精度でも競争力のある結果を示しており、3つのベンチマーク全てにおいてLLaMA-MoEの各バリエーションおよびEMoEを上回っている。

表1: 25%のスパース性(Sp.)におけるゼロショット精度(%)。スパース化された全ての手法は2,000サンプルでファインチューニングされている。密なベースライン(Dense)はファインチューニングされていない。

**Llama-2 7B**

| 手法 | Sp. | PIQA | WinoG. | ARC-E | ARC-C | HellaS. |
|---|---|---|---|---|---|---|
| Dense(密モデル) | 0% | 78.78 | 69.06 | 74.58 | 46.16 | 76.00 |
| SliceGPT | 20% | 65.71 | 62.88 | 59.76 | 33.21 | 51.34 |
| SLEB | 20% | 73.13 | 58.98 | 57.90 | 33.02 | 62.47 |
| LLaMA-MoE | 25% | 49.35 | 50.28 | 54.04 | 26.37 | 25.77 |
| LLaMA-MoE-v2 | 25% | 63.55 | 59.35 | 63.77 | 34.81 | 54.89 |
| EMoE | 25% | 72.47 | 64.68 | 58.94 | 35.75 | 60.80 |
| **Ours(提案手法)** | 25% | **74.34** | 65.77 | 67.09 | 40.35 | 69.36 |

**Llama-2 70B**

| 手法 | Sp. | PIQA | WinoG. | ARC-E | ARC-C | HellaS. |
|---|---|---|---|---|---|---|
| Dense(密モデル) | 0% | 82.70 | 77.98 | 80.98 | 57.34 | 83.84 |
| SliceGPT | 20% | 68.91 | 70.06 | 64.56 | 41.14 | 56.26 |
| SLEB | 20% | 77.39 | 65.55 | 62.37 | 40.11 | 68.39 |
| LLaMA-MoE | 25% | 51.95 | 66.50 | 59.09 | 32.40 | 27.57 |
| LLaMA-MoE-v2 | 25% | 66.79 | 66.57 | 62.42 | 38.19 | 59.57 |
| EMoE | 25% | 76.34 | 72.33 | 63.47 | 43.62 | 66.19 |
| **Ours(提案手法)** | 25% | **78.49** | 73.49 | 73.32 | 49.86 | 76.12 |

**Qwen-2.5-7B**

| 手法 | Sp. | PIQA | WinoG. | ARC-E | ARC-C | HellaS. |
|---|---|---|---|---|---|---|
| Dense(密モデル) | 0% | 79.82 | 73.16 | 77.36 | 51.02 | 78.86 |
| SliceGPT | 20% | 66.19 | 66.51 | 61.88 | 36.69 | 53.21 |
| SLEB | 20% | 74.95 | 61.76 | 59.95 | 35.80 | 64.41 |
| LLaMA-MoE | 25% | 49.63 | 53.21 | 57.05 | 28.64 | 25.65 |
| LLaMA-MoE-v2 | 25% | 64.25 | 62.71 | 65.41 | 37.59 | 56.06 |
| EMoE | 25% | 73.98 | 65.41 | 60.63 | 38.48 | 62.71 |
| **Ours(提案手法)** | 25% | **75.93** | 69.36 | 70.59 | 43.86 | 72.21 |

**Qwen-3-30B-A3B**

| 手法 | Sp. | PIQA | WinoG. | ARC-E | ARC-C | HellaS. |
|---|---|---|---|---|---|---|
| Dense(密モデル) | 0% | 84.51 | 79.18 | 84.43 | 57.88 | 87.44 |
| SliceGPT | 20% | 70.60 | 71.58 | 66.48 | 41.85 | 58.41 |
| SLEB | 20% | 79.16 | 66.01 | 70.08 | 42.11 | 71.74 |
| LLaMA-MoE | 25% | 52.18 | 54.48 | 62.50 | 30.77 | 28.32 |
| LLaMA-MoE-v2 | 25% | 65.54 | 67.71 | 71.27 | 41.99 | 62.78 |
| EMoE | 25% | 74.76 | 70.50 | 65.78 | 43.12 | 70.62 |
| **Ours(提案手法)** | 25% | **80.23** | 74.84 | 76.75 | 48.80 | 80.71 |

表2: Llama-2 7Bにおける25%スパース性(S3A3E8)でのより広範な評価。

| 手法 | MMLU (%) | HumanEval | GSM8K (%) |
|---|---|---|---|
| LLaMA-MoE | 35.09 | 7.58 | 7.41 |
| LLaMA-MoE-v2 | 38.02 | 9.32 | 10.09 |
| EMoE | 43.11 | 10.29 | 12.55 |
| **Ours(提案手法)** | **44.02** | **11.22** | **13.01** |

### 5.3 アブレーション研究

**学習不要 vs ファインチューニング済みの性能。** 図4は、25%スパース性におけるデータ効率を示している。この手法は、ファインチューニングを一切行わずに構築した直後でも妥当な性能を達成しており、解析的なルーターの初期化が有効であることを示している。性能は追加データによってすぐに頭打ちになり、わずか1,024サンプルでほぼ最適な結果に到達する。

表3: Llama-2 7B(25%スパース性)における学習不要 vs ファインチューニング済みの比較。

| 手法 | 条件 | MMLU (%) | PPL Wiki | PPL C4 |
|---|---|---|---|---|
| LLaMA-MoE-v2 | 学習不要 | 30.33 | >10k | >7k |
| LLaMA-MoE-v2 | ファインチューニング済み | 38.02 | 8.68 | 19.76 |
| Ours(提案手法) | 学習不要 | 42.50 | 7.32 | 11.98 |
| Ours(提案手法) | ファインチューニング済み(2k) | **44.02** | **5.92** | **11.21** |

表3は、Llama-2 7BにおいてLLaMA-MoE-v2との比較を示している。我々の学習不要モデルは42.50%のMMLU-5shotを達成しており、これはLLaMA-MoE-v2をファインチューニングした後の38.02%を上回る。2,000サンプルを用いると44.02%に達し、性能向上の大部分がファインチューニングではなく解析的な再構成そのものから生じていることを示している。

**キャリブレーションに対する感度。** 表4は、Llama-2 7Bにおいてキャリブレーション元(WikiText-2とC4)とそのサイズを変化させたものである。WikiText-2については、サンプル数を8から64に増やすと、MMLUがわずかに向上し(44.02→44.89)、パープレキシティもわずかに減少する。パープレキシティは両方のキャリブレーション元で一貫しており、多様な活性化パターン(3.2節)が特定のデータに依存するものではなく、事前学習済みFFNに内在する性質であることを裏付けている。必要なサンプル数が少ない(8例)ことは、必要に応じてドメイン固有のキャリブレーションを行うことを実用的なものにしている。

さらにドメインに依存しないことを確認するため、数学・科学データ(Nemotron-Post-Training-Dataset-v1(Nathawani et al., 2025))とコードデータ(Open-Coder(Huang et al., 2024))を用いて共有エキスパートのニューロンを個別にプロファイリングし、両者の重なり(overlap)を測定した。重なりは数学/科学で84%、数学/コードで86%、科学/コードで80%であり、共有エキスパートの選択がドメイン固有の偶然の産物ではなく、モデルに内在する構造を反映していることを裏付けている。

**クラスタリングとルーティング。** クラスタリングとルーティングそれぞれの貢献を切り分けるため、表5は同一の設定(25%スパース性、2,000サンプルのファインチューニング)のもとで各手法を比較している。MoEfication(Zhang et al., 2021)、G-MoEfication(Lee et al., 2024、その非ReLUへの一般化版)、Read-ME(Cai et al., 2024)は、それぞれMMLUで35.17%、36.37%、31.24%しか達成しないのに対し、我々の手法は44.02%に達する。それぞれの手法のルーターを我々の解析的ルーターに置き換えると結果が+2〜6ポイント改善し、さらに我々の活性化ベースのクラスタリングと共有エキスパートへ切り替えると、さらに+5〜7ポイントの改善が得られ、両方の要素がそれぞれ独立に貢献していることが確認された。

重みベースの手法は、パラメータが似ているニューロンは似た機能を持つと仮定しているが、入力依存の挙動を無視している。我々のアプローチは共活性化するニューロンをグループ化することで、エキスパートの構造を実際の使用パターンに整合させている。特筆すべき点として、MoEficationも共活性化グラフによるクラスタリングを検討していたが、彼らのパラメータベースの手法の方が良い性能を示していた。我々のアプローチは以下の点で異なる。(1) 我々は高頻度のニューロンを共有エキスパートへ明示的に分離する、(2) 我々はルーターを代表ニューロンから解析的に導出する。この組み合わせにより、MoEficationの最良の手法と比べて、ほぼ+9ポイントの改善が得られる。

### 5.4 分析と考察

**効率性の比較。** 表6はトークン予算と変換にかかる時間を比較している。LLaMA-MoEは継続的な事前学習を行うのに対し、我々の手法は事後学習による再構成に焦点を当てている点に注意されたいが、それでもなお、どちらの手法を選ぶか検討する実務者にとって総コストは重要な判断材料となりうる。LLaMA-MoE-v1/v2はそれぞれ2,000億/70億トークン(数週間/数日)を必要とするのに対し、我々の手法はわずか400万トークン(2,000サンプル×2,048トークン)しか使用せず、end-to-endで46分(解析的な構築だけなら4.5分)で完了する。

表7はFLOPs(浮動小数点演算数)とスループットを報告している。Llama-2 7Bでは、25%のスパース性によりFLOPsが16.6%削減され、スループットが14.8%向上する。Qwen3-30B-A3Bへの階層的な適用では、18.5%のFLOPs削減と14.3%のスループット向上が得られる。

**負荷分散。** 図5はエキスパートの利用状況を示している。負荷分散を行わない場合、最終層では活性化の偏りが見られる。我々の適応的なバイアス機構は負荷を均一に再分配し、これによってデプロイ時に完全な高速化のポテンシャルを達成できるようになる。

**活性化スパース性との直交性。** 表8は、我々のエキスパートレベルの再構成が、ニューロンレベルの活性化スパース性(WINA(Chen et al., 2025))と直交していることを示している。両方を組み合わせると、Llama-2 7Bにおいて27.2%のTFLOPs削減と22.0%のスループット向上が得られ、これらが異なる非効率性を対象としていることが確認できる。

**産業応用における高速化。** 表9は、メモリ律速および計算律速(バッチサイズ400超)のシナリオにおける、Qwen-2.5 72Bでの推論高速化を示している。32kコンテキストの計算律速な設定では、S1A5E8構成が最大1.17倍の高速化を達成する。

**パープレキシティとスパース性のトレードオフ。** 表10は、16個の総エキスパート数のもとでスパース性を変化させたときの、Llama-2 7BにおけるWikiText-2パープレキシティを調べている。興味深いことに、スパース性0.125(87.5%のニューロンが活性化する)のとき、我々の変換されたモデルは密なベースラインをわずかに上回る(5.25 vs 5.27)。これは、この再構成が暗黙的な正則化(regularization)の効果を持つ可能性を示唆している。

**エキスパート構成の影響。** 図6は、25%スパース性における異なるエキスパート構成を比較している。S6A6E16はPIQAとARC-Easyで最も高い性能を達成する一方、S3A9E16はWinoGrandeで最も良い性能を示しており、最適な構成はタスクの特性に依存することを示している。

**自己整合性(Self-Consistency)による恩恵。** 表11は、25%スパース性における $k$サンプルの自己整合性(投票)を評価している。興味深いことに、スパースなモデルは密なモデルよりも自己整合性からより多くの恩恵を受ける。Llama-2 7Bでは、$k$を1から5に増やすと、我々の手法では+4.72ポイントの向上が見られるのに対し、密なモデルではわずか+0.57ポイントの向上にとどまる。Qwen3-30B-A3Bでは、この差はほぼ完全に埋まる(+6.91ポイント vs +0.58ポイント)。これは、スパースなルーティングに起因するばらつきが、デプロイ時に効果的に平均化されうることを示唆している。

表4: Llama-2 7B(25%スパース性)におけるキャリブレーション感度。パープレキシティはキャリブレーション元・サイズによらず一貫している。

| 元データ | $n$ | MMLU (%) | PPL Wiki | PPL C4 |
|---|---|---|---|---|
| WikiText-2 | 8 | 44.02 | 5.92 | 11.21 |
| WikiText-2 | 32 | 44.63 | 5.72 | 11.15 |
| WikiText-2 | 64 | **44.89** | **5.69** | **10.98** |
| C4 | 8 | 42.31 | 7.04 | 9.17 |
| C4 | 32 | 43.25 | 6.92 | 9.07 |
| C4 | 64 | **43.39** | **6.78** | **9.02** |

図4: データ効率:ファインチューニングのサンプル数に対する性能と構築時間(25%スパース性)。

表5: Llama-2 7Bにおけるクラスタリングとルーティングのアブレーション(MMLU)。全て25%スパース性・2,000サンプルのファインチューニングを使用。G-MoEfication(Lee et al., 2024)は、MoEficationを非ReLUモデルへ拡張したものである。

| 手法 | エキスパートのグループ化 | ルーター | MMLU (%) |
|---|---|---|---|
| MoEfication | パラメータのK-means | MLP | 35.17 |
| G-MoEfication | パラメータのK-means | MLP | 36.37 |
| READ-ME | ドメイン認識型 | グローバル | 31.24 |
| MoEfication + ours | パラメータのK-means | 解析的 | 37.33 |
| G-MoEfication + ours | パラメータのK-means | 解析的 | 39.21 |
| READ-ME + ours | ドメイン認識型 | 解析的 | 36.79 |
| **Ours(提案手法)** | 活性化 + 共有エキスパート | 解析的 | **44.02** |

表6: Llama-2 7Bにおけるトークン予算と変換時間の比較。E2E:ファインチューニングを含むend-to-end時間、Construct:再構成のみの時間。

| 手法 | トークン予算 | E2E時間 | 構築時間 |
|---|---|---|---|
| **Ours(提案手法)** | **400万** | **46分** | **4.5分** |
| LLaMA-MoE-v1 | 2,000億 | 数週間 | 6分† |
| LLaMA-MoE-v2 | 70億 | 数日 | 8分† |

† 分割のみ。学習時間は含まない。

表7: 効率性:FLOPs、MACs、スループット。

| モデル | 手法 | FLOPs (T/G) | MACs (G) | スループット (tok/s) |
|---|---|---|---|---|
| Llama-2 7B | Dense(密モデル) | 1.69T | 845.7 | 45.9 |
| Llama-2 7B | Ours(提案手法, 25%) | 1.41T (-16.6%) | 707.4 (-16.4%) | 52.7 (+14.8%) |
| Qwen3-30B-A3B | Dense(密モデル) | 778.7G | 389.3 | 1.19 |
| Qwen3-30B-A3B | Ours(提案手法, 階層適用) | 634.9G (-18.5%) | 331.3 (-14.9%) | 1.36 (+14.3%) |

表8: Llama-2 7B(25%スパース性)における、ニューロンレベルの活性化スパース性(WINA)と我々のエキスパートレベルのDense-to-MoE再構成の直交性。WINAはよりきめ細かいニューロンレベルで動作するのに対し、我々の手法はFFNをルーティングエキスパートへ再構成する。両者を組み合わせると効率の向上が加算的に得られる。

| 手法 | TFLOPs (↓) | GMACs (↓) | tokens/s (↑) |
|---|---|---|---|
| Dense(ベースライン) | 1.69 | 845.71 | 45.88 |
| WINA (25%スパース性) | 1.31 (-22.5%) | 691.19 (-18.3%) | 51.76 (+12.8%) |
| Ours(提案手法, 25%スパース性) | 1.41 (-16.6%) | 707.36 (-16.3%) | 52.67 (+14.8%) |
| **Ours + WINA** | **1.23 (-27.2%)** | **625.53 (-26.0%)** | **55.97 (+22.0%)** |

図5: 負荷分散:負荷分散を行う前(左)と行った後(右)のエキスパート利用状況。

表9: Qwen-2.5 72Bにおける、異なるコンテキスト長・シナリオでの推論高速化(25%スパース性)。S$x$A$y$E$z$:共有$x$個+アクティブなルーティング$y$個/合計$z$個のエキスパート。

| 構成 | メモリ律速 4kコンテキスト | メモリ律速 32kコンテキスト | 計算律速 4kコンテキスト | 計算律速 32kコンテキスト |
|---|---|---|---|---|
| S1A5E8 | 1.08倍 | 1.15倍 | 1.12倍 | **1.17倍** |
| S3A3E8 | 1.06倍 | 1.13倍 | 1.11倍 | 1.15倍 |
| S2A4E8 | 1.05倍 | 1.12倍 | 1.10倍 | 1.12倍 |
| S4A8E16 | 1.02倍 | 1.10倍 | 1.08倍 | 1.11倍 |
| S6A6E16 | 1.03倍 | 1.08倍 | 1.07倍 | 1.10倍 |
| S3A9E16 | 1.02倍 | 1.05倍 | 1.05倍 | 1.09倍 |

表10: Llama-2 7B(16エキスパート)におけるパープレキシティとスパース性の関係。値が小さいほど良い。

| スパース性 | Dense(密モデル) | 0.75 | 0.625 | 0.5 | 0.375 | 0.25 | 0.125 |
|---|---|---|---|---|---|---|---|
| PPL ↓ | 5.27 | 12.73 | 9.56 | 7.71 | 6.55 | 5.78 | **5.25** |

表11: 25%スパース性における $k$サンプル自己整合性の効果。精度(%)。

**Llama-2 7B**

| 手法 | $k$ | PIQA | ARC-E | ARC-C | 平均 |
|---|---|---|---|---|---|
| Dense(密モデル) | 1 | 78.78 | 74.58 | 46.16 | 66.51 |
| Dense(密モデル) | 5 | 79.21 | 75.29 | 46.75 | 67.08 |
| Ours(提案手法) | 1 | 74.34 | 67.09 | 40.35 | 60.59 |
| Ours(提案手法) | 5 | 77.52 | 73.88 | 44.54 | 65.31 |

**Qwen3-30B-A3B**

| 手法 | $k$ | PIQA | ARC-E | ARC-C | 平均 |
|---|---|---|---|---|---|
| Dense(密モデル) | 1 | 84.51 | 84.43 | 57.88 | 75.61 |
| Dense(密モデル) | 5 | 85.11 | 85.33 | 58.12 | 76.19 |
| Ours(提案手法) | 1 | 80.23 | 76.75 | 48.80 | 68.59 |
| Ours(提案手法) | 5 | 84.56 | 84.75 | 57.19 | 75.50 |

図6: 25%スパース性におけるエキスパート構成の影響。S$x$A$y$E$z$は、合計$z$個のエキスパートのうち共有エキスパート$x$個+アクティブなルーティングエキスパート$y$個を表す。

## 6 さらなる考察

**より広い影響と将来の方向性。** 我々の研究は、LLM推論における大きな計算オーバーヘッドを削減するための解析的な事後学習フレームワークを提示するものであり、これにより計算資源の限られた環境における研究や実運用において、強力なモデルをより利用しやすくする。純粋な高速化技術という側面を超えて、この手法の解析的な性質は、FFNの内部の働きを解釈するための新しい視点を提供する。活性化統計量に基づいてニューロンを「共有」と「ルーティング」のエキスパートへ明確にグループ化することは、これらの層内における機能的な専門化(functional specialization)についての経験的な証拠を与える。将来の研究では、この方法論を活用して、LLM内で知識がどのようにエンコードされ処理されているかを分析できるかもしれない。今後の課題としては、この解析的な再構成のアプローチを、Attentionヘッドなどのtransformerの他の部分へと拡張することが有望な方向性である。さらに、ルーター構築のためのより高度な解析的技術を探求することで、事後的なアプローチの効率性を犠牲にすることなく、完全に学習されたルーターとの残された性能差を埋められる可能性がある。

**他の効率化技術との互換性。** この解析的な再構成は、ほとんどのシステムレベル・モデルレベルの効率化手法と直交しており、それらと組み合わせることができる。実際には、FFNの再構成は事後的な量子化(例:AWQ/QAT)とうまく統合できる。これは、この操作が層のインターフェースを保持するためであり、少量のキャリブレーションを行えば、量子化の前でも後でも適用でき、精度を維持できる。同様に、Attention側の最適化(KVキャッシュ圧縮、投機的デコーディング、Attentionのスパース性)は異なるボトルネックを対象としており、相補的である。構造的プルーニング(例:SliceGPT、SLEB)と我々の動的なエキスパートルーティングは異なる状況を対象としている。プルーニングは全ての入力に対して静的に容量を削減するのに対し、我々の手法はトークンごとに条件付きで容量を活性化する。同様に、学習不要な活性化スパース性の手法(例:TEAL、WINA)はよりきめ細かいニューロンレベルで動作し、我々のルーティングエキスパートの内部に適用してさらにFLOPsを削減することができる。実運用においては、end-to-endでの高速化を実現するために、負荷分散とバッチ処理のポリシーが依然として重要である。我々が組み込んだバイアス適応の仕組みは、エキスパートへのアクセス集中(ホットスポット化)を緩和し、メモリ律速・計算律速の両方の設定において利用率を改善する。全体として、軽量なキャリブレーションと再構成のステップを経た後、このフレームワークはFFNの置き換えとして機能し、量子化、キャッシング、プルーニング、サービング(推論提供)の最適化と組み合わせることで、実用上の高速化の範囲を広げることができる。

## 7 結論

我々は、小さなキャリブレーションデータセットとわずか数分の計算(Llama-2 7Bで4.5分)のみを用いて、FFNをスパースなMoEアーキテクチャへ解析的に再構成する事後学習フレームワークを提案した。ニューロンの活性化頻度の多様な分布を活用することで、この手法はニューロンを共有エキスパートとルーティングエキスパートへ分割し、代表的なニューロン統計量からルーターを構築する。これにより、任意の軽量なファインチューニングによって高い性能を実現できる。この手法は密なモデルにも既存のMoEモデルにも適用でき、階層的なスパース性を実現できるほか、量子化やニューロンレベルの活性化スパース性といった技術とも直交しており、性能の高いスパースなLLMを実用的かつ軽い学習コストで導入するための道を提供する。

## 8 限界

我々のフレームワークには3つの主な限界がある。第一に、活性化プロファイリングの質はキャリブレーションデータセットに依存する。データが対象ドメインを代表するものであるときに性能が最も良くなるが、この手法はキャリブレーションセットのサイズに対しては比較的頑健である。第二に、スパースなルーティングが持つ離散的な性質は、生成における分散(ばらつき)を高める傾向があるが、これは自己整合性によって緩和できる。第三に、我々の評価は英語のベンチマークとデコーダのみのtransformer(Llama-2、Qwen)に焦点を当てており、多言語やエンコーダ・デコーダ型、あるいは状態空間モデル(state-space)アーキテクチャへの拡張は今後の課題として残されている。

## References

Saleh Ashkboos, Maximilian L Croci, Marcelo Gennari do Nascimento, Torsten Hoefler, and James Hensman. 2024. Slicegpt: Compress large language models by deleting rows and columns. *arXiv preprint arXiv:2401.15024*.

Yonatan Bisk, Rowan Zellers, Jianfeng Gao, Yejin Choi, and 1 others. 2020. Piqa: Reasoning about physical commonsense in natural language. In *Proceedings of the AAAI conference on artificial intelligence*, volume 34, pages 7432–7439.

Ruisi Cai, Yeonju Ro, Geon-Woo Kim, Peihao Wang, Babak Ehteshami Bejnordi, Aditya Akella, Zhangyang Wang, and 1 others. 2024. Read-me: Refactorizing llms as router-decoupled mixture of experts with system co-design. *Advances in Neural Information Processing Systems*, 37:116126–116148.

Sihan Chen, Dan Zhao, Jongwoo Ko, Colby Banbury, Huiping Zhuang, Luming Liang, and Tianyi Chen. 2025. Wina: Weight informed neuron activation for accelerating large language model inference. *arXiv preprint arXiv:2505.19427*.

Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. 2018. Think you have solved question answering? try arc, the ai2 reasoning challenge. *arXiv preprint arXiv:1803.05457*.

Damai Dai, Chengqi Deng, Chenggang Zhao, RX Xu, Huazuo Gao, Deli Chen, Jiashi Li, Wangding Zeng, Xingkai Yu, Y Wu, and 1 others. 2024. Deepseek-moe: Towards ultimate expert specialization in mixture-of-experts language models. *arXiv preprint arXiv:2401.06066*.

Nan Du, Yanping Huang, Andrew M Dai, Simon Tong, Dmitry Lepikhin, Yuanzhong Xu, Maxim Krikun, Yanqi Zhou, Adams Wei Yu, Orhan Firat, and 1 others. 2022. Glam: Efficient scaling of language models with mixture-of-experts. In *International Conference on Machine Learning*, pages 5547–5569. PMLR.

William Fedus, Barret Zoph, and Noam Shazeer. 2022. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. *Journal of Machine Learning Research*, 23(120):1–39.

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2021. Lora: Low-rank adaptation of large language models. *arXiv preprint arXiv:2106.09685*.

Siming Huang, Tianhao Cheng, Jason Klein Liu, Jiaran Hao, Liuyihan Song, Yang Xu, J. Yang, J. H. Liu, Chenchen Zhang, Linzheng Chai, Ruifeng Yuan, Zhaoxiang Zhang, Jie Fu, Qian Liu, Ge Zhang, Zili Wang, Yuan Qi, Yinghui Xu, and Wei Chu. 2024. Opencoder: The open cookbook for top-tier code large language models.

Roy Jonker and Ton Volgenant. 1988. A shortest augmenting path algorithm for dense and sparse linear assignment problems. In *DGOR/NSOR: Papers of the 16th Annual Meeting of DGOR in Cooperation with NSOR/Vorträge der 16. Jahrestagung der DGOR zusammen mit der NSOR*, pages 622–622. Springer.

Diederik P Kingma. 2014. Adam: A method for stochastic optimization. *arXiv preprint arXiv:1412.6980*.

Aran Komatsuzaki, Joan Puigcerver, James Lee-Thorp, Carlos Riquelme Ruiz, Basil Mustafa, Joshua Ainslie, Yi Tay, Mostafa Dehghani, and Neil Houlsby. 2022. Sparse upcycling: Training mixture-of-experts from dense checkpoints. *arXiv preprint arXiv:2212.05055*.

Jaeseong Lee, Seung-won Hwang, Wonpyo Park, and Mingyi Ji. 2024. Breaking relu barrier: Generalized moefication for dense pretrained models. In *Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing*, pages 10097–10107.

Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu, Dehao Chen, Orhan Firat, Yanping Huang, Maxim Krikun, Noam Shazeer, and Zhifeng Chen. 2020. Gshard: Scaling giant models with conditional computation and automatic sharding. *arXiv preprint arXiv:2006.16668*.

Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, and 1 others. 2024a. Deepseek-v3 technical report. *arXiv preprint arXiv:2412.19437*.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. 2024b. Visual instruction tuning. *Advances in neural information processing systems*, 36.

James Liu, Pragaash Ponnusamy, Tianle Cai, Han Guo, Yoon Kim, and Ben Athiwaratkun. 2024c. Training-free activation sparsity in large language models. *arXiv preprint arXiv:2408.14690*.

Zichang Liu, Jue Wang, Tri Dao, Tianyi Zhou, Binhang Yuan, Zhao Song, Anshumali Shrivastava, Ce Zhang, Yuandong Tian, Christopher Re, and 1 others. 2023. Deja vu: Contextual sparsity for efficient llms at inference time. In *International Conference on Machine Learning*, pages 22137–22176. PMLR.

Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. 2016. Pointer sentinel mixture models. *arXiv preprint arXiv:1609.07843*.

Dhruv Nathawani, Igor Gitman, Somshubra Majumdar, Evelina Bakhturina, Ameya Sunil Mahableshwarkar, , Jian Zhang, and Jane Polak Scowcroft. 2025. Nemotron-Post-Training-Dataset-v1.

Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, and 1 others. 2019. Pytorch: An imperative style, high-performance deep learning library. *Advances in neural information processing systems*, 32.

Zehua Pei, Hui-Ling Zhen, Xianzhi Yu, Sinno Jialin Pan, Mingxuan Yuan, and Bei Yu. 2024. Fusegpt: Learnable layers fusion of generative pre-trained transformers. *arXiv preprint arXiv:2411.14507*.

Zihan Qiu, Zeyu Huang, and Jie Fu. 2023. Unlocking emergent modularity in large language models. *arXiv preprint arXiv:2310.10908*.

Xiaoye Qu, Daize Dong, Xuyang Hu, Tong Zhu, Weigao Sun, and Yu Cheng. 2024. Llama-moe v2: Exploring sparsity of llama from perspective of mixture-of-experts with post-training. *arXiv preprint arXiv:2411.15708*.

Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. 2021. Winogrande: An adversarial winograd schema challenge at scale. *Communications of the ACM*, 64(9):99–106.

Jiwon Song, Kyungseok Oh, Taesu Kim, Hyungjun Kim, Yulhwa Kim, and Jae-Joon Kim. 2024. Sleb: Streamlining llms through redundancy verification and elimination of transformer blocks. *arXiv preprint arXiv:2402.09025*.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, and 1 others. 2023. Llama 2: Open foundation and fine-tuned chat models. *arXiv preprint arXiv:2307.09288*.

Ziteng Wang, Jun Zhu, and Jianfei Chen. 2024. Remoe: Fully differentiable mixture-of-experts with relu routing. *arXiv preprint arXiv:2412.14711*.

T Wolf. 2019. Huggingface's transformers: State-of-the-art natural language processing. *arXiv preprint arXiv:1910.03771*.

Haoyuan Wu, Haisheng Zheng, Zhuolun He, and Bei Yu. 2024. Parameter-efficient sparsity crafting from dense to mixture-of-experts for instruction tuning on general tasks. *arXiv preprint arXiv:2401.02731*.

Yuanhang Yang, Shiyi Qi, Wenchao Gu, Chaozheng Wang, Cuiyun Gao, and Zenglin Xu. 2024. Xmoe: Sparse models with fine-grained and adaptive expert selection. *arXiv preprint arXiv:2403.18926*.

Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. 2019. Hellaswag: Can a machine really finish your sentence? *arXiv preprint arXiv:1905.07830*.

Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, and 1 others. 2022. Opt: Open pretrained transformer language models. *arXiv preprint arXiv:2205.01068*.

Zhengyan Zhang, Yankai Lin, Zhiyuan Liu, Peng Li, Maosong Sun, and Jie Zhou. 2021. Moefication: Transformer feed-forward layers are mixtures of experts. *arXiv preprint arXiv:2110.01786*.

Haizhong Zheng, Xiaoyan Bai, Xuesheng Liu, Z Morley Mao, Beidi Chen, Fan Lai, and Atul Prakash. 2024. Learn to be efficient: Build structured sparsity in large language models. *arXiv preprint arXiv:2402.06126*.

Zexuan Zhong, Mengzhou Xia, Danqi Chen, and Mike Lewis. 2024. Lory: Fully differentiable mixture-of-experts for autoregressive language model pre-training. *arXiv preprint arXiv:2405.03133*.

Tong Zhu, Xiaoye Qu, Daize Dong, Jiacheng Ruan, Jingqi Tong, Conghui He, and Yu Cheng. 2024. Llama-moe: Building mixture-of-experts from llama with continual pre-training. In *Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing*, pages 15913–15923.
