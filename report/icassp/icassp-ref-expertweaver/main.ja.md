# ExpertWeaver：GLUの活性化パターンによって密なLLMに内在するMoEを解き明かす

Ziyu Zhao、Tong Zhu、Zhi Zhang、Tiantian Fan、Jinluan Yang、Kun Kuang、Zhongyu Wei、Fei Wu、Yu Cheng

arXiv:2602.15521v1 [cs.CL] 2026年2月17日

## 概要

Mixture-of-Experts（MoE）アーキテクチャは、少数のエキスパートだけを疎に活性化することで、計算効率を保ちながらモデル容量を効果的に拡張する。しかし、高品質なMoEをゼロから学習するには非常に大きなコストがかかる。有望な代替策は、事前学習済みの密なモデルを疎なMoEへ変換することである。既存のdense-to-MoE手法は、主に二つに分かれる。一つは、性能と推論効率のバランスを取るため、密なモデルを中程度の疎性（例：25%）を持つMoEへ変換する動的構造的プルーニングである。もう一つは、事前学習済みの密なモデルを利用して、非常に高い疎性を持つMoEを初期化するダウンサイクリングである。しかし既存手法は、密なモデル内部にある本来の活性化パターンを壊すため、エキスパート構成が最適でなくなる。本研究では、Gated Linear Unit（GLU）の仕組みが、dense-to-MoE変換の自然な設計図になると主張する。GLUの細粒度なニューロン単位の活性化パターンを調べると、常に活性化される汎用ニューロンと、入力に応じて動的に活性化される特化ニューロンから成る、粗粒度の内在的なMoE構造が見える。そこで、活性化パターンに基づいてニューロンを分割し、層ごとに構成を調整しながら共有エキスパートと特化ルーティング・エキスパートを構築する、学習不要のフレームワークExpertWeaverを提案する。実験では、ExpertWeaverが、学習不要の動的構造的プルーニング手法としても、優れたMoE初期化のためのダウンサイクリング戦略としても、既存手法を大幅に上回ることを示す。

## 1. はじめに

Mixture-of-Experts（MoE）アーキテクチャは、少数のエキスパートだけを活性化することで、効率を維持しながらモデル容量を拡張する。しかし、MoEモデルをゼロから学習するには高い費用が必要であり、事前学習済みの密なモデルを疎なMoEアーキテクチャへ変換する研究が盛んになっている。これらの研究は、目標とする疎性と目的に応じて大きく二つの流れに分けられる。第一はDynamic Structural Pruningであり、密なモデルを中程度の疎性（例えば25%）を持つMoEへ変換し、性能と効率のバランスを取る（Gao et al., 2025; Pei et al., 2025; Nishu et al., 2025; Zheng et al., 2024）。第二はDowncyclingである。これは、モデルを拡大するupcycling（He et al., 2024; Komatsuzaki et al., 2022）とは反対に、事前学習済みの密なMLPを小さなエキスパートへ分割することで、疎性75%以上のMoEを初期化し、MoEをゼロから学習する莫大なコストを避ける（Zhang et al., 2022b; Zhu et al., 2024; Qu et al., 2024）。しかし既存手法は本来の活性化パターンを壊し、追加のルーター学習を必要とし、層ごとの違いも無視するため、エキスパート構成が最適でなくなる。詳しい関連研究は付録Dに示す。

本研究では、これらの限界を乗り越える鍵はGLUの仕組みにあると考える。GLUは、入力の文脈に基づいてニューロンの活性化を動的に制御する追加のゲート重みを用いるため、各ニューロンの役割と重要性を示す有用な手掛かりを与える。したがって、密なモデルをMoEへ変換する自然な設計図になる（Shazeer, 2020）。

§2.2で行うGLUの分析から、本手法の基盤となる重要な知見を得た。第一に、GLUのゲート信号は自然にトークン単位の活性化疎性を生む（Takeaway 1）。さらに、このパターンから、共有エキスパートに適した汎用ニューロンと、ルーティングされるエキスパートに適したタスク固有のクラスタの両方が見つかる（Takeaway 2、3）。また、変動係数（Coefficient of Variation; CV）によってニューロンの特化度を定量化すると、汎用ニューロンと特化ニューロンの分布は層をまたいで体系的に変化することが分かった。これは、全ての層に同じエキスパート構成を適用するのではなく、層ごとに異なる構成が必要であることを示す（Takeaway 4）。これらを合わせると、細粒度のニューロン単位のGLU活性化から、粗粒度のエキスパート単位のMoEアーキテクチャを構築するための完全な設計図が得られる。

これらの知見に基づき、GLUの活性化パターンに導かれて密なモデルをMoEへ変換する、学習不要の新手法ExpertWeaverを提案する。処理は三段階から成る。まず、多タスクの較正データセットに対するGLU活性化を記録し、全ニューロンの活性化パターンを取得する。次に、これらの活性化のCVを計算して層を考慮した構成を決め、各層における共有エキスパートの正確な大きさと、ルーティングされるエキスパート群の規模を決める。最後に、層ごとの構成に従ってニューロンを分割する。最も一貫して活性化するニューロンを一つの共有エキスパートにまとめ、残る特化ニューロンを活性化パターンに基づいてルーティング・エキスパートへクラスタリングする。MoEルーターもGLUゲートのセントロイドから学習なしで構築する。この一連の処理により、密なモデルのニューロンを複数のエキスパートへ「織り込み」、構造化されたMoEアーキテクチャを作る。

図1．多様なタスクにおけるニューロン活性化パターン。Flan-v2の一部を用い、Qwen2.5-7Bの中間層における活性化パターンを可視化した。a) 異なるタスク間でのニューロン活性化分布。b) 個々のタスククラスタ内でのニューロン活性化分布。同じクラスタに属するタスクは箱で囲んでいる。

異なる層では異なるエキスパート構成が必要であり、全層に一つの構成を当てはめればよいわけではない。この観察は、細粒度のニューロン単位のGLU活性化から、粗粒度のエキスパート単位のMoE構造を作るための設計図を与える。

本研究の主な貢献は次のとおりである。

- 密なLLMにおけるGLU活性化パターンを体系的に分析し、細粒度の活性化パターンが構造的性質を持ち、粗粒度の活性化を持つMoE構築の自然な設計図になることを示す。
- GLU活性化パターンを利用して密なモデルをMoEアーキテクチャへ変換する、学習不要の新しいフレームワークExpertWeaverを提案する。
- 包括的な実験により、ExpertWeaverが多様な下流タスクにおいて、構造的プルーニングとMoEダウンサイクリングの両方で既存手法を大幅に上回ることを示す。

## 2. 準備

### 2.1 背景

#### Gated Linear Unit

現在の高性能LLMの多くはGLUアーキテクチャを採用しており、SwiGLUが最も一般的な変種である（Shazeer, 2020）。密なGLU層は、$W_{gate}$、$W_{up}$、$W_{down}$という三つの重み行列を用いて入力$x$を処理する。計算全体は次のように表される。

$$y = (\mathrm{Swish}(xW_{gate}) \odot (xW_{up})) W_{down}.$$

要素積$odot$は、入力の射影と動的ゲートを組み合わせ、各トークンに対するニューロンの活性化強度を調整する。このゲート機構によって、密なGLUモデルには内在的な動的疎性が生じる。これは、事前学習済みの密なモデルを構造化された疎なMoEへ効率的に変換する鍵である。

#### Mixture-of-Experts

MoEアーキテクチャは、LLMの密なFeed-Forward Network（FFN）を、複数の小さなエキスパートネットワークとゲート機構に置き換える。具体的には、MoE層は$N$個のエキスパート$(E_1,\ldots,E_N)$と、入力トークンごとに活性化するエキスパートを動的に選ぶルーターネットワーク$G$から成る。

$$y=\sum_{i\in\mathrm{TopK}(G(x))}g(x)_iE_i(x); \tag{1}$$

$$E_i(x)=\left(\mathrm{Swish}(xW_{gate}^{(i)})\odot(xW_{up}^{(i)})\right)W_{down}^{(i)},$$

ここで$g(x)_i$はエキスパート$i$に対する正規化済みゲート重みを表し、各エキスパートは独自のパラメータを持つ。トークンごとに一部のエキスパートだけを活性化することで、MoEはモデル容量を保ちながら計算コストを大幅に削減する。

## 2.2 動機

§2.1で述べたように、GLUとMoEは活性化パターンの粒度が異なる。GLUは、各トークンについてニューロンの活性化強度を調整する動的ゲートによって、ニューロン単位の細粒度の活性化を実現する。一方MoEは、トークン全体を選択されたエキスパートへ送ることで、エキスパート単位の粗粒度で構造的な活性化を行う。このニューロンレベルとエキスパートレベルの違いから、次の疑問が生じる。GLUの細粒度なニューロン単位の活性化パターンを集約し、粗粒度のエキスパート単位で活性化されるMoEの構築に役立てられるだろうか。事前学習済みの密なLLMにおけるGLU活性化パターンを探索的に分析した結果、ExpertWeaverの基盤となる四つの知見を得た。

### 2.2.1 GLUは自然な疎性信号である

§2.1で述べたように、GLUのゲート機構$\mathrm{Swish}(xW_{gate})$は、データに依存する動的なフィルターとして働く。この内在的な信号を使って構造化された疎性を直接導入できるか調べるため、ゲート活性化スコアに基づいてFFNへ明示的に疎性を導入する単純な変更、AbsTopk-GLUを定義する。

$$\mathrm{AbsTopk\text{-}GLU}(x)=(\mathrm{AbsTopK}(\mathrm{Swish}(xW_{gate}),k))\odot(xW_{up}). \tag{2}$$

ここで$\mathrm{AbsTopK}(v,k)$は、$|v|$の上位$k$個の値を残し、それ以外をゼロにする。これにより、各トークンについて最も強く活性化されたニューロンだけが保持される。

**Takeaway 1：GLUのゲート機構は、密なLLM内部に自然な構造的疎性を生む。**表1に示すように、ニューロンの50%だけを活性化するAbsTopK-GLUは、Wanda（Sun et al., 2023）やSparseGPT（Frantar & Alistarh, 2023）などの代表的な非構造的プルーニング手法を一貫して大幅に上回る。この性能は、密なモデルに存在する内在的な構造的疎性を示し、GLUのゲートスコアを活性化パターン制御に用いることの妥当性を裏付ける。

表1．疎性50%におけるAbsTopk-GLUと非構造的プルーニングの性能比較。括弧内は評価時のショット数であり、注記がないものはゼロショットである。

| 手法 | MMLU(5) | HellaSwag(10) | ARC-e | ARC-C(25) | PiQA | 平均 |
|---|---:|---:|---:|---:|---:|---:|
| LLaMA3-8B Dense | 65.3 | 82.1 | 77.9 | 57.9 | 80.8 | 72.8 |
| Wanda | 55.8 | 75.0 | 72.0 | 51.3 | 77.3 | 66.3 |
| SparseGPT | 57.4 | 75.5 | 72.0 | 50.3 | 78.1 | 66.7 |
| Magnitude | 48.3 | 41.9 | 53.3 | 38.7 | 65.8 | 49.6 |
| AbsTopk-GLU | 58.8 | 79.4 | 72.4 | 51.6 | 78.6 | 68.2 |
| Qwen2.5-7B Dense | 74.2 | 80.3 | 77.8 | 63.8 | 80.0 | 75.2 |
| Wanda | 69.2 | 74.1 | 75.5 | 55.5 | 78.1 | 70.5 |
| SparseGPT | 69.8 | 75.9 | 74.7 | 56.7 | 78.3 | 71.1 |
| Magnitude | 64.2 | 49.5 | 46.3 | 33.6 | 63.8 | 51.5 |
| AbsTopk-GLU | 70.8 | 78.6 | 76.6 | 59.7 | 78.9 | 72.9 |

この細粒度のエキスパート分割は、各ニューロンを個別のエキスパートとして扱うため、ハードウェア実装には現実的でない。理論上のFLOPsは減るものの、TopK選択のオーバーヘッドと分散メモリアクセスが実用化を妨げる。それでもこの実験は、GLUゲート信号によるニューロン単位の活性化制御を、より粗粒度でハードウェアに適したブロックの制御へ拡張できるという重要な知見を示す。これが、ニューロンをグループ化して大きなエキスパートを作る本研究の動機である。

### 2.2.2 GLU活性化パターンによる汎用ニューロンと特化ニューロンの識別

GLU活性化パターンの構造を調べるため、Flan-v2コレクションの48タスク、10タスククラスタについて、各タスク5個のfew-shotサンプルから活性化パターンを記録して分析した（詳細は付録H）。図1は、次の観察につながる二つの相補的な現象を明確に示している。

**Takeaway 2：異なるタスクで一貫して活性化される、汎用的に重要なニューロンの中核集合が存在する。**図1a)に示すように、タスク領域に関係なく高い活性化を示すニューロンの一貫した部分集合が観察された。これらのニューロンはタスクに依存しない知識を符号化していると仮説を立てる。この観察から、タスク非依存の知識を捉えて利用する共有エキスパートをMoEに設けることで、パラメータ効率と全体性能を高められると考えられる。

**Takeaway 3：特化ニューロンはタスク固有の共活性化パターンを示す。**図1b)の活性化パターン類似度ヒートマップは、切り詰めて正規化したニューロン活性化から作成したものであり、明確なブロック対角構造を示す。これは、意味的に関連するタスクが、特化ニューロンの活性化パターンでも高い類似性を持つことを意味する。すなわちニューロンは、自然に共活性化する機能的なクラスタを形成している。

### 3. 手法

この節では、密なLLMを効率的なMoEへ変換する学習不要のフレームワークExpertWeaverを導入する。

図2．層をまたいだニューロンの変動係数。

Takeaway 2と3を合わせると、密なモデルからMoEへ変換するための設計図が得られる。すなわち、汎用ニューロンから共有エキスパートを作り、特化ニューロンを共活性化パターンに基づいてクラスタリングしてルーティング・エキスパートを作るのである。タスクの意味に沿って活性化クラスタが形成されるため、多タスクの活性化パターンに基づくクラスタリングによって、ルーティング・エキスパートを構築する原理的な方法が得られる。

### 2.2.3 層を考慮したエキスパート割り当て

§2.2.2で汎用ニューロンと特化ニューロンを発見したことを踏まえ、次に、汎用ニューロンと特化ニューロンの比率が層によってどのように変化するかを調べる。この層ごとの挙動を定量化するため、各ニューロンの活性化の一貫性を変動係数（CV）で測定する。CVは標準偏差を平均で割った値である（Abdi, 2010）。

$$CV(a_j)=\frac{\sigma_j}{\mu_j+\epsilon}. \tag{3}$$

ここで$\mu_j=\mathbb{E}_{t\in T}[\bar a_{j,t}]$、$\sigma_j=\mathrm{STD}_{t\in T}[\bar a_{j,t}]$は、較正セット$D_{calib}$に含まれるタスク全体での、ニューロンの平均絶対活性化$\bar a_{j,t}$の平均と標準偏差である。CVが低い場合は一貫して活性化する汎用ニューロンを示し、CVが高い場合は選択的に活性化する特化ニューロンを示す。

**Takeaway 4：層ごとにニューロンの特化度は異なる。**図2は、境界の層（浅い層と深い層）ではCVが一貫して低く、汎用ニューロンが多いことを示す。一方、中間層ではCVが広く分布し、高度に特化したニューロンが多い。この層ごとの異質性のため、全層に一様なMoE変換を行うのではなく、層固有のエキスパート構成が必要になる。

## 3. 手法

ExpertWeaverは、次の三段階で密なLLMを効率的なMoEアーキテクチャへ変換する。(a) 各層について、多タスクのGLU活性化パターンを取得する。(b) その活性化のCVを利用し、共有エキスパートとルーティング・エキスパートの層固有の比率を決める。(c) 汎用ニューロンから共有エキスパートを作り、特化ニューロンをクラスタリングしてルーティング・エキスパートを作り、MoEルーターを構築する。

### 3.1 多タスクのゲート活性化パターンの取得

標準的なSwiGLUベースのFFN層（Shazeer, 2020）は、三つの重み行列$W_{gate}\in\mathbb{R}^{d_{model}\times d_{ffn}}$、$W_{up}\in\mathbb{R}^{d_{model}\times d_{ffn}}$、$W_{down}\in\mathbb{R}^{d_{ffn}\times d_{model}}$から成る。ここで$d_{model}$はモデルの隠れ次元、$d_{ffn}$は中間層の次元である。表記を簡潔にするため層の添字は省略する。ニューロン$j\in\{1,\ldots,d_{ffn}\}$について、ニューロン・スライス$s_j$を次のように定義する。

$$s_j=((W_{gate})_{:,j},(W_{up})_{:,j},(W_{down})_{j,:}).$$

ニューロン・スライス同士は独立している。ExpertWeaverは、ニューロン・スライスの集合$\{s_j\}_{j=1}^{d_{ffn}}$を異なるエキスパート群へ分割する。

§2.2で示したように、GLUの活性化パターンはモデルに内在する構造について豊富な情報を与える。そこでFlan-v2の多タスク較正セット$D_{calib}$（42タスク、各5サンプル）を使い、頑健な活性化信号を取得する。$D_{calib}$の各サンプル$j$について、ニューロン$i$のトークン平均ゲート活性化を$a_{ij}=\mathrm{mean}(\mathrm{Swish}(x_jW_{gate}))_i$として計算する。ここで$x_j$はサンプル内の全トークンを表す。これにより、ニューロン$i$の活性化プロファイル$a_i=[a_{i1},a_{i2},\ldots,a_{iM}]$が得られる。$M=|D_{calib}|$である。全プロファイルを行列$A=[a_1,a_2,\ldots,a_{d_{ffn}}]$に集め、以後の処理の主要な信号として使う。

### 3.2 層を考慮したエキスパート割り当て

汎用ニューロンと特化ニューロンの比率が層によって変わるという観察（Takeaway 4）に基づき、各層の共有エキスパート比率を決める層適応型の割り当て戦略を提案する。

#### 層ごとのニューロン特化度の定量化

まず、各層$\ell$の機能的特化度を定量化する。ニューロンの活性化パターン$A_\ell$に基づいてCVを計算する。CVが高いニューロンは特定の入力でだけ活性化するため、高度に特化している。次に、その層における特化比率$r_\ell$を、CVが特化閾値$\tau$を超えるニューロンの割合として定義する。

$$r_\ell=\frac{1}{d_{ffn}}\sum_{j=1}^{d_{ffn}}I[CV_j>\tau]. \tag{4}$$

ここで$I[\cdot]$は指示関数、$\tau$は特化度の閾値である。

#### 共有エキスパートの大きさの動的割り当て

次に、この特化比率を使って、その層のニューロンのうち共有エキスパートへ割り当てる割合$\alpha_\ell$を決める。基本原理は、特化度の高い層（$r_\ell$が大きい層）ほど共有エキスパートを小さくすることである。$\alpha_\ell$は線形写像によって計算する。

$$\alpha_\ell=\alpha_{max}-(\alpha_{max}-\alpha_{min})\cdot r_\ell. \tag{5}$$

ここで$\alpha_{min}$と$\alpha_{max}$は、共有エキスパートに割り当てるニューロン比率の上下限を定める。フレームワークでは、合計$d_{ffn}$個のニューロンを$N_e$個のエキスパートへ分配し、各エキスパートに固定容量$d_{expert}=d_{ffn}/N_e$個のニューロンを割り当てる。共有ニューロンから作れる完全なエキスパート数によって、共有エキスパート数$N_{se,\ell}$を次のように決める。

$$d_{s,\ell}=\mathrm{round}(\alpha_\ell\cdot d_{ffn}),\quad N_{se,\ell}=\mathrm{round}(d_{s,\ell}/d_{expert}). \tag{6}$$

残りの$N_{re,\ell}=N_e-N_{se,\ell}$個をルーティング・エキスパートとする。各入力トークンでは、必要な疎性に従って合計$k$個のエキスパートを活性化する。これは全ての$N_{se,\ell}$個の共有エキスパートと、ルーターが選ぶ上位$k-N_{se,\ell}$個のルーティング・エキスパートから成る。

### 3.3 MoE層の構築

共有エキスパートの層ごとの割り当て$N_{se,\ell}$と、ルーティング・エキスパートの割り当て$N_{re,\ell}$が決まったら、FFNのニューロンを共有部分とルーティング部分へ分割する。

#### 共有エキスパートの構築

共有エキスパートは、最も汎用的に活性化するニューロンから作る。絶対値付き平均活性化スコアが最も高い$d_{expert}\cdot N_{se,\ell}$個のニューロンを選び、共有ニューロンプール$I_s$を形成する。これらのニューロン・スライスを連結し、一つの統合された共有エキスパートの重み行列を作る。

$$W_{gate/up}^{(s)}=\mathrm{CONCAT}((W_{gate/up})_{:,j})_{j\in I_s} \tag{7}$$

$$W_{down}^{(s)}=\mathrm{CONCAT}((W_{down})_{j,:})_{j\in I_s}. \tag{8}$$

この共有エキスパートは常に活性化され、異なる文脈に共通する知識を捉えて統合する。

#### ルーティング・エキスパートの構築

特化ニューロンのプール$I_{-s}$を形成する残りの$N_{re,\ell}\cdot d_{expert}$個のニューロンを、$N_{re,\ell}$個のルーティング・エキスパートへ分割する。同じエキスパートで頻繁に共活性化するニューロンをまとめるため、活性化パターンベクトル$\{a_i\}_{i\in I_{-s}}$に対して、バランス型K-Meansクラスタリング（Malinen & Fränti, 2014）を行う（バランス型K-Meansの詳細は付録I）。この処理により、特化ニューロンは$N_{re,\ell}$個の互いに重ならないクラスタ$\{C_1,\ldots,C_{N_{re,\ell}}\}$に分割される。各クラスタは正確に$d_{expert}$個のニューロンを含み、一つのルーティング・エキスパートに対応する。$i$番目のエキスパートの重み行列は、クラスタ$C_i$内の全ニューロンの重みを連結して作る。

$$W_{gate/up}^{(i)}=\mathrm{CONCAT}(W_{gate/up,j})_{j\in C_i} \tag{9}$$

$$W_{down}^{(i)}=\mathrm{CONCAT}(W_{down,j})_{j\in C_i}. \tag{10}$$

#### MoEルーターの構築

前段階で得たクラスタ構造を利用し、MoEルーターを学習不要で構築する。重要な点は、各ニューロンの元のゲートベクトル$(W_{gate})_{:,j}$が、そのニューロンの活性化パターンを制御していることである。したがって、各エキスパート・クラスタ内のこれらのベクトルのセントロイドは、そのエキスパートの代表的な活性化挙動を自然に表す。そこで、$W_{router}\in\mathbb{R}^{d_{model}\times N_{e,\ell}}$を作り、各ルーティング・エキスパート$i$について、クラスタ内の全ニューロンのゲートベクトルを平均して代表ゲートベクトルを計算する。

クラスタに割り当てられたゲートベクトルを平均し、次のようにルーターを構成する。

$$\bar w_{gate}^{(i)}=\frac{1}{|C_i|}\sum_{j\in C_i}(W_{gate})_{:,j},\quad W_{router}=\left[\bar w_{gate}^{(1)},\ldots,\bar w_{gate}^{(N_{e,\ell})}\right]. \tag{11}$$

元のゲート重みを直接使うことで、学習なしにルーターを構築しながら、事前学習中に獲得されたニューロン活性化パターンを保つ。

### 3.4 実運用におけるExpertWeaver：動的構造的プルーニングとダウンサイクリング

#### 3.4.1 学習不要の動的構造的プルーナー

低い疎性レベルで追加学習なしに動作させる場合、ExpertWeaverは動的構造的プルーニング手法とみなせる。任意の入力$x$に対し、ルーターは再構成されたルーターのロジットに基づいて上位$k$個のルーティング・エキスパートを選択する。$T(x)=\mathrm{TopK}(xW_g)$とする。最終出力は、常に活性化する共有エキスパートと、選択されたルーティング・エキスパートの出力の直接和である。

$$y=E_{shared}(x)+\sum_{i\in T(x)}E_i(x). \tag{12}$$

エキスパート出力を重み付きで組み合わせるのではなく、ここでのゲートは構造的プルーニング機構として働き、順伝播から除外するニューロンを動的に選択する点に注意されたい。活性化された各エキスパート内部ではGLU機構が正確な活性化値を引き続き決めるため、事前学習済みモデルの元のニューロン単位の計算の整合性が保たれる。

#### 3.4.2 密なLLMのMoEへのダウンサイクリング

モデル・ダウンサイクリングは、大規模な事前学習済み密モデルを計算効率の高いMoEへ変換することを目指す。密なモデルの重みを優れた初期値として利用し、ゼロから学習する莫大なコストを避ける。ExpertWeaverでは、密なモデルを疎なMoEアーキテクチャへ変換した後、継続事前学習（CPT）によってさらに最適化する。

**初期化。** ダウンサイクリングでは疎性が高く、下流タスクの要件に応じて構成されることが多い。そのため、全層で固定かつ一様な共有エキスパート比率を使う。これにより、後続の学習段階に向けて頑健で構造化された出発点を得る。

**継続事前学習。** CPTでは、最適化の柔軟性を高め、エキスパートが異なる特化を学習できるよう、標準的なsoftmaxルーターへ切り替える。softmaxルーターはゲート重み$g(x)=\mathrm{Softmax}(xW_{router})$を計算する。層の出力は、共有エキスパートとTop-Kのルーティング・エキスパートの重み付き組み合わせとなる。

$$y=E_{shared}(x)+\sum_{i\in\mathrm{TopK}(g(x))}g(x)_i\cdot E_i(x). \tag{13}$$

全損失は、次トークン予測損失$L_{NTP}$と、トークンをエキスパート間に均等に分配するための補助負荷分散損失$L_{LB}$の二つから成る。

$$L_{total}=L_{NTP}+\lambda L_{LB};\quad L_{LB}=N_{e,\ell}\cdot\sum_{i=1}^{N_{e,\ell}}f_i\cdot P_i. \tag{14}$$

ここで$f_i$はエキスパート$i$へルーティングされたトークンの割合、$P_i$はバッチ内の平均ルーター確率である。

## 4. 実験

ExpertWeaverを二つの場面で評価する。第一に、低疎性設定での学習不要の動的構造的プルーナーとして、§4.1で学習不要のベースラインと比較する。第二に、高疎性設定でのモデル・ダウンサイクリングの初期化戦略として、§4.2で同程度の規模のベースラインと比較し、得られたMoEの利点を示す。

### 4.1 動的構造的プルーニングのためのExpertWeaver

**実験設定。** Qwen2.5-7BとLlama3-8Bを用い、ExpertWeaverをFLAP（An et al., 2024）、LLM-Pruner（Ma et al., 2023）、CMoE（Pei et al., 2025）などの学習不要ベースラインと比較する。CMoEはLlama系列モデルにしか実装されていないため、CMoEとの比較はLlama3-8Bだけで行う。先行研究の設定（Ma et al., 2023）に従い、全手法の目標疎性を25%とする。MMLU、HellaSwag、ARC-e、ARC-c、PIQAという広く使われる五つのベンチマークで性能を評価する。ExpertWeaverには、$\alpha_{min}=0.2$、$\alpha_{max}=0.7$、特化閾値$\tau=0.6$、エキスパート粒度64というデフォルトのハイパーパラメータを使う。共有エキスパートとルーティング・エキスパートの具体的な層別割り当ては付録N、ハイパーパラメータの詳細なアブレーションは付録Bに示す。多様なタスクの活性化パターンを捉える較正セットとして、10種類のタスククラスタにまたがるFlan-v2の42タスクから各5例をサンプリングする（詳細は付録H）。

表3．ダウンサイクリング性能の比較。太字は最良、下線は次点を示す。灰色で示されるモデル（Qwen2.5-{7B, 3B, 1.5B}、Llama-3.2-3B、OLMoE）は、より多くのデータで学習されたモデルであるため参考として掲載する。ExpertWeaverモデル（OLMoベースおよびQwen2.5ベース）は200Bトークンで学習した。OLMoベース版の活性化パラメータは1B（7B-A1B）、Qwen2.5ベース版は3.5B（7B-A3.5B）である。OLMoEについては500Bトークンのチェックポイント（*）と比較する。OpenMoEの空欄は結果が利用できないことを示す。

| モデル | MMLU(5) | HellaSwag(10) | ARC-e | ARC-c(25) | PIQA | WinoGrande | LogiQA | SciQ | 平均 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-7B | 74.1 | 80.2 | 77.5 | 63.7 | 79.7 | 73.2 | 36.4 | 95.2 | 72.5 |
| Qwen2.5-3B | 65.6 | 74.6 | 73.9 | 56.5 | 78.8 | 68.1 | 33.5 | 95.2 | 68.3 |
| Qwen2.5-1.5B | 60.9 | 68.0 | 72.5 | 54.9 | 75.9 | 63.8 | 31.9 | 93.4 | 65.2 |
| Llama-3.2-3B | 56.1 | 76.4 | 71.6 | 50.5 | 77.4 | 69.9 | 30.6 | 92.7 | 65.7 |
| OLMo-7B | 30.7 | 77.1 | 68.7 | 45.1 | 79.6 | 66.5 | 27.5 | 88.6 | 60.8 |
| OPT-2.7B | 25.8 | 61.4 | 54.4 | 34.0 | 74.8 | 60.8 | 25.8 | 78.9 | 52.0 |
| Pythia-2.8B | 26.8 | 60.7 | 58.8 | 36.7 | 73.6 | 59.6 | 28.1 | 83.2 | 53.4 |
| INCITE-Base-3B | 27.2 | 64.7 | 61.7 | 40.3 | 73.9 | 63.5 | 27.5 | 85.6 | 55.6 |
| Open-LLaMA-3B-v2 | 26.8 | 71.4 | 63.3 | 36.9 | 77.9 | 63.1 | 28.1 | 88.0 | 57.3 |
| Sheared-LLaMA-2.7B | 27.3 | 71.0 | 63.3 | 40.1 | 76.9 | 65.0 | 28.3 | 87.5 | 57.6 |
| Gemma-2-2b | 53.0 | 69.0 | 36.9 | 41.6 | 67.5 | 51.9 | 22.7 | 75.8 | 53.7 |
| SmolLM2-1.7B | 50.4 | 72.6 | 73.4 | 53.2 | 76.0 | 65.8 | 30.1 | 84.3 | 63.2 |
| LLaMA-MoE-v1-3.5B | 26.8 | 73.3 | 65.6 | 44.2 | 77.9 | 65.5 | 29.7 | 87.6 | 58.8 |
| OpenMoE-3B-9B | — | 56.5 | 50.6 | 33.3 | 65.7 | 51.9 | — | — | — |
| OLMoE-1B-7B | 53.8 | 79.6 | 76.3 | 55.6 | 80.1 | 68.4 | 29.3 | 94.9 | 67.2 |
| OLMoE-1B-7B* | 28.4 | 70.2 | 71.0 | 43.9 | 77.1 | 63.5 | 28.7 | 88.5 | 58.9 |
| LLaMA-MoE-v2-3.5B | 40.9 | 53.7 | 57.0 | 40.2 | 67.9 | 56.1 | 30.7 | 88.8 | 54.4 |
| ExpertWeaver-E64-A14-S2 (OLMo-7B) | 45.0 | 61.2 | 69.3 | 38.8 | 74.5 | 62.1 | 28.5 | 91.8 | 58.9 |
| ExpertWeaver-E64-A14-S2 (Qwen2.5-7B) | 45.6 | 73.7 | 72.4 | 56.3 | 78.0 | 65.3 | 29.0 | 87.7 | 63.5 |

図3．ExpertWeaverフレームワーク。(a) MLP層のGLUは三つの重み行列を含み、同じ色は対応するニューロン・スライスを表す。(b) 多タスク較正データセットでニューロン活性化パターンを取得する。(c) CVを計算して共有エキスパートとルーティング・エキスパートの予算を決める。(d) 活性化パターンに従ってニューロンをクラスタリングし、一つの共有エキスパートと複数のルーティング・エキスパートを作る。

**主な結果。** 表2は、疎性25%におけるExpertWeaverと他の学習不要構造的プルーニング・ベースラインの比較をまとめたものである。主な発見は次のとおりである。(1) ExpertWeaverは全ての競合手法を一貫して上回り、LLaMA3-8BではCMoEに対して相対5.6%、Qwen2.5-7BではLLM-Prunerに対して3.1%の改善を達成した。(2) dense-to-MoE手法は、FLAPやLLM-Prunerのような静的プルーニングを大幅に上回る。パラメータを永久に削除する代わりに、入力文脈に応じてパラメータ部分集合を選択的に活性化するため、避けられない知識損失を回避できる。(3) ExpertWeaverがCMoEを上回る理由は三つある。第一に、GLU活性化パターンを包括的に分析することで、変換中も元の活性化構造を維持する。第二に、多タスク較正セットが多様なニューロン共活性化パターンを捉え、より良いエキスパートクラスタリングを可能にする（アブレーションは付録G）。第三に、層固有の構成によって、各層の特化特性に合わせた適応的なMoE構築が可能になる。さらに、より複雑なタスクで他の構造的プルーニング手法と比較し、推論モデル上でも評価した結果を付録Eに示す。

表2．疎性25%における構造的プルーニング手法の比較。

| 手法 | MMLU(5) | HellaSwag(10) | ARC-e | ARC-c | PiQA | 平均 |
|---|---:|---:|---:|---:|---:|---:|
| LLaMA3-8B Dense | 65.3 | 82.1 | 77.9 | 57.9 | 80.8 | 72.8 |
| LLM-Pruner | 24.2 | 51.3 | 58.9 | 32.4 | 74.4 | 48.2 |
| FLAP | 33.4 | 48.0 | 50.0 | 29.3 | 68.3 | 45.8 |
| CMoE | 41.6 | 65.9 | 63.1 | 41.5 | 73.9 | 57.2 |
| ExpertWeaver | 47.0 | 69.8 | 64.4 | 44.3 | 76.3 | 60.4 |
| Qwen2.5-7B Dense | 74.2 | 80.3 | 77.8 | 63.8 | 80.0 | 75.2 |
| LLM-Pruner | 55.9 | 72.2 | 71.0 | 49.1 | 77.0 | 65.0 |
| FLAP | 54.7 | 58.5 | 67.3 | 42.2 | 70.8 | 58.7 |
| ExpertWeaver | 61.6 | 72.3 | 71.5 | 53.5 | 76.3 | 67.0 |

### 4.2 ExpertWeaverによるダウンサイクリング

**実験設定。** ExpertWeaverをダウンサイクリング戦略として評価するため、二つのMoE変種を初期化する。第一のExpertWeaver（Qwen2.5-7B）は密なQwen2.5-7Bから初期化し、MLP層を62個のルーティング・エキスパートと2個の共有エキスパートに分割する。各トークンでは14個のルーティング・エキスパートと2個の共有エキスパートを活性化する（総パラメータ7B、活性化パラメータ3.5B）。このモデルをFineWeb-Eduデータセット（Penedo et al., 2024）の200BトークンでCPTする。OLMoEベースラインとの公平な比較のため、第二のExpertWeaver（OLMo-7B）はOLMo-7Bベースモデルから初期化し、OLMoEと同じデータセットで200Bトークン継続事前学習する。両モデルをMMLU、HellaSwag、ARC-e、ARC-c、PIQA、WinoGrande、LogiQA、SciQで評価し、パラメータ数と学習予算が近い密モデルおよびMoEベースラインと比較する（詳細は付録J）。CPT後、Llama-MoE-V2（Qu et al., 2024）に似た二段階の教師ありファインチューニング（SFT）を行い、まず一般的な会話能力、次にコードと数学能力に重点を置く。得られた指示チューニングモデルをMMLU、ARC-c、GSM8K、HumanEval、IFEvalで他の指示チューニング・ベースラインと比較する（詳細は付録L、結果は付録C）。

**主な結果。** 表3は、パラメータ数と学習予算が同程度のモデルとExpertWeaverを比較した結果である。(1) 学習予算とパラメータ数が近いモデルの中で、ExpertWeaver（Qwen2.5-7B）は平均63.5で最良の性能を達成し、最も強いMoEベースラインOLMoE-1B-7B*を4.6ポイント上回った。(2) 公平な比較のためOLMo-7Bもダウンサイクルした。200Bトークンだけの継続事前学習後、ExpertWeaver（OLMo-7B）は58.9を達成した。これは元のOLMo-7Bの性能60.5の97.35%に相当する。元モデルに近いだけでなく、500Bトークンをゼロから学習したOLMoE-1B-7B*と同等である。スコアの小さな差には、OLMoE-MixよりOLMoのDolmaデータセットの品質が低い可能性も関係するが、この結果はExpertWeaverの有効性を強く示す。(3) 密な対応モデルと比較すると、ExpertWeaver（Qwen2.5-7B）はMLPパラメータの4分の1だけを活性化しながら、元のQwen2.5-7Bの87.6%の性能を保つ。さらに、活性化パラメータ数が近いQwen2.5-3BとLlama-3.2-3Bに対して、それぞれ93.0%と96.7%の性能を達成した。これらのモデルはそれぞれ18T、9Tトークンという大規模データセットで学習されているため、この結果はダウンサイクリングが高性能MoEを作る効率的な道であることを示す。

図4．ダウンサイクリング、アップサイクリング、ゼロからの学習の比較。同じOLMoEモデル構成を用いて、学習損失、評価損失、下流タスク性能を三つのMoE初期化方式で比較した。

図5．異なるMoE初期化戦略における学習損失の比較。

**他のMoE初期化手法との比較。** 図5は最初の5,000ステップ（約20Bトークン）の学習損失を示し、ExpertWeaverをダウンサイクリング戦略として使う有効性を明らかにする。ランダム初期化とLlama-MoE初期化の両方に比べ、提案手法は学習全体を通じて一貫して低い損失を示す。これはExpertWeaverがMoEにより有効な出発点を与え、収束を速め、最終性能を高めることを示す。緑色の損失曲線と他の曲線との持続的な差は、変換中に密モデルの基礎的知識をよりよく保存できていることを裏付ける。

**ダウンサイクリング、アップサイクリング、ゼロからの学習の直接比較。** より公平に比較してExpertWeaverの有効性を示すため、1Tトークンで事前学習されたOLMo-1.3Bをダウンサイクリングの基礎モデルに使った。直接比較するため、疎なアップサイクリング（Muennighoff et al., 2024）も行い、575MのOLMoモデルを1Tトークンで学習して、活性化パラメータ676Mの1.3B OLMoEモデルを作った。ダウンサイクリングとアップサイクリングを、ゼロから学習したベースラインと比較した。詳細なモデル構成は付録Kに示す。

図4に示す学習損失、評価損失、下流タスク性能から、いくつかの知見が得られる。初期には、ダウンサイクリングとアップサイクリングの両方が、ゼロから学習するベースラインより速く収束する。しかし長期間学習すると、アップサイクルされたモデルの性能はゼロから学習したベースラインへ戻っていく。対照的に、ExpertWeaverによるダウンサイクリングは常に最高の性能と収束を示し、学習全体で優位性を保つ。大きなモデルから豊かな特徴空間を受け継ぎ、パラメータ複製を避けるため、ダウンサイクリングは、局所最適に陥る危険のある重み複製型アップサイクリングより多様なエキスパートと高い最適化上限を提供する。

### 5. 結論

本研究では、密なモデルを高性能MoEへ変換する問題を調べた。GLUの活性化パターンは、潜在的なニューロンの特化を見つける豊富な情報源であり、自然に効果的なエキスパート構築を可能にすることを示した。この観察に基づき、活性化パターンに従ってニューロンを層ごとに共有エキスパートとルーティング・エキスパートへ分割する、学習不要の手法ExpertWeaverを導入した。実験により、ExpertWeaverは推論効率のためのゼロショット・プルーニングでも、モデル・ダウンサイクリングのためのMoE初期化でも、既存手法を大幅に上回ることを示した。

## References

Abdi, H. Coefficient of variation. Encyclopedia of research design, 1(5):169–171, 2010.

Allal, L. B., Lozhkov, A., Bakouch, E., Blá́zquez, G. M., Penedo, G., Tunstall, L., Marafioti, A., Kydlí́ček, H., Lajarín, A. P., Srivastav, V., et al. Smollm2: When smol goes big–data-centric training of a small language model. arXiv preprint arXiv:2502.02737, 2025.

An, Y., Zhao, X., Yu, T., Tang, M., and Wang, J. Fluctuation-based adaptive structured pruning for large language models. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pp. 10865–10873, 2024.

Biderman, S., Schoelkopf, H., Anthony, Q. G., Bradley, H., O’Brien, K., Hallahan, E., Khan, M. A., Purohit, S., Prashanth, U. S., Raff, E., et al. Pythia: A suite for analyzing large language models across training and scaling. In International Conference on Machine Learning, pp. 2397–2430. PMLR, 2023.

Bisk, Y., Zellers, R., Gao, J., Choi, Y., et al. Piqa: Reasoning about physical commonsense in natural language. In Proceedings of the AAAI conference on artificial intelligence, volume 34, pp. 7432–7439, 2020.

Chen, L., Li, J., Dong, X., Zhang, P., He, C., Wang, J., Zhao, F., and Lin, D. Sharegpt4v: Improving large multimodal models with better captions. In European Conference on Computer Vision, pp. 370–387. Springer, 2024.

Chen, M., Tworek, J., Jun, H., Yuan, Q., de Oliveira Pinto, H. P., Kaplan, J., Edwards, H., Burda, Y., Joseph, N., Brockman, G., Ray, A., Puri, R., Krueger, G., Petrov, M., Khlaaf, H., Sastry, G., Mishkin, P., Chan, B., Gray, S., Ryder, N., Pavlov, M., Power, A., Kaiser, L., Bavarian, M., Winter, C., Tillet, P., Such, F. P., Cummings, D., Plappert, M., Chantzis, F., Barnes, E., Herbert-Voss, A., Guss, W. H., Nichol, A., Paino, A., Tezak, N., Tang, J., Babuschkin, I., Balaji, S., Jain, S., Saunders, W., Hesse, C., Carr, A. N., Leike, A., Achiam, J., Misra, V., Morikawa, E., Radford, A., Knight, M., Brundage, M., Murati, M., Mayer, K., Welinder, P., McGrew, B., Amodei, D., McCandlish, S., Sutskever, I., and Zaremba, W. Evaluating large language models trained on code. 2021.

Chung, H. W., Hou, L., Longpre, S., Zoph, B., Tay, Y., Fedus, W., Li, Y., Wang, X., Dehghani, M., Brahma, S., et al. Scaling instruction-finetuned language models. Journal of Machine Learning Research, 25(70):1–53, 2024.

Clark, P., Cowhey, I., Etzioni, O., Khot, T., Sabharwal, A., Schoenick, C., and Tafjord, O. Think you have solved question answering? try arc, the ai2 reasoning challenge. arXiv preprint arXiv:1803.05457, 2018.

Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., Plappert, M., Tworek, J., Hilton, J., Nakano, R., et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

Frantar, E. and Alistarh, D. Sparsegpt: Massive language models can be accurately pruned in one-shot. In International conference on machine learning, pp. 10323–10337. PMLR, 2023.

Gao, S., Hua, T., Shirkavand, R., Lin, C.-H., Tang, Z., Li, Z., Yuan, L., Li, F., Zhang, Z., Ganjdanesh, A., et al. Tomoe: Converting dense large language models to mixture-of-experts through dynamic structural pruning. arXiv preprint arXiv:2501.15316, 2025.

Geng, X. and Liu, H. Openllama: An open reproduction of llama, 2023.

He, E., Khattar, A., Prenger, R., Korthikanti, V., Yan, Z., Liu, T., Fan, S., Aithal, A., Shoeybi, M., and Catanzaro, B. Upcycling large language models into mixture of experts. arXiv preprint arXiv:2410.07524, 2024.

Hendrycks, D., Burns, C., Basart, S., Zou, A., Mazeika, M., Song, D., and Steinhardt, J. Measuring massive multitask language understanding. arXiv preprint arXiv:2009.03300, 2020.

Komatsuzaki, A., Puigcerver, J., Lee-Thorp, J., Ruiz, C. R., Mustafa, B., Ainslie, J., Tay, Y., Dehghani, M., and Houlsby, N. Sparse upcycling: Training mixture-of-experts from dense checkpoints. arXiv preprint arXiv:2212.05055, 2022.

Li, J., Du, L., Zhao, H., Zhang, B.-w., Wang, L., Gao, B., Liu, G., and Lin, Y. Infinity instruct: Scaling instruction selection and synthesis to enhance language models. arXiv preprint arXiv:2506.11116, 2025.

Liu, J., Cui, L., Liu, H., Huang, D., Wang, Y., and Zhang, Y. Logiqa: A challenge dataset for machine reading comprehension with logical reasoning. arXiv preprint arXiv:2007.08124, 2020.

Ma, X., Fang, G., and Wang, X. Llm-pruner: On the structural pruning of large language models. In Advances in Neural Information Processing Systems, 2023.

Malinen, M. I. and Fränti, P. Balanced k-means for clustering. In Joint IAPR international workshops on statistical techniques in pattern recognition (SPR) and structural and syntactic pattern recognition (SSPR), pp. 32–41. Springer, 2014.

Muennighoff, N., Soldaini, L., Groeneveld, D., Lo, K., Morrison, J., Min, S., Shi, W., Walsh, P., Tafjord, O., Lambert, N., et al. Olmoe: Open mixture-of-experts language models. arXiv preprint arXiv:2409.02060, 2024.

Nakamura, T., Akiba, T., Fujii, K., Oda, Y., Yokota, R., and Suzuki, J. Drop-upcycling: Training sparse mixture of experts with partial re-initialization. arXiv preprint arXiv:2502.19261, 2025.

Nishu, K., Mehta, S., Abnar, S., Farajtabar, M., Horton, M., Najibi, M., Nabi, M., Cho, M., and Naik, D. From dense to dynamic: Token-difficulty driven moefication of pretrained llms. arXiv preprint arXiv:2502.12325, 2025.

Pei, Z., Zou, L., Zhen, H.-L., Yu, X., Liu, W., Pan, S. J., Yuan, M., and Yu, B. Cmoe: Converting mixture-of-experts from dense to accelerate llm inference. arXiv preprint arXiv:2502.04416, 2025.

Penedo, G., Kydlí́ček, H., Lozhkov, A., Mitchell, M., Raffel, C. A., Von Werra, L., Wolf, T., et al. The fineweb datasets: Decanting the web for the finest text data at scale. Advances in Neural Information Processing Systems, 37:30811–30849, 2024.

Qu, X., Dong, D., Hu, X., Zhu, T., Sun, W., and Cheng, Y. Llama-moe v2: Exploring sparsity of llama from perspective of mixture-of-experts with post-training. arXiv preprint arXiv:2411.15708, 2024.

Sakaguchi, K., Bras, R. L., Bhagavatula, C., and Choi, Y. Winogrande: An adversarial winograd schema challenge at scale. Communications of the ACM, 64(9):99–106, 2021.

Shazeer, N. Glu variants improve transformer. arXiv preprint arXiv:2002.05202, 2020.

Sun, M., Liu, Z., Bair, A., and Kolter, J. Z. A simple and effective pruning approach for large language models. arXiv preprint arXiv:2306.11695, 2023.

Team, G., Riviere, M., Pathak, S., Sessa, P. G., Hardin, C., Bhupatiraju, S., Hussenot, L., Mesnard, T., Shahriari, B., Ramé, A., et al. Gemma 2: Improving open language models at a practical size. arXiv preprint arXiv:2408.00118, 2024a.

Team, Q. et al. Qwen2 technical report. arXiv preprint arXiv:2407.10671, 2(3), 2024b.

Teknium. Openhermes 2.5: An open dataset of synthetic data for generalist llm assistants, 2023. URL https://huggingface.co/datasets/teknium/OpenHermes-2.5.

Weber, M., Fu, D., Anthony, Q., Oren, Y., Adams, S., Alexandrov, A., Lyu, X., Nguyen, H., Yao, X., Adams, V., et al. Redpajama: an open dataset for training large language models. Advances in neural information processing systems, 37:116462–116492, 2024.

Welbl, J., Liu, N. F., and Gardner, M. Crowdsourcing multiple choice science questions. arXiv preprint arXiv:1707.06209, 2017.

Xia, M., Gao, T., Zeng, Z., and Chen, D. Sheared llama: Accelerating language model pre-training via structured pruning. arXiv preprint arXiv:2310.06694, 2023.

Xue, F., Zheng, Z., Fu, Y., Ni, J., Zheng, Z., Zhou, W., and You, Y. Openmoe: An early effort on open mixture-of-experts language models. arXiv preprint arXiv:2402.01739, 2024.

Yu, L., Jiang, W., Shi, H., Yu, J., Liu, Z., Zhang, Y., Kwok, J. T., Li, Z., Weller, A., and Liu, W. Metamath: Bootstrap your own mathematical questions for large language models. arXiv preprint arXiv:2309.12284, 2023.

Zellers, R., Holtzman, A., Bisk, Y., Farhadi, A., and Choi, Y. Hellaswag: Can a machine really finish your sentence? arXiv preprint arXiv:1905.07830, 2019.

Zhang, S., Roller, S., Goyal, N., Artetxe, M., Chen, M., Chen, S., Dewan, C., Diab, M., Li, X., Lin, X. V., et al. Opt: Open pre-trained transformer language models. arXiv preprint arXiv:2205.01068, 2022a.

Zhang, Z., Lin, Y., Liu, Z., Li, P., Sun, M., and Zhou, J. Moefication: Transformer feed-forward layers are mixtures of experts. In Findings of the Association for Computational Linguistics: ACL 2022, pp. 877–890, 2022b.

Zheng, H., Bai, X., Liu, X., Mao, Z. M., Chen, B., Lai, F., and Prakash, A. Learn to be efficient: Build structured sparsity in large language models. Advances in Neural Information Processing Systems, 37:101969–101991, 2024.

Zhou, C., Liu, P., Xu, P., Iyer, S., Sun, J., Mao, Y., Ma, X., Efrat, A., Yu, P., Yu, L., et al. Lima: Less is more for alignment. Advances in Neural Information Processing Systems, 36:55006–55021, 2023a.

Zhou, J., Lu, T., Mishra, S., Brahma, S., Basu, S., Luan, Y., Zhou, D., and Hou, L. Instruction-following evaluation for large language models. arXiv preprint arXiv:2311.07911, 2023b.

Zhu, T., Qu, X., Dong, D., Ruan, J., Tong, J., He, C., and Cheng, Y. Llama-moe: Building mixture-of-experts from llama with continual pre-training. arXiv preprint arXiv:2406.16554, 2024.
