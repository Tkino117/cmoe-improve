# マルチタスク学習のためのLoRAベース混合エキスパートによる適応型共有エキスパート

Minghao Yang　Ren Togo　Guang Li　Takahiro Ogawa　Miki Haseyama  
北海道大学  
{yang, togo, guang, ogawa, mhaseyama}@lmd.ist.hokudai.ac.jp

## 要旨

混合エキスパート（Mixture-of-Experts; MoE）は、マルチタスク学習（Multi-Task Learning; MTL）のための強力な枠組みとして注目されている。しかし、既存のMoE–MTL手法は、単一タスク用に事前学習されたバックボーンに依存することが多く、単一タスク学習からマルチタスク学習へ移行する過程（STL-to-MTL）で、冗長な適応と非効率な知識共有に悩まされる。これらの問題に対処するため、我々は低ランク適応（Low-Rank Adaptation; LoRA）に基づくMoEの内部に、適応型共有エキスパート（adaptive shared experts; ASE）を提案する。共有エキスパートにはルーターが計算したゲーティング重みを割り当て、スパースエキスパートと共同で正規化する。この設計により、STLからMTLへの移行が容易になり、エキスパートの専門化と協調が向上する。さらに、LoRAエキスパートの数を増やす一方で各エキスパートのランクを比例して下げることで、細粒度エキスパートを導入する。これにより、パラメータ予算をほぼ同等に保ちながら、より効果的な知識共有が可能になる。PASCAL-Contextベンチマークで統一された学習設定のもと広範な実験を行った結果、ASEはさまざまな構成で一貫して性能を向上させ、細粒度設計がMTLに有効であることを確認した。

> **背景込みでのわかりやすい説明**
> 複数の仕事を同時に学ぶMTLでは、すべての仕事に同じ計算をさせると、仕事ごとの違いを捉えにくい。MoEは複数の「専門家」から入力に合うものを選ぶ仕組みだが、単一タスク用モデルからMTLへ移ると、専門家の役割が重複したり、知識をうまく共有できなかったりする。本研究のASEは、特定のタスク専用ではなく必要に応じて共有エキスパートも使うように重みを調整する。LoRAは大規模モデル全体を更新せず、小さな低ランク行列だけを学習する方法であり、さらに専門家を細かく分けても総パラメータ数を抑えられる点が重要である。

**索引語—** マルチタスク学習、混合エキスパート、LoRA、適応型ゲーティング、共有エキスパート

## 1. はじめに

大規模な深層ニューラルネットワークは、音声 [3–5]、視覚 [6–8]、自然言語処理 [9–11] を含むさまざまな領域 [1,2] で大きな進歩をもたらしてきた。こうした進歩を支える主要な技術の一つが、混合エキスパート（Mixture-of-Experts; MoE）アーキテクチャ [7, 10, 12] である。MoEは入力ごとに少数のエキスパートだけを有効化することで、計算量を扱いやすく保ちながらモデルの容量を拡張する。この考えに基づき、近年はマルチタスク学習（MTL）へのMoEの応用が研究されている [6, 13–15]。MTLでは異なるタスクを同時に学習し、共有表現とタスク固有の専門化の両方を活用する。

> **背景込みでのわかりやすい説明**
> 大きなモデルは多くの知識を保持できるが、毎回すべての部分を動かすと計算が重い。MoEは複数の専門家を用意し、入力に応じて一部だけを選ぶことでこの問題を緩和する。MTLでは、画像の物体認識や深度推定のような複数の仕事を同時に扱うため、共通して使える特徴と仕事ごとの特徴を両立させる必要がある。

しかし、既存のMoE–MTL手法は通常、単一タスク学習（STL）で事前学習したバックボーンを基盤としている。複数のタスクに対応するには、このバックボーンの表現をタスク固有のものからタスク非依存のものへ移行させなければならない（STL-to-MTL）。活性化されるエキスパートがスパースで入力ごとに変化するため、各エキスパートがこの移行を個別に学習することを強いられる。その結果、異なるエキスパートが重複する知識を何度も学習する一方、タスク非依存の特徴は十分に捉えられないという非効率な適応が生じる。したがって、効率性と、タスクを効果的に専門化する能力の両方が損なわれる。

> **背景込みでのわかりやすい説明**
> ここでの問題は、単一タスク用のモデルを複数タスク用に作り替える作業を、各専門家が別々に繰り返してしまう点にある。例えば、全タスクに共通する「物体の輪郭を見つける」知識を、複数の専門家がそれぞれ覚え直すと、パラメータも学習時間も無駄になる。しかも、専門家が担当タスクに特化する前に共通基盤を作る段階で、学習が不十分になりうる。

自然な解決策は、STL-to-MTL移行に必要な共通知識を捉える共有エキスパートを組み込むことである [9, 10]。これにより、タスク固有エキスパートの冗長な適応を減らし、より細かな専門化を可能にできる。しかし、MoE–MTLにおける共有エキスパートには重大な制約がある。第一に、一般的な実装では共有エキスパートの出力を、スパースに活性化されたエキスパートの出力へ直接加える [9, 10]。これは暗黙のうちに、共有エキスパートへ固定された大きなゲーティング重みを与えることになる。そのため、共有エキスパートとスパースエキスパートの間に不均衡が生じる。第二に、共有エキスパートはすべてのタスクにまたがって共同最適化されるため、その影響が過度に大きいと、学習後半で勾配の衝突が増幅されることが多い [16]。さらに、MoE–MTLモデルは、スパース計算であっても複数のエキスパートを活性化してルーティングするため、メモリ使用量とパラメータ更新量が増え、学習コストも高い [11, 15]。

> **背景込みでのわかりやすい説明**
> 共有エキスパートは重複学習を減らす有望な部品だが、単純にその出力を足すだけでは、常に強く効きすぎる。すべてのタスクに同じ共有出力を強く押しつけると、タスクごとに望ましい更新方向が異なるため、勾配が互いに反対を向く「勾配の衝突」が起きやすい。また、選択する専門家を増やすほど、選ばれた部分だけを計算するMoEの利点があっても、保持すべきパラメータや中間結果が増える。

これらの課題を解決するため、我々は三つの革新を備えた新しいMoE–MTLフレームワークを提案する。第一に、適応型共有エキスパート（ASE）を導入する。ASEの寄与はルーターが計算するゲーティング重みによって決まり、選択されたスパースエキスパートと一緒に正規化される。この設計は、共同正規化によって安定した均衡の取れた出力を実現する。STL-to-MTLへの移行中、共有エキスパートは当初主導的な役割を担うが、徐々に影響を小さくするため、勾配の衝突を緩和できる。第二に、計算効率を確保するため、各エキスパートを低ランク適応モジュール（LoRA）[17]として実装する。これにより表現力を保ちながら、パラメータ数とFLOPのオーバーヘッドを大幅に削減する。第三に、細粒度エキスパートという概念を導入する [10, 18]。LoRAエキスパートの数を増やす一方でランクを比例して下げることで、パラメータ予算を同程度に保ちつつ、より細かな専門化とタスク間のより効果的な協調を実現する。我々はPASCAL-Context [19]のMTLベンチマークで手法を評価した。実験結果は、異なる設定に対するフレームワークの頑健性と有効性に加え、細粒度構成がマルチタスク性能を改善する利点を示した。

> **背景込みでのわかりやすい説明**
> ASEの核心は、共有エキスパートを常に同じ強さで使うのではなく、入力と学習状態に応じてルーターが重みを調整することにある。学習初期には共通知識を使いやすくし、各タスクの違いが学べるようになるにつれて共有部分への依存を下げる。LoRA化は、専門家を増やすことによる計算・記憶コストを抑えるための手段である。ランクはLoRAの更新行列の幅に相当し、専門家数とランクの一方を増やして他方を下げれば、同じ予算で役割を細分化できる。

我々の貢献は以下のようにまとめられる。

- MTL向けの初のASE設計を導入し、より効果的なSTL-to-MTL転移、勾配衝突の低減、エキスパート専門化の向上を実現する。
- 表現力を維持しながら計算量とパラメータのオーバーヘッドを大幅に削減する、MTL向けの効率的なLoRA–MoE設計を提案する。
- 細粒度エキスパートがMTLにおけるより良い専門化と協調をもたらし、性能と安定性を向上させることを実証する。

> **背景込みでのわかりやすい説明**
> この三点は、提案手法の価値を「共有の仕方」「軽量化」「専門化の細かさ」という異なる観点から整理したものである。つまり、ASEで学習の流れを調整し、LoRAで実用的なコストに抑え、細粒度化で専門家同士の役割分担を改善する、という構成である。

## 2. 手法

### 2.1. 準備

**混合エキスパート。** MoE [9–11]では、Transformerブロック内のフィードフォワードネットワーク（FFN）のようなサブモジュールを複製して、$N$個の並列なエキスパートの集合を作り、入力ごとにその一部だけを活性化する。形式的には、各入力$x in mathbb{R}^{d_{in}}$に対して、ルーターが全エキスパートのゲーティング・ロジットを計算し、次のようにゲーティングスコアを導出する。

$$
g = \operatorname{Softmax}(W_gx) \in \mathbb{R}^{N}, \qquad W_g \in \mathbb{R}^{N \times d_{in}},
\tag{1}
$$

ここで、$g$はゲーティングスコア、$W_g$は学習可能なルーターのパラメータを表す。次にルーターは上位$k$個のエキスパートを選択し、最終出力を次のように計算する。

$$
y = x + \sum_{i \in \mathcal{T}} g_i E_i(x), \qquad \mathcal{T} = \operatorname{TopK}(g,k),
\tag{2}
$$

ここで、$E_i$は選択された集合$\mathcal{T}$に含まれる$i$番目のエキスパート、$g_i$はそのゲーティングスコアである。入力ごとに少数のエキスパートしか活性化しないが、学習中はすべてのエキスパートを保存して更新しなければならない。そのため、メモリと計算量のオーバーヘッドは相当なものになる [11, 15]。

> **背景込みでのわかりやすい説明**
> ルーターは、入力を見て「どの専門家にどの程度任せるか」を決める司令塔である。Softmaxによってスコアの合計が1になるため、スコアは専門家への配分比率として解釈できる。Top-$k$はそのうち上位だけを実際に使う仕組みである。ただし、使わない専門家も次の入力では必要になるため、モデルとしては全員分を保持する必要があり、ここがMoEの隠れたコストになる。

**MoEのLoRA変種（LoRA-MoE）。** LoRA-MoE [11, 15]は、パラメータ効率を高めたMoEの変種である。ここでは各エキスパートを、凍結されたベース重みに追加する低ランク適応（LoRA）モジュール [17]で置き換える。

$$
y = x + W_0x + \sum_{i \in \mathcal{T}} g_i E_i(x),
\tag{3}
$$

ここで、$W_0 \in \mathbb{R}^{d_{out} \times d_{in}}$は凍結されたベース重みを表す。各エキスパートは、次のLoRAモジュールとして実装される。

$$
E(x) = \Delta Wx = BAx,
\tag{4}
$$

ここで、更新行列$\Delta W \in \mathbb{R}^{d_{out} \times d_{in}}$は、二つの低ランク行列$A \in \mathbb{R}^{r \times d_{in}}$と$B \in \mathbb{R}^{d_{out} \times r}$に分解される。ランクは$r \ll \min(d_{in}, d_{out})$である。この設計は、MoEの柔軟性を保ちながら学習可能なパラメータ数とFLOPを大幅に削減するため、MTLの状況に適している。そこで本研究では、LoRA-MoEを基盤としてMTL向けに拡張する。

> **背景込みでのわかりやすい説明**
> 式(3)では、元の大きな重み$W_0$は変更せず、エキスパートごとに小さな補正だけを学習する。式(4)の$BA$は、巨大な更新行列を直接持つ代わりに、細い行列二つの積で表す方法である。例えば入力・出力の次元が大きくても、ランク$r$を小さくすれば学習量は大きく減る。これにより、複数タスク用に複数のエキスパートを用意しても、通常のMoEより現実的な規模にできる。

### 2.2. 適応型共有エキスパート

既存のMoE-MTLフレームワークでは、バックボーンをタスク固有の表現からタスク非依存の表現へ適応させる必要がある。MoEはエキスパートをスパースに活性化するため、各エキスパートがこのSTL-to-MTL移行を孤立して学習しなければならず、冗長な適応と共通特徴の不十分な捕捉につながる。この非効率性は、全体の有効性とエキスパートの専門化の両方を損なう。

> **背景込みでのわかりやすい説明**
> この節では、前節で述べた「共通知識を誰が学ぶか」という問題を具体的に解決する。タスクごとに選ばれる専門家だけに移行を任せると、共通部分が重複して学習される。そこで、複数タスクが共有する基礎的な変換を担当するエキスパートを導入する。

**LoRA-MoEにおける共有エキスパート。** 共有エキスパート [9, 10]は、タスク間の共通知識を捉えて統合するために導入された。これにより、タスク固有エキスパートの冗長な適応を軽減し、より細かな専門化とタスク間のより良い協調を可能にする。LoRA-MoEと組み合わせる場合、$N$個のエキスパートの中に$S$個の共有エキスパート$E^s$を含め、その出力を次のように直接加える。

$$
y = x + W_0x + \sum_{i \in \mathcal{T}} g_i E_i(x) + \sum_{i=1}^{S} E_i^s(x).
\tag{5}
$$

この直接加算は、各共有エキスパートに固定ゲーティングスコア$g_i^s=1$を与えて、$\sum_{i=1}^{S} g_i^s E_i^s(x)$を加えることと等価である。スパースエキスパートのゲーティングスコアは、Softmax後には$\sum_{i \in \mathcal{T}}g_i < 1$の合計になる（式(1)）。一方、共有エキスパートの総寄与$\sum_{i=1}^{S}g_i^s=S$は混合を支配し、スパースエキスパートと共有エキスパートの間に不均衡を生む。さらにこの不均衡は、ベース重み$W_0$とLoRAエキスパート$\Delta W$の間のスケールも乱す。実際、$\sum_{i \in \mathcal{T}}g_i+\sum_{i=1}^{S}g_i^s>S$となる。共有エキスパートが全タスクにまたがって学習されるMTL設定では、このように支配的で固定された寄与が、学習後半の勾配衝突をさらに悪化させる。

> **背景込みでのわかりやすい説明**
> 式(5)の問題は、共有エキスパートの出力に「1」という大きな固定係数を掛けている点である。通常のSoftmaxの重みは合計1に近い小さな値だが、共有側だけが無条件に1で加わるため、共有部分が強すぎる。共有エキスパートが複数あれば寄与はさらに積み上がり、元のバックボーンやタスク固有部分との数値的な釣り合いも崩れる。

**提案する適応型共有エキスパート。** これらの問題に対処するため、図1に示すように、寄与をルーターが動的に計算し、スパースエキスパートと一緒に正規化する適応型共有エキスパート（ASE）を提案する。具体的には、スパースエキスパートと共有エキスパートについて別々にロジットを計算する。

$$
z = W_gx \in \mathbb{R}^{N-S}, \qquad W_g \in \mathbb{R}^{(N-S)\times d_{in}},
\tag{6}
$$

$$
z^s = W_sx \in \mathbb{R}^{S}, \qquad W_s \in \mathbb{R}^{S\times d_{in}},
\tag{7}
$$

ここで、$z$と$z^s$はそれぞれスパースエキスパートと共有エキスパートのロジットを表す。活性化されるエキスパート数を同じに保つため、選択対象を$\mathcal{T}=\operatorname{TopK}(z,k-S)$に減らす。最後に、スパース側と共有側のロジットを共同で正規化する。

$$
g_i^s = \frac{\exp(z_i^s)}{\sum_{j\in\mathcal{T}}\exp(z_j)+\sum_{j=1}^{S}\exp(z_j^s)}, \qquad i=1,\ldots,S,
\tag{8}
$$

$$
g_i = \frac{\exp(z_i)}{\sum_{j\in\mathcal{T}}\exp(z_j)+\sum_{j=1}^{S}\exp(z_j^s)}, \qquad i\in\mathcal{T}.
\tag{9}
$$

このSoftmax後の正規化により、活性化されたすべてのエキスパートの寄与の合計が1になる。その結果、スパースエキスパートと共有エキスパートのバランスが取れ、LoRAモジュールのスケールも安定する。同時に、適応型ゲーティングによって、STL-to-MTLの初期段階では共有エキスパートがより強く寄与し、後の段階ではその重みを徐々に下げられるため、勾配衝突が緩和される。最終出力は次のように定義される。

$$
y = x + W_0x + \sum_{i\in\mathcal{T}}g_iE_i(x) + \sum_{i=1}^{S}g_i^sE_i^s(x).
\tag{10}
$$

> **背景込みでのわかりやすい説明**
> 式(8)と式(9)では、共有側とスパース側を同じ分母に入れている。したがって、どちらかが強く選ばれれば、全体の予算の中で他方の重みは自然に下がる。さらに学習が進むとルーターがタスクに応じた配分を学び、初期には共有知識、後期にはタスク固有の専門性を重視できる。これは固定重みで共有側を押しつける方法より、学習の段階に適した柔軟な制御である。

全体として、提案設計はスパースエキスパートと共有エキスパートの寄与のバランスを取り、勾配衝突を緩和し、LoRA-MoEフレームワーク内でより効率的なSTL-to-MTL転移を可能にする。

> **背景込みでのわかりやすい説明**
> ASEは新しいエキスパートを単に追加するのではなく、共有と専門化の間の配分を学習可能にした点に意味がある。これにより、共通の土台を使う利点と、タスクごとに異なる処理を行う利点を同時に利用できる。

**図1.** 提案する適応型共有エキスパートの概略。凍結されたFFN $W_0$に、スパースエキスパート$E_i(\cdot; A_i, B_i)$と適応型共有エキスパート$E_i^s(\cdot; A_i^s, B_i^s)$を組み合わせる。これらの寄与には、タスク固有ルーターが計算した適応型ゲーティング重みを割り当て、スパースエキスパートと共同で正規化する。

### 2.3. LoRA-MoEによるマルチタスク学習

ここでは、提案する適応型共有エキスパートとLoRA-MoEを統一されたTransformerバックボーンに統合した、全体のフレームワークを説明する。我々の設計は標準的なMTL設定に従う。この設定では、タスクは共通のバックボーンを共有しながら、タスク固有の構成要素を保持する。

**タスク固有のルーティングとヘッド。** 各タスク$t\in\{1,\ldots,T\}$について、まず入力$x$に学習可能なタスク埋め込み$e_t$を加え、$x_t=x+e_t$とする。その後、拡張された入力を共有バックボーンで独立に処理する。タスク固有のエキスパート選択を可能にするため、Transformerブロック$B^{(\ell)}$内の各層$\ell\in\{1,\ldots,L\}$で、各タスクはエキスパートのルーティングとスコア計算に自身のタスク固有ルーター$R_t^{(\ell)}$を使用する。これにより、エキスパートはパラメータのレベルでは共有されるが、活性化はタスクに依存する。隠れ状態$h_t^{(\ell-1)}$が与えられたとき、現在のブロックの更新は次のようになる。

$$
h_t^{(\ell)}=B^{(\ell)}(h_t^{(\ell-1)};R_t^{(\ell)}), \qquad h_t^{(0)}=x_t.
\tag{11}
$$

すべての$L$層を通過した後、隠れ状態をタスク固有ヘッド$H_t$に入力して予測を生成する。

$$
\hat{y}_t=H_t(h_t^{(L)}).
\tag{12}
$$

全体として、すべてのタスクが同じバックボーンとエキスパートを共有するが、タスク埋め込み$e_t$、ルーター$\{R_t^{(\ell)}\}_{\ell=1}^{L}$、出力ヘッド$\{H_t\}$はタスクごとに異なる。この設計により、効率的なパラメータ共有を実現しつつ、タスク固有の柔軟性を保てる。明確化のため、タスク$t$に対するエンドツーエンドの写像は次のように書ける。

$$
\hat{y}_t=H_t\left(F\left(x+e_t;\{R_t^{(\ell)}\}_{\ell=1}^{L}\right)\right),
\tag{13}
$$

ここで、$F(\cdot;\{R_t^{(\ell)}\})$は、LoRA–MoE層と適応型共有エキスパートを備え、タスク固有ルーターによってルーティングされる共有ViTを表す。

> **背景込みでのわかりやすい説明**
> 同じ画像を処理しても、タスク埋め込みによって「これはどの仕事のための入力か」をモデルに伝える。バックボーンの重みそのものは共有する一方、ルーターがタスクごとに異なる専門家を選び、最後のヘッドもタスクごとに出力形式を変える。例えば、分類ならクラス確率、深度推定なら画素ごとの深度を出すという役割分担である。

**パラメータ化戦略。** 効率化のため、バックボーンを事前学習済みのSTLモデルから初期化し、LoRAを用いてファインチューニングする。具体的には、すべてのエキスパートをLoRAモジュールとして実装し、バックボーン層にもLoRA更新を適用する。これに対して、ルーター$\{R_t^{(\ell)}\}_{\ell=1}^{L}$とタスク固有ヘッド$\{H_t\}$は、全パラメータを学習する方式でゼロから訓練する。このパラメータ化により、大部分のパラメータを軽量に保ちながら、ルーティングと予測の構成要素は十分な表現力を維持できる。

> **背景込みでのわかりやすい説明**
> すでに単一タスクで得られた知識を持つバックボーン全体を更新すると、MTLの学習コストが大きくなる。そこで、バックボーンと専門家の変更部分をLoRAに限定し、タスクの振り分けを決めるルーターと最終出力を作るヘッドだけは完全なパラメータで学習する。共有知識は低コストで再利用し、タスク固有の判断は十分な自由度で学習する設計である。

## 3. 実験

### 3.1. 実験設定

**データセットと評価指標。** 我々はPASCAL-Context [19]を用いてMTLフレームワークの有効性を評価する。このデータセットには10,103枚の画像が含まれ、エッジ検出（$Edge$）、意味分割（$Seg.$）、人体部位分割（$H.Parts$）、表面法線（$Norm.$）、顕著性検出（$Sal.$）という五つのタスクのアノテーションが付いている。

> **背景込みでのわかりやすい説明**
> 一つの画像に対して、物体の境界、物体の種類、人体の部位、面の向き、目立つ領域を同時に推定する。これらは互いに関連しつつも正解の形式が異なるため、MTLで共有知識とタスク固有知識の両方を検証しやすい組み合わせである。

各タスクについて、標準的な評価指標を採用する。$Seg.$、$H.Parts$、$Sal.$には平均Intersection over Union（mIoU）[20]、$Norm.$には平均誤差（mErr）[21]、$Edge$には最適データセットF値（odsF）[22]を用いる。odsFの計算には密な閾値探索が必要で計算コストが高いため、アブレーション研究では$Edge$についてBalanced Cross-Entropy Loss（BCE）[23]も報告する。先行研究 [13, 14]に従い、全タスクにわたって、ベースラインのSTLモデル$b$に対するMTLモデル$m$の平均相対性能低下$\Delta_m$を用いてMTLモデルを評価する。

$$
\Delta_m = \frac{1}{T}\sum_{t=1}^{T}(-1)^{l_t}\frac{M_{m,t}-M_{b,t}}{M_{b,t}},
$$

ここで、$M_{m,t}$と$M_{b,t}$はそれぞれMTLモデルとSTLモデルにおけるタスク$t$の評価指標を表し、$l_t=1$は値が小さいほど性能が良い場合に設定する。

> **背景込みでのわかりやすい説明**
> 指標には「大きいほど良い」ものと「小さいほど良い」ものが混在する。$(-1)^{l_t}$はこの向きをそろえるための符号であり、各タスクの改善・悪化を同じ基準で平均できるようにする。STLを基準にすることで、複数タスクを同時に扱った結果、どれだけ性能を保てたかを測れる。

**MoE損失。** ルーターを正則化するため、Mod-Squad損失 [13]を用いる。この損失はタスクとエキスパートの相互情報量を最大化し、スパースでありながら強いタスク–エキスパート依存関係を促す。この制約は、より効果的なエキスパート専門化を促進し、提案する適応型共有エキスパート設計を補完する。

> **背景込みでのわかりやすい説明**
> ルーターが毎回ほぼ同じ専門家を選ぶと、MoEの分業が機能しない。Mod-Squad損失は、タスクの種類と選ばれる専門家の間に意味のある対応関係を作りつつ、必要な専門家だけを使うように学習を誘導する。

**実装の詳細。** STLで事前学習したViT-smallバックボーン [24]の各フィードフォワードネットワーク（FFN）層に、適応型共有エキスパート（ASE）とLoRA-MoEを組み込む。各層にはランク$r=4$のLoRAエキスパートを16個置き、そのうち1個を適応型共有エキスパートとする。また、top-$k$は3に設定する。パラメータ効率のよい学習のため、バックボーンにはランク4のLoRAを適用し、新たに追加したモジュール（ルーターとタスク固有ヘッド）は全パラメータで学習する。すべての実験を単一のNVIDIA RTX 6000 Ada GPU上で行う。メモリ使用量を抑えるため、入力解像度を$224\times224$に下げ、40エポック学習する。

**細粒度エキスパート設定。** MTLでのエキスパートの専門化と協調をさらに高めるため、細粒度エキスパートの考え方 [9, 18]を採用する。これは、パラメータ予算を同程度に保つため、エキスパート数を増やす一方でLoRAランクを比例して下げる方法である。$N/k/S/r$と表す三段階の構成を検討する。ここで、$N$は層ごとのエキスパート総数、$k$はスパースに活性化されるエキスパート数、$S$は適応型共有エキスパート数、$r$は各エキスパートのLoRAランクである。$(16/3/1/4)$は初期設定に対応し、$(32/6/2/2)$と$(64/12/4/1)$はそれぞれ中粒度と高粒度の変種である。$r$を下げることでエキスパートのパラメータ数は釣り合うが、ルーターのコストは$N$とともに増加するため、$(64/12/4/1)$は計算量が大きい。したがって、主に$(16/3/1/4)$と$(32/6/2/2)$の結果を報告し、$(64/12/4/1)$は補足的な比較としてのみ含める。

> **背景込みでのわかりやすい説明**
> 例えば、専門家を16個・ランク4で持つ構成を、専門家32個・ランク2に置き換える。個々の専門家は小さくなるが、役割の選択肢が増えるため、タスクごとの細かな分業を期待できる。ただし、全専門家を見て選ぶルーターの処理は増えるので、専門家を増やせば無条件に速くなるわけではない。

**図2.** ベースライン、単純な共有エキスパート、提案するASEの比較。$Edge$の評価指標にはBCEを用いる。$\Delta_m$はViTベースのSTLモデルを基準に計算する。

### 3.2. 評価結果

ここでは、我々のフレームワークの実験結果を示す。各実験では、通常のLoRA-MoEをベースラインとする。アブレーション研究では$Edge$にBCEを報告し、ViT-baseのSTLモデルに対して$\Delta_m$を計算する。他のMTL手法との比較では、$Edge$にodsFを用い、一般的な慣行に従ってResNet-18ベースのSTLモデルに対して$\Delta_m$を計算する。

**単純な共有エキスパートとの比較。** まず、ASEをベースライン（通常のLoRA-MoE）および単純な共有エキスパートを加えたその変種と比較する。図2に示すように、単純な共有エキスパートは、不均衡と勾配衝突の問題（2.2節）により明確な性能低下を招く。これに対し、ASEはこの傾向を逆転させ、一貫した改善を示す。これらの結果は、ASEが単純な設計の制約を緩和するだけでなく、MTLにおける共有エキスパートの潜在能力を引き出すことを示している。

> **背景込みでのわかりやすい説明**
> 共有エキスパートを入れるだけでは、共有側が強くなりすぎて逆効果になる。ASEは共有側の重みを学習して調整するため、同じ「共有」というアイデアでも、固定的に足す方法との差が性能に現れる。

**活性化エキスパート数に関するアブレーション。** 共有エキスパート数$S$を1に固定し、スパースに活性化されるエキスパート数を$k\in\{3,4,5,6,7\}$の範囲で変化させる。図3(a)に示すように、ASEはすべての$k$でベースラインより一貫して高い$\Delta_m$を示し、top-$k$の選択に対して頑健であることが分かる。さらに、活性化されるエキスパート総数$k_{tot}=k+S$を同じにして比較しても、ASEはベースラインを上回る。

**細粒度エキスパート設定。** ベースラインとASEの両方について、3.1節に従った細粒度設定をさらに採用する。図3(b)に示すように、ASEはすべての設定でベースラインを一貫して改善し、64エキスパート構成で最も顕著な向上が見られる。加えて、エキスパートの粒度を細かくすると改善はほぼ線形に増加する。これは、特にASEと組み合わせた場合に、細粒度設計がタスク協調とエキスパート専門化を促進する有効な方法であることを示す。

> **背景込みでのわかりやすい説明**
> ASEの効果は、特定のtop-$k$だけに依存しない。専門家をより細かく分けるほど、各専門家が担当する役割を絞れるため性能が伸びる傾向がある。ただし、これは同じ予算でランクを調整した条件での結果であり、単純にモデルを巨大化した効果ではない。

**PASCAL-Contextでの結果。** 表1に異なるMTL手法の結果をまとめる。MTAN [25]、NDDR-CNN [26]、Cross-Stitch [27]を含む古典的なResNet-18ベースの手法に加え、ViTベースのマルチタスクベースラインと提案するLoRA-MoE変種を比較する。公平性を確保するため、すべての手法を入力解像度$224\times224$、40エポックという同じ設定で学習する。

通常のMTL-ViTと比べると、LoRA-MoEベースラインはタスク全体で明確な改善を示す。例えば、$Seg.$は69.1から73.8 mIoUに上昇し、$\Delta_m$は+2.68%から+6%超に上がる。これを基盤として、提案するASEはさらに性能を向上させる。具体的に、$(32/6/2/2)$構成では、ASEが$Seg.$で74.0 mIoU、$H.Parts$で60.3 mIoU、平均$\Delta_m$で**+7.58%**という最高の総合結果を達成し、ベースラインを一貫して上回る。注目すべきことに、これらの改善は通常のViT-baseベースラインに対するパラメータ増加がわずか約4%で実現されている。これは、LoRA-MoE設計がモデルサイズのオーバーヘッドを大きく増やさずに大幅な性能向上をもたらす効率性を示している。

> **背景込みでのわかりやすい説明**
> LoRA-MoEは、通常のMTL-ViTよりも少ない追加負担で各タスクの性能を押し上げる。さらにASEを加えると、共有知識とタスク固有知識の配分が改善され、細粒度構成では$\Delta_m$が+7.58%に達する。つまり、単一タスクモデルを基準にしても、複数タスク化による性能低下を抑えるだけでなく、平均的には改善できている。

ResNet-18ベースの手法と比較しても、我々の手法は明確な優位性を示す。例えば、Cross-Stitch [27]はResNet-18を$\Delta_m=+0.66\%$まで改善するのに対し、我々のASEは同じ統一設定で+7%を超える。

**エキスパートの活性化頻度の可視化。** 図4は、タスク$Norm.$について、層をまたいだエキスパートの活性化頻度を示す。具体的には、学習初期（エポック1）ではASEがほとんどの層で支配的になり、STL-to-MTL移行を橋渡しする役割が強調される。学習が進むと（エポック3）、スパースエキスパートの分布はより鋭くなり、タスク固有性が増す一方、ASEは勾配衝突を緩和するため寄与を徐々に減らす。最終段階（エポック40）ではエキスパート選択が安定し、明確な専門化が見られる。この動的な移行は、ASEが学習段階ごとに影響力を適応的に調整し、初期の効果的な知識転移を促進しながら、後期の勾配衝突を緩和することを示す。さらに、ASEの共同正規化は、スパースエキスパートと共有エキスパートの間だけでなく、LoRAエキスパートの出力と凍結重みの間でも安定したバランスを保証する。

> **背景込みでのわかりやすい説明**
> 活性化頻度の変化は、ASEが学習中に役割を変えることを直接示す。最初は共有エキスパートが共通の土台を作り、各タスクの専門家が十分に育つと、タスク固有エキスパートへ主役を譲る。提案手法はこの動きを設計上の仮定として固定するのではなく、ルーターの学習結果として実現している。

**表1.** PASCAL-ContextデータセットにおけるMTL手法の比較。最良の結果を**太字**で示す。

| 手法 | バックボーン | $Seg.$ (mIoU)↑ | $Norm.$ (mErr)↓ | $H.Parts$ (mIoU)↑ | $Sal.$ (mIoU)↑ | $Edge$ (odsF)↑ | $\Delta_m$ (%)↑ | Params (M)↓ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| STL | ResNet-18 | 60.1 | **15.4** | 51.0 | 63.2 | 50.8 | 0.00 | **11** |
| MTL | ResNet-18 | 58.4 | 15.9 | 51.8 | 62.6 | 50.3 | -1.29 | **11** |
| MTAN [25] | ResNet-18 | 58.9 | 16.4 | 51.9 | 62.2 | 50.6 | -1.74 | **11** |
| NDDR-CNN [26] | ResNet-18 | 59.7 | 15.5 | 51.7 | 62.7 | 51.5 | +0.13 | **11** |
| Cross-Stitch [27] | ResNet-18 | 60.4 | **15.4** | 51.6 | 63.1 | 51.7 | +0.66 | **11** |
| MTL-ViT | ViT-base | 69.1 | 16.2 | 54.8 | 61.9 | 49.9 | +2.68 | 104 |
| Baseline (16/4/0/4) | LoRA-MoE ViT-base | 73.7 | 17.5 | 59.2 | 62.9 | 53.7 | +6.06 | 107 |
| Baseline (32/8/0/2) | LoRA-MoE ViT-base | 73.8 | 17.4 | 59.3 | 62.9 | 53.3 | +6.11 | 108 |
| **提案手法 (16/3/1/4)** | LoRA-MoE ViT-base | **74.0** | 17.3 | 60.1 | 63.2 | **55.3** | +7.49 | 107 |
| **提案手法 (32/6/2/2)** | LoRA-MoE ViT-base | **74.0** | 17.2 | **60.3** | **63.3** | 54.9 | **+7.58** | 108 |

**図3.** 異なるtop-$k$および細粒度設定におけるASE。$Edge$にはBCEを用い、ViTベースのSTLモデルに対して$\Delta_m$を計算する。(a) $S=1$におけるtop-$k$に対する$\Delta_m$の折れ線グラフ。横軸は選択したtop-$k$、縦軸は$\Delta_m$を示す。(b) 細粒度割り当てにおける$\Delta_m$の棒グラフ。

**図4.** 層をまたいだエキスパートの活性化頻度。タスク$Norm.$について、全層におけるエキスパートの活性化頻度を、エポック1、3、40（全40エポック中）で可視化する。縦軸は層のインデックス、横軸は16個のエキスパート（ASE 1個を含む）を表す。カラーバーは正規化された頻度を示す。

## 4. 結論

本論文では、MoEベースのMTLにおける単純な共有エキスパートの制約に対処するため、LoRA-MoEフレームワーク内に適応型共有エキスパート（ASE）を導入した。これにより、STL-to-MTL移行と、効果的なエキスパートの専門化・協調の両方を促進する。さらに、MTL向けMoEに細粒度エキスパートの考え方を組み込んだ。PASCAL-Contextでの広範な実験により、異なる設定におけるASEの頑健性と有効性、および細粒度構成がマルチタスク性能を改善する利点を実証した。

> **背景込みでのわかりやすい説明**
> 結論として、ASEは共有知識を活用しながら、学習後半にはタスクごとの専門化へ移れる仕組みである。LoRAによる軽量化と細粒度化を組み合わせることで、モデルを大きくしすぎずにMTLの性能を高められることを実験で確認した。

## References

[1] Licheng Jiao, Mengjiao Wang, Xu Liu, Lingling Li, Fang Liu, Zhixi Feng, Shuyuan Yang, and Biao Hou, “Multiscale deep learning for detection and recognition: A comprehensive survey,” *IEEE Trans. Neural. Netw. Learn. Syst.*, vol. 36, no. 4, pp. 5900–5920, 2024.

[2] Weilin Cai, Jiyong Jiang, Fan Wang, Jing Tang, Sunghun Kim, and Jiayi Huang, “A survey on mixture of experts in large language models,” *IEEE Trans. Knowl. Data Eng.*, vol. 37, no. 7, pp. 3896–3915, 2025.

[3] Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever, “Robust speech recognition via large-scale weak supervision,” in *Proc. ICML*, 2023.

[4] Weiran Wang, Rohit Prabhavalkar, Haozhe Shan, Zhong Meng, Dongseong Hwang, Qiujia Li, Khe Chai Sim, Bo Li, James Qin, Xingyu Cai, Adam Stooke, Chengjian Zheng, Yanzhang He, Tara Sainath, and Pedro Moreno Mengibar, “Massive end-to-end speech recognition models with time reduction,” in *Proc. NAACL*, 2024, pp. 6206–6217.

[5] Yangyang Meng, Jinpeng Li, Guodong Lin, Yu Pu, Guanbo Wang, Hu Du, Zhiming Shao, Yukai Huang, Ke Li, and Wei-Qiang Zhang, “Dolphin: A large-scale automatic speech recognition model for eastern languages,” *arXiv preprint arXiv:2503.20212*, 2025.

[6] Tianlong Chen, Xuxi Chen, Xianzhi Du, Abdullah Rashwan, Fan Yang, Huizhong Chen, Zhangyang Wang, and Yeqing Li, “Adamv-moe: Adaptive multi-task vision mixture-of-experts,” in *Proc. ICCV*, 2023, pp. 17300–17311.

[7] Peng Jin, Bo Zhu, Li Yuan, and Shuicheng Yan, “Moh: Multi-head attention as mixture-of-head attention,” *arXiv preprint arXiv:2410.11842*, 2024.

[8] Leonid Karlinsky, Assaf Arbelle, Abraham Daniels, Ahmed Nassar, Amit Alfassi, Bo Wu, and et al., “Granite vision: A lightweight, open-source multimodal model for enterprise intelligence,” *arXiv preprint arXiv:2502.09927*, 2025.

[9] Samyam Rajbhandari, Conglong Li, Zhewei Yao, Minjia Zhang, Reza Yazdani Aminabadi, Ammar Ahmad Awan, Jeff Rasley, and Yuxiong He, “DeepSpeed-MoE: Advancing mixture-of-experts inference and training to power next-generation AI scale,” in *Proc. ICML*, 2022, pp. 18332–18346.

[10] Damai Dai, Chengqi Deng, Chenggang Zhao, R. X. Xu, Huazuo Gao, Deli Chen, Jiashi Li, Wangding Zeng, Xingkai Yu, Y. Wu, Zhenda Xie, Y. K. Li, Panpan Huang, Fuli Luo, Chong Ruan, Zhifang Sui, and Wenfeng Liang, “Deepseek-Moe: Towards ultimate expert specialization in mixture-of-experts language models,” *arXiv preprint arXiv:2401.06066*, 2024.

[11] Shihan Dou, Enyu Zhou, Yan Liu, Songyang Gao, Wei Shen, Limao Xiong, Yuhao Zhou, Xiao Wang, Zhiheng Xi, Xiaoran Fan, Shiliang Pu, Jiang Zhu, Rui Zheng, Tao Gui, Qi Zhang, and Xuanjing Huang, “LoRA-MoE: Alleviating world knowledge forgetting in large language models via MoE-style plugin,” in *Proc. ACL*, 2024, pp. 1932–1945.

[12] Dmitry Lepikhkin, HyoukJoon Lee, Yuanzhong Xu, Dehao Chen, Orhan Firat, Yanping Huang, Maxim Krikun, Noam Shazeer, and Zhifeng Chen, “Gshard: Scaling giant models with conditional computation and automatic sharding,” in *Proc. ICLR*, 2021.

[13] hanxue liang, Zhiwen Fan, Rishov Sarkar, Ziyu Jiang, Tianlong Chen, Kai Zou, Yu Cheng, Conghao Hao, and Zhangyang Wang, “M3vit: Mixture-of-experts vision transformer for efficient multi-task learning with model-accelerator co-design,” in *Proc. NeurIPS*, 2022, pp. 28441–28457.

[14] Zitian Chen, Yiqing Shen, Mingyu Ding, Zhenfang Chen, Hengshuang Zhao, Erik Learned-Miller, and Chuang Gan, “Mod-squad: Designing mixtures of experts as modular multi-task learners,” in *Proc. CVPR*, 2023, pp. 11828–11837.

[15] Yuqi Yang, Peng-Tao Jiang, Qibin Hou, Hao Zhang, Jinwei Chen, and Bo Li, “Multi-task dense prediction via mixture of low-rank experts,” in *Proc. CVPR*, 2024, pp. 27927–27937.

[16] Simon Vandenhende, Stamatis Georgoulis, Wouter Van Gansbeke, Marc Proesmans, Dengxin Dai, and Luc Van Gool, “Multi-task learning for dense prediction tasks: A survey,” *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 44, no. 7, pp. 3614–3633, 2022.

[17] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen, “LoRA: Low-rank adaptation of large language models,” in *Proc. ICLR*, 2022.

[18] Jan Ludziejewski, Jakub Krajewski, Kamil Adamczewski, Maciej Pióro, Michal Krutul, Szymon Antoniak, Kamil Ciebi era, Krystian Król, Tomasz Odrzygóźdź, Piotr Sankowski, Marek Cygan, and Sebastian Jaszczur, “Scaling laws for fine-grained mixture of experts,” in *Proc. ICLR Workshop*, 2024.

[19] Roozbeh Mottaghi, Xiangjie Chen, Xiaobai Liu, Nam-Gyu Cho, Seong-Whan Lee, Sanja Fidler, Raquel Urtasun, and Alan Yuille, “The role of context for object detection and semantic segmentation in the wild,” in *Proc. CVPR*, 2014, pp. 891–898.

[20] Mark Everingham, Luc Van Gool, Christopher K. I. Williams, John Winn, and Andrew Zisserman, “The pascal visual object classes (voc) challenge,” *Int. J. Comput. Vis.*, vol. 88, no. 2, pp. 303–338.

[21] Nathan Silberman, Derek Hoiem, Pushmeet Kohli, and Rob Fergus, “Indoor segmentation and support inference from rgbd images,” in *Proc. ECCV*, 2012, pp. 746–760.

[22] Pablo Arbeláez, Michael Maire, Charless Fowlkes, and Jitendra Malik, “Contour detection and hierarchical image segmentation,” *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 33, no. 5, pp. 898–916, 2011.

[23] Saining Xie and Zhuowen Tu, “Holistically-nested edge detection,” in *Proc. ICCV*, 2015, pp. 1395–1403.

[24] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby, “An image is worth 16x16 words: Transformers for image recognition at scale,” in *Proc. ICLR*, 2021.

[25] Shikun Liu, Edward Johns, and Andrew J. Davison, “End-to-end multi-task learning with attention,” in *Proc. CVPR*, 2019, pp. 1871–1880.

[26] Yuan Gao, Jiayi Ma, Mingbo Zhao, Wei Liu, and Alan L. Yuille, “Nddr-cnn: Layerwise feature fusing in multi-task cnns by neural discriminative dimensionality reduction,” in *Proc. CVPR*, 2019, pp. 3200–3209.

[27] Ishan Misra, Abhinav Shrivastava, Abhinav Gupta, and Martial Hebert, “Cross-stitch networks for multi-task learning,” in *Proc. CVPR*, 2016, pp. 3994–4003.

<!-- PAPER-TRANSLATE-JA:END -->
