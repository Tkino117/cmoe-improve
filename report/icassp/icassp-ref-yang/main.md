# ADAPTIVE SHARED EXPERTS WITH LORA-BASED MIXTURE OF EXPERTS FOR MULTI-TASK LEARNING

Minghao Yang  Ren Togo  Guang Li  Takahiro Ogawa  Miki Haseyama  
Hokkaido University  
{yang, togo, guang, ogawa, mhaseyama}@lmd.ist.hokudai.ac.jp

## ABSTRACT

Mixture-of-Experts (MoE) has emerged as a powerful framework for multi-task learning (MTL). However, existing MoE–MTL methods often rely on single-task pretrained backbones and suffer from redundant adaptation and inefficient knowledge sharing during the transition from single-task to multi-task learning (STL-to-MTL). To address these limitations, we propose adaptive shared experts (ASE) within a low-rank adaptation (LoRA) based MoE, where shared experts are assigned router-computed gating weights jointly normalized with sparse experts. This design facilitates STL-to-MTL transition, enhances expert specialization, and cooperation. Furthermore, we incorporate fine-grained experts by increasing the number of LoRA experts while proportionally reducing their rank, enabling more effective knowledge sharing under a comparable parameter budget. Extensive experiments on the PASCAL-Context benchmark, under unified training settings, demonstrate that ASE consistently improves performance across diverse configurations and validates the effectiveness of fine-grained designs for MTL.

**Index Terms—** Multi-Task Learning, Mixture-of-Experts, LoRA, Adaptive Gating, Shared Expert

## 1. INTRODUCTION

Large-scale deep neural networks have driven significant advances in various domains [1,2], including speech [3–5], vision [6–8], and natural language processing [9–11]. A key technique of these advances is the Mixture-of-Experts (MoE) architecture [7, 10, 12], which expands model capacity while keeping manageable computation by activating only a small subset of experts for each input. Motivated by this, recent studies have explored MoE for multi-task learning (MTL) [6, 13–15], where different tasks are trained jointly to exploit both shared representations and task-specific specialization.

Despite these advantages, existing MoE–MTL approaches typically built upon a single-task learning (STL) pretrained backbone. To serve multiple tasks, this backbone requires shifting from task-specific to task-agnostic representations (STL-to-MTL). Due to sparse and varying activation, each expert is compelled to learn this transition independently. This results in inefficient adaptation, where different experts may redundantly relearn overlapping knowledge, while task-agnostic features are insufficiently captured. Consequently, both efficiency and effective task specialization are compromised.

A natural solution is to incorporate a shared expert [9, 10] dedicated to capturing common knowledge for the STL-to-MTL transition, thereby relieving task-specific experts from redundant adaptation and enabling finer specialization. However, shared experts in MoE–MTL face critical limitations. First, common implementations add the shared expert’s output directly to that of sparsely activated experts, implicitly assigning it a fixed, dominant gating weight [9, 10]. This introduces an imbalance between the shared and sparse experts. Second, because the shared expert is jointly optimized across all tasks, its disproportionately large influence often amplifies gradient conflicts in later training stages [16]. Moreover, MoE–MTL models are also expensive to train, since activating and routing multiple experts increases the memory footprint and parameter updates even under sparse computation [11, 15].

To overcome these challenges, we propose a new MoE–MTL framework with three innovations. First, we introduce an adaptive shared expert (ASE) whose contribution is governed by a router-computed gating weight and normalized together with the selected sparse experts. This design achieves stable and balanced outputs through joint normalization. During the transition of STL-to-MTL, the shared experts initially take the lead but gradually reduce their influence, which helps alleviate gradient conflicts. Second, to ensure computational efficiency, we implement each expert as a low-rank adaptation module (LoRA) [17], substantially reducing parameter and FLOP overhead while preserving expressivity. Third, we introduce the notion of fine-grained experts [10, 18]: by increasing the number of LoRA experts while proportionally lowering their rank, we maintain a comparable parameter budget but achieve finer specialization and more effective cooperation across tasks. We evaluate our method on the PASCAL-Context [19] MTL benchmark. Experimental results demonstrate the robustness and effectiveness of our framework across different settings, as well as the benefits of fine-grained configurations for improving multi-task performance.

Our contributions are summarized as follows.

- We introduce the first ASE design for MTL, enabling more effective STL-to-MTL transfer, reducing gradient conflicts, and enhancing expert specialization.
- We propose an efficient LoRA–MoE design for MTL that preserves expressivity while substantially reducing computational and parameter overhead.
- We demonstrate that fine-grained experts provide better specialization and cooperation in MTL, leading to improved performance and stability.

## 2. METHODOLOGY

### 2.1. Preliminaries

**Mixture-of-Experts.** MoE [9–11] duplicates a sub-module, such as the feed-forward network (FFN) in a Transformer block, into a set of $N$ parallel experts and activates only a small subset per input. Formally, for each input $x \in \mathbb{R}^{d_{in}}$, a router computes gating logits over all experts and derives gating scores as:

$$
g = \operatorname{Softmax}(W_gx) \in \mathbb{R}^{N}, \qquad W_g \in \mathbb{R}^{N \times d_{in}},
\tag{1}
$$

where $g$ denotes the gating scores and $W_g$ are trainable router parameters. The router then selects the top-$k$ experts and computes the final output as:

$$
y = x + \sum_{i \in \mathcal{T}} g_i E_i(x), \qquad \mathcal{T} = \operatorname{TopK}(g,k),
\tag{2}
$$

where $E_i$ is the $i$-th expert in the selected set $\mathcal{T}$ and $g_i$ is its gating score. Although only a few experts are activated per input, all experts must be stored and updated during training, resulting in considerable memory and computation overhead [11, 15].

**LoRA variant of MoE (LoRA-MoE).** LoRA-MoE [11, 15] is a parameter-efficient variant of MoE, where each expert is replaced by a low-rank adaptation (LoRA) module [17] added to the frozen base weights:

$$
y = x + W_0x + \sum_{i \in \mathcal{T}} g_i E_i(x),
\tag{3}
$$

where $W_0 \in \mathbb{R}^{d_{out} \times d_{in}}$ denotes frozen base weights. Here, each expert is implemented as a LoRA module:

$$
E(x) = \Delta Wx = BAx,
\tag{4}
$$

where the update matrix $\Delta W \in \mathbb{R}^{d_{out} \times d_{in}}$ is factorized into two low-rank matrices $A \in \mathbb{R}^{r \times d_{in}}$ and $B \in \mathbb{R}^{d_{out} \times r}$, with rank $r \ll \min(d_{in}, d_{out})$. This design substantially reduces trainable parameters and FLOPs while preserving the flexibility of MoE, making it suitable for MTL scenarios. Hence, in this work, we build upon and extend LoRA-MoE for MTL.

### 2.2. Adaptive Shared Expert

Existing MoE-MTL frameworks require the backbone to adapt from task-specific to task-agnostic representations. Because MoE activates experts sparsely, each expert must learn this STL-to-MTL transition in isolation, which leads to redundant adaptation and insufficient capture of common features. This inefficiency compromises both overall effectiveness and expert specialization.

**Shared Expert with LoRA-MoE.** The shared expert [9, 10] was introduced to capture and consolidate common knowledge across tasks, thus relieving task-specific experts from redundant adaptation and enabling finer specialization and better cooperation across tasks. When combined with LoRA-MoE, the $N$ experts include $S$ shared experts $E^s$, whose outputs are directly added as:

$$
y = x + W_0x + \sum_{i \in \mathcal{T}} g_i E_i(x) + \sum_{i=1}^{S} E_i^s(x).
\tag{5}
$$

This direct addition is equivalent to adding $\sum_{i=1}^{S} g_i^s E_i^s(x)$ with fixed gating scores $g_i^s=1$ for each shared expert. Since the gating scores of sparse experts sum to $\sum_{i \in \mathcal{T}}g_i < 1$ after softmax (Eq. 1), the total shared contribution $\sum_{i=1}^{S}g_i^s=S$ dominates the mixture, creating imbalance between sparse and shared experts. Moreover, this imbalance also disrupts the scale between the base weights $W_0$ and the LoRA experts $\Delta W$, since $\sum_{i \in \mathcal{T}}g_i+\sum_{i=1}^{S}g_i^s>S$. In MTL settings, where shared experts are trained across all tasks, such dominant and fixed contributions further aggravate gradient conflicts in later training stages.

**Proposed Adaptive Shared Expert.** To address these problems, we propose adaptive shared experts (ASE), as illustrated in Fig. 1, whose contribution is dynamically computed by the router and normalized together with sparse experts. Specifically, we compute logits for sparse and shared experts separately:

$$
z = W_gx \in \mathbb{R}^{N-S}, \qquad W_g \in \mathbb{R}^{(N-S)\times d_{in}},
\tag{6}
$$

$$
z^s = W_sx \in \mathbb{R}^{S}, \qquad W_s \in \mathbb{R}^{S\times d_{in}},
\tag{7}
$$

where $z$ and $z^s$ denote the logits for sparse and shared experts, respectively. To maintain the same number of active experts, we reduce the selection to $\mathcal{T}=\operatorname{TopK}(z,k-S)$. Finally, sparse and shared logits are normalized jointly:

$$
g_i^s = \frac{\exp(z_i^s)}{\sum_{j\in\mathcal{T}}\exp(z_j)+\sum_{j=1}^{S}\exp(z_j^s)}, \qquad i=1,\ldots,S,
\tag{8}
$$

$$
g_i = \frac{\exp(z_i)}{\sum_{j\in\mathcal{T}}\exp(z_j)+\sum_{j=1}^{S}\exp(z_j^s)}, \qquad i\in\mathcal{T}.
\tag{9}
$$

This post-softmax normalization ensures that the contributions of all activated experts sum to one, thereby balancing sparse and shared experts as well as stabilizing the scale of the LoRA modules. At the same time, the adaptive gating enables the shared experts to contribute more strongly during the early STL-to-MTL phase, while gradually reducing their weight in later stages, thus alleviating gradient conflicts. The final output is defined as:

$$
y = x + W_0x + \sum_{i\in\mathcal{T}}g_iE_i(x) + \sum_{i=1}^{S}g_i^sE_i^s(x).
\tag{10}
$$

Overall, the proposed design balances the contributions of sparse and shared experts, mitigates gradient conflicts, and enables a more efficient STL-to-MTL transfer within the LoRA-MoE framework.

**Fig. 1.** An illustration of the proposed adaptive shared experts. The frozen FFN $W_0$ is combined with sparse experts $E_i(\cdot; A_i, B_i)$ and adaptive shared experts $E_i^s(\cdot; A_i^s, B_i^s)$, whose contributions are assigned adaptive gating weights computed by task-specific routers and normalized jointly with sparse experts.

### 2.3. Multi-Task Learning with LoRA-MoE

We now describe the overall framework that integrates the proposed adaptive shared expert with LoRA-MoE into a unified Transformer backbone. Our design follows a standard MTL setup, in which tasks share a common backbone while retaining task-specific components.

**Task-specific routing and heads.** For each task $t\in\{1,\ldots,T\}$, the input $x$ is first augmented with a learnable task embedding $e_t$, as: $x_t=x+e_t$. The augmented input is then processed independently by the shared backbone. To enable task-specific expert selection, at each layer $\ell\in\{1,\ldots,L\}$ within Transformer block $B^{(\ell)}$, each task employs its own task-specific router $R_t^{(\ell)}$ for expert routing and score computing. This ensures that experts are shared at the parameter level but activated in a task-dependent manner. Given the hidden state $h_t^{(\ell-1)}$, the update of the current block is:

$$
h_t^{(\ell)}=B^{(\ell)}(h_t^{(\ell-1)};R_t^{(\ell)}), \qquad h_t^{(0)}=x_t.
\tag{11}
$$

After passing through all $L$ layers, the hidden state is fed into a task-specific head $H_t$ to produce the prediction:

$$
\hat{y}_t=H_t(h_t^{(L)}).
\tag{12}
$$

Overall, all tasks share the same backbone and experts, but differ in their task embeddings $e_t$, routers $\{R_t^{(\ell)}\}_{\ell=1}^{L}$, and output heads $\{H_t\}$. This design enables efficient parameter sharing while retaining task-specific flexibility. For clarity, the end-to-end mapping for task $t$ can be written as:

$$
\hat{y}_t=H_t\left(F\left(x+e_t;\{R_t^{(\ell)}\}_{\ell=1}^{L}\right)\right),
\tag{13}
$$

where $F(\cdot;\{R_t^{(\ell)}\})$ denotes the shared ViT with LoRA–MoE layers and adaptive shared experts, routed by task-specific routers.

**Parameterization strategy.** For efficiency, the backbone is initialized from a pretrained STL model and fine-tuned using LoRA. Specifically, all experts are implemented as LoRA modules, and LoRA updates are also applied to the backbone layers. In contrast, routers $\{R_t^{(\ell)}\}_{\ell=1}^{L}$ and task-specific heads $\{H_t\}$ are trained from scratch in full-parameter mode. This parameterization ensures the majority of parameters remain lightweight, while the routing and prediction components maintain full expressivity.

## 3. EXPERIMENTS

### 3.1. Experiment Setup

**Dataset and Evaluation Metrics.** We evaluate the effectiveness of our MTL framework on PASCAL-Context [19], which contains 10,103 images with five task annotations of edge detection ($Edge$), semantic segmentation ($Seg.$), human parts segmentation ($H.Parts$), surface normals ($Norm.$), and saliency detection ($Sal.$).

For each task, we adopt standard evaluation metrics: mean intersection over union (mIoU) [20] for $Seg.$, $H.Parts$, and $Sal.$; mean error (mErr) [21] for $Norm.$; and the optimal dataset F-measure (odsF) [22] for $Edge$. Since computing odsF involves dense thresholding and is computationally expensive, we additionally report the Balanced Cross-Entropy Loss (BCE) [23] for $Edge$ in the ablation study. Following previous works [13, 14], we adopt the average relative performance drop $\Delta_m$ to evaluate an MTL model $m$ with respect to the baseline STL model $b$ over all tasks:

$$
\Delta_m = \frac{1}{T}\sum_{t=1}^{T}(-1)^{l_t}\frac{M_{m,t}-M_{b,t}}{M_{b,t}},
$$

where $M_{m,t}$ and $M_{b,t}$ denote the evaluation metric of task $t$ for the MTL and STL models, respectively, and $l_t=1$ if a lower value indicates better performance.

**MoE Loss.** We employ the Mod-Squad loss [13] to regularize the routers. This loss maximizes the mutual information between tasks and experts, thereby encouraging sparse yet strong task–expert dependencies. Such a constraint promotes more effective expert specialization and complements our adaptive shared expert design.

**Implementation Details.** We integrate adaptive shared experts (ASE) with LoRA-MoE into each feed-forward network (FFN) layer of a ViT-small backbone pretrained on STL [24]. Each layer contains 16 LoRA experts with rank $r=4$, including a single adaptive shared expert, and we set the top-$k$ to 3. LoRA with rank 4 is applied to the backbone for parameter-efficient training, while newly added modules (routers and task-specific heads) are trained in full. All experiments are conducted on a single NVIDIA RTX 6000 Ada GPU. To reduce memory usage, all experiments are conducted with a reduced input resolution of $224\times224$ for 40 training epochs.

**Fine-Grained Expert Settings.** To further enhance expert specialization and cooperation in MTL, we adopt the notion of fine-grained experts [9, 18], where the number of experts is increased while their LoRA rank is proportionally reduced to maintain a comparable parameter budget. We consider three levels of configurations, denoted as $(N/k/S/r)$, where $N$ is the total number of experts per layer, $k$ is the number of sparsely activated experts, $S$ is the number of adaptive shared experts, and $r$ is the LoRA rank of each expert. The $(16/3/1/4)$ configuration corresponds to the initial setup, while $(32/6/2/2)$ and $(64/12/4/1)$ represent medium- and high-granularity variants. Although decreasing $r$ balances the parameter count of experts, the router cost still grows with $N$, making $(64/12/4/1)$ more computationally demanding. Thus, we mainly report results for $(16/3/1/4)$ and $(32/6/2/2)$, and include $(64/12/4/1)$ only as a supplementary comparison.

**Fig. 2.** Comparison between baseline, naive shared expert, and the proposed ASE. BCE is used as the evaluation metric for $Edge$. The $\Delta_m$ is computed with respect to a ViT-based STL model.

### 3.2. Evaluation Results

We now present the experimental results of our framework. For each experiment, we denote the vanilla LoRA-MoE as our baseline. For ablation studies, we report BCE for $Edge$ and compute $\Delta_m$ against a ViT-base STL model. For comparisons with other MTL methods, we adopt odsF for $Edge$ and compute $\Delta_m$ with respect to an STL model based on ResNet-18, following common practice.

**Comparison With the Naive Shared Expert.** We first compare our ASE with the baseline (vanilla LoRA-MoE), and its variant with a naive shared expert. As shown in Fig. 2, the naive shared expert leads to clear performance degradation due to imbalance and gradient conflict issues (Sec. 2.2). In contrast, ASE reverses this trend with consistent improvement. These results highlight the effectiveness of ASE in not only mitigating the limitations of the naive design but also unlocking the potential of shared experts in MTL.

**Ablation on the Number of Activated Experts.** We vary the number of sparsely activated experts $k\in\{3,4,5,6,7\}$ while fixing the number of shared experts $S$ to 1. As shown in Fig. 3(a), our ASE consistently yields higher $\Delta_m$ than the baseline across all $k$, indicating robustness to the choice of top-$k$. Moreover, when com-

paring under the same total activated experts $k_{tot}=k+S$, ASE still outperforms the baseline.

**Fine-grained expert settings.** We further adopt the fine-grained settings following Sec. 3.1 for both baseline and ASE. As illustrated in Fig. 3(b), ASE consistently improves over the baseline across all settings, with the most pronounced gain observed in the 64-expert configuration. In addition, finer expert granularity yields nearly linear improvements, highlighting the effectiveness of fine-grained designs in facilitating task cooperation and expert specialization, particularly when combined with ASE.

**Results on PASCAL-Context.** Table 1 summarizes the results of different MTL methods. We compare against classical ResNet-18 based methods, including MTAN [25], NDDR-CNN [26], and Cross-Stitch [27], as well as ViT-based multi-task baselines and our proposed LoRA-MoE variants. Importantly, all methods are trained under the same setting with an input resolution of $224\times224$ and 40 training epochs to ensure fairness.

Compared with the vanilla MTL-ViT, the LoRA-MoE baselines show clear improvements across tasks, e.g., $Seg.$ increases from 69.1 to 73.8 mIoU and $\Delta_m$ rises from +2.68% to over +6%. Building on this, our proposed ASE further improves performance. Specifically, under $(32/6/2/2)$ configuration, ASE achieves the best overall results with 74.0 mIoU on $Seg.$, 60.3 mIoU on $H.Parts$, and an average $\Delta_m$ of **+7.58%**, demonstrating consistent gains over the baseline. Notably, these improvements are achieved with only a marginal ~4% increase in parameters over the plain ViT-base baseline, highlighting the efficiency of the LoRA-MoE design in delivering substantial performance gains without significant model size overhead.

When compared with ResNet-18 based methods, our approach shows a clear advantage. For example, Cross-Stitch [27] improves ResNet-18 to $\Delta_m=+0.66\%$, whereas our ASE achieves more than +7% under the same unified setting.

**Visualization of the Activation Frequency of Experts.** Fig. 4 illustrates the activation frequency of experts across layers for task $Norm.$. Specifically, at the early training stage (epoch 1), ASE dominates most layers, emphasizing its role in bridging the STL-to-MTL transition. As training progresses (epoch 3), the distribution of sparse experts becomes sharper and increasingly task-specific, while ASE gradually reduces its contribution to alleviate gradient conflicts. By the final stage (epoch 40), expert selection stabilizes, exhibiting clear specialization. This dynamic transition demonstrates that ASE adaptively adjusts its influence throughout different training phases, thereby facilitating effective early knowledge transfer and mitigating later gradient conflicts. Moreover, the joint normalization of ASE ensures a stable balance between sparse and shared experts, as well as the output of the LoRA-experts and the frozen weights.

**Table 1.** Comparisons with MTL methods on the PASCAL-Context dataset. Best results are highlighted in **bold**.

| Method | Backbone | $Seg.$ (mIoU)↑ | $Norm.$ (mErr)↓ | $H.Parts$ (mIoU)↑ | $Sal.$ (mIoU)↑ | $Edge$ (odsF)↑ | $\Delta_m$ (%)↑ | Params (M)↓ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| STL | ResNet-18 | 60.1 | **15.4** | 51.0 | 63.2 | 50.8 | 0.00 | **11** |
| MTL | ResNet-18 | 58.4 | 15.9 | 51.8 | 62.6 | 50.3 | -1.29 | **11** |
| MTAN [25] | ResNet-18 | 58.9 | 16.4 | 51.9 | 62.2 | 50.6 | -1.74 | **11** |
| NDDR-CNN [26] | ResNet-18 | 59.7 | 15.5 | 51.7 | 62.7 | 51.5 | +0.13 | **11** |
| Cross-Stitch [27] | ResNet-18 | 60.4 | **15.4** | 51.6 | 63.1 | 51.7 | +0.66 | **11** |
| MTL-ViT | ViT-base | 69.1 | 16.2 | 54.8 | 61.9 | 49.9 | +2.68 | 104 |
| Baseline (16/4/0/4) | LoRA-MoE ViT-base | 73.7 | 17.5 | 59.2 | 62.9 | 53.7 | +6.06 | 107 |
| Baseline (32/8/0/2) | LoRA-MoE ViT-base | 73.8 | 17.4 | 59.3 | 62.9 | 53.3 | +6.11 | 108 |
| **Ours (16/3/1/4)** | LoRA-MoE ViT-base | **74.0** | 17.3 | 60.1 | 63.2 | **55.3** | +7.49 | 107 |
| **Ours (32/6/2/2)** | LoRA-MoE ViT-base | **74.0** | 17.2 | **60.3** | **63.3** | 54.9 | **+7.58** | 108 |

**Fig. 3.** ASE under different top-$k$ and fine-grained settings. We adopt BCE for $Edge$, and compute $\Delta_m$ relative to a ViT-based STL model. (a) Line plot of $\Delta_m$ versus top-$k$ with $S=1$. The x-axis shows the chosen top-$k$ and the y-axis shows $\Delta_m$. (b) Bar chart of $\Delta_m$ under fine-grained allocations.

**Fig. 4.** Activation frequency of experts across layers. We visualize the activation frequency of experts for task $Norm.$ across all layers at epochs 1, 3, and 40 (out of 40). Here, the y-axis represents the layer index, and the x-axis represents the 16 experts, including one ASE. The color bar indicates the normalized frequency.

## 4. CONCLUSION

In this paper, we introduced adaptive shared experts (ASE) within a LoRA-MoE framework to address the limitations of naive shared experts in MoE-based MTL, facilitating both STL-to-MTL transition and effective expert specialization and cooperation. In addition, we incorporated the notion of fine-grained experts into MoE for MTL. Extensive experiments on PASCAL-Context demonstrated the robustness and effectiveness of ASE across different settings, as well as the benefits of fine-grained configurations for improving multi-task performance.

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
