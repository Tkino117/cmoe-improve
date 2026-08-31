ExpertWeaver: Unlocking the Inherent MoE in Dense LLMs with GLU
                                                                        Activation Patterns


                                             Ziyu Zhao 1 2 Tong Zhu 3 Zhi Zhang 4 Tiantian Fan 4 Jinluan Yang 1 Kun Kuang 1 Zhongyu Wei 5 Fei Wu 1
                                                                                           Yu Cheng 6


                                                                   Abstract                                1. Introduction
arXiv:2602.15521v1 [cs.CL] 17 Feb 2026




                                                                                                           The Mixture-of-Experts (MoE) architecture effectively ex-
                                                Mixture-of-Experts (MoE) effectively scales
                                                                                                           pands model capacity while maintaining efficiency by ac-
                                                model capacity while preserving computational
                                                                                                           tivating only sparse subsets of experts. However, train-
                                                efficiency through sparse expert activation. How-
                                                                                                           ing MoE models from scratch is rather expensive, driving
                                                ever, training high-quality MoEs from scratch is
                                                                                                           growing interest in converting pretrained dense models into
                                                prohibitively expensive. A promising alternative
                                                                                                           sparse MoE architectures. These efforts can be broadly cat-
                                                is to convert pretrained dense models into sparse
                                                                                                           egorized into two distinct lines of research based on target
                                                MoEs. Existing dense-to-MoE methods fall into
                                                                                                           sparsity and objectives. The first one is Dynamic Structural
                                                two categories: dynamic structural pruning
                                                                                                           Pruning, which converts dense models into MoE architec-
                                                that converts dense models into MoE architec-
                                                                                                           tures with moderate sparsity (e.g., 25%) to balance perfor-
                                                tures with moderate sparsity to balance perfor-
                                                                                                           mance and efficiency (Gao et al., 2025; Pei et al., 2025;
                                                mance and inference efficiency, and downcy-
                                                                                                           Nishu et al., 2025; Zheng et al., 2024). The second category
                                                cling approaches that use pretrained dense mod-
                                                                                                           is Downcycling (versus upcycling that expands models (He
                                                els to initialize highly sparse MoE architectures.
                                                                                                           et al., 2024; Komatsuzaki et al., 2022)), which initializes
                                                However, existing methods break the intrinsic ac-
                                                                                                           highly sparse MoEs (75%+ sparsity) by splitting pretrained
                                                tivation patterns within dense models, leading to
                                                                                                           dense MLPs into smaller experts, avoiding the prohibitive
                                                suboptimal expert construction. In this work, we
                                                                                                           costs of training sparse MoEs from scratch (Zhang et al.,
                                                argue that the Gated Linear Unit (GLU) mech-
                                                                                                           2022b; Zhu et al., 2024; Qu et al., 2024). However, exist-
                                                anism provides a natural blueprint for dense-
                                                                                                           ing methods break the inherent activation patterns, neces-
                                                to-MoE conversion. We show that the fine-
                                                                                                           sitating additional router training, and ignore layer-specific
                                                grained neural-wise activation patterns of GLU
                                                                                                           differences, leading to suboptimal expert construction. The
                                                reveal a coarse-grained structure, uncovering an
                                                                                                           detailed related work is shown in Appendix D.
                                                inherent MoE architecture composed of consis-
                                                tently activated universal neurons and dynam-              In this work, we argue that the key to overcoming these lim-
                                                ically activated specialized neurons. Leverag-             itations lies in the GLU mechanism (Shazeer, 2020), which
                                                ing this discovery, we introduce ExpertWeaver,             provides a natural blueprint for dense-to-MoE conversion
                                                a training-free framework that partitions neurons          by revealing the model’s intrinsic functional structure. The
                                                according to their activation patterns and con-            GLU mechanism employs additional gating weights to dy-
                                                structs shared experts and specialized routed ex-          namically control neuron activations based on the input
                                                perts with layer-adaptive configurations. Our ex-          context, providing meaningful indicators of each neuron’s
                                                periments demonstrate that ExpertWeaver signif-            role and importance for effective MoE expert construction.
                                                icantly outperforms existing methods, both as              Our analysis of the GLU mechanism in §2.2 yields key in-
                                                a training-free dynamic structural pruning tech-           sights that serve as the cornerstone of our method. Firstly,
                                                nique and as a downcycling strategy for superior           GLU gating signals naturally induce token-level activation
                                                MoE initialization.                                        sparsity (Takeaway 1). Moreover, these patterns reveal
                                                                                                           both universal neurons ideal for a shared expert and task-
                                                                                                           specific clusters suitable for routed experts (Takeaways 2
                                               1
                                                 Zhejiang University 2 Shanghai Innovation Institute       & 3). Additionally, by quantifying neuron specialization
                                         3
                                          Shanghai AI Laboratory 4 Bytedance Seed 5 Fudan Univer-
                                                                                                           with the Coefficient of Variation (CV), we find that the dis-
                                         sity 6 Chinese University of Hong Kong. Correspondence to:
                                         Ziyu Zhao <benzhao.styx@gmail.com>.                               tribution of universal versus specialized neurons varies sys-
                                                                                                           tematically across layers, indicating that different layers
                                         Preliminary work.

                                                                                                       1


                   ExpertWeaver: Unlocking the Inherent MoE in Dense LLMs with GLU Activation Patterns




                         a) Routing Distribution within a Layer                            b) Routing Distribution Similarity between Tasks

Figure 1. Neuron activation patterns across diverse tasks. We visualize the middle layer’s activation patterns from Qwen2.5-7B on
a subset of Flan-v2. a) Activation distribution of neurons across different tasks. b) Activation distribution of neurons within individual
task clusters, where tasks belonging to the same cluster are enclosed in boxes.


require different expert configurations rather than a one-              • We systematically analyze GLU activation patterns in
size-fits-all approach (Takeaway 4). Together, these ob-                  dense LLMs and find that their fine-grained activation
servations provide a complete blueprint for constructing a                patterns exhibit structural properties that provide a natu-
coarse-grained, expert-level MoE architecture from fine-                  ral blueprint for constructing MoEs with coarse-grained
grained, neural-level GLU activations.                                    activation.
Based on these findings, we propose ExpertWeaver, a                     • We propose ExpertWeaver, a novel training-free frame-
novel, training-free method that effectively converts a                   work that leverages GLU activation patterns to convert
dense model into MoE guided by GLU activation patterns.                   dense models into MoE architectures.
The process consists of three key stages. a) Firstly, the acti-         • Comprehensive experiments demonstrate that Exper-
vation pattern of all neurons is captured by recording their              tWeaver significantly outperforms existing methods in
GLU activations from a multi-task calibration dataset. b)                 both structural pruning and MoE downcycling across di-
Then, the CVs of these activations are calculated to per-                 verse downstream tasks.
form a layer-aware configuration, determining the precise
size of the shared expert and the pool of routed experts for
each layer. c) Finally, neurons are partitioned according to            2. Preliminaries
layer-wise configuration by grouping the most consistently              2.1. Background
active neurons into a single shared expert, while remain-
ing specialized neurons are clustered into routed experts               Gated Linear Unit. Most current high-performance
based on their activation patterns. The MoE router is also              LLMs have adopted the GLU architecture, with SwiGLU
constructed in a training-free manner from the GLU gating               being the most commonly used variant (Shazeer, 2020). A
centroids. This entire process successfully “weaves” the                dense GLU layer processes an input x using three distinct
neurons of a dense model into multiple experts, construct-              weight matrices: Wgate , Wup , and Wdown . The entire com-
ing a well-structured MoE architecture.                                 putation can be expressed as:

We conduct extensive experiments to validate the effective-                      y = (Swish(xWgate ) ⊙ (xWup )) Wdown .
ness of ExpertWeaver across two categories of methods. As
                                                                        The element-wise product ⊙ combines the input projec-
a training-free dynamic structural pruning method, Exper-
                                                                        tion with a dynamic gate that modulates neuron activation
tWeaver significantly outperforms existing structural prun-
                                                                        strength for each token. This gating mechanism provides
ing baselines on multiple benchmarks. Furthermore, when
                                                                        dense GLU models with intrinsic dynamic sparsity, which
applied as a downcycling strategy, our method successfully
                                                                        we identify as the key to efficiently converting pretrained
initializes a highly sparse MoE model that, with limited
                                                                        dense models into structured sparse MoEs.
continued pretraining, achieves superior performance com-
pared to both established MoE and dense baselines with
comparable parameters and training budgets. The main                    Mixture-of-Experts. The MoE architecture replaces
contributions of this paper are as follows:                             dense Feed-Forward Networks (FFNs) in LLMs with mul-
                                                                        tiple smaller expert networks and a gating mechanism for

                                                                    2


                   ExpertWeaver: Unlocking the Inherent MoE in Dense LLMs with GLU Activation Patterns

sparse activation. Specifically, an MoE layer consists of             Table 1. Performance of AbsTopk-GLU vs. unstructured
                                                                      pruning at 50% sparsity. (Numbers in parentheses denote the
N experts (E1 , . . . , EN ) and a router network G that dy-          number of shots for evaluation; no annotation means zero-shot.)
namically selects which experts to activate for each input
token:                                                                 Method        MMLU(5)   HellaSwag(10)    ARC-e     ARC-C(25)   PiQA   Avg.
                   X                                                                           LLaMA3-8B, 50% sparsity
        y=                   g(x)i Ei (x);                             Dense           65.3         82.1          77.9      57.9      80.8   72.8
                                                                       Wanda           55.8         75.0          72.0      51.3      77.3   66.3
              i∈TopK(G(x))                                  (1)        SparseGPT       57.4         75.5          72.0      50.3      78.1   66.7
                                                                     Magnitude       48.3         41.9          53.3      38.7      65.8   49.6
                          (i)         (i)          (i)
   Ei (x) =       Swish(xWgate ) ⊙ (xWup  )       Wdown ,              AbsTopk-GLU     58.8         79.4          72.4      51.6      78.6   68.2
                                                                                               Qwen2.5-7B, 50% sparsity
                                                                       Dense           74.2         80.3          77.8      63.8      80.0   75.2
where g(x)i represents the normalized gating weight for                Wanda           69.2         74.1          75.5      55.5      78.1   70.5
expert i, and each expert has its own parameters. By acti-             SparseGPT       69.8         75.9          74.7      56.7      78.3   71.1
                                                                       Magnitude       64.2         49.5          46.3      33.6      63.8   51.5
vating only a subset of experts per token, MoE maintains               AbsTopk-GLU     70.8         78.6          76.6      59.7      78.9   72.9
model capacity while significantly reducing computational
costs.

2.2. Motivation                                                       sparsity of dense models and validates the use of GLU
                                                                      gate scores as a reliable mechanism for controlling acti-
As shown in §2.1, GLU and MoE represent two distinct                  vation patterns. However, this fine-grained expert parti-
granularities of activation patterns. GLU implements fine-            tioning, where each neuron is treated as an individual ex-
grained activation at the neuron level through a dynamic              pert, is impractical for hardware implementation. While
gate that modulates neuron activation strength for each               it reduces theoretical FLOPs, the overhead of TopK selec-
token. In contrast, MoE employs coarse-grained activa-                tion and scattered memory access prevents real-world ap-
tion by routing entire tokens to selected experts, establish-         plications. Nonetheless, this experiment reveals the crucial
ing macroscopic, structural activation at the expert level.           insight that the GLU gating signal’s control over neuron-
This difference between neural-level and expert-level acti-           level activation can be extended to manage coarse-grained
vations raises an intriguing question: can the fine-grained           experts, motivating our approach of grouping neurons into
neuron-level activation patterns of GLU be aggregated to              larger, hardware-friendly blocks.
inform the construction of a coarse-grained, expert-level
activated MoE architecture? Our exploratory analysis of               2.2.2. I DENTIFYING U NIVERSAL AND S PECIALIZED
the GLU activation patterns in pretrained dense LLMs                         N EURONS VIA GLU ACTIVATION PATTERNS
yields four key insights that form the cornerstone of Ex-
pertWeaver.                                                           To investigate the structure of GLU activation patterns, we
                                                                      analyzed activation patterns recorded for 5 few-shot sam-
2.2.1. GLU AS A NATURAL S PARSITY S IGNAL                             ples per task in the Flan-v2 collection (48 tasks, 10 task
                                                                      clusters; details in Appendix H). The results in Figure 1
As established in §2.1, the gating mechanism in a GLU,                clearly illustrate two distinct and complementary phenom-
Swish(xWgate ), acts as a dynamic, data-dependent filter.             ena, which inform the following observations.
To investigate the feasibility of leveraging inherent sig-
nals to directly enforce structured sparsity, we introduce Takeaway 2: There exists a core set of universally im-
AbsTopk-GLU, a simple modification that explicitly intro-  portant neurons that are consistently activated across
duces sparsity into FFNs based on their gating activation  different tasks. As shown in Figure 1a), we observed that
scores:                                                    a consistent subset of neurons showed high activation re-
                                                           gardless of the task domain. We hypothesize that these
AbsTopk-GLU(x) = (AbsTopK(Swish(xWgate ), k))⊙(xWup ).neurons encode task-agnostic knowledge. This observation
                                                       (2) motivates the design of MoE architectures that incorpo-
Here, AbsTopK(v, k) preserves the top k values in |v|      rate a shared expert specifically dedicated to capturing and
while zeroing out the rest, effectively retaining only the leveraging this task-agnostic knowledge, thereby enhanc-
most activated neurons for each token.                     ing both parameter efficiency and overall performance.
Takeaway 1: The GLU’s gating mechanism naturally                      Takeaway 3: Specialized neurons exhibit task-specific
induces structured sparsity within dense LLMs. As                     co-activation patterns. As illustrated in Figure 1b), the
shown in Table 1, AbsTopK-GLU, which activates only                   activation pattern similarity heatmap, generated from trun-
50% of neurons, consistently and significantly surpasses              cated and normalized neuron activations, reveals clear
established unstructured pruning methods such as Wanda                block-diagonal structures, where semantically related tasks
(Sun et al., 2023) and SparseGPT (Frantar & Alistarh,                 show high similarity in specialized neuron activation pat-
2023). This performance highlights the inherent structural            terns. This demonstrates that neurons naturally form co-

                                                                  3

ExpertWeaver: Unlocking the Inherent MoE in Dense LLMs with GLU Activation Patterns
                     27
                     26
                     25
                                                                                             3. Methodology
                     24
                     23
                     22
                                                                                             This section introduces ExpertWeaver, a training-free
                     21
                     20                                                                      framework that converts dense LLMs into efficient MoE
                     19
                     18                                                                      architectures through a three-stage process: a) Capturing
                     17
                     16
                     15
                                                                                             multi-task GLU activation patterns for each layer; b) Lever-
          Layer ID




                     14
                     13
                                                                                             aging the CVs of these activations to determine the layer-
                     12
                     11                                                                      specific ratio of shared to routed experts; c) Constructing
                     10
                      9                                                                      shared experts from universal neurons, clustering special-
                      8
                      7
                      6
                                                                                             ized neurons into routed experts, and building the MoE
                      5
                      4
                                                                                             router.
                      3
                      2
                      1
                      0
                          0.00     0.15   0.30            0.45       0.60   0.75
                                                                                             3.1. Capturing Multi-Task Gating Activation Patterns
                                          Coefficient of Variation


  Figure 2. Neuron Coefficient of Variation Across Layers.
                                                                                             A standard SwiGLU-based FFN layer (Shazeer, 2020) con-
                                                                                             sists of three weight matrices: Wgate ∈ Rdmodel ×dffn , Wup ∈
                                                                                             Rdmodel ×dffn , and Wdown ∈ Rdffn ×dmodel . Here, dmodel is the
                                                                                             model’s hidden dimension and dffn is the dimension of the
activation clusters organized by task semantics, provid-                                     intermediate layer. Layer indices are omitted for notational
ing a principled approach for constructing routed experts                                    simplicity. We define a neuron slice sj for each neuron
through clustering based on multi-task activation patterns.                                  j ∈ {1, . . . , dffn } as:

Takeaways 2&3 collectively provide a blueprint for dense-
                                                                                                       sj = ((Wgate ):,j , (Wup ):,j , (Wdown )j,: ).
to-MoE conversion by forming shared experts from univer-
sal neurons and constructing routed experts through clus-
tering specialized neurons based on their co-activation pat-                                 The neuron slices are independent of each other. Exper-
terns.                                                                                       tWeaver aims to partition the neuron slices {sj }dj=1
                                                                                                                                                ffn
                                                                                                                                                    into
                                                                                             different expert groups.
2.2.3. L AYER -AWARE E XPERT A LLOCATION                                                     As established in §2.2, GLU activation patterns are a rich
Building on our discovery of universal and specialized neu-                                  source of information about the model’s inherent struc-
rons in §2.2.2, we next investigate how the ratio of univer-                                 ture. We use a multi-task calibration set Dcalib from Flan-
sal to specialized neurons varies across layers. To quantify                                 v2 (42 tasks, 5 samples each) to capture robust activa-
this layer-wise behavior, we measure each neuron’s acti-                                     tion signals. For each sample j in Dcalib , we compute
vation consistency using the Coefficient of Variation (CV),                                  the token-averaged gate activation for neuron i as aij =
which is defined as the ratio of the standard deviation to the                               mean(Swish(xj Wgate ))i , where xj represents all tokens in
mean (Abdi, 2010):                                                                           sample j. This gives the activation profile of neuron i as
                                                                                             ai = [ai1 , ai2 , . . . , aiM ] with M = |Dcalib |. We collect
                                                                                             all activation profiles into a matrix A = [a1 , a2 , . . . , adffn ],
                                                          σj                                 which serves as the primary signal for our subsequent pro-
                                   CV(aj ) =                   ,                   (3)
                                                        µj + ϵ                               cess.

                                                                                             3.2. Layer-Aware Expert Allocation
where µj = Et∈T [āj,t ] and σj = STDt∈T [āj,t ] are the
mean and standard deviation of the neuron’s average abso-                                    Building on our observation that the ratio of universal to
lute activation, āj,t , across the tasks in our calibration set                             specialized neurons varies across layers (Takeaway 4), we
Dcalib . A low CV indicates a universal neuron with con-                                     propose a layer-adaptive allocation strategy that determines
sistent activation, while a high CV points to a specialized                                  the shared expert ratio for each layer.
neuron that activates selectively.
Takeaway 4: Layers exhibit different levels of neuron                                        Quantifying Layer-wise Neural Specialization. First,
specialization. Figure 2 shows that boundary layers (shal-                                   we quantify the functional specialization of each layer ℓ.
low and deep) consistently have low CV scores, indicat-                                      We compute the CV based on each neuron’s activation pat-
ing universal neuron behavior, while middle layers exhibit                                   tern Aℓ . A high CV indicates that a neuron is highly spe-
diverse CVs with many highly specialized neurons. This                                       cialized, activating only for specific inputs. We then define
layer-wise heterogeneity requires layer-specific expert con-                                 the layer’s overall specialization ratio, rℓ , as the fraction
figurations rather than uniform MoE conversion.                                              of neurons whose CV exceeds a CV threshold (τ ) used to

                                                                                         4


                                         ExpertWeaver: Unlocking the Inherent MoE in Dense LLMs with GLU Activation Patterns

                    up_proj                                                                                                                3.3. MoE Layer Construction
                                                                                     Multi-task Fewshot Samples
                                                          down_proj                                                                        Once the layer-specific allocations of shared experts
                                                                                                              silu&abs                     (Nse,ℓ ) and routed experts (Nre,l ) are determined, we pro-
                                                                                         x
                                                                                                                                           ceed to partition the FFN’s neurons into shared and routed
                                                                                 y
       x                                                                                          gate_proj
                                  silu

                                               Activation Signal                                            activation pattern
                                                                                                                                           experts.
                    gate_proj                       a) GLU in MLP                     b) Record Activation Pattern


                                                                                                      Expert 1
                                                                                                                                           Constructing Shared Experts. The shared expert is
                                                             activation clustering
                                                                                                                                           formed from the most universally active neurons. We se-
                                                                                                                     MoE Gate
                                                                                                      Avg

           activation pattern            CVs of Neurons
                                                                                                                                           lect the top dexpert · Nse,ℓ neurons with the highest abso-
                                                                                                     Expert 2
                                                                                                                                           lute average activation scores to form the shared neuron
                                                             most activated
                                                                                                     Avg
                                                                                                                                           pool, indexed by Is . These neuron slices are then con-
                                                                                                    Shared Expert
     Quantify Neuron Specialization

           c) Layer-wise Configuration
                                      Shared Expert Budget
                                                                                                                                           catenated to form the weight matrices of a single, consol-
                                                                       d) Expert and Gate Construction
                                                                                                                                           idated shared expert. Specifically, the weight matrices for
                                                                                                                                           the shared expert are constructed by concatenating the cor-
Figure 3. The ExpertWeaver Framework. a) The GLU in the
MLP layer contains three weight matrices, where the same color                                                                             responding neural slices from the original dense layer:
denotes corresponding neuron slices. b) Neuron activation pat-
terns are captured using a multi-task calibration dataset. c) The                                                                                          (s)
                                                                                                                                                       Wgate/up = CONCAT((Wgate/up ):,j )               (7)
CVs are computed to determine the budget for shared vs. routed                                                                                                         j∈Is
experts. (d) Neurons are clustered according to their activation                                                                                            (s)
patterns to form one shared expert and multiple routed experts.                                                                                          Wdown = CONCAT((Wdown )j,: )                   (8)
                                                                                                                                                                       j∈Is


                                                                                                                                           This shared expert is always activated, capturing and con-
identify specialized neurons:
                                                                                                                                           solidating common knowledge across varying contexts.
                                                       dffn
                                                   1 X
                                      rℓ =                  I[CVj > τ ],                                                         (4)       Constructing Routed Experts. The remaining Nre,ℓ ·
                                                  dffn j=1
                                                                                                                                           dexpert neurons, which form the specialized pool (indexed
                                                                                                                                           by I−s ), are partitioned into Nre,ℓ routed experts. To
where I[·] is the indicator function and τ is a specialization
                                                                                                                                           group neurons that are frequently co-activated into the
threshold.
                                                                                                                                           same expert, we employ a balanced K-Means clustering al-
                                                                                                                                           gorithm (Malinen & Fränti, 2014) on their activation pat-
Dynamic Allocation of Shared Expert Size. Next, we                                                                                         tern vectors {ai }i∈I−s (see Appendix I for details of bal-
use this specialization ratio to determine the proportion of                                                                               anced K-Means). This process partitions the set of special-
neurons, αℓ , to be allocated to the shared expert in that                                                                                 ized neurons into Nre,ℓ disjoint clusters {C1 , . . . , CNre,ℓ },
layer. The core principle is that more specialized layers                                                                                  where each cluster contains exactly dexpert neurons and cor-
(higher rℓ ) require a smaller shared expert. We compute αℓ                                                                                responds to a single routed expert. The weight matrices for
using a linear mapping:                                                                                                                    the i-th expert are formed by concatenating the weights of
                                                                                                                                           all neurons in cluster Ci :
                        αℓ = αmax − (αmax − αmin ) · rℓ ,                                                                        (5)
                                                                                                                                                            (i)
where αmin and αmax define the bounds for the shared                                                                                                     Wgate/up = CONCAT(Wgate/up,j )                 (9)
                                                                                                                                                                         j∈Ci
expert neuron ratio. In our framework, the total dffn neu-                                                                                                       (i)
rons are distributed among Ne experts, giving each expert                                                                                                 Wdown = CONCAT(Wdown,j )                     (10)
                                                                                                                                                                         j∈Ci
a fixed capacity of dexpert = dffn /Ne neurons. The number
of shared experts, Nse,ℓ , is then determined by the number
of full experts that can be formed from the allocated shared                                                                               Constructing the MoE Router. We construct our MoE
neurons by:                                                                                                                                router in a training-free manner by leveraging the clustering
                                                                                                                                           structure established in the previous step. The key insight
 ds,ℓ = round(αℓ · dffn ),    Nse,ℓ = round(ds,ℓ /dexpert ).                                                                               is that the original gating vector of each neuron, (Wgate ):,j ,
                                                         (6)                                                                               controls that neuron’s activation patterns. Therefore, the
The remaining Nre,l = Ne − Nse,ℓ experts are designated                                                                                    centroid of these vectors within each expert cluster natu-
as routed experts. For each input token, we activate a to-                                                                                 rally captures that expert’s representative activation behav-
tal of k experts according to the required sparsity. This                                                                                  ior. Thus, we construct the router Wrouter ∈ Rdmodel ×Ne,ℓ by
includes activating all Nse,ℓ shared experts, plus the top                                                                                 calculating a representative gating vector for each routed
k − Nse,ℓ routed experts as determined by the router.                                                                                      expert i through averaging the gating vector of all neurons

                                                                                                                                       5


                       ExpertWeaver: Unlocking the Inherent MoE in Dense LLMs with GLU Activation Patterns

assigned to that cluster:                                                      Table 2. Comparison with structured pruning methods under
                                                                               25% sparsity.
   (i)       1 X                           h
                                              (1)              (N )
                                                                      i
 w̄gate =         (Wgate ):,j ,   Wrouter = w̄gate , . . . , w̄gatee,ℓ .        Method         MMLU(5)    HellaSwag(10)     ARC-e   ARC-c    PiQA    Avg.
            |Ci |                                                                                        LLaMA3-8B, 25% sparsity
                j∈Ci
                                                        (11)                    Dense            65.3           82.1         77.9   57.9      80.8   72.8
                                                                                LLM-Pruner       24.2           51.3         58.9   32.4      74.4   48.2
By directly using the original gating weights, this approach                    FLAP             33.4           48.0         50.0   29.3      68.3   45.8
constructs the router without training while preserving the                     CMoE             41.6           65.9         63.1   41.5      73.9   57.2
                                                                                ExpertWeaver     47.0           69.8         64.4   44.3      76.3   60.4
neuron activation patterns captured during pretraining.
                                                                                                         Qwen2.5-7B, 25% sparsity
                                                                                Dense            74.2           80.3         77.8   63.8      80.0   75.2
3.4. ExpertWeaver in Practice: Dynamic Structural                               LLM-Pruner       55.9           72.2         71.0   49.1      77.0   65.0
                                                                                FLAP             54.7           58.5         67.3   42.2      70.8   58.7
     Pruning and Downcycling                                                    ExpertWeaver     61.6           72.3         71.5   53.5      76.3   67.0

3.4.1. T RAINING - FREE DYNAMIC S TRUCTURAL
       P RUNER                                                                 bination of the shared expert and the top-k routed experts:
When operating at low sparsity levels without requiring                                                   X
additional training, ExpertWeaver can be viewed as a dy-                            y = Eshared (x) +              g(x)i · Ei (x).    (13)
namic structural pruning method. For any given input x,                                                     i∈TopK(g(x))

the router selects the top-k routed experts based on the log-                  The total loss function consists of two components: the
its from the reconstructed router, T (x) = TopK(xWg ).                         next-token prediction loss (LNTP ) and an auxiliary load-
The final output is the direct sum of the outputs from the                     balancing loss (LLB ) to encourage a balanced distribution
always-active shared experts and these selected routed ex-                     of tokens across experts.
perts:
                                   X                                                                                                Ne,ℓ
               y = Eshared (x) +        Ei (x)           (12)                                                                       X
                                                                                 Ltotal = LNTP + λLLB ;           LLB = Ne,ℓ ·             fi · Pi , (14)
                                      i∈T (x)
                                                                                                                                    i=1
It is worth noting that, instead of using a weighted com-                      where fi is the fraction of tokens routed to expert i and Pi
bination of expert outputs, our gate acts as a structured                      is its average router probability within a batch.
pruning mechanism, dynamically selecting which neurons
to exclude from the forward pass. This preserves the in-
tegrity of the original neuron-level computations from the                     4. Experiments
pretrained model, as the GLU mechanism within each acti-                       We evaluate ExpertWeaver in the following two scenarios.
vated expert still governs the precise activation values.                      First, as a training-free dynamic structured pruner for low-
                                                                               sparsity settings, we benchmark it against some training-
3.4.2. D OWNCYCLING D ENSE LLM S INTO M O E S
                                                                               free baselines in Section 4.1. Second, as an initialization
Model downcycling aims to convert large, pretrained dense                      strategy for model downcycling in higher-sparsity settings,
models into computationally efficient MoEs, using the                          we demonstrate its advantages by comparing our result-
dense model’s weights as a superior initialization to avoid                    ing MoE models against similarly scaled baselines in Sec-
the prohibitive costs of training from scratch. Exper-                         tion 4.2.
tWeaver performs downcycling by converting dense mod-
els into sparse MoE architectures, followed by continued                       4.1. ExpertWeaver for Dynamic Structural Pruning
Pretraining (CPT) for further optimization.
                                                                               Experimental Setup. We benchmark ExpertWeaver
                                                                               against training-free baselines including FLAP (An et al.,
Initialization. Since the sparsity in downcycling is often                     2024), LLM-Pruner (Ma et al., 2023), and CMoE (Pei
higher and configured according to downstream require-                         et al., 2025) on Qwen2.5-7B and Llama3-8B models. Since
ments, we use a fixed, uniform shared expert ratio across                      CMoE is only implemented for Llama-series models, we
all layers. This provides a robust and well-structured start-                  only compared it on Llama3-8B. Following previous set-
ing point for the subsequent training phase.                                   tings (Ma et al., 2023), we set the target sparsity level to
                                                                               25% for all methods. Model performance is assessed on
Continued Pretraining. During CPT, we switch to a                              five widely-used benchmarks: MMLU, HellaSwag, ARC-
standard softmax router to provide greater optimization                        e, ARC-c, and PIQA. For ExpertWeaver, we use our default
flexibility and enable experts to learn distinct specializa-                   hyperparameters: a shared expert ratio with αmin = 0.2
tions. The softmax router computes gating weights g(x) =                       and αmax = 0.7, a specialization threshold of τ = 0.6,
Softmax(xWrouter ). The layer’s output is a weighted com-                      and an expert granularity of 64. The specific layer-wise

                                                                           6

ExpertWeaver: Unlocking the Inherent MoE in Dense LLMs with GLU Activation Patterns

allocation between shared and routed experts is shown in            Table 3. Downcycling Performance Comparison. Best re-
                                                                    sults are in bold, second-best are underlined. Models in gray
Appendix N and the detailed ablation studies on hyperpa-            (Qwen2.5-{7B, 3B, 1.5B}, Llama-3.2-3B, and OLMoE) are for
rameters is shown in Appendix B. To construct an effec-             context, as they were trained on significantly more data. Our
tive calibration set that captures diverse multi-task activa-       ExpertWeaver models (OLMo-based and Qwen2.5-based) were
tion patterns, we sample 5 instances from each of the 42            trained on 200B tokens. The OLMo-based version has 1B active
tasks in the Flan-v2 dataset, which spans 10 distinct task          parameters (7B-A1B), and the Qwen2.5-based version has 3.5B
                                                                    active parameters (7B-A3.5B). We compare against the 500B to-
clusters (details in Appendix H).
                                                                    ken checkpoint (*) of OLMoE. Blank entries for OpenMoE indi-
                                                                    cate unavailable results.
Main Results. Table 2 summarizes the performance                    Model
                                                                    Dense Models
                                                                                                          MMLU(5)   HellaSwag(10)   ARC-e   ARC-c(25)   PIQA   WinoGrande   LogiQA   SciQ   Average


comparison between ExpertWeaver and other training-free             Qwen2.5-7B
                                                                    Qwen2.5-3B
                                                                                                            74.1
                                                                                                            65.6
                                                                                                                        80.2
                                                                                                                        74.6
                                                                                                                                     77.5
                                                                                                                                     73.9
                                                                                                                                              63.7
                                                                                                                                              56.5
                                                                                                                                                        79.7
                                                                                                                                                        78.8
                                                                                                                                                                  73.2
                                                                                                                                                                  68.1
                                                                                                                                                                             36.4
                                                                                                                                                                             33.5
                                                                                                                                                                                     95.2
                                                                                                                                                                                     95.2
                                                                                                                                                                                             72.5
                                                                                                                                                                                             68.3
                                                                    Qwen2.5-1.5B                            60.9        68.0         72.5     54.9      75.9      63.8       31.9    93.4    65.2
structural pruning baselines at 25% sparsity. From these            Llama-3.2-3B
                                                                    OLMo-7B
                                                                                                            56.1
                                                                                                            30.7
                                                                                                                        76.4
                                                                                                                        77.1
                                                                                                                                     71.6
                                                                                                                                     68.7
                                                                                                                                              50.5
                                                                                                                                              45.1
                                                                                                                                                        77.4
                                                                                                                                                        79.6
                                                                                                                                                                  69.9
                                                                                                                                                                  66.5
                                                                                                                                                                             30.6
                                                                                                                                                                             27.5
                                                                                                                                                                                     92.7
                                                                                                                                                                                     88.6
                                                                                                                                                                                             65.7
                                                                                                                                                                                             60.8
                                                                    OPT-2.7B                                25.8        61.4         54.4     34.0      74.8      60.8       25.8    78.9    52.0
results, we observe the following key findings: (1) Ex-             Pythia-2.8B
                                                                    INCITE-Base-3B
                                                                                                            26.8
                                                                                                            27.2
                                                                                                                        60.7
                                                                                                                        64.7
                                                                                                                                     58.8
                                                                                                                                     61.7
                                                                                                                                              36.7
                                                                                                                                              40.3
                                                                                                                                                        73.6
                                                                                                                                                        73.9
                                                                                                                                                                  59.6
                                                                                                                                                                  63.5
                                                                                                                                                                             28.1
                                                                                                                                                                             27.5
                                                                                                                                                                                     83.2
                                                                                                                                                                                     85.6
                                                                                                                                                                                             53.4
                                                                                                                                                                                             55.6

pertWeaver consistently outperforms all competing ap-               Open-LLaMA-3B-v2
                                                                    Sheared-LLaMA-2.7B
                                                                    Gemma-2-2b
                                                                                                            26.8
                                                                                                            27.3
                                                                                                            53.0
                                                                                                                        71.4
                                                                                                                        71.0
                                                                                                                        69.0
                                                                                                                                     63.3
                                                                                                                                     63.3
                                                                                                                                     36.9
                                                                                                                                              40.1
                                                                                                                                              41.6
                                                                                                                                              52.6
                                                                                                                                                        77.9
                                                                                                                                                        76.9
                                                                                                                                                        67.5
                                                                                                                                                                  63.1
                                                                                                                                                                  65.0
                                                                                                                                                                  51.9
                                                                                                                                                                             28.1
                                                                                                                                                                             28.3
                                                                                                                                                                             22.7
                                                                                                                                                                                     88.0
                                                                                                                                                                                     87.5
                                                                                                                                                                                     75.8
                                                                                                                                                                                             57.3
                                                                                                                                                                                             57.6
                                                                                                                                                                                             53.7
proaches, achieving a 5.6% relative improvement over                SmolLM2-1.7B
                                                                    MoE Models
                                                                                                            50.4        72.6         73.4     53.2      76.0      65.8       30.1    84.3    63.2


CMoE on LLaMA3-8B and a 3.1% advantage over LLM-                    LLaMA-MoE-v1-3.5B
                                                                    OpenMoE-3B-9B
                                                                                                            26.8
                                                                                                             -
                                                                                                                        73.3
                                                                                                                        56.5
                                                                                                                                     65.6
                                                                                                                                     50.6
                                                                                                                                              44.2
                                                                                                                                              33.3
                                                                                                                                                        77.9
                                                                                                                                                        65.7
                                                                                                                                                                  65.5
                                                                                                                                                                  51.9
                                                                                                                                                                             29.7
                                                                                                                                                                              -
                                                                                                                                                                                     87.6
                                                                                                                                                                                      -
                                                                                                                                                                                             58.8
                                                                                                                                                                                              -
                                                                    OLMoE-1B-7B                             53.8        79.6         76.3     55.6      80.1      68.4       29.3    94.9    67.2
Pruner on Qwen2.5-7B. (2) Dense-to-MoE methods signif-              OLMoE-1B-7B*
                                                                    LLaMA-MoE-v2-3.5B
                                                                                                            28.4
                                                                                                            40.9
                                                                                                                        70.2
                                                                                                                        53.7
                                                                                                                                     71.0
                                                                                                                                     57.0
                                                                                                                                              43.9
                                                                                                                                              40.2
                                                                                                                                                        77.1
                                                                                                                                                        67.9
                                                                                                                                                                  63.5
                                                                                                                                                                  56.1
                                                                                                                                                                             28.7
                                                                                                                                                                             30.7
                                                                                                                                                                                     88.5
                                                                                                                                                                                     88.8
                                                                                                                                                                                             58.9
                                                                                                                                                                                             54.4
                                                                    ExpertWeaver-E64-A14-S2(OLMo-7B)        45.0        61.2         69.3     38.8      74.5      62.1       28.5    91.8    58.9
icantly surpass static pruning approaches like FLAP and             ExpertWeaver-E64-A14-S2(Qwen2.5-7B)     45.6        73.7         72.4     56.3      78.0      65.3       29.0    87.7    63.5


LLM-Pruner. Rather than permanently removing param-
eters, these methods achieve sparsity by selectively acti-
vating parameter subsets based on input context, avoid-             V2 (Qu et al., 2024), focusing first on general conversa-
ing inevitable knowledge loss. (3) ExpertWeaver outper-             tional abilities and then on code and math skills. The result-
forms CMoE for three key reasons. First, our comprehen-             ing instruction-tuned model is then compared with other
sive analysis of GLU activation patterns ensures that the           instruction-tuned baselines on MMLU, ARC-c, GSM8K,
original activation structure remains intact during conver-         HumanEval, and IFEval. Further training details are avail-
sion. Second, our multi-task calibration set enables better         able in Appendix L and the comparation results is shown
expert clustering by capturing diverse neural co-activation         in Appendix C.
patterns (ablation in Appendix G). Third, our layer-specific
configuration allows adaptive MoE construction tailored to          Main Results. Table 3 presents the results of compar-
each layer’s specialization characteristics. We also com-           ing ExpertWeaver with other models of comparable pa-
pared our method against other structural pruning tech-             rameters and training budgets, yielding the following ob-
niques on more complex tasks and evaluated its perfor-              servations. (1) Among models with comparable training
mance on a reasoning model, as detailed in Appendix E.              budgets and parameter counts, our ExpertWeaver(Qwen2.5-
                                                                    7B) achieves the best average performance (63.5), outper-
                                                                    forming the strongest MoE baseline (OLMoE-1B-7B*) by
4.2. Downcycling with ExpertWeaver.
                                                                    4.6 points. (2) To ensure a fair comparison, we also
Experiment Setup. To evaluate ExpertWeaver as a                     downcycled OLMo-7B. After just 200B tokens of con-
downcycling strategy, we initialize two MoE variants. The           tinued pretraining, our ExpertWeaver(OLMo-7B) achieves a
first, ExpertWeaver (Qwen2.5-7B), is initialized from the           score of 58.9. This performance represents 97.35% of
dense Qwen2.5-7B model. It divides the MLP layers into              the original OLMo-7B model’s performance (60.5). No-
62 routed experts and 2 shared experts, activating 14 routed        tably, this result not only nears the performance of the
and 2 shared experts per token (7B total parameters, 3.5B           original model but also matches the performance of the
active). This model undergoes CPT for 200B tokens on                OLMoE-1B-7B* model, which was trained from scratch
the FineWeb-Edu dataset (Penedo et al., 2024). For a                on more than double the data (500B tokens). While a
fairer comparison with the OLMoE baseline, we also ini-             slight difference in scores may be attributed, in part, to
tialize a second variant, ExpertWeaver (OLMo-7B), from              the potentially lower quality of the Dolma dataset used
the OLMo-7B base model. This variant is continually pre-            by OLMo compared to OLMoE-Mix (Muennighoff et al.,
trained for 200B tokens on the same dataset used for OL-            2024), this result strongly highlights the effectiveness of
MoE. The performance of both models is evaluated on a               our ExpertWeaver method. It demonstrates the capacity
wide range of tasks, including MMLU, HellaSwag, ARC-                to efficiently downcycle a large dense model into a high-
e, ARC-c, PIQA, WinoGrande, LogiQA, and SciQ, and                   performing MoE architecture with only a minimal amount
compared against dense and MoE baselines with similar               of continued training data. (3) Compared to its dense coun-
parameter counts and training budgets (see Appendix J for           terpart, ExpertWeaver(Qwen2.5-7B) retains 87.6% of the base
details). Following CPT, we conduct a two-stage super-              Qwen2.5-7B’s performance while activating only a quar-
vised fine-tuning (SFT) process, similar to Llama-MoE-              ter of the MLP parameters. This remarkable performance

                                                                7


                         ExpertWeaver: Unlocking the Inherent MoE in Dense LLMs with GLU Activation Patterns




Figure 4. Comparison of Downcycling, Upcycling, and From-Scratch Training. Comparison of training loss, evaluation loss, and
downstream task performance using the same OLMoE model configuration under three different MoE initialization paradigms.


retention, achieved with a modest 200B tokens of train-                            son and demonstrate the effectiveness of ExpertWeaver, we
ing, strongly validates the effectiveness of our downcycling                       used OLMo-1.3B (pre-trained on 1T tokens) as the base
strategy. Furthermore, when compared to dense models                               model for our downcycling process. For direct contrast,
with similarly activated parameter counts like Qwen2.5-3B                          we also performed sparse upcycling (Muennighoff et al.,
and Llama-3.2-3B, our model achieves 93.0% and 96.7%                               2024), training a 575M OLMo model for 1T tokens to yield
of their respective performance. Given that these models                           a 1.3B OLMoE model (with 676M activated parameters).
were trained on massive datasets (18T and 9T tokens, re-                           Both our downcycling method and the upcycling approach
spectively), this result further demonstrates that our down-                       are compared against a baseline trained from scratch. De-
cycling method is a highly efficient path to creating perfor-                      tailed model configurations are provided in Appendix K.
mant MoE models.                                                                   As illustrated in Figure 4, which plots the training loss,
           4.00
                                                                                   evaluation loss, and downstream task performance, the re-
                                                           Random Init
           3.75                                            Llama MoE Init          sults lead to several key observations. Initially, both the
                                                           Ours
           3.50
                                                                                   downcycling and upcycling strategies exhibit faster con-
           3.25
                                                                                   vergence compared to the from-scratch baseline. How-
                                                                                   ever, over a longer training period, the performance of the
     lm loss




           3.00

           2.75
                                                                                   upcycled model regresses toward that of the from-scratch
           2.50
                                                                                   baseline. In contrast, our downcycling approach with Ex-
           2.25
                                                                                   pertWeaver consistently achieves the best performance and
           2.00
                  1000         2000          3000   4000                5000
                                                                                   convergence, maintaining its superiority throughout the en-
                                      Step                                         tire training process. By inheriting a richer feature space
Figure 5. Training Loss Comparison with Different MoE Ini-                         from a larger model and avoiding parameter duplication,
tialization Strategies.                                                            downcycling offers more diverse experts and a higher opti-
                                                                                   mization ceiling than weight-duplicating upcycling, which
                                                                                   risks local optima.
Compare with other MoE initialization methods. Fig-
ure 5 shows the training loss for the first 5,000 steps (ap-
proximately 20B tokens), highlighting the effectiveness of                         5. Conclusion
our ExpertWeaver as a downcycling strategy. Compared to                            In this paper, we investigate the problem of converting
both random and Llama-MoE initializations, our method                              dense models into high-performance MoEs. We show that
consistently yields a lower training loss throughout the en-                       GLU activation patterns provide a rich source for identify-
tire training process. This demonstrates that ExpertWeaver                         ing latent neuron specialization, which naturally enables ef-
provides a more effective starting point for the MoE, en-                          fective expert construction. Based on this observation, we
abling faster convergence and superior final performance.                          introduce ExpertWeaver, a training-free method that parti-
The persistent gap between our loss curve (green) and the                          tions neurons into shared and routed experts in a layer-wise
others validates that our strategy better preserves the foun-                      manner based on their activation patterns. Experiments
dational knowledge of the dense model during conversion.                           demonstrate that ExpertWeaver significantly outperforms
                                                                                   existing methods in both zero-shot pruning for inference
A Direct Comparison of Downcycling, Upcycling, and                                 efficiency and MoE initialization for model downcycling.
From-Scratch Training. To provide a fairer compari-

                                                                               8


                  ExpertWeaver: Unlocking the Inherent MoE in Dense LLMs with GLU Activation Patterns

References                                                             question answering? try arc, the ai2 reasoning challenge.
                                                                       arXiv preprint arXiv:1803.05457, 2018.
Abdi, H. Coefficient of variation. Encyclopedia of research
  design, 1(5):169–171, 2010.                                        Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H.,
Allal, L. B., Lozhkov, A., Bakouch, E., Blázquez, G. M.,              Kaiser, L., Plappert, M., Tworek, J., Hilton, J., Nakano,
  Penedo, G., Tunstall, L., Marafioti, A., Kydlı́ček, H.,             R., et al. Training verifiers to solve math word problems.
  Lajarı́n, A. P., Srivastav, V., et al. Smollm2: When                 arXiv preprint arXiv:2110.14168, 2021.
  smol goes big–data-centric training of a small language            Frantar, E. and Alistarh, D. Sparsegpt: Massive language
  model. arXiv preprint arXiv:2502.02737, 2025.                        models can be accurately pruned in one-shot. In Inter-
An, Y., Zhao, X., Yu, T., Tang, M., and Wang, J.                       national conference on machine learning, pp. 10323–
  Fluctuation-based adaptive structured pruning for large              10337. PMLR, 2023.
  language models. In Proceedings of the AAAI Confer-
                                                                     Gao, S., Hua, T., Shirkavand, R., Lin, C.-H., Tang, Z.,
  ence on Artificial Intelligence, volume 38, pp. 10865–
                                                                       Li, Z., Yuan, L., Li, F., Zhang, Z., Ganjdanesh, A.,
  10873, 2024.
                                                                       et al. Tomoe: Converting dense large language models
Biderman, S., Schoelkopf, H., Anthony, Q. G., Bradley,                 to mixture-of-experts through dynamic structural prun-
  H., O’Brien, K., Hallahan, E., Khan, M. A., Purohit, S.,             ing. arXiv preprint arXiv:2501.15316, 2025.
  Prashanth, U. S., Raff, E., et al. Pythia: A suite for an-
  alyzing large language models across training and scal-            Geng, X. and Liu, H. Openllama: An open reproduction of
  ing. In International Conference on Machine Learning,                llama, 2023.
  pp. 2397–2430. PMLR, 2023.
                                                                     He, E., Khattar, A., Prenger, R., Korthikanti, V., Yan, Z.,
Bisk, Y., Zellers, R., Gao, J., Choi, Y., et al. Piqa: Reason-         Liu, T., Fan, S., Aithal, A., Shoeybi, M., and Catanzaro,
  ing about physical commonsense in natural language. In               B. Upcycling large language models into mixture of ex-
  Proceedings of the AAAI conference on artificial intelli-            perts. arXiv preprint arXiv:2410.07524, 2024.
  gence, volume 34, pp. 7432–7439, 2020.
                                                                     Hendrycks, D., Burns, C., Basart, S., Zou, A., Mazeika,
Chen, L., Li, J., Dong, X., Zhang, P., He, C., Wang, J.,               M., Song, D., and Steinhardt, J. Measuring mas-
  Zhao, F., and Lin, D. Sharegpt4v: Improving large multi-             sive multitask language understanding. arXiv preprint
  modal models with better captions. In European Confer-               arXiv:2009.03300, 2020.
  ence on Computer Vision, pp. 370–387. Springer, 2024.
                                                                     Komatsuzaki, A., Puigcerver, J., Lee-Thorp, J., Ruiz,
Chen, M., Tworek, J., Jun, H., Yuan, Q., de Oliveira Pinto,            C. R., Mustafa, B., Ainslie, J., Tay, Y., Dehghani, M.,
  H. P., Kaplan, J., Edwards, H., Burda, Y., Joseph, N.,               and Houlsby, N. Sparse upcycling: Training mixture-
  Brockman, G., Ray, A., Puri, R., Krueger, G., Petrov,                of-experts from dense checkpoints. arXiv preprint
  M., Khlaaf, H., Sastry, G., Mishkin, P., Chan, B., Gray,             arXiv:2212.05055, 2022.
  S., Ryder, N., Pavlov, M., Power, A., Kaiser, L., Bavar-
  ian, M., Winter, C., Tillet, P., Such, F. P., Cummings,            Li, J., Du, L., Zhao, H., Zhang, B.-w., Wang, L., Gao, B.,
  D., Plappert, M., Chantzis, F., Barnes, E., Herbert-Voss,            Liu, G., and Lin, Y. Infinity instruct: Scaling instruc-
  A., Guss, W. H., Nichol, A., Paino, A., Tezak, N., Tang,             tion selection and synthesis to enhance language models.
  J., Babuschkin, I., Balaji, S., Jain, S., Saunders, W.,              arXiv preprint arXiv:2506.11116, 2025.
  Hesse, C., Carr, A. N., Leike, J., Achiam, J., Misra,
  V., Morikawa, E., Radford, A., Knight, M., Brundage,               Liu, J., Cui, L., Liu, H., Huang, D., Wang, Y., and Zhang,
  M., Murati, M., Mayer, K., Welinder, P., McGrew, B.,                 Y. Logiqa: A challenge dataset for machine reading
  Amodei, D., McCandlish, S., Sutskever, I., and Zaremba,              comprehension with logical reasoning. arXiv preprint
  W. Evaluating large language models trained on code.                 arXiv:2007.08124, 2020.
  2021.
                                                                     Ma, X., Fang, G., and Wang, X. Llm-pruner: On the struc-
Chung, H. W., Hou, L., Longpre, S., Zoph, B., Tay, Y.,                tural pruning of large language models. In Advances in
  Fedus, W., Li, Y., Wang, X., Dehghani, M., Brahma,                  Neural Information Processing Systems, 2023.
  S., et al. Scaling instruction-finetuned language mod-
  els. Journal of Machine Learning Research, 25(70):1–               Malinen, M. I. and Fränti, P. Balanced k-means for clus-
  53, 2024.                                                           tering. In Joint IAPR international workshops on statis-
                                                                      tical techniques in pattern recognition (SPR) and struc-
Clark, P., Cowhey, I., Etzioni, O., Khot, T., Sabharwal, A.,          tural and syntactic pattern recognition (SSPR), pp. 32–
  Schoenick, C., and Tafjord, O. Think you have solved                41. Springer, 2014.

                                                                 9

ExpertWeaver: Unlocking the Inherent MoE in Dense LLMs with GLU Activation Patterns

Muennighoff, N., Soldaini, L., Groeneveld, D., Lo, K.,              Weber, M., Fu, D., Anthony, Q., Oren, Y., Adams, S.,
 Morrison, J., Min, S., Shi, W., Walsh, P., Tafjord, O.,             Alexandrov, A., Lyu, X., Nguyen, H., Yao, X., Adams,
 Lambert, N., et al. Olmoe: Open mixture-of-experts lan-             V., et al. Redpajama: an open dataset for training large
 guage models. arXiv preprint arXiv:2409.02060, 2024.                language models. Advances in neural information pro-
                                                                     cessing systems, 37:116462–116492, 2024.
Nakamura, T., Akiba, T., Fujii, K., Oda, Y., Yokota, R.,
  and Suzuki, J. Drop-upcycling: Training sparse mixture            Welbl, J., Liu, N. F., and Gardner, M. Crowdsourc-
  of experts with partial re-initialization. arXiv preprint          ing multiple choice science questions. arXiv preprint
  arXiv:2502.19261, 2025.                                            arXiv:1707.06209, 2017.

Nishu, K., Mehta, S., Abnar, S., Farajtabar, M., Horton, M.,        Xia, M., Gao, T., Zeng, Z., and Chen, D. Sheared llama:
  Najibi, M., Nabi, M., Cho, M., and Naik, D. From dense              Accelerating language model pre-training via structured
  to dynamic: Token-difficulty driven moefication of pre-             pruning. arXiv preprint arXiv:2310.06694, 2023.
  trained llms. arXiv preprint arXiv:2502.12325, 2025.              Xue, F., Zheng, Z., Fu, Y., Ni, J., Zheng, Z., Zhou,
                                                                      W., and You, Y. Openmoe: An early effort on open
Pei, Z., Zou, L., Zhen, H.-L., Yu, X., Liu, W., Pan, S. J.,
                                                                      mixture-of-experts language models. arXiv preprint
  Yuan, M., and Yu, B. Cmoe: Converting mixture-of-
                                                                      arXiv:2402.01739, 2024.
  experts from dense to accelerate llm inference. arXiv
  preprint arXiv:2502.04416, 2025.                                  Yu, L., Jiang, W., Shi, H., Yu, J., Liu, Z., Zhang, Y., Kwok,
                                                                      J. T., Li, Z., Weller, A., and Liu, W. Metamath: Boot-
Penedo, G., Kydlı́ček, H., Lozhkov, A., Mitchell, M., Raf-           strap your own mathematical questions for large lan-
  fel, C. A., Von Werra, L., Wolf, T., et al. The fineweb             guage models. arXiv preprint arXiv:2309.12284, 2023.
  datasets: Decanting the web for the finest text data at
  scale. Advances in Neural Information Processing Sys-             Zellers, R., Holtzman, A., Bisk, Y., Farhadi, A., and Choi,
  tems, 37:30811–30849, 2024.                                         Y. Hellaswag: Can a machine really finish your sen-
                                                                      tence? arXiv preprint arXiv:1905.07830, 2019.
Qu, X., Dong, D., Hu, X., Zhu, T., Sun, W., and Cheng,
  Y. Llama-moe v2: Exploring sparsity of llama from per-            Zhang, S., Roller, S., Goyal, N., Artetxe, M., Chen, M.,
  spective of mixture-of-experts with post-training. arXiv            Chen, S., Dewan, C., Diab, M., Li, X., Lin, X. V.,
  preprint arXiv:2411.15708, 2024.                                    et al. Opt: Open pre-trained transformer language mod-
                                                                      els. arXiv preprint arXiv:2205.01068, 2022a.
Sakaguchi, K., Bras, R. L., Bhagavatula, C., and Choi, Y.
                                                                    Zhang, Z., Lin, Y., Liu, Z., Li, P., Sun, M., and Zhou, J.
  Winogrande: An adversarial winograd schema challenge
                                                                      Moefication: Transformer feed-forward layers are mix-
  at scale. Communications of the ACM, 64(9):99–106,
                                                                      tures of experts. In Findings of the Association for Com-
  2021.
                                                                      putational Linguistics: ACL 2022, pp. 877–890, 2022b.
Shazeer, N. Glu variants improve transformer.         arXiv         Zheng, H., Bai, X., Liu, X., Mao, Z. M., Chen, B.,
  preprint arXiv:2002.05202, 2020.                                    Lai, F., and Prakash, A. Learn to be efficient: Build
                                                                      structured sparsity in large language models. Advances
Sun, M., Liu, Z., Bair, A., and Kolter, J. Z. A simple and
                                                                      in Neural Information Processing Systems, 37:101969–
  effective pruning approach for large language models.
                                                                      101991, 2024.
  arXiv preprint arXiv:2306.11695, 2023.
                                                                    Zhou, C., Liu, P., Xu, P., Iyer, S., Sun, J., Mao, Y., Ma, X.,
Team, G., Riviere, M., Pathak, S., Sessa, P. G., Hardin,              Efrat, A., Yu, P., Yu, L., et al. Lima: Less is more for
  C., Bhupatiraju, S., Hussenot, L., Mesnard, T., Shahri-             alignment. Advances in Neural Information Processing
  ari, B., Ramé, A., et al. Gemma 2: Improving open                  Systems, 36:55006–55021, 2023a.
  language models at a practical size. arXiv preprint
  arXiv:2408.00118, 2024a.                                          Zhou, J., Lu, T., Mishra, S., Brahma, S., Basu, S.,
                                                                      Luan, Y., Zhou, D., and Hou, L. Instruction-following
Team, Q. et al. Qwen2 technical report. arXiv preprint                evaluation for large language models. arXiv preprint
  arXiv:2407.10671, 2(3), 2024b.                                      arXiv:2311.07911, 2023b.

Teknium.     Openhermes 2.5: An open dataset of                     Zhu, T., Qu, X., Dong, D., Ruan, J., Tong, J., He, C.,
  synthetic data for generalist llm assistants, 2023.                 and Cheng, Y. Llama-moe: Building mixture-of-experts
  URL https://huggingface.co/datasets/                                from llama with continual pre-training. arXiv preprint
  teknium/OpenHermes-2.5.                                             arXiv:2406.16554, 2024.

                                                               10


                  ExpertWeaver: Unlocking the Inherent MoE in Dense LLMs with GLU Activation Patterns

A. Impact Statement
ExpertWeaver is a training-free method to convert pretrained dense LLMs into sparse MoE models using GLU activation
patterns. By activating fewer parameters per token, it can reduce inference compute and energy, potentially lowering
deployment cost and environmental impact. As with other efficiency improvements, cheaper inference may also enable
broader misuse (e.g., spam or misinformation), and the conversion does not itself address issues such as bias or unsafe
generation. We recommend standard safety evaluation and monitoring when deploying converted models.

B. Ablation Studies




                            a) Ablation on Shared Expert Ratio             b) Ablation on Threshold       c) Ablation on Expert Granularity



Figure 6. Ablation studies on key hyperparameters of ExpertWeaver. a) MMLU performance heatmap for different shared expert
ratios, controlled by the hyperparameters αmin and αmax in Eq. 5. The diagonal where αmin = αmax represents a static configuration
with a uniform ratio across all layers. b) MMLU performance varying specialization threshold τ from Eq. 4. c) MMLU performance
varying expert granularity.


As shown in Figure 6, we conduct a series of ablation studies to investigate the impact of key hyperparameters.
a) Shared Expert Ratio: The heatmap in Figure 6a) explores the impact of the shared expert ratio, controlled by αmin
and αmax from Eq. 5. The results reveal that our layer-aware dynamic allocation strategy significantly outperforms static
configurations. The diagonal, where αmin = αmax , represents a static setting with a uniform ratio across all layers, and
its performance is lower than that of the off-diagonal regions. The optimal performance is achieved within the range of
αmin ∈ [0.2, 0.4] and αmax ∈ [0.5, 0.7], with the peak at (0.2, 0.7). We adopted αmin = 0.2 and αmax = 0.7 as our default
setting. b) Specialization Threshold: We study the impact of the specialization threshold τ from Eq. 4, which determines
whether a neuron is classified as specialized based on its CV score. The model demonstrates robust performance across
a range of τ values, with optimal results achieved at τ = 0.6. We adopt τ = 0.6 as the default configuration for all
experiments. c) Expert Granularity: The total number of experts significantly influences performance. The MMLU
score peaks when the number of experts is around 64 and 128, declining with either too few or too many experts. This
suggests an optimal granularity that balances functional diversity and specialization efficiency. To prioritize inference
efficiency while maintaining strong performance, we adopt 64 as our default expert granularity.

                                              Table 4. Instruct Model Performance Comparison

                 Method                             MMLU(5)      ARC-C(25)          GSM8K             HumanEval           IFEval              Avg.
                 LLaMA-MoE-3B-7B                        28.24      44.03              4.62              12.02              28.10              23.40
                 OLMoE-1B-7B                            53.79      55.63              40.94             40.48              35.49              45.27
                 LLaMA-MoE-v2                           40.90      40.20              55.00             51.20              36.00              44.66
                 ExpertWeaver-Instruct                  50.60      69.80              57.10             50.20              33.10              52.16



C. SFT Performance
As shown in Table 4, our instruction-tuned model, ExpertWeaver-Instruct, achieves superior performance against other
MoE baselines. It achieves a top-ranking average score of 52.16, establishing a clear lead over comparable MoE mod-
els like OLMoE-1B-7B (45.27) and LLaMA-MoE-v2 (44.66). This result demonstrates the effectiveness of the Exper-

                                                                      11


                  ExpertWeaver: Unlocking the Inherent MoE in Dense LLMs with GLU Activation Patterns

tWeaver methodology in creating a robust foundation for supervised fine-tuning. We also show the serving efficiency of
the ExpertWeaver-Instruct model in Appendix F.

D. Related Work
Structural Pruning. Structural pruning improves LLM efficiency by removing entire components like neurons or atten-
tion heads. To avoid costly retraining, recent training-free methods use importance scores to identify and remove structures.
For example, LLM-Pruner (Ma et al., 2023) analyzes gradients, while FLAP (An et al., 2024) measures output feature sta-
bility. These methods perform static pruning by permanently removing weights. In contrast, methods that convert dense
models into MoEs, such as ToMoE (Gao et al., 2025), can be considered a form of dynamic structural pruning. Since
the goal of converting a dense model to an MoE aligns with that of structural pruning (reducing computational cost while
preserving performance), we also categorize ExpertWeaver as a dynamic structural pruning method. It achieves this by
dynamically selecting a sparse subset of neuron blocks for each token at inference time.

Model Upcycling. Model upcycling (Team et al., 2024b; Muennighoff et al., 2024; He et al., 2024; Nakamura et al.,
2025; Komatsuzaki et al., 2022) offers a cost-effective strategy for creating larger sparse models by merging smaller,
pretrained dense models, thus avoiding the expense of from-scratch training. A common technique is to replicate the FFN
layers from one or more dense models to form the experts of a new, larger MoE, followed by a fine-tuning phase to learn
the routing logic. While effective, this paradigm faces two emerging challenges. First, the concentration of research and
computational resources on frontier models means that the capabilities of state-of-the-art large models are advancing at
a pace that smaller models struggle to match. Upcycling from smaller, less capable models may therefore not match the
performance of downcycling from a larger, more advanced one. Second, because upcycling often relies on parameter
duplication to initialize experts, it can lead to a lack of diversity that predisposes the model to fall into local optima during
long-term optimization. In contrast, downcycling, as implemented by ExpertWeaver, leverages the rich, non-redundant
internal structure of a single large model, providing a more robust foundation for sustained performance gains.

Model Downcycling. Model downcycling converts large pretrained dense models into computationally efficient MoEs,
aiming to retain performance while gaining inference speed through FFN neuron partitioning. Moefication (Zhang et al.,
2022b) pioneered this field by splitting FFN parameters into functional partitions as experts with learned routers, but was
originally designed for ReLU-based networks and struggles with modern GLU-based architectures. LTE (Zheng et al.,
2024) trains efficiency-aware models to amplify inherent activation sparsity through efficiency loss penalties, but requires
substantial computational overhead during the training phase and does not directly leverage inherent model structures for
expert creation. ToMoE (Gao et al., 2025) uses dynamic structural pruning with frozen weights to discover expert structures
via learned routing, but still requires substantial training overhead for the routing modules. Llama-MoE (Zhu et al., 2024;
Qu et al., 2024) partitions FFN parameters through clustering followed by extensive continual pretraining with 200B tokens,
but ignores the model’s internal structural patterns, resulting in poor MoE performance that fails to preserve the original
dense model’s capabilities. CMoE (Pei et al., 2025) achieves training-free conversion using balanced clustering with
analytically constructed routers, yet lacks detailed activation signal analysis and employs fixed configurations across layers,
leading to suboptimal expert partitioning that fails to capture layer-specific specializations. ExpertWeaver introduces a
novel training-free dense-to-MoE technique that leverages intrinsic GLU activation patterns to form experts and construct
routers, enabling both training-free applications and supporting efficient downcyling for further CPT.

E. Comparision on Complex Tasks and Reasoning Models
To compare the capabilities of different methods in more complex scenarios, we first investigate the performance of
Qwen2.5-7B on GSM8k and HumanEval under various sparsities. Furthermore, to assess performance on models special-
ized for reasoning, we explore the effectiveness of these methods on the DeepSeek-R1-Distill-Qwen-7B model, evaluating
it on the GPQADiamond and LiveCodeBench benchmarks.
The results, presented in Table 5, reveal the limitations of static structural pruning on complex tasks. At 25% sparsity on
GSM8k, static methods like LLM-Pruner and FLAP almost completely fail; the dash (-) for LLM-Pruner indicates its accu-
racy dropped to zero. In stark contrast, ExpertWeaver, which retains all model parameters, maintains strong performance.
This advantage holds at 12.5% sparsity, where ExpertWeaver continues to significantly outperform static methods on both
GSM8k and HumanEval. The same trend is observed on the specialized reasoning model, DeepSeek-R1-Distill-Qwen-

                                                               12

ExpertWeaver: Unlocking the Inherent MoE in Dense LLMs with GLU Activation Patterns

7B, where ExpertWeaver again achieves the highest scores on GPQADiamond and LiveCodeBench. This demonstrates
that ExpertWeaver’s dynamic pruning approach is substantially more effective at preserving critical reasoning and coding
capabilities, especially at higher sparsity levels where static methods suffer from irreversible information loss.

Table 5. Training-Free Pruning on Code Benchmarks. We compare ExpertWeaver against static pruning methods on several code-
related benchmarks. The base models used for each benchmark are specified in the table.

                                      GSM8k          HumanEval        GPQADiamond        LiveCodeBench
                  Base Model                Qwen2.5-7B                  DeepSeek-R1-Distill-Qwen-7B
                  Sparsity         25%     12.5%    25%     12.5%                   12.5%
                  LLM-Pruner        2.0         -    14.6      10.4         33.3               6.1
                  FLAP             14.8         -    57.6       9.8         36.3               2.3
                  ExpertWeaver     34.9      18.9    64.6      32.9         37.3               16.8



F. Serving Efficiency
To investigate the serving efficiency of our model, we conduct a comprehensive benchmark comparing ExpertWeaver
(E64-A14-S2, derived from Qwen2.5-7B) with the dense Qwen2.5-7B model using the vLLM framework. All tests are run
on a single GPU with ‘tensor parallel size=1’ and GPU memory utilization set to 90% to maximize the memory allocated
to the KV Cache. We simulate a high-load scenario by sending 1024 random requests with an average input length of
512 tokens at an infinite rate, evaluating the models’ peak performance at a maximum concurrency of 128 sequences. The
evaluation spans three generations of NVIDIA GPUs—A100 (Ampere) and H100 (Hopper)—to ensure a comprehensive
and fair assessment.
The results, presented in Table 6, show ExpertWeaver demonstrates superior performance on both platforms. It achieves
higher throughput (RPS, OTPS, and TTPS) and lower latency for both the first token (TTFT) and subsequent tokens
(TPOT). This clear-cut advantage across all metrics confirms the inference efficiency of the ExpertWeaver.

Table 6. Inference throughput comparison across different GPU architectures. RPS: Requests Per Second. OTPS: Output Tokens
Per Second. TTPS: Total Tokens Per Second. TTFT: Time To First Token. TPOT: Time Per Output Token. ITL: Inter-Token Latency.

                 GPU      Method           RPS ↑     OTPS ↑       TTPS ↑     TTFT ↓     TPOT ↓        ITL ↓
                                                                               (ms)       (ms)         (ms)
                          ExpertWeaver      22.88    2928.86     14644.31      440.6        40.3       40.3
                 A100
                          Qwen2.5-7B        18.85    2413.22     12066.10      675.4        47.8       48.2
                          ExpertWeaver      44.04    5637.42     28187.10      542.2         9.8        9.8
                 H100
                          Qwen2.5-7B        41.41    5299.96     26499.78     1404.5        18.5       18.5



G. Ablation Studies on the Calibration Set
To investigate the impact of the calibration set on ExpertWeaver’s performance, we conducted two ablation studies, with
results presented in Table 7.

Impact of Data Quantity and Diversity. We first analyzed the sensitivity to the calibration set’s size and diversity. We
created several calibration sets by randomly sampling different proportions (25%, 50%, 75%, and 100%) of the tasks from
our default Flan-v2 collection. For each selected task, we used 10 samples, meaning that as the proportion increases,
both the number of tasks (diversity) and the total number of samples (quantity) grow. The results show that performance
generally improves with a larger and more diverse calibration set, with the best result (67.0) achieved using 100% of the
tasks. Notably, using just 50% of the tasks already yields a strong average score of 66.1, which is over 98% of the final
performance. This demonstrates that while diversity is beneficial, ExpertWeaver is data-efficient and does not require an
excessively large calibration set to achieve robust performance.

                                                            13


                  ExpertWeaver: Unlocking the Inherent MoE in Dense LLMs with GLU Activation Patterns

Impact of Data Source. We also compared our default multi-task calibration set (Flan-v2) against a variant calibrated
solely on a general-domain corpus (C4), denoted as ExpertWeaverC4 . The results clearly demonstrate the benefit of using
a multi-task dataset. Our default ExpertWeaver achieves a superior average score of 67.0, outperforming ExpertWeaverC4
(65.8). This suggests that capturing a wide range of activation patterns from diverse tasks is crucial for identifying a truly
robust and generalizable functional structure within the dense model, leading to a more effective expert partition.

       Table 7. Ablation Studies on the Calibration Set. We analyze the impact of calibration set size, diversity, and source.

            Method                             MMLU         HellaSwag(10)        ARC-e       ARC-c(25)       PiQA      Avg.
                                                      Qwen2.5-7B, 25% sparsity
            Dense                                74.2             80.3               77.8        63.8         80.0     75.2
            Ablation on Data Source
            ExpertWeaverF lan (default)          61.6             72.3               71.5        53.5         76.3     67.0
            ExpertWeaverC4                       61.2             71.6               72.1        49.1         74.8     65.8
            Ablation on Data Quantity & Diversity
            ExpertWeaverF lan25%          59.2                    69.0               68.7        47.3         75.4     63.9
            ExpertWeaverF lan50%          60.6                    71.2               71.5        51.7         75.5     66.1
            ExpertWeaverF lan75%          60.1                    71.3               72.5        46.8         77.3     65.6
            ExpertWeaverF lan100%         61.6                    72.3               71.5        53.5         76.3     67.0



H. Details of the Calibration Set
To construct a diverse, multi-task calibration set for analyzing neuron activation patterns, we sampled from the Flan-v2
collection (Chung et al., 2024). Flan-v2 is a large-scale dataset consisting of a mixture of publicly available NLP datasets
that have been formatted into an instruction-tuning style. This diversity makes it an ideal source for a calibration set
intended to capture a wide range of functional specializations.
For our calibration set, Dcalib , we selected a representative subset of 48 distinct tasks from 10 different task clusters within
Flan-v2. For each of these 48 tasks, we randomly sampled 5 few-shot examples, resulting in a total of 240 samples in our
calibration set. This carefully curated subset ensures that our analysis of neuron activation patterns is based on a broad
and balanced distribution of tasks, from reading comprehension and summarization to commonsense reasoning and natural
language inference.
Table 8 lists the 10 task clusters and the 48 specific tasks used in our calibration set.

                                       Table 8. Tasks from Flan-v2 used in the calibration set.

 Task Cluster                 Description                                             Datasets
 Reading Comprehension        Answers questions based on provided passages.           squad v1, squad v2, drop, duorc, quac, record
 Summarization                Creates a shorter version of a document.                xsum, cnn dailymail, samsum, multi news
 Translation                  Translates text across multiple languages.              wmt14 en-fr, wmt14 en-de, wmt14 en-ro
 Commonsense Reasoning        Understands everyday scenarios.                         boolq, piqa, siqa, cosmos qa, hellaswag, winogrande
 Natural Language Inference   Determines logical relationship between sentences.      mnli, qnli, rte, wnli, anli
 Coreference Resolution       Identifies expressions referring to the same entity.    wsc, dpr, winogender
 Sentiment Analysis           Determines sentiment polarity.                          imdb, sentiment140, yelp polarity
 Question Answering           Answers questions without external knowledge.           arc, openbookqa, race, trivia qa
 Paraphrase Detection         Generates alternative phrasings of sentences.           mrpc, qqp
 Structure-to-Text            Generates text from structured data.                    common gen, e2e nlg, dart



I. Balanced K-Means Clustering
We employ balanced K-Means clustering to partition specialized neurons Ir into Nre,ℓ routed experts of equal capacity
dexpert . Given activation pattern vectors Ar = {ai }i∈Ir , the algorithm finds clusters C1 , . . . , CNre,ℓ that solve:

                                                                   14


                 ExpertWeaver: Unlocking the Inherent MoE in Dense LLMs with GLU Activation Patterns


                                                        Nre,ℓ
                                                         X X
                                           min                      ||ai − µk ||2
                                     C1 ,...,CNre,ℓ
                                                         k=1 i∈Ck

                                                 s.t.   |Ck | = dexpert     ∀k ∈ {1, . . . , Nre,ℓ }
                                                                                                                         (15)
                                                        Cj ∩ C k = ∅        ∀j ̸= k
                                                        Nre,ℓ
                                                         [
                                                                Ck = Ir
                                                         k=1

where µk is the centroid of cluster Ck .
Algorithm:

 1. Initialize: Sample Nre,ℓ centroids µ1 , . . . , µNre,ℓ from Ar .
 2. Assign: Solve the minimum-cost perfect matching problem with fixed centroids. We use a greedy approximation:
    iteratively assign neurons to the nearest available cluster slot.
 3. Update: Recompute centroids as:
                                                                    1       X
                                                         µk ←                      ai   ∀k                               (16)
                                                                  dexpert
                                                                            i∈Ck

 4. Iterate: Repeat steps 2-3 until convergence.

This ensures functionally coherent and structurally uniform experts for efficient MoE implementation.

J. Details of the compared baselines
J.1. Baselines for Structured Pruning
For the structured pruning evaluation, we compare ExpertWeaver with the following training-free methods:

   • LLM-Pruner (Ma et al., 2023) is a task-agnostic, training-free structural pruning method that identifies and removes
     redundant structures by analyzing gradient information and parameter magnitudes. It aims to preserve the model’s
     generalization capabilities by focusing on the interconnectedness of model components.
   • FLAP (An et al., 2024) is a training-free pruning framework that prunes large language models at the FFN-layer
     level. It uses a metric based on output feature stability to identify and remove less important neurons, offering a
     computationally efficient alternative to methods that require gradient computation.
   • CMoE (Pei et al., 2025) is a training-free method that converts dense models into MoEs using balanced clustering.
     It constructs routers analytically but uses a fixed configuration across all layers, which may not capture layer-specific
     functional specializations.

J.2. Baselines for Model Downcycling
In the model downcycling experiments, we compare our ExpertWeaver-initialized model against a variety of both dense
and MoE models with comparable parameter counts and training budgets.

Dense Models.     We include several strong, publicly available dense models as baselines:

   • OPT-2.7B (Zhang et al., 2022a), Pythia-2.8B (Biderman et al., 2023), INCITE-Base-3B (Weber et al., 2024), Open-
     LLaMA-3B-v2 (Geng & Liu, 2023), and Sheared-LLaMA-2.7B (Xia et al., 2023) are all well-established language
     models in the 2.7B-3B parameter range.
   • Gemma-2-2B (Team et al., 2024a) is a recent, highly-performant model from Google.
   • SmolLM2-1.7B (Allal et al., 2025) is another strong baseline known for its efficiency and performance at a smaller
     scale.

                                                                    15

ExpertWeaver: Unlocking the Inherent MoE in Dense LLMs with GLU Activation Patterns

MoE Models.     We also compare against several existing MoE models:

  • LLaMA-MoE-v1 (Zhu et al., 2024) and LLaMA-MoE-v2 (Qu et al., 2024) are MoE variants of the LLaMA architec-
    ture. They are created by partitioning the FFN parameters of a dense model into experts and then applying extensive
    continued pretraining.
  • OpenMoE (Xue et al., 2024) is an open-source MoE model series.
  • OLMoE (Muennighoff et al., 2024) is a family of open language models with a MoE architecture, developed by
    AI2. We compare against a checkpoint trained on 500B tokens to ensure a fair comparison with our model’s training
    budget.

J.3. Baselines for Instruct Model Performance
For evaluating the performance of our instruction-tuned model, ExpertWeaver-Instruct, we compare it against the fol-
lowing instruction-tuned MoE baselines:

  • LLaMA-MoE-3B-7B and LLaMA-MoE-v2 are the instruction-tuned versions of the LLaMA-MoE models described
    above.
  • OLMoE-1B-7B is the instruction-tuned version of the OLMoE model.

K. Model Architecture for OLMo Downcycling Experiment
The model architecture for the OLMo Downcycling experiment is shown in Table 9.
                             Table 9. Model Configurations of OLMo Downcycling Experiment

                                            OLMo 575M        OLMo 1.3B       OLMoE 1.3B-A676M
                      Model Dimension            2048             2048                2048
                      FFN Dimension              1024             8192                1024
                      Attention Heads             16               16                  16
                      Key/Value Heads             16               16                  16
                      Layers                      16               16                  16
                      Vocabulary Size           50280            50280               50280
                      Weight Typing              True             True                True
                      Context Length             4096             4096                4096
                      Expert Granularity           -                -                2 in 8


L. Training Configurations
This section provides the detailed configurations for both the continued pre-training (CPT) and supervised fine-tuning
(SFT) phases.

L.1. Continued Pre-training (CPT)
The CPT phase was conducted on the ExpertWeaver-E64-A14-S2 model for 200 billion tokens using the FineWeb-Edu
dataset (Penedo et al., 2024) using 128 H100 GPUs. Key hyperparameters are listed in Table 10. We use megatron-swift1
for model training.

L.2. Supervised Fine-Tuning
The SFT process consists of two stages with identical hyperparameters. Following the methodology of (Muennighoff et al.,
2024), we use a fixed learning rate of 2e-5, a global batch size of 128, and do not use an auxiliary loss for 2 epochs. The
training for both stages was conducted on 64 H100 GPUs.
   1
       https://swift.readthedocs.io/en/latest/Megatron-SWIFT/Quick-start.html

                                                            16


                 ExpertWeaver: Unlocking the Inherent MoE in Dense LLMs with GLU Activation Patterns

                                     Table 10. Hyperparameters for Continued Pre-training.

                                       Hyperparameter                       Value
                                       Optimizer                            AdamW
                                       Learning Rate                        4e-4
                                       Min Learning Rate                    4e-5
                                       Global Batch Size                    1024
                                       Sequence Length                      4096
                                       Training Iterations                  50,000
                                       Warmup Iterations                    100
                                       Auxiliary Loss Coeff (λ)             0.01

Stage 1: General Conversational Tuning. This stage focuses on general conversational abilities, using a dataset mixture
of LIMA (Zhou et al., 2023a), OpenHermes (Teknium, 2023), ShareGPT (Chen et al., 2024), and BAAI Infinity Instruct
(Li et al., 2025).

Stage 2: Code and Math Tuning. This stage hones the model’s capabilities in code and mathematics, using a dataset
mixture of BAAI (Li et al., 2025) and MetaMathQA (Yu et al., 2023), with a small amount of conversational data from
Stage 1.

M. Evaluation Datasets
We evaluate our models on a comprehensive suite of benchmarks to assess their capabilities across various reasoning and
knowledge domains.

• MMLU (Hendrycks et al., 2020) (Measuring Massive Multitask Language Understanding) is a broad benchmark de-
  signed to measure knowledge acquired during pre-training. It covers 57 subjects across STEM, humanities, social sci-
  ences, and more, making it a robust test of world knowledge and problem-solving ability.
• HellaSwag (Zellers et al., 2019) is a commonsense reasoning benchmark that challenges models to complete a sentence
  by choosing the most plausible ending from four options. It is designed to be difficult for models that rely on superficial
  statistical patterns.
• ARC (Clark et al., 2018) (AI2 Reasoning Challenge) is a question-answering dataset containing grade-school level
  science questions. We use both the Easy (ARC-e) and Challenge (ARC-c) sets, which are designed to be answerable
  with simple retrieval or multi-hop reasoning, respectively.
• PIQA (Bisk et al., 2020) (Physical Interaction Question Answering) is a commonsense reasoning benchmark focused
  on physical interactions. It presents two possible solutions to everyday situations, and the model must choose the more
  physically plausible one.
• WinoGrande (Sakaguchi et al., 2021) is a large-scale dataset for commonsense reasoning, formulated as a Winograd
  Schema Challenge. It requires resolving pronouns in ambiguous sentences, which is challenging for models without a
  deep understanding of context.
• LogiQA (Liu et al., 2020) is a dataset designed for logical reasoning. It consists of reading comprehension questions
  from professional logic exams, requiring the model to perform complex logical operations.
• SciQ (Welbl et al., 2017) is a question-answering dataset containing science exam questions from various domains,
  primarily focused on physics, chemistry, and biology.
• GSM8K (Cobbe et al., 2021) (Grade School Math 8K) is a dataset of high-quality, linguistically diverse grade school
  math word problems, designed to test multi-step mathematical reasoning.
• HumanEval (Chen et al., 2021) is a benchmark for evaluating code generation. It consists of 164 programming problems
  with function signatures, docstrings, and unit tests to assess the functional correctness of the generated code.

                                                              17


                  ExpertWeaver: Unlocking the Inherent MoE in Dense LLMs with GLU Activation Patterns

• IFEval (Zhou et al., 2023b) (Instruction Following Evaluation) is a benchmark for evaluating a model’s ability to follow
  instructions. It consists of a set of prompts with explicit constraints that the model’s response must adhere to.

                                                         Shared vs Routed Expert Ratio Across Layers
                                    1.0                                                                Shared Expert Ratio
                                                                                                       Routed Expert Ratio

                                    0.8



                                    0.6
                            Ratio




                                    0.4



                                    0.2



                                    0.0
                                          0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27
                                                                          Layer ID

Figure 7. Shared vs. Routed Expert Ratio Across Layers. The figure shows the layer-wise configuration of shared and routed expert
ratios for Qwen2.5-7B, as determined by ExpertWeaver.


N. Layer-wise Expert Configuration
Figure 7 shows the exact configuration of ExpertWeaver at 25% sparsity. This configuration is derived from the layer-aware
expert allocation strategy in § 3.2. The resulting U-shaped distribution, with more shared experts in the initial and final
layers and more routed experts in the middle layers, demonstrates ExpertWeaver’s ability to automatically tailor expert
composition to each layer’s specific needs, contrasting with the uniform approaches used by other methods.

                  Layer 0                                                 Layer 11                                           Layer 23




                                          Figure 8. Expert Specialization in the ExpertWeaver Model.


O. Expert Specialization
Figure 8 presents a detailed visualization of the expert routing patterns within our ExpertWeaver model. To create these
heatmaps, we sampled 20 instances from subsets of the Red Pajama dataset (such as github and arxiv) at different model
layers. The results shows that, in the shallow layers (e.g., Layer 0), tokens from various domains tend to activate a broad
range of experts. Although many experts are utilized, the activation patterns between tasks are still distinguishable. The
routing in deeper layers (e.g., Layer 23) becomes highly concentrated and specialized, with tokens from a specific domain
consistently routed to a small and distinct set of experts. This results also reveal that shallow layers are responsible
for processing common, foundational knowledge, while experts in deeper layers undergo functional differentiation to
efficiently handle domain-specific information.

                                                                          18
