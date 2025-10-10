## Linear Representation of Sequence Models

Any sequence model can be written in the linear form

$$
o_t = \sum_{i=1}^t x_i M_{t,i} = \sum_{i=1}^t x_i W_v \; g(\{x_j\}_{j=1}^t, t,i) \; W_o
$$

where $M_{t,i}$ or equivalently $g(\cdot)$ defines the context mixing kernel between tokens $i$ and $t$. The choice of $g$ encodes different inductive biases. 

### Additive (Order-Independent) Scaling

Softmax attention adopts

$$g_{softmax}(t,i) = \frac{\exp(q_tK_i^T)}{\sum_{j=1}^t \exp(q_tk_j^T)}$$

providing global competition independent of temportal order. This treats positions symmetrically after normalization.

### Multiplicative (Order-Dependent) Scaling

Linear RNNs and SSMs (Mamba, S4) adopt

$$g_{RNN} (t,i) = \frac{\prod_{j=1}^t A_j}{\prod_{j=1}^i A_j} = \prod_{j=i+1}^t A_j$$

Softmax captures what to attend to via content similarity. Prefix-products capture how information decays over time. Existing models choose only one paradigm, sacrificing either temporal locality or content-based retrieval.

## Cartan Decoposition

We interpret context kernels through the Cartan decomposition

$$g(t,i) = \hat q_t [R_t\sum_{t,i}\R_i^T]\hat k_i$$

where $R_t, R_i \in SO(2n)$ encode directional geometry (generalized RoPE) and $\sum_{t,i} \in SPD(2n)$ encodes magnitude scaling.

## Prefix-product Sigmoid * Softmax (Different Input)

- Full

$$
q_t \frac{\prod_{j=1}^t \sigma(s_j)}{\sum_{j=1}^t \exp(p_j + \log \Delta_j)} \cdot \frac{\exp(p_i + \log \Delta_i)}{\prod_{j=1}^i \sigma (s_j)} k_i
$$

Assumption 1. Transformer+ is not implemented correctly? It should be:

$$
\text{Attn, RoPE: } g \leftarrow \hat{q}_t [R_t \frac{\| q_t \|}{\| k_i \|} I R_i^\top] \hat{k}_i \, ?
$$

Assumption 2. The comparison mechanism can be more direct:

$$
\text{Non-causal 1D-generator: } g \leftarrow \hat{q}_t [R_t \frac{\exp(s_i)}{\sum_{j=1}^t \exp(s_j)} R_i^\top] \hat{k}_i
$$

$$
\text{Causal 1D-generator: } g \leftarrow \hat{q}_t [R_t \frac{\prod_{j=1}^t D_j}{\prod_{j=1}^i D_j} R_i^\top] \hat{k}_i
$$

Assumption 3. Inductive bias is incorporated mainly in the isometry generator. The performance of Mamba, LRU... on LRA is purely relied on the causal generator. 

Assumption 4. Commutablility of $
\text{SO}(2n) \cdot \text{SPD}(2n) \cdot \text{SO}(2n)
$ with $SPD(n) \otimes I_2$? We don't need that: let the model decide when to retain the commutability.

Assumption 5. SPD(2n) provide us a lot of good properties. Nonetheless, breaking the SPD(2n) is the critical part for the great performance enhancement in linear attention.

Sometimes we don't need these continuous group isometry properties, inner product space properties, etc. (which happens a lot in real world data!) Let the model and data know when to break it in a specific $(t,i)$ pair is the key to performance enhancement. -- We need to let the model decide when to use the isometry and inner product properties provided by the two-side SO(n) group.

The edibility of the isometry generator in the inner product metric space of query and key is the key to the performance margin. However, breaking the property of this generator is the key to more enhancement, which is actually applying gating mechanism to this isometry.

Actually, allowing negative values in the diagonal matrix is introducing reflections for the $\text{SO}(2n)$. If we have 1 negative value in $Diag(2)$, then the corresponding $\text{SO}(2)$ is reflected. If we have 2 negative values in the diagonal elements, then the $\text{SO}(2)$ is rotated 180 degree, meaning that the diagonal elements can trigger the flipping effect of inner product by itself.


$$
\text{ZeroS Non-causal: } g \leftarrow \hat{q}_t [R_t [\frac{\exp(s_i)}{\sum_{j=1}^t \exp(s_j)} - \sigma_{0,t} \frac1t - \sigma_{1,t} \frac{s_i - \bar{s}_i}{t} - \sigma_{2,t} \cdots] R_i^\top] \hat{k}_i
$$

$$
\text{ZeroS Causal: } g \leftarrow \hat{q}_t [R_t [\frac{\exp(s_i)}{\sum_{j=1}^t \exp(s_j)} - \frac{\prod_{j=1}^t \sigma_{0,j}}{\prod_{j=1}^i \sigma_{0,j}} \frac1t - \frac{\prod_{j=1}^t \sigma_{1,j}}{\prod_{j=1}^i \sigma_{1,j}} \frac{s_i - \bar{s}_i}{t} - \frac{\prod_{j=1}^t \sigma_{2,j}}{\prod_{j=1}^i \sigma_{2,j}} \cdots] R_i^\top] \hat{k}_i
$$


Getting rid of the bilinear form?

$$
x_i W_v [I\bm{1}^\top] K_i R_i [\frac{\exp(s_i)}{\sum_{j=1}^t \exp(s_j)} - \sigma_{0,t} \frac1t - \sigma_{1,t} \frac{s_i - \bar{s}_i}{t} - \sigma_{2,t} \cdots] R_t^\top Q_t [\bm{1} I] W_o
$$


Chain of Evidence:

Sequence models can be expressed in linear form, which facilitates stability analysis, avoids the misleading aspects of stream form, and helps in probing the essence. From this perspective, attention and linear RNN differ only in the inductive bias introduced through scaling.

In the linear form, if there is a bilinear term, it becomes attention; if not, it corresponds to "attention-free" models such as Mamba and Linear RNN with $O(d)$ hidden states. Both rely on the mechanism of group isometry, which essentially constructs a $t/i$ contrast mechanism from a $(t,i)$ pair.

A typical case exists in the Cartan decomposition of $\text{SO}(2n) \cdot \text{SPD}(2n) \cdot \text{SO}(2n)$.

We introduce two kinds of contrast mechanisms for the intermediate SPD in typical Cartan decomposition: softmax and prefix product. The two sides are responsible for rotation, while the middle is responsible for scaling. Scaling introduces inductive bias based on the contrast mechanism, such as order-dependent vs. order-independent, corresponding respectively to linear RNN and attention.

Stability analysis and review: convex, affine, and matrix space.

Insight: Gated linear attention essentially operates in the bilinear diagonal space, and stability should also be analyzed in this space.

Expressivity analysis: Most of the above methods use SPD to scale SO(2). But when gated linear attention claims it can modify the context—does it really? In fact, it relies on negative weights.

The necessity of stable negative weights: negative weights correspond to reflections of SO(2). The extension to SO(2n) generates $2^n$ channels, enhancing group representation capacity. Negative weights disrupt Cartan decomposition and group isometry. On one hand, they can be regarded as gating over whether group isometry is used; on the other hand, they actually generate $2^n$ subsets of isometry, significantly boosting capacity.

How to obtain numerically stable negative weights? First, negative weights should originate from the contrast mechanism. The contrast mechanism is the reason why Cartan decomposition parameterization works, not the other way around. Second, zero-sum weights are a very good choice. Therefore, zero-sum weights based on the contrast mechanism are what we need most.
