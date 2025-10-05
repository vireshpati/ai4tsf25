
### Softmax

- Full

$$
q_t (\sum_{j=1}^t \exp(s_j))^{-1} \cdot \exp(s_i) k_i
$$

- 0th-order

$$
q_t \frac1t \cdot k_i
$$

- 1st-order

$$
q_t \frac1t \cdot s_i k_i - q_t \frac1{t^2} \sum_{j=1}^t s_j \cdot k_i
$$

- Full - 0th - 1st (Matrix form)

$$
q_t (\sum_{j=1}^t \exp(s_j))^{-1} \cdot \exp(s_i) k_i + q_t [\frac1t \frac1t \sum_{j=1}^t s_j  - \frac1t] \cdot k_i - q_t \frac1t \cdot s_i k_i
$$

- Full - 0th - 1st (Matrix form, Gated)

$$
q_t \sigma^h_t (\sum_{j=1}^t \exp(s_j))^{-1} \cdot \exp(s_i) k_i + q_t [\frac1{t^2} (\sigma^h_t -\sigma^1_t) \sum_{j=1}^t s_j  - \frac1t \sigma^h_t] \cdot k_i + q_t [\sigma^1_t - \sigma^h_t] \frac1t \cdot s_i k_i
$$

- Full - 0th - 1st (Matrix form, Ordered)

$$
q_t (\sum_{j=1}^t \exp(s_j))^{-1} \cdot \exp(s_i) k_i + q_t [\frac1{t^2} (-\sigma^1_t) \sum_{j=1}^t s_j  - \frac1t] \cdot k_i + q_t [\sigma^1_t] \frac1t \cdot s_i k_i
$$

- Full - 0th - 1st (Matrix form, Gated, Sequential)

$$
q_t \Pi^h_t (\sum_{j=1}^t \exp(s_j))^{-1} \cdot \frac1{\Pi^h_i} \exp(s_i) k_i + q_t \Pi^h_t (\frac1{t^2} \sum_{j=1}^t s_j - \frac1t) \cdot \frac1{\Pi^h_i} k_i - q_t \Pi^1_t \frac1{t^2} \sum_{j=1}^t s_j \cdot \frac1{\Pi^1_i} k_i + q_t \Pi^1_t \frac1t \cdot \Pi^1_i s_i k_i - q_t \Pi^h_t \frac1t \cdot \frac1{\Pi^h_i} s_i k_i
$$

- 2nd-order

$$
S^2 = \sum_{j=1}^t (s_j - \bar{s}_t)^2 = \sum_{j=1}^t s_j^2 - 2 \bar{s}_t \sum_{j=1}^t s_j + t \bar{s}_t^2
$$


$$
p_i^{(2)} = \frac1{2t} (s_i - \bar{s}_t)^2 - \frac1{2t^2} S^2 \\
= \frac1{2t} s_i^2 - \frac1t \bar{s}_t s_i + \frac1{2t} \bar{s}_t^2 - \frac1{2t^2} \sum_{j=1}^t s_j^2 + \frac1{t^2} \bar{s}_t \sum_{j=1}^t s_j - \frac1{2t} \bar{s}_t^2 \\
= \frac1{2t} s_i^2 - \frac1t \bar{s}_t s_i - \frac1{2t^2} \sum_{j=1}^t s_j^2 + \frac1{t^2} \bar{s}_t \sum_{j=1}^t s_j \\
= \frac1{2t} s_i^2 - \frac1{t^2} (\sum_{j=1}^t s_j) s_i - \frac1{2t^2} \sum_{j=1}^t s_j^2 + \frac1{t^3} (\sum_{j=1}^t s_j)^2
$$

$$
q_t \frac1{2t} \cdot s_i^2 k_i - q_t \frac1{t^2} \sum_{j=1}^t s_j \cdot s_i k_i + q_t [\frac1{t^3} (\sum_{j=1}^t s_j)^2 - \frac1{2t^2} \sum_{j=1}^t s_j^2] \cdot k_i
$$

- Full - 0th - 1st - 2nd (Matrix form)

$$
q_t (\sum_{j=1}^t \exp(s_j))^{-1} \cdot \exp(s_i) k_i + q_t [\frac1{t^2} \sum_{j=1}^t s_j  - \frac1t - \frac1{t^3} (\sum_{j=1}^t s_j)^2 + \frac1{2t^2} \sum_{j=1}^t s_j^2] \cdot k_i + q_t [\frac1{t^2} \sum_{j=1}^t s_j - \frac1t] \cdot s_i k_i - q_t \frac1{2t} \cdot s_i^2 k_i
$$

- Full - 0th - 1st - 2nd (Matrix form, Gated)

$$
\begin{aligned}
\sigma_t^h q_t (\sum_{j=1}^t \exp(s_j))^{-1} \cdot \exp(s_i) k_i &+ \sigma_t^h q_t [\frac1{t^2} \sum_{j=1}^t s_j  - \frac1t - \frac1{t^3} (\sum_{j=1}^t s_j)^2 + \frac1{2t^2} \sum_{j=1}^t s_j^2] \cdot k_i &+ \sigma_t^h q_t [\frac1{t^2} \sum_{j=1}^t s_j - \frac1t] \cdot s_i k_i &- \sigma_t^h q_t \frac1{2t} \cdot s_i^2 k_i \\ 0 &+ \sigma_t^2 q_t [\frac1{t^3} (\sum_{j=1}^t s_j)^2 - \frac1{2t^2} \sum_{j=1}^t s_j^2] \cdot k_i &- \sigma_t^2 q_t \frac1{t^2} \sum_{j=1}^t s_j \cdot s_i k_i &+ \sigma_t^2 q_t \frac1{2t} \cdot s_i^2 k_i \\ 0 &- \sigma_t^1 q_t \frac1{t^2} \sum_{j=1}^t s_j \cdot k_i & + \sigma_t^1 q_t \frac1t \cdot s_i k_i & + 0 \\ = q_t \sigma_t^h (\sum_{j=1}^t \exp(s_j))^{-1} \cdot \exp(s_i) k_i &+ q_t [-\frac{\sigma_t^h}{t} + \frac{\sigma_t^{h-1}}{t^2} \sum_{j=1}^t s_j + \frac{\sigma_t^{h-2}}{2t^2} \sum_{j=1}^t s_j^2 + \frac{\sigma_t^{2-h}}{t^3} (\sum_{j=1}^t s_j)^2] \cdot k_i &+ q_t [\frac{\sigma_t^{1-h}}t + \frac{\sigma_t^{h-2}}{t^2} \sum_{j=1}^t s_j] \cdot s_i k_i &+ q_t \frac{\sigma_t^{2-h}}{2t} \cdot s_i^2 k_i
\end{aligned}
$$


- Full Mahalanobis Softmax

$$
q_t^2 s_{t,i} - 2 q_t s_{t,i} k_i + s_{t,i} k_i^2
$$

$$
q_t^2 (\sum_{j=1}^t \exp(s_j))^{-1} \cdot \exp(s_i) - 2 q_t (\sum_{j=1}^t \exp(s_j))^{-1} \cdot \exp(s_i) k_i + (\sum_{j=1}^t \exp(s_j))^{-1} \cdot \exp(s_i) k_i^2
$$

$$
-\frac1t q_t^2 \cdot \mathbf{1} + 2 \frac1t q_t \cdot k_i - \frac1t \mathbf{1} \cdot k_i^2
$$

$$
-q_t^2 \frac1t \cdot s_i + 2 q_t \frac1t \cdot s_i k_i - \frac1t \mathbf{1} \cdot s_i k_i^2 + q_t^2 \frac1{t^2} \sum_{j=1}^t s_j \cdot \mathbf{1} - 2 q_t \frac1{t^2} \sum_{j=1}^t s_j \cdot k_i + \frac1{t^2} \sum_{j=1}^t s_j \cdot k_i^2
$$

$$
a_{t,i} = q_t^2 (\sum_{j=1}^t \exp(s_j))^{-1} \cdot \exp(s_i) - 2 q_t (\sum_{j=1}^t \exp(s_j))^{-1} \cdot \exp(s_i) k_i + (\sum_{j=1}^t \exp(s_j))^{-1} \cdot \exp(s_i) k_i^2 \\ + [q_t^2 \frac1{t^2} \sum_{j=1}^t s_j - \frac1t q_t^2] \cdot \mathbf{1} + [2 \frac1t q_t - 2 q_t \frac1{t^2} \sum_{j=1}^t s_j] \cdot k_i + [\frac1{t^2} \sum_{j=1}^t s_j - \frac1t \mathbf{1}] \cdot k_i^2 -q_t^2 \frac1t \cdot s_i + 2 q_t \frac1t \cdot s_i k_i - \frac1t \mathbf{1} \cdot s_i k_i^2
$$

### Prefix-product Sigmoid

- Full

$$
q_t \prod_{j=1}^t \sigma(s_j) \cdot (\prod_{j=1}^i \sigma(s_j))^{-1} k_i 
$$

- 0th-order

$$
q_t (\frac12)^t \cdot (\frac12)^{-i} k_i
$$

- 1st-order

$$
q_t \frac12 (\frac12)^t \sum_{j=1}^t s_j \cdot (\frac12)^{-i} k_i - q_t \frac12 (\frac12)^t \cdot \sum_{j=1}^i s_j (\frac12)^{-i} k_i 
$$

- High-order (Matrix form)

$$
q_t \prod_{j=1}^t \sigma(s_j) \cdot (\prod_{j=1}^i \sigma(s_j))^{-1} k_i - q_t [(\frac12)^t + \frac12 (\frac12)^t \sum_{j=1}^t s_j] \cdot (\frac12)^{-i} k_i + q_t \frac12 (\frac12)^t \cdot \sum_{j=1}^i s_j (\frac12)^{-i} k_i
$$

### Prefix-product Sigmoid * Softmax (Same Input)

- Full

$$
q_t \frac{\prod_{j=1}^t \sigma(s_j)}{\sum_{j=1}^t \exp(s_j)} \cdot \frac{\exp(s_i)}{\prod_{j=1}^i \sigma (s_j)} k_i
$$

- 0th-order

$$
q_t \frac1t (\frac12)^t \cdot (\frac12)^{-i} k_i
$$

- 1st-order

$$
q_t \frac1t (\frac12)^t (\frac12)^{-i} [s_i - \frac1t \sum_{j=1}^t s_j + \frac12 (\sum_{j=1}^t s_j - \sum_{j=1}^i s_j)] k_i \\
= q_t \frac1t (\frac12)^t \cdot s_i (\frac12)^{-i} k_i + q_t (\frac12 - \frac1t) \frac1t (\frac12)^t \sum_{j=1}^t s_j \cdot (\frac12)^{-i} k_i - q_t \frac12 \frac1t (\frac12)^t \cdot \sum_{j=1}^i s_j (\frac12)^{-i} k_i
$$

- High-order (Matrix form)
$$
q_t \frac{\prod_{j=1}^t \sigma(s_j)}{\sum_{j=1}^t \exp(s_j)} \cdot \frac{\exp(s_i)}{\prod_{j=1}^i \sigma (s_j)} k_i \\ - q_t [\frac1t (\frac12)^t (1 + (\frac12 - \frac1t) \sum_{j=1}^t s_j)] \cdot (\frac12)^{-i} k_i \\ - q_t \frac1t (\frac12)^t \cdot s_i (\frac12)^{-i} k_i \\ + q_t \frac12 \frac1t (\frac12)^t \cdot \sum_{j=1}^i s_j (\frac12)^{-i} k_i
$$

### Prefix-product Sigmoid * Softmax (Different Input)

- Full

$$
q_t \frac{\prod_{j=1}^t \sigma(s_j)}{\sum_{j=1}^t \exp(p_j + \log \Delta_j)} \cdot \frac{\exp(p_i + \log \Delta_i)}{\prod_{j=1}^i \sigma (s_j)} k_i
$$

- 0th-order

$$
q_t \frac1t (\frac12)^t \cdot (\frac12)^{-i} k_i
$$

- 1st-order

$$
q_t \frac1t (\frac12)^t (\frac12)^{-i} [p_i - \frac1t \sum_{j=1}^t p_j + \frac12 \sum_{j=1}^t s_j - \frac12 \sum_{j=1}^i s_j] k_i \\ 
= q_t \frac1t (\frac12)^t \cdot (\frac12)^{-i} p_i k_i + q_t \frac1t (\frac12)^t [\frac12 \sum_{j=1}^t s_j - \frac1t \sum_{j=1}^t p_j] \cdot (\frac12)^{-i} k_i - q_t \frac1t (\frac12)^t \cdot \frac12 \sum_{j=1}^i s_j (\frac12)^{-i} k_i 
$$

- High-order (Matrix form)

$$
q_t \frac{\prod_{j=1}^t \sigma(s_j)}{\sum_{j=1}^t \exp(p_j)} \cdot \frac{\exp(p_i)}{\prod_{j=1}^i \sigma (s_j)} k_i \\ - q_t [\frac1t (\frac12)^t (1 + \frac12 \sum_{j=1}^t s_j - \frac1t \sum_{j=1}^t p_j)] \cdot (\frac12)^{-i} k_i \\ - q_t \frac1t (\frac12)^t \cdot (\frac12)^{-i} p_i k_i \\ + q_t \frac1t (\frac12)^t \cdot \frac12 (\frac12)^{-i}  \sum_{j=1}^i s_j k_i 
$$


## Generalized Form

$$
o_t = \sum_{i=1}^t x_i M_{t,i} = \sum_{i=1}^t x_i W_v \; g(\{x_j\}_{j=1}^t, t,i) \; W_o
$$

KAK cartan decomposition:

$$
\text{SO}(2n) \cdot \text{SPD}(2n) \cdot \text{SO}(2n)
$$

$$
\text{Attn, RoPE: } g \leftarrow \hat{q}_t [R_t (\| q_t \|\| k_i \|) I R_i^\top] \hat{k}_i
$$

$$
\text{Linear RNN, Mamba: } g \leftarrow \frac{\prod_{j=1}^t A_j}{\prod_{j=1}^i A_j} = \prod_{j=i+1}^t A_j
$$

$$
\text{LRU, LPE: } g \leftarrow P \frac{\prod_{j=1}^t D_j}{\prod_{j=1}^i D_j} P^{-1}
$$

There's a comparison mechanisms used in this group isometry generator. 

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

We then define a certain function $s(\tau)$, where $\tau \in [1,t]$, and consider the integral:

$$
\int_{j+1}^t e^{s(t) - s(\tau)} d\tau
$$

$$
\exp(\int_{i}^t \ln s(\tau) d\tau) = \exp(\int_{1}^t \ln s(\tau) d\tau - \int_{1}^i \ln s(\tau) d\tau)
$$

Discretization:

$$
\exp(\sum_{j=1}^t \ln \sigma(s_j) \Delta_j - \sum_{j=1}^i \ln \sigma(s_j) \Delta_j) = \frac{\exp(\sum_{j=1}^t \ln \sigma(s_j) \Delta_j)}{\exp(\sum_{j=1}^i \ln \sigma(s_j) \Delta_j)}
$$

Prefix:

$$
\Delta_j = \exp(u_j) (\sum_{j=1}^t \exp(u_j))^{-1}
$$

$$
\exp(\sum_{j=1}^t \ln \sigma(s_j) \exp(u_j) (\sum_{j=1}^t \exp(u_j))^{-1} - \sum_{j=1}^i \ln \sigma(s_j) \exp(u_j) (\sum_{j=1}^t \exp(u_j))^{-1}) \\ = \exp((\sum_{j=1}^t \exp(u_j))^{-1} [\sum_{j=1}^t \ln \sigma(s_j) \exp(u_j)  - \sum_{j=1}^i \ln \sigma(s_j) \exp(u_j)])
$$

- 0th order term

  $$
  R_{t,i}^{(0)}=\Bigl(\tfrac12\Bigr)^{\displaystyle\sum_{j=i+1}^{t}\!\Delta_j} = \frac{(\tfrac12)^{\sum_{j=1}^{t}\!\Delta_j}}{(\tfrac12)^{\sum_{j=1}^{i}\!\Delta_j}}
  $$

- 1st order term

  $$
  R_{t,i}^{(1)}
    =\Bigl(\tfrac12\Bigr)^{\displaystyle\sum_{j=i+1}^{t}\!\Delta_j}
     \;\cdot\;
     \frac12\!\sum_{j=i+1}^{t}\!\Delta_j s_j
     = \frac12 \frac{(\tfrac12)^{\sum_{j=1}^{t}\!\Delta_j}}{(\tfrac12)^{\sum_{j=1}^{i}\!\Delta_j}} (\sum_{j=1}^{t} \Delta_j s_j - \sum_{j=1}^{i} \Delta_j s_j)
  $$


Riemann Softmax:

$$
\frac{\exp(s_i) \Delta_i}{\sum_{j=1}^t \exp(s_j) \Delta_j}
$$

Prefix delta:

$$
\frac{\exp(s_i) \frac{\Pi_t}{\Pi_i}}{\sum_{j=1}^t \exp(s_j) \frac{\Pi_t}{\Pi_j}} = \frac{\exp(s_i) (\Pi_i)^{-1}}{\sum_{j=1}^t \exp(s_j) (\Pi_j)^{-1}} = \frac{\exp(s_i - \ln \Pi_i)}{\sum_{j=1}^t \exp(s_j - \ln \Pi_j)}
$$

- 0th order term

$$
\frac{\Delta_i}{\sum_{j=1}^t \Delta_j}
$$

- 1st order term

$$
\frac{\Delta_i}{\sum_{j=1}^t \Delta_j} \cdot [s_i - \bar{s}_t] = \frac{\Delta_i}{\sum_{j=1}^t \Delta_j} \cdot [s_i - \frac{\sum_{j=1}^t \Delta_j s_j}{\sum_{j=1}^t \Delta_j}]
$$

- Full - 0th - 1st

$$
q_t (\sum_{j=1}^t \exp(s_j) \Delta_j)^{-1} \cdot (\exp(s_i) \Delta_i) k_i + q_t (\sum_{j=1}^t \Delta_j)^{-1} (\sigma_t^1 \frac{\sum_{j=1}^t \Delta_j s_j}{\sum_{j=1}^t \Delta_j} - 1) \cdot \Delta_i k_i - q_t \sigma_t^1 (\sum_{j=1}^t \Delta_j)^{-1} \cdot \Delta_i s_i k_i 
$$


Double softmax:

$$
q_t [\frac{\exp(s_i) \Delta_i}{\sum_{j=1}^t \exp(s_j) \Delta_j} x_i^\top x_i ] k_i^\top
$$

$$
q_t [\sum_{k=1}^t \frac{\exp(s_k) \Delta_k}{\sum_{j=1}^t \exp(s_j) \Delta_j} x_k^\top x_i ] k_i^\top = q_t (\sum_{j=1}^t \exp(s_j) \Delta_j)^{-1} [\sum_{k=1}^t \exp(s_k) \Delta_k x_k^\top ] x_i k_i^\top
$$


$$
q_t [\sum_{j=1}^t (x_i x_j^\top) x_j^\top x_j] k_i^\top = q_t [\sum_{j=1}^t x_j^\top x_j (x_j x_i^\top) I ] k_i^\top
$$

$$
q_t [\sum_{j=1}^t \alpha_{j,i} D_j] k_i^\top \\ = q_t [\sum_{j=1}^t D_j x_j^\top \frac{\exp(s_j) \Delta_j}{\sum_{k=1}^t \exp(s_k) \Delta_k} x_i ] k_i^\top = q_t (\sum_{k=1}^t \exp(s_k) \Delta_k)^{-1} [\sum_{j=1}^t D_j x_j^\top \exp(s_j) \Delta_j] x_i k_i^\top
$$

$$
v_i \otimes k_i [\sum_{j=1}^t \exp(s_j) \Delta_j q_j^\top \otimes  D_j ]  (\sum_{k=1}^t \exp(s_k) \Delta_k)^{-1}
$$

Combining Two Types:

$$
\sum_{i=1}^t v_i \frac{\exp(s_i)}{\sum_{j=1}^t \exp(s_j)} q_t D k_i^\top = (\sum_{j=1}^t \exp(s_j))^{-1} q_t \sum_{i=1}^t D k_i^\top v_i \exp(s_i) 
$$

$$
[q_t (\sum_{j=1}^t \exp(u_j))^{-1} \sum_{i=1}^t \exp(u_i) k_i^\top v_i \exp(s_i)] (\sum_{j=1}^t \exp(s_j))^{-1} 
$$

$$
x_i W_v K_i R_i^\top \Lambda_i^{-1} [k_i R_i^\top \Lambda_i^{-1} \Lambda_t R_t q_t^\top I] \Lambda_t R_t Q_t W_o
$$

$$
x_i W_v K_i R_i^\top \frac{\exp(\Lambda_i) \Delta_{t,i}}{\sum_{j=1}^t \exp(\Lambda_j) \Delta_{t,j}} R_t Q_t W_o
$$

$$
x_i W_v R_i^\top \frac{\exp(\Lambda_i) k_i^\top q_t}{[\sum_{j=1}^t \exp(\Lambda_j) k_j^\top] q_t} R_t W_o = x_i W_v R_i^\top \exp(\Lambda_i) \frac{k_i^\top q_t}{[\sum_{j=1}^t \exp(\Lambda_j) \otimes k_j^\top] q_t}  R_t W_o
$$

$$
q_t \frac{q_t \sigma(U_i) k_i^\top \exp(S_i) }{\sum_{j=1}^t q_t \sigma(U_j) k_j^\top \exp(S_j)} k_i^\top
$$

- Recursive Bilinear Summarizier:

$$
\Delta_{t,i}^{(n)} = q_t^{(n)} \frac{\exp(S_i^{(n)}) \Delta_{t,i}^{(n+1)}}{\sum_{j=1}^t \exp(S_j^{(n)}) \Delta_{t,j}^{(n+1)}} k_i^{(n)\top} = [q_t^{(n)} \odot (\sum_{j=1}^t (S_j^{(n)}) \Delta_{t,j}^{(n+1)})^{-1}] [\Delta_{t,i}^{(n+1)} I] [\exp(S_j^{(n)}) k_i^{(n)\top}], \, n \in \{1, \cdots, N+1\}
$$

$$
\Delta_{t,i}^{(\mathrm{N+1})} = \delta_i
$$

$$
o_t = \sum_{i=1}^t \Delta_{t,i}^{(1)} v_i
$$

- Efficient Calculation:

$$
\Delta_{t,i}^{(N)} =  q_t^{(N)} \frac{\exp(S_i^{(N)}) \delta_i}{\sum_{j=1}^t \exp(S_j^{(N)}) \delta_j} k_i^{(N)\top} = [q_t^{(N)} (\sum_{j=1}^t \exp(S_j^{(N)}) \delta_j)^{-1}] \cdot [\exp(S_i^{(N)}) \delta_i k_i^{(N)\top}]
$$

$$
\Delta_{t,i}^{(N-1)} = q_t^{(N-1)} \frac{\exp(S_i^{(N-1)}) \Delta_{t,i}^{(N)}}{\sum_{j=1}^t \exp(S_j^{(N-1)}) \Delta_{t,j}^{(N)}} k_i^{(N-1)\top} = [q_t^{(N-1)} (\sum_{j=1}^t l_t^{(N)} \cdot r_j^{(N)} \otimes \exp(S_j^{(N-1)}) )^{-1} \otimes l_t^{(N)}] \cdot [r_i^{(N)} \otimes \exp(S_i^{(N-1)}) k_i^{(N-1)\top}]
$$

Thus, we can let:

$$
\Delta_{t,i}^{(n)} = l_t^{(n)} \cdot r_i^{(n)}
$$

$$
l_t^{(n)} = q_t^{(n)} \oslash (\sum_{j=1}^t l_t^{(n+1)} \cdot r_j^{(n+1)} \otimes \exp(S_j^{(n)})), \, r_i^{(n)} = \exp(S_i^{(n)}) k_i^{(n)\top}
$$

With $l_t^{N+1} = \bm{1}$ and $r_i^{N+1} = \frac1d \delta_i \bm{1}$, we now have:

$$
\Delta_{t,i}^{(N-1)} = [l_t^{(N-1)} \otimes l_t^{(N)} \otimes l_t^{(N+1)}] \cdot [r_i^{(N+1)} \otimes r_i^{(N)} \otimes r_i^{(N-1)}] = [l_t^{(N-1)} \otimes l_t^{(N)}] \cdot [r_i^{(N)} \otimes r_i^{(N-1)}] \delta_i
$$

Then,

$$
\Delta_{t,i}^{(n)} = \bigotimes_{k=n}^{N+1} l_t^{(k)} \cdot \bigotimes_{k=N+1}^n r_i^{(k)}
$$

The output would be (Linear-time):

$$
o_t = \sum_{i=1}^t \Delta_{t,i}^{(1)} v_i = \sum_{i=1}^t [\bigotimes_{k=1}^{N+1} l_t^{(k)}] \cdot [\bigotimes_{k=N+1}^1 r_i^{(k)} \otimes v_i] = [\bigotimes_{k=1}^{N+1} l_t^{(k)}] \cdot \sum_{i=1}^t [\bigotimes_{k=N+1}^1 r_i^{(k)} \otimes v_i]
$$

Since $(a \otimes b) \cdot (c \otimes d) = (a \cdot c) (b \cdot d)$, the output would be (quadratic-time):

$$
o_t = \sum_{i=1}^t \prod_{k=1}^{N+1} (l_t^{(k)} r_i^{(k)}) v_i
$$

Time Complexity:

$$
O(T [\frac{D}{hN}]^{N} \frac{D}{h}) = O(T [\frac{D^{N+1}}{h^{N+1} N^N}])
$$

Element-wise:

$$
\Delta_{t,i}^{(n)} = Q_t^{(n)} \frac{\exp(S_i^{(n)}) \Delta_{t,i}^{(n+1)}}{\sum_{j=1}^t \exp(S_j^{(n)}) \Delta_{t,j}^{(n+1)}} K_i^{(n)} = [Q_t^{(n)} L_{t}^{(n+1)} \oslash (L_{t}^{(n+1)} \sum_{j=1}^t \exp(S_j^{(n)}) R_{j}^{(n+1)})] \odot [\exp(S_i^{(n)}) R_{i}^{(n+1)} K_i^{(n)}] = L_t^{(n)} \odot R_i^{(n)}
$$

$$
\Delta_{t,i}^{(n)} = [Q_t^{(n)} \oslash (\sum_{j=1}^t \exp(S_j^{(n)}) R_{j}^{(n+1)})] \odot [\exp(S_i^{(n)}) R_{i}^{(n+1)} K_i^{(n)}]
$$

$$
q_t \frac{\prod_{n=1}^N \exp(S_i^{(n)}) \Delta_{i}^{(n)}}{\sum_{j=1}^t \prod_{n=1}^N \exp(S_j^{(n)}) \Delta_{j}^{(n)}} k_i^\top
$$

Hyperbolic:

$$
\frac1t [q_t \frac{\exp(S_i) \Delta_i}{\sum_{j=1}^t [\exp(S_t) + \exp(S_i)] \Delta_i} k_i^\top - q_t \frac{\exp(S_t) \Delta_i}{\sum_{j=1}^t [\exp(S_t) + \exp(S_i)] \Delta_i} k_i^\top + 1] \\ = \frac1t [q_t \frac{\exp(S_i) \Delta_i}{\exp(S_t) \sum_{j=1}^t \Delta_i + \sum_{j=1}^t \exp(S_i) \Delta_i} k_i^\top - q_t \frac{\exp(S_t) \Delta_i}{\exp(S_t) \sum_{j=1}^t \Delta_i + \sum_{j=1}^t \exp(S_i) \Delta_i} k_i^\top + q_t \Delta_i k_i^\top]
$$

$$
q_t \frac{\exp(S_i) \Delta_i^2}{\exp(S_t) \sum_{j=1}^t \Delta_t + \sum_{j=1}^t \exp(S_i) \Delta_i} k_i^\top - q_t \frac{\exp(S_t) \Delta_t \Delta_i}{\exp(S_t) \sum_{j=1}^t \Delta_t + \sum_{j=1}^t \exp(S_i) \Delta_i} k_i^\top + q_t \Delta_i k_i^\top
$$


$O(T(D/ho)^o)$ O(T^2D)