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