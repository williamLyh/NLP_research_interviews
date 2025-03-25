# Basic knowledge
## Model size
For a 10B full precision model. It has $10e10$ parameters and each parameter has 32 bit (4 bytes). Therefore, its size is $10^{10} \times 4 /(1024)^3 \approx 40\text{Gb}$
If it's a half precision model: Then each parameter is 16 bit (2 bytes). Therefore, the size is ~20Gb.

# Model Parameters
For a decoder transformer, Batch size B, sequence len T, hidden dimension D, head number H.  
- LayerNorm parameters: $2D$, scale + bias.
- Multi-Head Attention: $4D^2$, Query, Key, Value and Output projection.
- FFN: $8D^2 + 4D$, Two layer MLP, the middle hidden state is usually $4D$, and the bias is also $4D$. Therefore $2\times D \times 4D + 4D$.
Above is the total parameter number of a decoder block.

# KV Cache (per token)
- Standard MHA: $2D$
- MQA: $2D/H$
- GQA: $2D/G$



# Training
## RLHF
3 steps: 1) Pretrain+(SFT) a LM. 2) Train a Reward Model (RM) and 3) Train LM with RL.
- RM training is what makes RLHF different from RL. RM can be another SFT-ed LM or Pretrained-LM 
with different or similar sizes.
Rewards can be human scores or ranking. Ranking can compare outputs from different models and has 
less noise and subjective preference.
- RM training general datasize is 50K.
- Applying RL to LM has been difficult. **Proximal Policy Optimization (PPO)** is what the current popular 
solution. Having a reference LM and an active LM. Compare the KL divergence between their outputs as 
part of the penalty to constrain the active LM not deviate too much.
[Reference link](https://huggingface.co/docs/trl/main/en/quickstart)
- DPO

## Context Length
More context length will lead to: 1) More GPU memory, 2) Time and space complexity and 3) long-term dependency.
Methods:
- Sparse attention mechanism
- ALiBi position encoding
- FlashAttention
- Retrieval-based

## Positional Embedding
1. Absolute Positional Embeddings: Assigns a unique vector to each position in the sequence.
    - Pros: Simple to implement
    - Cons: Limited to a fixed maximum sequence length; doesn't generalize well to unseen positions

2. Sinusoidal Positional Encodings: Uses sine and cosine functions of different frequencies to encode positions.
    - Pros: Can theoretically handle arbitrary sequence lengths; no additional parameters to learn
    - Cons: May not capture positional information as effectively as learned methods

3. Relative Positional Embeddings: Encodes relative distances between tokens rather than absolute positions.
    - Pros: Can handle longer sequences; better generalization to unseen lengths
    - Cons: Can be computationally expensive

4. Rotary Positional Embeddings (RoPE): Applies rotation to token embeddings based on their position
    - Pros: Theoretically unlimited sequence length; good extrapolation to unseen lengths; efficiently captures both absolute and relative positions.
    - Cons: Slightly more complex to implement than absolute embeddings.
    - Used by Llama models


## Training parallel
- Data Parallel

- Model Parallel
    - Pipeline parallel:
        - In pipeline parallelism, the model is split into stages, and each stage is placed on a different GPU.
        - The computation is done in a sequential manner, with different GPUs processing different stages in a pipeline.
        - This technique is often used to handle large models by distributing parts of the model that can be processed sequentially, such as different layers in a neural network or different components of a transformer model.
    - Tensor parallel: 
        - In tensor parallelism, the model's individual tensor operations (such as matrix multiplications) are split across multiple GPUs.
        - Rather than splitting the model into entire layers or stages, tensor parallelism divides up the work within each layer or operation.
        - This allows for parallel computation of large tensor operations, which is especially useful for handling large matrix computations or weight matrices that cannot fit into a single GPU's memory.
