# Basic knowledge
## Model size
For a 10B full precision model. It has $10e10$ parameters and each parameter has 32 bit (4 bytes). Therefore, its size is $10^{10} \times 4 /(1024)^3 \approx 40\text{Gb}$
If it's a half precision model: Then each parameter is 16 bit (2 bytes). Therefore, the size is ~20Gb.

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
