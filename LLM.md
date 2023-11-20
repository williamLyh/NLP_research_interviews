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
