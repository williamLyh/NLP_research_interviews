# NLP_research_interviews

## Generative AutoEncoder Models Comparison
### VAE vs Diffusion model

## Boosting

## Transformer-based models
### Attention vs LSTM
RNN is hard to parallelize because of its sequential nature. CNN can parallelize, but each filter/kernel can only see limited vectros. Filters from higher layer have potential to see more vectors. Self-attention layer has Q, K, V and token at any position can attend other tokens at any positions. Self-attention has similar idea with CNN, but has more parameters and therefore is more flexible and needs more training data, e.g. ViT has 300M training pictures, but if only trained on ImageNet, its performance is worse than CNN. 

## NLG Metrics
- BLEU
- ROUGE
